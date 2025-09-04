import cupy as cp
import gmsh
import math


class SpatialEvaluator:
    def __init__(self, mesh, sensor_points):
        self.mesh = mesh

        if not gmsh.isInitialized():
            gmsh.initialize()

        self._element_offset, _ = gmsh.model.mesh.getElementsByType(4)

        # Cache inverse Vandermonde directly on GPU
        self.invV = cp.asarray(
            self.mesh.reference_element_operators.inverse_vandermonde_3d,
            dtype=float
        )

        # Precompute cache: (element, phi_vector_on_gpu)
        self.sensor_cache = []
        for (x, y, z) in sensor_points:
            element = self.get_element(x, y, z)
            r, s, t = self._map_to_reference_tetrahedron(x, y, z, element)
            phi_vector = self._compute_phi_vector(r, s, t)   # returns cp.array
            self.sensor_cache.append((element, phi_vector))

    def get_element(self, x, y, z):
        dim = 3
        element = (
            gmsh.model.mesh.getElementByCoordinates(x, y, z, dim)[0]
            - self._element_offset[0]
        )
        return int(element)

    def _map_to_reference_tetrahedron(self, x, y, z, cell):
        cell_to_vertices = self.mesh.cell_to_vertices
        vx, vy, vz = self.mesh.x_vertex, self.mesh.y_vertex, self.mesh.z_vertex
        va, vb, vc, vd = cell_to_vertices[cell, :4]

        J = cp.array([
            [vx[vb] - vx[va], vx[vc] - vx[va], vx[vd] - vx[va]],
            [vy[vb] - vy[va], vy[vc] - vy[va], vy[vd] - vy[va]],
            [vz[vb] - vz[va], vz[vc] - vz[va], vz[vd] - vz[va]],
        ], dtype=float)

        b = cp.array([
            2*x + vx[va] - vx[vb] - vx[vc] - vx[vd],
            2*y + vy[va] - vy[vb] - vy[vc] - vy[vd],
            2*z + vz[va] - vz[vb] - vz[vc] - vz[vd],
        ], dtype=float)

        rst = cp.linalg.solve(J, b)
        return tuple(rst)

    def _compute_phi_vector(self, r, s, t):
        n = self.mesh.reference_element.n
        eval_basis = self.mesh.reference_element.eval_3d_basis_function

        phi_list = []
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    m = (
                        1
                        + (11 + 12*n + 3*n**2) * i / 6
                        + (2*n + 3) * j / 2
                        + k
                        - (2 + n) * i**2 / 2
                        - i*j
                        - j**2 / 2
                        + i**3 / 6
                    )
                    m = math.ceil(m - 1)
                    phi = eval_basis([r], [s], [t], i, j, k)[0]
                    if len(phi_list) <= m:
                        phi_list.extend([0.0] * (m - len(phi_list) + 1))
                    phi_list[m] = phi

        return cp.array(phi_list, dtype=float)   # stays on GPU forever

    def eval_all_sensors(self, field):
        """
        Evaluate the field at all pre-defined sensor points.
        Assumes `field` is already a CuPy array.
        """
        results = cp.zeros(len(self.sensor_cache), dtype=float)

        # Group sensors by element
        sensors_by_elem = {}
        for idx, (elem, phi_vec) in enumerate(self.sensor_cache):
            sensors_by_elem.setdefault(elem, []).append((idx, phi_vec))

        for elem, sensor_list in sensors_by_elem.items():
            values = field[:, elem]            # already CuPy
            weights = self.invV @ values       # GPU matvec
            for idx, phi_vec in sensor_list:
                results[idx] = weights @ phi_vec   # GPU dot

        return results
