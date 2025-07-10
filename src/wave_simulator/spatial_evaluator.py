import numpy as np
import math
import gmsh

class SpatialEvaluator:
    def __init__(self, mesh):
        self.mesh = mesh
        # Gmsh element offset for tetrahedra
        if not gmsh.isInitialized():
            gmsh.initialize()
        self._element_offset, _ = gmsh.model.mesh.getElementsByType(4)

    def get_element(self, x, y, z):
        """Find which element corresponds to a point in the mesh."""
        dim = 3
        element = gmsh.model.mesh.getElementByCoordinates(x, y, z, dim)[0] - self._element_offset[0]
        return element

    def eval_at_point(self, x, y, z, field):
        """Evaluate a given field at any point in the domain."""
        element = self.get_element(x, y, z)
        values = field[:, element]

        invV = self.mesh.reference_element_operators.inverse_vandermonde_3d
        weights = invV @ values

        r, s, t = self._map_to_reference_tetrahedron(x, y, z, element)

        solution = 0.0
        n = self.mesh.reference_element.n
        eval_basis = self.mesh.reference_element.eval_3d_basis_function

        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    m = (
                        1 + (11 + 12*n + 3*n**2) * i / 6 + (2*n + 3) * j / 2 + k
                        - (2 + n) * i**2 / 2 - i * j - j**2 / 2 + i**3 / 6
                    )
                    m = math.ceil(m - 1)
                    phi = eval_basis([r], [s], [t], i, j, k)
                    solution += weights[m] * phi

        return solution

    def _map_to_reference_tetrahedron(self, x, y, z, cell):
        """
        Maps a point (x, y, z) in physical space to reference coordinates (r, s, t)
        for a tetrahedral element defined by its vertices.
        """
        cell_to_vertices = self.mesh.cell_to_vertices

        vx = self.mesh.x_vertex
        vy = self.mesh.y_vertex
        vz = self.mesh.z_vertex

        va = cell_to_vertices[cell, 0]
        vb = cell_to_vertices[cell, 1]
        vc = cell_to_vertices[cell, 2]
        vd = cell_to_vertices[cell, 3]

        J = np.array([
            [vx[vb] - vx[va], vx[vc] - vx[va], vx[vd] - vx[va]],
            [vy[vb] - vy[va], vy[vc] - vy[va], vy[vd] - vy[va]],
            [vz[vb] - vz[va], vz[vc] - vz[va], vz[vd] - vz[va]]
        ])

        b = np.array([
            2*x + vx[va] - vx[vb] - vx[vc] - vx[vd],
            2*y + vy[va] - vy[vb] - vy[vc] - vy[vd],
            2*z + vz[va] - vz[vb] - vz[vc] - vz[vd]
        ])

        rst = np.linalg.solve(J, b)
        return tuple(rst)
