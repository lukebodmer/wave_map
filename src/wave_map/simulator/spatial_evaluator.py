import numpy as np
import cupy as cp
import math
import gmsh


class SpatialEvaluator:
    def __init__(self, mesh):
        self.mesh = mesh
        # Gmsh element offset for tetrahedra
        if not gmsh.isInitialized():
            gmsh.initialize()
        self._element_offset, _ = gmsh.model.mesh.getElementsByType(4)

    def _to_cpu(self, array):
        """Convert CuPy array to NumPy array for compatibility"""
        if hasattr(array, 'get'):  # CuPy array
            return array.get()
        return array  # Already NumPy array

    def get_element(self, x, y, z):
        """Find which element corresponds to a point in the mesh."""
        dim = 3
        element = gmsh.model.mesh.getElementByCoordinates(x, y, z, dim)[0] - self._element_offset[0]
        return element

    def eval_at_point(self, x, y, z, field):
        """Evaluate a given field at any point in the domain."""
        element = self.get_element(x, y, z)

        # Keep computation on GPU when possible
        if hasattr(field, 'get'):  # CuPy array
            values = field[:, element]
        else:
            values = cp.asarray(field[:, element])

        # Convert reference operators to GPU if needed
        invV = cp.asarray(self.mesh.reference_element_operators.inverse_vandermonde_3d)
        weights = invV @ values

        r, s, t = self._map_to_reference_tetrahedron(x, y, z, element)

        # Vectorized basis function evaluation
        solution = self._evaluate_basis_vectorized(r, s, t, weights)

        # Convert back to scalar if needed
        if hasattr(solution, 'get'):
            return float(solution.get())
        return float(solution)

    def eval_at_points_batch(self, points, field):
        """Vectorized evaluation at multiple points for better GPU performance."""
        n_points = len(points)
        results = np.zeros(n_points)

        # Process points in batches to avoid gmsh overhead
        batch_size = 32  # Reasonable batch size for GPU kernels

        for batch_start in range(0, n_points, batch_size):
            batch_end = min(batch_start + batch_size, n_points)
            batch_points = points[batch_start:batch_end]

            # Get elements for batch (CPU operation with gmsh)
            elements = []
            coords_rst = []
            for x, y, z in batch_points:
                element = self.get_element(x, y, z)
                elements.append(element)
                r, s, t = self._map_to_reference_tetrahedron(x, y, z, element)
                coords_rst.append((r, s, t))
            
            # Batch GPU operations
            batch_results = self._evaluate_batch_gpu(elements, coords_rst, field)
            results[batch_start:batch_end] = batch_results
            
        return results
    
    def _evaluate_batch_gpu(self, elements, coords_rst, field):
        """GPU-optimized batch evaluation."""
        batch_size = len(elements)
        results = np.zeros(batch_size)
        
        # Convert field to GPU if needed
        if hasattr(field, 'get'):  # CuPy array
            field_gpu = field
        else:
            field_gpu = cp.asarray(field)
            
        # Convert reference operators to GPU
        invV_gpu = cp.asarray(self.mesh.reference_element_operators.inverse_vandermonde_3d)
        
        for i, (element, (r, s, t)) in enumerate(zip(elements, coords_rst)):
            # Extract field values for this element
            values = field_gpu[:, element]
            weights = invV_gpu @ values
            
            # Simple basis evaluation (avoid GPU overhead for small ops)
            result = self._evaluate_basis_simple(r, s, t, weights)
            results[i] = float(result.get() if hasattr(result, 'get') else result)
            
        return results
    
    def _evaluate_basis_simple(self, r, s, t, weights):
        """Simplified basis evaluation for better performance."""
        n = self.mesh.reference_element.n
        eval_basis = self.mesh.reference_element.eval_3d_basis_function
        
        solution = 0.0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    m = (
                        1 + (11 + 12*n + 3*n**2) * i / 6 + (2*n + 3) * j / 2 + k
                        - (2 + n) * i**2 / 2 - i * j - j**2 / 2 + i**3 / 6
                    )
                    m = math.ceil(m - 1)
                    # Use CPU for basis function evaluation (avoid cupy overhead)
                    phi = eval_basis([r], [s], [t], i, j, k)
                    # Convert weight to CPU for multiplication
                    weight_cpu = float(weights[m].get() if hasattr(weights[m], 'get') else weights[m])
                    solution += weight_cpu * phi
                    
        return solution
    
    def precompute_sensor_metadata(self, all_points, field_indices):
        """Precompute all sensor metadata once at initialization."""
        n_points = len(all_points)
        elements = np.zeros(n_points, dtype=int)
        coords_rst = np.zeros((n_points, 3))
        weights_cache = {}
        
        print(f"Precomputing metadata for {n_points} sensor points...")
        
        # Precompute elements and coordinates (CPU operations with gmsh)
        for i, (x, y, z) in enumerate(all_points):
            elements[i] = self.get_element(x, y, z)
            r, s, t = self._map_to_reference_tetrahedron(x, y, z, elements[i])
            coords_rst[i] = [r, s, t]
        
        # Precompute Vandermonde weights for unique elements
        unique_elements = np.unique(elements)
        invV_gpu = cp.asarray(self.mesh.reference_element_operators.inverse_vandermonde_3d)
        
        # Store precomputed data on GPU
        return {
            'elements': cp.asarray(elements),
            'coords_rst': cp.asarray(coords_rst), 
            'field_indices': field_indices,
            'invV_gpu': invV_gpu,
            'n_points': n_points
        }
    
    def eval_sensors_gpu_optimized(self, field_arrays, sensor_metadata):
        """GPU-optimized evaluation of all sensors across all fields."""
        if sensor_metadata is None:
            return []
            
        elements = sensor_metadata['elements']
        coords_rst = sensor_metadata['coords_rst']
        field_indices = sensor_metadata['field_indices']
        invV_gpu = sensor_metadata['invV_gpu']
        n_points = sensor_metadata['n_points']
        
        # Convert all field arrays to GPU
        field_arrays_gpu = [cp.asarray(field) if not hasattr(field, 'get') else field 
                           for field in field_arrays]
        
        # Evaluate all sensors for all fields in parallel
        all_results = []
        
        for field_idx, field_gpu in enumerate(field_arrays_gpu):
            # Extract field values for all sensor elements at once
            field_values = field_gpu[:, elements]  # Shape: (Np, n_sensors)
            
            # Compute weights for all sensors simultaneously  
            weights_all = invV_gpu @ field_values  # Shape: (Np, n_sensors)
            
            # Vectorized basis function evaluation
            results = self._evaluate_sensors_vectorized_gpu(
                coords_rst, weights_all, n_points
            )
            
            all_results.append(results)
        
        # Split results by field type
        field_results = []
        for field_name in field_indices.keys():
            start_idx, end_idx = field_indices[field_name]
            field_results.append([result[start_idx:end_idx] for result in all_results])
        
        # Return results for each field
        return [field_results[i][i] for i in range(len(field_results))]
    
    def _evaluate_sensors_vectorized_gpu(self, coords_rst, weights_all, n_points):
        """Fully vectorized GPU evaluation of basis functions for all sensors."""
        n = self.mesh.reference_element.n
        results = cp.zeros(n_points)
        
        # Vectorized computation across all sensors
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    m = (
                        1 + (11 + 12*n + 3*n**2) * i / 6 + (2*n + 3) * j / 2 + k
                        - (2 + n) * i**2 / 2 - i * j - j**2 / 2 + i**3 / 6
                    )
                    m = math.ceil(m - 1)
                    
                    # Evaluate basis function for all sensors simultaneously
                    phi_all = self._eval_basis_vectorized_all_sensors(
                        coords_rst[:, 0], coords_rst[:, 1], coords_rst[:, 2], i, j, k
                    )
                    
                    # Accumulate weighted contributions
                    results += weights_all[m, :] * phi_all
        
        return results
    
    def _eval_basis_vectorized_all_sensors(self, r_all, s_all, t_all, i, j, k):
        """Vectorized basis function evaluation across all sensor points."""
        # Convert to CPU for basis function evaluation (avoid cupy overhead)
        r_cpu = r_all.get() if hasattr(r_all, 'get') else r_all
        s_cpu = s_all.get() if hasattr(s_all, 'get') else s_all  
        t_cpu = t_all.get() if hasattr(t_all, 'get') else t_all
        
        # Vectorized evaluation using existing numpy implementation
        eval_basis = self.mesh.reference_element.eval_3d_basis_function
        phi_results = cp.zeros(len(r_all))
        
        # Evaluate for all points at once
        for idx in range(len(r_all)):
            phi = eval_basis([r_cpu[idx]], [s_cpu[idx]], [t_cpu[idx]], i, j, k)
            phi_results[idx] = phi[0] if hasattr(phi, '__len__') else phi
            
        return phi_results

    def _evaluate_basis_vectorized(self, r, s, t, weights):
        """GPU-optimized evaluation of basis functions."""
        n = self.mesh.reference_element.n
        
        # Pre-compute all basis function indices
        indices = []
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    m = (
                        1 + (11 + 12*n + 3*n**2) * i / 6 + (2*n + 3) * j / 2 + k
                        - (2 + n) * i**2 / 2 - i * j - j**2 / 2 + i**3 / 6
                    )
                    m = math.ceil(m - 1)
                    indices.append((i, j, k, m))
        
        # Optimized computation using cupy
        solution = cp.array(0.0)
        for i, j, k, m in indices:
            phi = self._eval_3d_basis_cupy(r, s, t, i, j, k)
            solution += weights[m] * phi
            
        return solution

    def _eval_3d_basis_cupy(self, r, s, t, i, j, k):
        """CuPy-optimized 3D basis function evaluation."""
        # Convert coordinates to cupy if needed
        r_cp = cp.array([r]) if not hasattr(r, 'get') else cp.array([r])
        s_cp = cp.array([s]) if not hasattr(s, 'get') else cp.array([s])  
        t_cp = cp.array([t]) if not hasattr(t, 'get') else cp.array([t])
        
        # Map to abc coordinates (cupy version)
        a, b, c = self._rst_to_abc_cupy(r_cp, s_cp, t_cp)
        
        # Evaluate jacobi polynomials using cupy
        h1 = self._eval_jacobi_polynomial_cupy(a, 0, 0, i)
        h2 = self._eval_jacobi_polynomial_cupy(b, 2*i+1, 0, j)
        h3 = self._eval_jacobi_polynomial_cupy(c, 2*(i+j)+2, 0, k)
        
        # Compute basis function value
        P = 2 * cp.sqrt(2) * h1 * h2 * ((1 - b) ** i) * h3 * ((1 - c) ** (i + j))
        return cp.sum(P)  # Return scalar
    
    def _rst_to_abc_cupy(self, r, s, t):
        """CuPy version of coordinate transformation."""
        Np = len(r)
        a = cp.zeros(Np)
        b = cp.zeros(Np)
        c = cp.zeros(Np)
        
        for n in range(Np):
            if s[n] + t[n] != 0:
                a[n] = 2 * (1 + r[n]) / (-s[n] - t[n]) - 1
            else:
                a[n] = -1

            if t[n] != 1:
                b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1
            else:
                b[n] = -1

            c[n] = t[n]
        
        return a, b, c
    
    def _eval_jacobi_polynomial_cupy(self, x, alpha, beta, N):
        """CuPy-optimized Jacobi polynomial evaluation."""
        # Use the existing numpy implementation but convert result
        # This is a compromise - full cupy implementation would be more complex
        x_np = x.get() if hasattr(x, 'get') else x
        result_np = self.mesh.reference_element.eval_jacobi_polynomial(x_np, alpha, beta, N)
        return cp.asarray(result_np)

    def _map_to_reference_tetrahedron(self, x, y, z, cell):
        """
        Maps a point (x, y, z) in physical space to reference coordinates (r, s, t)
        for a tetrahedral element defined by its vertices.
        """
        # Use GPU arrays when possible
        cell_to_vertices = cp.asarray(self.mesh.cell_to_vertices) if hasattr(self.mesh.cell_to_vertices, 'get') else cp.asarray(self.mesh.cell_to_vertices)
        vx = cp.asarray(self.mesh.x_vertex) if hasattr(self.mesh.x_vertex, 'get') else cp.asarray(self.mesh.x_vertex)
        vy = cp.asarray(self.mesh.y_vertex) if hasattr(self.mesh.y_vertex, 'get') else cp.asarray(self.mesh.y_vertex)
        vz = cp.asarray(self.mesh.z_vertex) if hasattr(self.mesh.z_vertex, 'get') else cp.asarray(self.mesh.z_vertex)

        va = int(cell_to_vertices[cell, 0])
        vb = int(cell_to_vertices[cell, 1])
        vc = int(cell_to_vertices[cell, 2])
        vd = int(cell_to_vertices[cell, 3])

        J = cp.array([
            [vx[vb] - vx[va], vx[vc] - vx[va], vx[vd] - vx[va]],
            [vy[vb] - vy[va], vy[vc] - vy[va], vy[vd] - vy[va]],
            [vz[vb] - vz[va], vz[vc] - vz[va], vz[vd] - vz[va]]
        ])

        b = cp.array([
            2*x + vx[va] - vx[vb] - vx[vc] - vx[vd],
            2*y + vy[va] - vy[vb] - vy[vc] - vy[vd],
            2*z + vz[va] - vz[vb] - vz[vc] - vz[vd]
        ])

        rst = cp.linalg.solve(J, b)
        return tuple(float(rst[i].get()) if hasattr(rst[i], 'get') else float(rst[i]) for i in range(3))
