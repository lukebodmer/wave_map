import numpy as np
from wave_simulator.mesh import Mesh3d

class LinearAcoustics:
    def __init__(self,
                 mesh: Mesh3d,
                 source_center=None,
                 source_radius=None,
                 source_amplitude=None,
                 source_frequency=None,
                 ):
        self.mesh = mesh
        self.nodes_per_cell = self.mesh.reference_element.nodes_per_cell
        self.nodes_per_face = self.mesh.reference_element.nodes_per_face
        self.num_cells = self.mesh.num_cells
        self.num_faces = self.mesh.reference_element.num_faces
        self.p = np.zeros((self.nodes_per_cell, self.num_cells), order='F')
        self.u = np.zeros((self.nodes_per_cell, self.num_cells), order='F')
        self.v = np.zeros((self.nodes_per_cell, self.num_cells), order='F')
        self.w = np.zeros((self.nodes_per_cell, self.num_cells), order='F')
        self.source_center = np.array(source_center)
        self.source_radius = source_radius
        self.source_frequency = source_frequency# Hz
        self.source_amplitude = source_amplitude
        self.source_duration = 1 / self.source_frequency
        self._locate_source_nodes()
        self.set_initial_conditions()
        # Pre-cache material properties and constants for performance
        self.flux_terms_cached = False
        self._material_properties_cached = False
        self._inv_rho = None
        self._bulk = None
        self._precompute_indices()
        self._precompute_material_properties()
        # precompute spatial derivative matrices
        Dr = self.mesh.reference_element_operators.r_differentiation_matrix
        Ds = self.mesh.reference_element_operators.s_differentiation_matrix
        Dt = self.mesh.reference_element_operators.t_differentiation_matrix

        self.Dx = np.empty((self.nodes_per_cell, self.nodes_per_cell, self.num_cells))
        self.Dy = np.empty((self.nodes_per_cell, self.nodes_per_cell, self.num_cells))
        self.Dz = np.empty((self.nodes_per_cell, self.nodes_per_cell, self.num_cells))

        for k in range(self.num_cells):
            self.Dx[:,:,k] = self.mesh.drdx[:,k] * Dr + self.mesh.dsdx[:,k] * Ds + self.mesh.dtdx[:,k] * Dt
            self.Dy[:,:,k] = self.mesh.drdy[:,k] * Dr + self.mesh.dsdy[:,k] * Ds + self.mesh.dtdy[:,k] * Dt
            self.Dz[:,:,k] = self.mesh.drdz[:,k] * Dr + self.mesh.dsdz[:,k] * Ds + self.mesh.dtdz[:,k] * Dt

    def _precompute_material_properties(self):
        # Material properties (constant)
        if self._inv_rho is None:
            self._inv_rho = 1.0 / self.mesh.density
            self._bulk = self.mesh.density * (self.mesh.speed ** 2)

        # Material properties on faces
        ext = self.mesh.exterior_face_node_indices
        intr = self.mesh.interior_face_node_indices
        Npf = self.nodes_per_face
        Nf = self.num_faces
        K = self.num_cells
        self.rho_p = self.mesh.density.ravel('F')[ext].reshape((Npf*Nf, K), order='F')
        self.rho_m = self.mesh.density.ravel('F')[intr].reshape((Npf*Nf, K), order='F')
        self.c_p = self.mesh.speed.ravel('F')[ext].reshape((Npf*Nf, K), order='F')
        self.c_m = self.mesh.speed.ravel('F')[intr].reshape((Npf*Nf, K), order='F')
    
    def _precompute_indices(self):
        # Current precomputation...
        self.boundary_indices = self.mesh.boundary_face_node_indices
        self.interior_indices = self.mesh.interior_face_node_indices
        self.exterior_indices = self.mesh.exterior_face_node_indices
        self.source_nodes_boundary = self.boundary_indices[self.source_nodes]

    def set_initial_conditions(self, kind="none"):
        """Set initial conditions for testing the wave propagation."""
        # initialize zero value velocity and pressure fields
        # get vertex coordinates
        x = self.mesh.x
        y = self.mesh.y
        z = self.mesh.z

        # set fields
        if kind == "gaussian":
            # Gaussian pulse centered at (x0, y0, z0)
            center=(0.1250, 0.1250, 0.1250)
            sigma=0.01
            x0, y0, z0 = center
            # define pressure field to be a gaussian pulse 
            amplitude = 0.5
            self.p = amplitude*np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))

    def _reshape_to_rectangular(self, du, dv, dw, dp):
        # reshape jump matrices
        Npf = self.nodes_per_face
        num_faces = self.num_faces
        K = self.num_cells
        du = du.reshape((Npf*num_faces, K), order='F')
        dv = dv.reshape((Npf*num_faces, K), order='F')
        dw = dw.reshape((Npf*num_faces, K), order='F')
        dp = dp.reshape((Npf*num_faces, K), order='F')
        return du, dv, dw, dp

    # Pulse types
    def _compute_shifted_cosine(self, time):
        # shifted cosine wave
        f = self.source_frequency 
        a = self.source_amplitude
        pressure = a * (1 - 1 * np.cos(2 * np.pi * f * time)) 
        if time <= self.source_duration:
            return pressure 
        else:
            return 0.0

    def _compute_sin(self, time):
        # shifted cosine wave
        f = self.source_frequency 
        a = self.source_amplitude
        pressure = a * (np.sin(2 * np.pi * f * time)) 
        if time < self.source_duration:
            return pressure 
        else:
            return 0

    def _compute_gaussian_pulse(self, time):
        f = self.source_frequency
        a = self.source_amplitude
        t0 = self.source_duration / 2  # center time
        sigma = t0 / 5  # related to frequency; adjust if needed
        pulse = np.exp(-((time - t0)**2 / (2 * sigma**2)))
        return a * pulse

    def _compute_ricker_wavelet(self, time):
        f = self.source_frequency
        a = self.source_amplitude
        t0 = self.source_duration / 2  # center the wavelet in the source duration
        tau = time - t0
        wavelet = (1 - 2 * (np.pi * f * tau)**2) * np.exp(-(np.pi * f * tau)**2)
        return a * wavelet

    def _get_source_pressure(self, time):
        # choose which source pulse to use
        return self._compute_shifted_cosine(time)
        #return self._compute_sin(time)
        #return self._compute_gaussian_pulse(time)
        #return self._compute_ricker_wavelet(time)

    def _locate_source_nodes(self):
        exterior_values = self.mesh.exterior_face_node_indices
        boundary = self.mesh.boundary_face_node_indices
        tol = self.mesh.reference_element.NODE_TOLERANCE
        nodes_per_face = self.mesh.reference_element.nodes_per_face

        # Get global boundary face node coordinates
        x_b = self.mesh.x.ravel(order='F')[exterior_values][boundary]
        y_b = self.mesh.y.ravel(order='F')[exterior_values][boundary]
        z_b = self.mesh.z.ravel(order='F')[exterior_values][boundary]
            
        # Find nodes within circular source region
        in_source = ((x_b - self.source_center[0])**2 +
                     (y_b - self.source_center[1])**2 < self.source_radius**2 + tol) & \
                     (np.abs(z_b - self.source_center[2]) < tol)

        # locate only face where all nodes lie in the circle
        # convert node number to face number
        faces = np.where(in_source)[0] // nodes_per_face
        # get unique faces and their counts
        unique_vals, counts = np.unique(faces, return_counts=True)
        # only accept faces represented nodes_per_face times
        included_faces = unique_vals[counts == nodes_per_face]
        # convert back to node numbers 
        base = included_faces * nodes_per_face  # shape (N,)
        # Add ranges [0, 1, ..., 20] to each base
        offsets = np.arange(nodes_per_face)  # shape (21,)
        full_ranges = base[:, np.newaxis] + offsets  # shape (N, 21)
        # Flatten to a 1D array
        self.source_nodes = full_ranges.ravel()

    def _get_source_material_properties(self, source_nodes):
        # get material arrays
        rho = self.rho_p.ravel(order='F')[source_nodes] 
        c = self.c_p.ravel(order='F')[source_nodes] 

        # make sure the material is homogeneous over the source
        if np.all(rho == rho[0]):
            rho = rho[0]
        else:
            raise ValueError("rho values are not constant across source_nodes.")
        if np.all(c == c[0]):
            c = c[0]
        else:
            raise ValueError("c values are not constant across source_nodes.")
        return rho, c


    def _apply_source_boundary_condition(self, time, p_p):
        # get source amplitude
        source_pressure = self._get_source_pressure(time)

        # get source node indices 
        source_nodes = self.source_nodes_boundary  # Precomputed

        # Overwrite pressure at the source nodes with the shifted cosine pressure
        # model where the transducer meets the domain as an open boundary
        p_p[source_nodes] = source_pressure

        return p_p
 
    def _apply_boundary_conditions(self, time):
        # get interior values on cells
        u_m = self.u.ravel('F')[self.interior_indices]
        v_m = self.v.ravel('F')[self.interior_indices]
        w_m = self.w.ravel('F')[self.interior_indices]
        p_m = self.p.ravel('F')[self.interior_indices]
        
        u_p = self.u.ravel('F')[self.exterior_indices]
        v_p = self.v.ravel('F')[self.exterior_indices]
        w_p = self.w.ravel('F')[self.exterior_indices]
        p_p = self.p.ravel('F')[self.exterior_indices]
        
        # Use precomputed boundary indices
        boundary = self.boundary_indices

        nx = self.mesh.nx.ravel(order='F')
        ny = self.mesh.ny.ravel(order='F')
        nz = self.mesh.nz.ravel(order='F')

        # compute normal velocity on interior boundary cells
        ndotum = nx[boundary] * u_m[boundary] + ny[boundary] * v_m[boundary] + nz[boundary] * w_m[boundary]

        # compute perfectly reflecting boundary conditions
        u_p[boundary] = u_m[boundary]# - 2.0 * (ndotum) * nx[boundary]
        v_p[boundary] = v_m[boundary]# - 2.0 * (ndotum) * ny[boundary]
        w_p[boundary] = w_m[boundary]# - 2.0 * (ndotum) * nz[boundary]
        p_p[boundary] = 0 

        # apply source
        p_p = self._apply_source_boundary_condition(time, p_p)

        # reshape for matrix-matrix multiplication
        self.u_m, self.v_m, self.w_m, self.p_m = self._reshape_to_rectangular(u_m, v_m, w_m, p_m)
        self.u_p, self.v_p, self.w_p, self.p_p = self._reshape_to_rectangular(u_p, v_p, w_p, p_p)

    def _compute_homogeneous_material_flux(self):
        # homogeneous material fluxes
        flux_p = 0.5 * ((ndotup - ndotum) - (p_p - p_m))
        flux_u = 0.5 * (self.mesh.nx * ((p_p - p_m) - (ndotup - ndotum)))
        flux_v = 0.5 * (self.mesh.ny * ((p_p - p_m) - (ndotup - ndotum)))
        flux_w = 0.5 * (self.mesh.nz * ((p_p - p_m) - (ndotup - ndotum)))

    def _compute_upwind_flux(self):
        # upwind weak form flux
        normal_vel_jump = self.ndotup - self.ndotum
        pressure_jump = self.p_p - self.p_m

        self.flux_p = 0.5 * (self.c_m**2 * self.rho_m * normal_vel_jump - self.c_m * pressure_jump)
        self.flux_u = 0.5 * self.mesh.nx * ((1/self.rho_m) * (self.p_p - self.p_m) - self.c_m * normal_vel_jump)
        self.flux_v = 0.5 * self.mesh.ny * ((1/self.rho_m) * (self.p_p - self.p_m) - self.c_m * normal_vel_jump)
        self.flux_w = 0.5 * self.mesh.nz * ((1/self.rho_m) * (self.p_p - self.p_m) - self.c_m * normal_vel_jump)

    def _compute_rh_flux(self):
        # Normal vector components
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
    
        normal_vel_jump = self.ndotup - self.ndotum
        pressure_jump = self.p_p - self.p_m
    
        if self.flux_terms_cached == False:
            self.flux_denominator = self.c_m * self.rho_m + self.c_p * self.rho_p
            self.Z_p = self.c_p * self.rho_p
            self.K_m = self.c_m**2 * self.rho_m
            self.flux_terms_cached = True

        num = -self.Z_p * normal_vel_jump + pressure_jump
        common_term = num / self.flux_denominator 
    
        self.flux_p = -self.K_m * common_term
        self.flux_u = nx * self.c_m * common_term
        self.flux_v = ny * self.c_m * common_term
        self.flux_w = nz * self.c_m * common_term

    def _compute_xijun_he_flux(self):
        # flux from Xiun He 2025 - An effective discontinuous galerkin
        # weak form flux
        self.flux_p = 0.5 * (
            (self.rho_p * self.c_p**2 * self.u_p - self.rho_m * self.c_m**2 * self.u_m) * self.mesh.nx + \
            (self.rho_p * self.c_p**2 * self.v_p - self.rho_m * self.c_m**2 * self.v_m) * self.mesh.ny + \
            (self.rho_p * self.c_p**2 * self.w_p - self.rho_m * self.c_m**2 * self.w_m) * self.mesh.nz - \
            self.mu * (self.p_p - self.p_m)
        )
        self.flux_u = 0.5 * (self.mesh.nx * (((self.p_p / self.rho_p) - (self.p_m / self.rho_m)) - self.mu * (self.ndotup - self.ndotum)))
        self.flux_v = 0.5 * (self.mesh.ny * (((self.p_p / self.rho_p) - (self.p_m / self.rho_m)) - self.mu * (self.ndotup - self.ndotum)))
        self.flux_w = 0.5 * (self.mesh.nz * (((self.p_p / self.rho_p) - (self.p_m / self.rho_m)) - self.mu * (self.ndotup - self.ndotum)))

    def compute_rhs(self, u=None, v=None, w=None, p=None, time=0.0):
        """
        flux function based on Xijun He et al. 2025
        "An effective discontinuous Galerkin method for solving
        acoustic wave equations"
        page 6/14
        """
        # use stored fields in first RK time step iteration
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if p is None:
            p = self.p

        # get heterogeneous material matrices
        #self._get_material_face_values()

        # spatial derivative matrices
        #Dr = self.mesh.reference_element_operators.r_differentiation_matrix
        #Ds = self.mesh.reference_element_operators.s_differentiation_matrix
        #Dt = self.mesh.reference_element_operators.t_differentiation_matrix

        # local spatial derivatives on reference tetrahedron
        #drdx = self.mesh.drdx
        #drdy = self.mesh.drdy
        #drdz = self.mesh.drdz
        #dsdx = self.mesh.dsdx
        #dsdy = self.mesh.dsdy
        #dsdz = self.mesh.dsdz
        #dtdx = self.mesh.dtdx
        #dtdy = self.mesh.dtdy
        #dtdz = self.mesh.dtdz

        ## compute derivatives in physical space
        #dudx = drdx * (Dr @ self.u) + dsdx * (Ds @ self.u) + dtdx * (Dt @ self.u)
        #dvdy = drdy * (Dr @ self.v) + dsdy * (Ds @ self.v) + dtdy * (Dt @ self.v)
        #dwdz = drdz * (Dr @ self.w) + dsdz * (Ds @ self.w) + dtdz * (Dt @ self.w)
        #dpdx = drdx * (Dr @ self.p) + dsdx * (Ds @ self.p) + dtdx * (Dt @ self.p)
        #dpdy = drdy * (Dr @ self.p) + dsdy * (Ds @ self.p) + dtdy * (Dt @ self.p)
        #dpdz = drdz * (Dr @ self.p) + dsdz * (Ds @ self.p) + dtdz * (Dt @ self.p)

        # Replace derivative calculations with:
        dudx = np.einsum('ijk,jk->ik', self.Dx, u)
        dvdy = np.einsum('ijk,jk->ik', self.Dy, v)
        dwdz = np.einsum('ijk,jk->ik', self.Dz, w)
        dpdx = np.einsum('ijk,jk->ik', self.Dx, p)
        dpdy = np.einsum('ijk,jk->ik', self.Dy, p)
        dpdz = np.einsum('ijk,jk->ik', self.Dz, p)

        # apply boundary conditions
        self._apply_boundary_conditions(time)

        # compute normal velocity at interior boundary and exterior boundary 
        self.ndotum = self.mesh.nx * self.u_m + self.mesh.ny * self.v_m + self.mesh.nz * self.w_m
        self.ndotup = self.mesh.nx * self.u_p + self.mesh.ny * self.v_p + self.mesh.nz * self.w_p

        # compute flux
        self._compute_rh_flux()

        ## get necessary matricies for integral computation
        face_scale = self.mesh.surface_to_volume_jacobian
        lift = self.mesh.reference_element_operators.lift_matrix

        ## inverse density and bulk modulus (rho c^2) - cached for performance
        if self._inv_rho is None:
            self._inv_rho = 1.0 / self.mesh.density
            self._bulk = self.mesh.density * (self.mesh.speed ** 2)

        self.rhs_p = -self._bulk * (dudx + dvdy + dwdz) - lift @ (face_scale * self.flux_p)
        self.rhs_u = -self._inv_rho * dpdx - lift @ (face_scale * self.flux_u)
        self.rhs_v = -self._inv_rho * dpdy - lift @ (face_scale * self.flux_v)
        self.rhs_w = -self._inv_rho * dpdz - lift @ (face_scale * self.flux_w)

        return self.rhs_u, self.rhs_v, self.rhs_w, self.rhs_p
