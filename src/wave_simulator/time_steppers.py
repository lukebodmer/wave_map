import numpy as np
from logging import getLogger
from wave_simulator.physics import LinearAcoustics


class LowStorageRungeKutta:
    def __init__(
        self,
        physics: LinearAcoustics,
        t_initial: float,
        t_final: float = None,
        number_of_timesteps: int = None
    ):
        if (t_final is None) == (number_of_timesteps is None):
            raise ValueError("Specify exactly one of 't_final' or 'number_of_timesteps'.")

        self.physics = physics
        self.t_initial = t_initial
        self.t_final = t_final
        self.num_time_steps = number_of_timesteps
        self.t = t_initial
        self.current_time_step = 0

        # Runge-Kutta residual storage
        nodes_per_cell = physics.mesh.reference_element.nodes_per_cell
        num_cells = physics.mesh.num_cells
        self.rk4a = np.array([0.0,
                              -567301805773.0 / 1357537059087.0,
                              -2404267990393.0 / 2016746695238.0,
                              -3550918686646.0 / 2091501179385.0,
                              -1275806237668.0 / 842570457699.0])

        self.rk4b = np.array([1432997174477.0 / 9575080441755.0,
                              5161836677717.0 / 13612068292357.0,
                              1720146321549.0 / 2090206949498.0,
                              3134564353537.0 / 4481467310338.0,
                              2277821191437.0 / 14882151754819.0])
        self.res_u = np.zeros((nodes_per_cell, num_cells))
        self.res_v = np.zeros((nodes_per_cell, num_cells))
        self.res_w = np.zeros((nodes_per_cell, num_cells))
        self.res_p = np.zeros((nodes_per_cell, num_cells))
        self.source_position = (0.5, 0.5, 0.00)
        self.source_radius = 0.021
        self.source_frequency = 10
        self.source_amplitude = 10000
        #self.source_duration = 0.10 # in seconds
        #self.wavelength = self.physics.max_speed / self.source_frequency  
        #self.source_duration = 1 / self.source_frequency
        self.damping_coefficient = 20.0 # for exponential damping
        self.gaussian_decay_scale = 0.001 # for gaussian damping
        self.tau = (3 - np.sqrt(3))/6
        #self._get_source_nodes()
        self._compute_time_step_size_hesthaven()
        samples_per_cycle = int(round(1 / (self.dt * self.source_frequency)))
        self.source_duration = samples_per_cycle * self.dt
        #self.source_duration = 0.01
        self._log_info()

#    def _compute_time_step_size_hesthaven(self):
#        n = self.physics.mesh.reference_element.n
#        #surface_to_volume_jacobian = self.physics.mesh.surface_to_volume_jacobian
#        #c = self.physics.max_speed
#
#        c = np.max(self.physics.mesh.speed) # used in time_stepper.py
#        d = self.physics.mesh.smallest_diameter
#        cfl_factor = 0.9
#        #dt = 1.0 * (1.0 / (np.max(np.max(surface_to_volume_jacobian)) * n * n * c))
#        dt = cfl_factor * (d / (n * n * c))
#        #dt = cfl_factor * (d / (10 * c))
#        # correct dt for integer # of time steps
#        self.num_time_steps = int(np.ceil(self.t_final/ dt))
#        self.dt = (self.t_final / self.num_time_steps)

    def _compute_time_step_size_hesthaven(self):
        p = self.physics.mesh.reference_element.n
        # TODO HACK: fix c so all runs have same dt
        #c = np.max(self.physics.mesh.speed)
        c = 2
        d = self.physics.mesh.smallest_diameter
        cfl_safety = 0.95  # Slightly under max for safety
        dt = cfl_safety * (d / ((2 * p + 1) * c))

        # set dt based on either number of timesteps or total time
        if self.t_final is not None:
            self.dt = dt
            self.num_time_steps = int(np.ceil(self.t_final / dt))
            # change t_final to a multiple of dt
            self.t_final = self.num_time_steps * self.dt
        else:
            self.num_time_steps = self.num_time_steps
            self.dt = dt
            self.t_final = self.num_time_steps * self.dt

    def _compute_time_step_size_xijun(self):
        """
        from Xijun He et al. 2023 - Modeling 3D elastic Wave Propagation
        page 6/14
        """
        d = self.physics.mesh.smallest_diameter
        c = self.physics.max_speed
        alpha = 0.2 # max is 0.503 for eta1 = 0.33 and eta2 = 0.87
        dt = (alpha * d) / c
        # correct dt for integer # of time steps
        self.num_time_steps = int(np.ceil(self.t_final/ dt))
        self.dt = (self.t_final / self.num_time_steps)

    def _get_node_xyz(self):
        x = self.physics.mesh.x.ravel(order='F')
        y = self.physics.mesh.y.ravel(order='F')
        z = self.physics.mesh.z.ravel(order='F')
        return x,y,z

    def _get_source_nodes(self):
        # abreviate long lists
        npc = self.physics.mesh.reference_element.nodes_per_cell
        npf = self.physics.mesh.reference_element.nodes_per_face
        # get nodes for mesh and source
        x,y,z = self._get_node_xyz()
        x0, y0, z0 = self.source_position

        # find only nodes in the source
        self.source_nodes = np.where(
            (np.isclose(z, 0, atol=1e-7)) &  # Check if z ≈ 0 within a tolerance
            (np.sqrt((x - x0)**2 + (y - y0)**2) < self.source_radius)
        )
        # convert nodes to elements
        self.source_elements = self.source_nodes[0] // npc 

        # get unique elements and counts
        unique_elements, counts = np.unique(self.source_elements, return_counts=True)

        # keep only elements that have a cound of nodes_per_face
        elements = unique_elements[counts == npf]

        # get all nodes that belong to these elements
        all_element_nodes = np.concatenate([
            np.arange(elem * npc, (elem + 1) * npc)
            for elem in elements
        ])

        # intersect with source_nodes to get the actual nodes on boundary face
        final_nodes = np.intersect1d(all_element_nodes, self.source_nodes)

        self.source_face_nodes = final_nodes
       
    def _get_amplitude(self):
        # Get acmplitude
        if self.t >= self.source_duration:
            # exponentially decay signal to kill it
            exp_decay = np.exp(-self.damping_coefficient* (self.t - self.source_duration))
            amplitude = self.source_amplitude * exp_decay 
        else:
            # apply signal as normal
            amplitude = self.source_amplitude
        return amplitude

    def _compute_source_value(self, t):
        amplitude = self._get_amplitude()
        f = self.source_frequency
        return amplitude * np.sin(2 * np.pi * f * t)
        
    def _apply_source_term(self, field, t):
        # get the value of the source
        source_value = self._compute_source_value(t)
        # ravel K_bar for indexing
        field = field.ravel(order='F')

        # apply the value to the source nodes
        field[self.source_nodes] += source_value

        # reshape field 
        field = field.reshape((self.physics.mesh.reference_element.nodes_per_cell,
                               self.physics.mesh.num_cells),
                              order='F')
        return field


    def advance_time_step_rk_with_force_term(self):
        """
        advance forward in time by dt
        Algorithm by Xijun He et al. 2023
        "modeling 3D elastic waves propagation in Ti Media"
        page 5/14
        """
        dt = self.dt
        eta1 = 0.33
        tau = (3 - np.sqrt(3))/6
        eta2 = 0.87

        # compute K
        time = self.t + self.tau * self.dt
        LC_u, LC_v, LC_w, LC_p = self.physics.compute_rhs(time=time)
        LC2_u, LC2_v, LC2_w, LC2_p = self.physics.compute_rhs(LC_u, LC_v, LC_w, LC_p, time=time)
        LC3_u, LC3_v, LC3_w, LC3_p = self.physics.compute_rhs(LC2_u, LC2_v, LC2_w, LC2_p, time=time)
         
        K_u = LC_u + tau * dt * LC2_u + eta1 * (tau * dt)**2 * LC3_u
        K_v = LC_v + tau * dt * LC2_v + eta1 * (tau * dt)**2 * LC3_v
        K_w = LC_w + tau * dt * LC2_w + eta1 * (tau * dt)**2 * LC3_w
        K_p = LC_p + tau * dt * LC2_p + eta1 * (tau * dt)**2 * LC3_p

        #K_p  = self._apply_source_term(K_p, t)

        # compute Kbar
        time = self.t + (1 - self.tau) * self.dt
        T_u = self.physics.u + (1 - 2 * tau) * dt * K_u
        T_v = self.physics.v + (1 - 2 * tau) * dt * K_v
        T_w = self.physics.w + (1 - 2 * tau) * dt * K_w
        T_p = self.physics.p + (1 - 2 * tau) * dt * K_p
        LT_u, LT_v, LT_w, LT_p = self.physics.compute_rhs(T_u, T_v, T_w, T_p, time=time)
        LT2_u, LT2_v, LT2_w, LT2_p = self.physics.compute_rhs(LT_u, LT_v, LT_w, LT_p, time=time)
        LT3_u, LT3_v, LT3_w, LT3_p = self.physics.compute_rhs(LT2_u, LT2_v, LT2_w, LT2_p, time=time)

        Kbar_u = LT_u + tau * dt * LT2_u + eta2 * (tau * dt)**2 * LT3_u
        Kbar_v = LT_v + tau * dt * LT2_v + eta2 * (tau * dt)**2 * LT3_v
        Kbar_w = LT_w + tau * dt * LT2_w + eta2 * (tau * dt)**2 * LT3_w
        Kbar_p = LT_p + tau * dt * LT2_p + eta2 * (tau * dt)**2 * LT3_p

        # apply source term
        #Kbar_p = self._apply_source_term(Kbar_p, t)

        # update fields
        self.physics.u = self.physics.u + (dt/2) * (K_u + Kbar_u)
        self.physics.v = self.physics.v + (dt/2) * (K_v + Kbar_v)
        self.physics.w = self.physics.w + (dt/2) * (K_w + Kbar_w)
        self.physics.p = self.physics.p + (dt/2) * (K_p + Kbar_p)
        
        self.t += self.dt  # Increment time
        self.current_time_step += 1

    def advance_time_step(self):
        """advance forward in time by dt"""
        dt = self.dt

        for i in range(5):  # inner multi-stage Runge-Kutta loop
            rhs_u, rhs_v, rhs_w, rhs_p = self.physics.compute_rhs(time=self.t)

            # initiate, increment Runge-Kutta residuals and update fields
            self.res_u = self.rk4a[i] * self.res_u + dt * rhs_u
            self.physics.u = self.physics.u + self.rk4b[i] * self.res_u
            self.res_v = self.rk4a[i] * self.res_v + dt * rhs_v
            self.physics.v = self.physics.v + self.rk4b[i] * self.res_v
            self.res_w = self.rk4a[i] * self.res_w + dt * rhs_w
            self.physics.w = self.physics.w + self.rk4b[i] * self.res_w
            self.res_p = self.rk4a[i] * self.res_p + dt * rhs_p
            self.physics.p = self.physics.p + self.rk4b[i] * self.res_p

        self.t += self.dt  # Increment time
        self.current_time_step += 1

    def _log_info(self):
        logger = getLogger("simlog")
        logger.info(f"Final time: {self.t_final}s")
        logger.info(f"Time step size: {self.dt:.6g}s")
        logger.info(f"Total Number of timesteps: {self.num_time_steps}")
