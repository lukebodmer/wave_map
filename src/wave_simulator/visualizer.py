import numpy as np
import pyvista as pv
import panel as pn
import matplotlib.pyplot as plt
import gmsh


class Visualizer:
    def __init__(self, mesh_data, data, grid=True):
        # create plotter
        self.plotter = pv.Plotter(off_screen=False)

        # set visualizer data
        self.set_data(mesh_data, data)

        # create grid
        if grid:
            self._show_grid()

        # set camera
        self.set_camera()

    def set_data(self, mesh_data, data):
        self.data = data
        self.mesh = mesh_data
        self.extract_data(data)
        self.plotter.clear()

    def extract_data(self, data):
        self.x = self.mesh["x"]
        self.y = self.mesh["y"]
        self.z = self.mesh["z"]
        self.nx = self.mesh["nx"]
        self.ny = self.mesh["ny"]
        self.nz = self.mesh["nz"]
        self.num_cells = self.mesh["num_cells"]
        self.fields = data["fields"]
        self.p = self.fields["p"]
        self.u = self.fields["u"]
        self.v = self.fields["v"]
        self.w = self.fields["w"]
        self.speed = self.mesh["speed_per_cell"]
        self.interior_face_node_indices = self.mesh["interior_face_node_indices"]
        self.boundary_node_indices = self.mesh["boundary_node_indices"]
        self.jacobians = self.mesh["cell_jacobians"]
        self.inclusion_center = self.mesh["inclusion_center"]
        self.inclusion_radius = self.mesh["inclusion_radius"]
        self.vertex_coordinates = self.mesh["vertex_coordinates"]
        self.cell_to_vertices = self.mesh["cell_to_vertices"]
        self.face_node_indices = self.mesh["reference_element"].face_node_indices
        self.boundary_face_node_indices = self.mesh["boundary_face_node_indices"]

        self.tracked_fields = data.get("simulator", {}).get("tracked_fields", {})
        self.energy_data = data.get("simulator", {}).get("energy_data", [])
        self.kinetic_data = data.get("simulator", {}).get("kinetic_data", [])
        self.potential_data = data.get("simulator", {}).get("potential_data", [])
        self.source_data = data.get("simulator", {}).get("source_data", None)

        self.dt = data["dt"]
        self.t_final = data["t_final"]
        self.current_time = data["current_time"]
        self.current_time_step = data["current_time_step"]
        self.output_path = data.get("output_path", "./")
        self.save_image_interval = data["save_image_interval"]
        self.save_data_interval  = data["save_data_interval"]
        self.save_points_interval = data["save_points_interval"]
        self.save_energy_interval = data["save_energy_interval"]
        self.sensor_coordinates = data["sensor_coordinates"]

        self.get_domain_parameters()

    def get_domain_parameters(self):
        # get minimum coordinate values
        self.x_min = np.min(self.x)
        self.y_min = np.min(self.y)
        self.z_min = np.min(self.z)

        # get maximum coordinate values
        self.x_max = np.max(self.x)
        self.y_max = np.max(self.y)
        self.z_max = np.max(self.z)

    def set_camera(self):
        camera_position = [
            # position
            (self.x_max * 3.1, self.y_max * 2.3, self.z_max * 1.1),
            # looking at
            (self.x_max * 0.3, self.y_max * 0.3, self.z_max * 0.3),
            # up direction
            (0, 0, 1)]
        self.plotter.camera_position = camera_position

    def _show_grid(self):
        self.plotter.show_grid()

    def add_nodes_3d(self, field):
        """Plot nodes on the mesh with colors and opacity based on solution values."""
        # Extract x, y, z coordinates for the nodes
        x = self.x.ravel(order='F')
        y = self.y.ravel(order='F')
        z = self.z.ravel(order='F')
        
        # Stack into nodal points
        node_coordinates = np.column_stack((x, y, z))
        
        # Flatten the solution matrix to align with the coordinates
        field = np.ravel(self.fields[field], order='F')
        #field[self.mesh.exterior_face_node_indices] = 0
        #opacity = np.abs(field)

        # Add the points to the plot with colors and opacity
        self.plotter.add_points(
            node_coordinates,
            scalars=field,
            cmap="seismic",
            #opacity='linear',
            #opacity=opacity,
            opacity=[0.9, 0.7, 0.5, 0.5, 0, 0.5, 0.5, 0.7, 0.9],
            #opacity=[0.01, 0.05, 0.06,  0.08, 0.09, 0.2, 0.3],
            #clim=[-.00001,.00001],
            clim=[-.10,.10],
            point_size=10,
            render_points_as_spheres=True
        )

    def add_node_list(self, nodes):
        
        # get interior nodes
        interior_values = self.interior_face_node_indices
        x = self.x.ravel(order='F')[interior_values]
        y = self.y.ravel(order='F')[interior_values]
        z = self.z.ravel(order='F')[interior_values]

        # get nodes in list
        x = x[nodes]
        y = y[nodes]
        z = z[nodes]

        # Stack into nodal points
        node_coordinates = np.column_stack((x, y, z))

        # Add the points to the plot
        self.plotter.add_points(
            node_coordinates,
            point_size=10,
            render_points_as_spheres=True
        )
        
    def add_cell_nodes(self, cell_list):
        # Extract x, y, z coordinates for the nodes in the specified cells 
        x = self.x[:, cell_list].flatten()
        y = self.y[:, cell_list].flatten()
        z = self.z[:, cell_list].flatten()
        
        # Stack into nodal points
        node_coordinates = np.column_stack((x, y, z))
        
        # Add the points to the plot
        self.plotter.add_points(
            node_coordinates,
            color="blue",
            point_size=10,
            render_points_as_spheres=True
        )

    def add_all_boundary_nodes(self):
        """Plot boundary nodes on the mesh."""
        # Extract x, y, z coordinates for the boundary nodes
        boundary_nodes = self.boundary_node_indices
        x = self.x.ravel(order="F")[boundary_nodes]
        y = self.y.ravel(order="F")[boundary_nodes]
        z = self.z.ravel(order="F")[boundary_nodes]
        
        # Stack into boundary nodal points
        boundary_points_to_plot = np.column_stack((x, y, z))
        
        # Add the boundary points to the plot
        self.plotter.add_points(
            boundary_points_to_plot,
            color="green",
            point_size=10,
            render_points_as_spheres=True
        )


    def add_cells(self, cell_list):
        """Highlight specific cells on the mesh."""
        # Get Jacobian values for all cells (using first element of each column)
        jacobian_values = self.jacobians
            
        # Normalize Jacobian values to create a color map
        cmap = plt.cm.viridis  # You can choose any colormap
        norm = plt.Normalize(vmin=np.min(jacobian_values), vmax=np.max(jacobian_values))
            
        for cell in cell_list:
            # Get the Jacobian value for the current cell
            jacobian_value = jacobian_values[cell]
                
            # Normalize and map the Jacobian value to color
            color = cmap(norm(jacobian_value))[:3]  # Use only the RGB channels
                
            # Create the mesh for the highlighted cell
            cell_mesh = pv.UnstructuredGrid(
                np.hstack([[4], self.cell_to_vertices[cell]]).flatten(),
                [pv.CellType.TETRA],
                self.vertex_coordinates
            )
                
            # Apply color based on the Jacobian value
            self.plotter.add_mesh(
                cell_mesh,
                color=color,
                opacity=0.5
            )

    def add_sensors(self):
        """Add red dots at all pressure receiver locations."""
        if not hasattr(self, 'sensor_coordinates') or not self.sensor_coordinates:
            print("No pressure receiver locations to display.")
            return

        # Convert list of coordinates to numpy array
        sensor_points = np.array(self.sensor_coordinates)

        # Add the points to the plot as red spheres
        self.plotter.add_points(
            sensor_points,
            color="black",
            point_size=14,
            render_points_as_spheres=True
        )

    def add_cell_normals(self, cell_list):
        face_node_indices = self.face_node_indices
        for cell in cell_list:
            x_origin = self.x[:, cell][face_node_indices].ravel(order='F')
            y_origin = self.y[:, cell][face_node_indices].ravel(order='F')
            z_origin = self.z[:, cell][face_node_indices].ravel(order='F')
            
            # Stack into origin points
            normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))
            
            # Extract normal vectors for the given element
            nx = self.mesh.nx[:, cell]
            ny = self.mesh.ny[:, cell]
            nz = self.mesh.nz[:, cell]
            
            # Stack into normal vectors
            normal_vector_direction = np.column_stack((nx, ny, nz))
            
            # Add normal vectors as arrows
            self.plotter.add_arrows(
                normal_vector_origin,
                normal_vector_direction,
                mag=0.08,
                color='red'
            )

    def add_boundary_normals(self):
        """Plot normal vectors for all elements."""
        # Get boundary node indices
        boundary_node_indices = self.boundary_node_indices
        boundary_face_node_indices = self.boundary_face_node_indices
        
        # Get normal vector origins for all boundary nodes 
        x_origin = self.x.ravel(order='F')[boundary_node_indices]
        y_origin = self.y.ravel(order='F')[boundary_node_indices]
        z_origin = self.z.ravel(order='F')[boundary_node_indices]
        
        normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))

        # Extract normal vectors for the given element
        nx = self.nx.ravel(order='F')[boundary_face_node_indices]
        ny = self.ny.ravel(order='F')[boundary_face_node_indices]
        nz = self.nz.ravel(order='F')[boundary_face_node_indices]
        
        # Stack into normal vectors
        normal_vector_direction = np.column_stack((nx, ny, nz))
    
        self.plotter.add_arrows(
            normal_vector_origin,
            normal_vector_direction,
            mag=0.05,
            color='red'
        )
 

    def add_cell_averages(self, field):
        """
        Plot cell averages for a given solution.
        Each cell's average solution value is computed and visualized.
        """
        # Construct a cells object to make a pyvista unstructuredGrid
        cells = np.zeros(self.num_cells * 5, dtype='int')
        index = 0
        for i in range(self.num_cells * 5):
            if i % 5 == 0:
                cells[i] = 4
            else:
                cells[i] = index
                index += 1
    
        # calculate average of each cell
        cell_averages = np.mean(field, axis=0)
        # designate cell type of tetrahedron
        cell_types = np.repeat(np.array([pv.CellType.TETRA]), self.num_cells)
        # get coordinates from mesh
        coordinates = self.vertex_coordinates[self.cell_to_vertices.ravel()]

        # create unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            coordinates
        )

        # add to plotter
        self.plotter.add_mesh(
            grid,
            scalars=cell_averages,
            #opacity=np.abs(cell_averages),
            #opacity=[0.9, 0.7, 0.5, 0.5,0.3, 0, 0.3, 0.5, 0.5, 0.7, 0.9],
            #opacity=[0.9, 0.7, 0.5,  0, 0.5, 0.7, 0.9],
            opacity=[0.01, 0.05, 0.06,  0.08, 0.09, 0.2, 0.3],
            clim=[0,200],
            #cmap='seismic',
            cmap='hsv',
            smooth_shading=True
        )

    def add_wave_speed(self):
        """ plot the wavespeed of each element """
        # Construct a cells object to make a pyvista unstructuredGrid
        cells = np.zeros(self.num_cells * 5, dtype='int')
        index = 0
        for i in range(self.num_cells * 5):
            if i % 5 == 0:
                cells[i] = 4
            else:
                cells[i] = index
                index += 1
        cell_types = np.repeat(np.array([pv.CellType.TETRA]), self.num_cells)
        points = self.vertex_coordinates[self.cell_to_vertices.ravel()]
    
        # create a pyvista unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            points
        )
           
        # add to plotter
        self.plotter.add_mesh(
            grid,
            scalars=self.speed,
            opacity=0.05#'linear'#abs(wave_speed)
        )

    def add_mesh(self):
        """Add the edges of the entire 3D mesh as translucent wireframe."""
        # Construct a cells object to make a pyvista unstructuredGrid
        cells = np.zeros(self.num_cells * 5, dtype='int')
        index = 0
        for i in range(self.num_cells * 5):
            if i % 5 == 0:
                cells[i] = 4
            else:
                cells[i] = index
                index += 1
    
        cell_types = np.repeat(np.array([pv.CellType.TETRA]), self.num_cells)
        points = self.vertex_coordinates[self.cell_to_vertices.ravel()]
    
        # create a pyvista unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            points,
        )
    
        # Add wireframe rendering of the mesh
        #self.plotter.add_mesh(
        #    grid,
        #    style='wireframe',    # Only edges
        #    color='black',        # Edge color
        #    line_width=1.0,       # Line thickness
        #    opacity=0.4           # Optional: make it slightly translucent
        #)
    
       # # Optional: Add clip planes if still desired
        #for norm in ['-z', '-x', '-y']:
        for norm in ['-z']:
            self.plotter.add_mesh_clip_plane(
                grid,
                normal=norm,
                crinkle=True,
                interaction_event='always',
                normal_rotation=False,
                color='black',
                style='wireframe',
                opacity=0.4
            )
 
    def add_mesh_boundary(self):
        """ Plot mesh edges on boundary """
        # create cells and cell_types to pyvista unstructured grid
        cells = np.hstack([np.full((self.num_cells, 1), 4), self.cell_to_vertices]).flatten()
        cell_types = np.full(self.num_cells, pv.CellType.TETRA) 

        # create unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            self.mesh.vertex_coordinates
        )
        # Add mesh
        self.plotter.add_mesh(
            grid,
            style='wireframe',
            color='black'
        )

    def visualize_array(self, array):
        """
        Visualizes a 2D NumPy array using a colormap.
        """
        cmap = 'viridis'
        if array.ndim != 2:
            raise ValueError("Input array must be 2D")
    
        plt.figure(figsize=(6, 5))
        plt.imshow(array, cmap=cmap, aspect='auto')
        
        plt.colorbar()
    
        plt.title("2D Array Visualization")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

    def plot_reference_nodes_3d(self):
        """ plot the nodes for the reference finite element """
        # Get nodes
        nodes = self.mesh["reference_element"].nodes
        
        # Create a PyVista point cloud
        point_cloud = pv.PolyData(nodes)
        
        # Create a plotter
        plotter = pv.Plotter()

        # Add mesh
        plotter.add_mesh(
            point_cloud,
            color='red',
            point_size=10,
            render_points_as_spheres=True
        )
        
        # Add axes labels
        plotter.show_grid()
        
        # show plot
        d = self.mesh["reference_element"].d
        n = self.mesh["reference_element"].n
        plotter.show(title=f"Lagrange Element Nodes (d={d}, n={n})")

    def add_inclusion_boundary(self):

        # Add the spherical inclusion
        sphere_center = self.inclusion_center
        sphere_radius = self.inclusion_radius
        sphere = pv.Sphere(center=sphere_center,
                           radius=sphere_radius,
                           theta_resolution=10,
                           phi_resolution=10)
        self.plotter.add_mesh(sphere,
                              color="#ccdee6",
                              opacity=0.1,
                              show_edges=True)

    def save(self):
        file_name = f't_{self.current_time_step:0>8}.png'
        output_file = f'{self.output_path}/images/{file_name}'
    
        try:
            # Backup current plotter
            old_plotter = self.plotter
    
            # Create a new off-screen plotter
            self.plotter = pv.Plotter(off_screen=True)
            self.plotter.clear()
            self.set_camera()
            self.add_inclusion_boundary()
            self.add_nodes_3d("p")
            self._show_grid()
            self.plotter.screenshot(output_file)
            self.plotter.close()

        except Exception as e:
            print(f"Failed to save screenshot to {output_file}: {e}")
            raise
        finally:
            # Restore original plotter
            self.plotter = old_plotter

    def plot_energy(self, show=False):
        #num_steps = len(self.energy_data)
        interval = self.save_energy_interval
        t = self.current_time_step
        dt = self.dt #dt * self.data["save_energy_interval"]
        num_steps = t // interval + 1
        time_array = np.arange(num_steps) * dt
        fig, ax = plt.subplots()
        ax.plot(time_array, self.energy_data[:num_steps], marker='o', label='Total Energy')
        ax.plot(time_array, self.kinetic_data[:num_steps], marker='x', label='KE')
        ax.plot(time_array, self.potential_data[:num_steps], marker='*', label='PE')
        ax.legend()
        ax.set_title(f'Global Energy (reflective boundary conditions)')
        ax.set_ylabel('Energy')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
            return
        return fig

    def plot_tracked_points(self, show=False):
        """
        Plot each field (pressure or velocity component) at each tracked point on a separate subplot,
        but only include data up to current_time_step // save_points_interval.
        Keep the x-axis representing the full simulation time.
        """
        if not self.tracked_fields:
            print("No tracked field data to plot.")
            return
    
        interval = self.data['save_points_interval']
        dt = self.dt
        t = self.current_time_step
        total_steps = next(iter(self.tracked_fields.values()))["data"].shape[1]
        num_steps = t // interval  # only show data up to this step
    
        # Time array for the truncated data
        time_array = np.arange(num_steps) * dt * interval
    
        color_map = {
            "pressure": "purple",
            "x": "blue",
            "y": "green",
            "z": "red"
        }
        label_map = {
            "pressure": "Pressure",
            "x": "Velocity (u)",
            "y": "Velocity (v)",
            "z": "Velocity (w)"
        }

        total_plots = sum(len(entry["points"]) for entry in self.tracked_fields.values())
        fig, axes = plt.subplots(total_plots, 1, figsize=(10, 20), sharex=True)

        if total_plots == 1:
            axes = [axes]

        plot_idx = 0
        for field_key, entry in self.tracked_fields.items():
            color = color_map.get(field_key, "black")
            label = label_map.get(field_key, field_key)
            for point_idx, (x, y, z) in enumerate(entry["points"]):
                data_series = entry["data"][point_idx][:num_steps]  # truncate the data
                ax = axes[plot_idx]
                ax.plot(
                    time_array,
                    data_series,
                    color=color,
                    linewidth=1.5,
                    marker='o',
                    markersize=4,
                    label=label
                )
                ax.set_title(f"{label} at (x={x:.3f}, y={y:.3f}, z={z:.3f})")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, self.t_final - self.dt)  # set x-limits to full time
                plot_idx += 1
    
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()

        if show:
            plt.show()
            return

        return fig

    def plot_sensor_data_as_matrix(self, show=False):
        """
        Plot pressure data from tracked_fields as a matrix, where each row corresponds
        to a tracked point and each column to a time step. Colors indicate pressure magnitude.
        """
        if "pressure" not in self.tracked_fields:
            print("No pressure data in tracked_fields to plot.")
            return
    
        interval = self.data['save_points_interval']
        t = self.current_time_step
        num_steps = t // interval
    
        pressure_entry = self.tracked_fields["pressure"]
        data_matrix = pressure_entry["data"][:, :num_steps]  # (n_points, time)
        points = pressure_entry["points"]
    
        fig, ax = plt.subplots(figsize=(12, 12))
        vmax = np.abs(data_matrix).max()
        vmin = -vmax

        cax = ax.imshow(data_matrix, aspect='auto', cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
        n_rows = data_matrix.shape[0]
        for i in range(1, n_rows):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)

        ax.set_ylabel("Sensor Index")
        ax.set_xlabel("Time Step Index")
        ax.set_xticks(np.linspace(0, num_steps-1, min(10, num_steps), dtype=int))
        ax.set_yticks(np.arange(len(points)))
    
        # Optional: label y-axis with coordinates
        y_labels = [f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in points]
        ax.set_yticklabels(y_labels, fontsize=8)

        fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.01, shrink=0.8)
    
        plt.tight_layout()
    
        if show:
            plt.show()
            return
    
        return fig

    def get_pyvista_pane(self):
        plotter = pv.Plotter(off_screen=True)
        self.plotter = plotter
        self.set_camera()
        self._show_grid()
        self.add_inclusion_boundary()
        self.add_nodes_3d("p")
        return pn.pane.VTK(self.plotter.ren_win)

    def plot_source(self, source_data):
        num_steps = len(source_data)
        dt = self.time_stepper.dt
        time_array = np.arange(num_steps) * dt
        fig, ax = plt.subplots()
        ax.plot(time_array, source_data, marker='o', label='Source')
        ax.legend()
        ax.set_title(f'Source Pressure')
        ax.set_ylabel('Pressure')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        plt.show()

    def show(self):
        self.plotter.show()

    def clear(self):
        self.plotter.clear()
