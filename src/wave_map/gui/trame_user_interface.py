import os
import numpy as np
import pyvista as pv
from pyvista.trame.ui import plotter_ui
import matplotlib.pyplot as plt
from trame.app import TrameApp
from trame.decorators import change
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
from trame.widgets import matplotlib
from pathlib import Path
import toml
import pickle

from wave_map.loggers.logger import Logger
from wave_map.simulator.visualizer import Visualizer

# PyVista offscreen rendering
pv.OFF_SCREEN = True

LOGGER_DIR = "data/gui_looger/"
BATCH_DATA_DIR = "data/simulation_batch_data"


class TrameGui(TrameApp):
    def __init__(self, server=None):
        super().__init__(server)

        pv.global_theme.colorbar_orientation = 'vertical'

        # get logger
        self.logger = Logger(log_path=Path(LOGGER_DIR) / "log.txt", name="guilog")

        # --- Browser tab title ---
        self.server.state.trame__title = "SBI"

        # --- PyVista plotters ---
        self.plotter = pv.Plotter()         # simulation pressure data
        self.plotter_wave = pv.Plotter()    # real image 
        self.plotter_pred = pv.Plotter()    # predicted prediction

        # --- State variables ---
        self.state.batch_name = None
        self.state.batch_list = []
        self.state.simulation_hash = None
        self.state.simulation_hash_list = []
        self.state.timestep_data = None
        self.state.timestep_list = []
        self.state.filter_kspace = False
        self.state.show_prediction = False
        #self.state.simulation_parameters = {}
        self.state.simulation_parameters_dict = {}


        # Store loaded mesh
        self.mesh_data = None
        self.visualizer = None
        self.visualizer_wave = None

        # Populate batch list
        self._refresh_batches()

        # Build UI
        self._build_ui()

    # --- Server-side state change listeners ---
    @change("batch_name")
    def on_batch_name_change(self, batch_name, **kwargs):
        """Update simulations when batch changes."""
        self._refresh_simulations(batch_name)
        self.state.timestep_list = []
        self.state.timestep_data = None

    @change("simulation_hash")
    def on_simulation_hash_change(self, simulation_hash, **kwargs):
        """Update timestep list and show parameters when simulation changes."""
        # Always refresh timesteps
        self._refresh_timesteps(self.state.batch_name, simulation_hash)
    
        # Reset parameters if nothing selected
        if not simulation_hash or not self.state.batch_name:
            self.state.simulation_parameters_dict = {}
            return
    
        try:
            # Path to simulation parameter file
            param_file = (
                Path(BATCH_DATA_DIR)
                / self.state.batch_name
                / "parameter_files"
                / f"{simulation_hash}.toml"
            )
    
            if not param_file.exists():
                self.state.simulation_parameters_dict = {
                    "Error": {"message": f"No parameter file found at {param_file}"}
                }
                return
    
            # Helper to prettify values
            def prettify(value):
                if isinstance(value, list):
                    if all(isinstance(x, (int, float)) for x in value):
                        # flat numeric list
                        return ", ".join(str(x) for x in value)
                    elif all(isinstance(x, list) for x in value):
                        # list of lists
                        return "\n".join(str(row) for row in value)
                    else:
                        return str(value)
                return str(value)
    
            # Load and prettify parameters
            raw_params = toml.load(param_file)
            sim_params = {
                section: {k: prettify(v) for k, v in content.items()}
                for section, content in raw_params.items()
            }
    
            # Store in state for the UI
            self.state.simulation_parameters_dict = sim_params
    
        except Exception as e:
            self.logger.error(f"Failed to load simulation parameters: {e}")
            self.state.simulation_parameters_dict = {
                "Error": {"message": f"Failed to load: {e}"}
            }

    @change("timestep_data")
    def on_timestep_data_change(self, **kwargs):
        timestep_file = self.state.timestep_data
        if not timestep_file:
            return
        batch_name = self.state.batch_name
        simulation_hash = self.state.simulation_hash
        if not batch_name or not simulation_hash:
            return

        sim_dir = Path(BATCH_DATA_DIR) / batch_name / "simulations" / simulation_hash
        data_dir = sim_dir / "data"

        # --- Load timestep data ---
        timestep_path = data_dir / timestep_file
        with open(timestep_path, "rb") as f:
            timestep_data = pickle.load(f)

        # --- Load mesh data ---
        batch_meta_path = Path(BATCH_DATA_DIR) / batch_name / "batch_metadata.toml"
        batch_meta = toml.load(batch_meta_path)

        mesh_hash = None
        for key, value in batch_meta.get("mesh", {}).items():
            if value.get("simulation_hash") == simulation_hash:
                mesh_hash = key
                break
        if mesh_hash is None:
            self.logger.error(f"No mesh found for simulation '{simulation_hash}' in batch '{batch_name}'")
            return

        mesh_dir = Path(BATCH_DATA_DIR) / batch_name / "meshes" / mesh_hash
        mesh_file = mesh_dir / "mesh.pkl"

        if not mesh_file.exists():
            self.logger.error(f"Mesh file not found: {mesh_file}")
            return

        with open(mesh_file, "rb") as f:
            self.mesh_data = pickle.load(f)

        # --- Load k-space prediction if available ---
        kspace_pred_file = Path(BATCH_DATA_DIR) / batch_name / "kspace_predictions" / f"{simulation_hash}.pkl"
        if kspace_pred_file.exists():
            try:
                with open(kspace_pred_file, "rb") as f:
                    kspace_data = pickle.load(f)

                voxel_pred = np.fft.ifftn(np.fft.ifftshift(kspace_data))
                voxel_pred = np.real(voxel_pred)

                pv_grid = pv.ImageData()
                pv_grid.dimensions = voxel_pred.shape
                pv_grid.spacing = (
                    1.0 / voxel_pred.shape[0],
                    1.0 / voxel_pred.shape[1],
                    1.0 / voxel_pred.shape[2],
                )
                pv_grid.origin = (0, 0, 0)
                pv_grid["values"] = voxel_pred.flatten(order="F")
        
                self.plotter_pred.clear()
                self.plotter_pred.add_volume(
                    pv_grid,
                    scalars="values",
                    opacity="sigmoid",
                    #shade=True,
                    clim=[0, 0.9],
                    cmap="viridis",
                )
                self.plotter_pred.show_grid()

                self.logger.info(f"Loaded k-space prediction for {simulation_hash}")
        
                # show panel
                self.state.show_prediction = True
        
            except Exception as e:
                self.logger.error(f"Failed to load k-space prediction: {e}")
                self.state.show_prediction = False
        else:
            self.logger.info(f"No k-space prediction found for {simulation_hash}")
            self.state.show_prediction = False

        # --- Initialize visualizers ---
        self.visualizer = Visualizer(self.mesh_data, timestep_data, plotter=self.plotter)
        self.visualizer.add_nodes_3d("p")
        self.visualizer._show_grid()
        self.visualizer.add_sensors()

        self.visualizer_wave = Visualizer(self.mesh_data, timestep_data, plotter=self.plotter_wave)
        self.visualizer_wave.add_wave_speed()
        self.visualizer_wave._show_grid()

        # Update Matplotlib figures
        fig_sensor = self.visualizer.plot_sensor_data_as_matrix(show=False)
        if fig_sensor is not None:
            self.sensor_matrix_widget.update(fig_sensor)
            
        fig_energy = self.visualizer.plot_energy(show=False)
        if fig_energy is not None:
            self.energy_widget.update(fig_energy)

        # synchronize cameras if needed
        self.plotter_pred.camera_position = self.plotter_wave.camera_position
        
        # force render so grids appear without clicking
        self.plotter_wave.render()
        self.plotter_pred.render()


    # --- Helpers ---
    def _list_batches(self):
        if not os.path.exists(BATCH_DATA_DIR):
            return []
        return [name for name in os.listdir(BATCH_DATA_DIR) if os.path.isdir(os.path.join(BATCH_DATA_DIR, name))]

    @change("filter_kspace")
    def on_filter_toggle(self, filter_kspace, **kwargs):
        """Refresh simulations when filter changes."""
        self._refresh_simulations(self.state.batch_name)

    def _list_simulations(self, batch_name):
        if batch_name is None:
            return []
        sim_base = os.path.join(BATCH_DATA_DIR, batch_name, "simulations")
        if not os.path.exists(sim_base):
            return []
    
        sims = [name for name in os.listdir(sim_base) if os.path.isdir(os.path.join(sim_base, name))]
    
        # Apply filter if enabled
        if self.state.filter_kspace:
            sims = [
                sim for sim in sims
                if (Path(BATCH_DATA_DIR) / batch_name / "kspace_predictions" / f"{sim}.pkl").exists()
            ]
    
        return sims

    def _list_timesteps(self, batch_name, simulation_hash):
        if not batch_name or not simulation_hash:
            return []
        sim_dir = Path(BATCH_DATA_DIR) / batch_name / "simulations" / simulation_hash / "data"
        if not sim_dir.exists():
            return []
        # Create a list of dictionaries with 'text' for display and 'value' for backend use
        timestep_files = []
        for f in sim_dir.glob("*.pkl"):
            # Extract the number after 't' and remove leading zeros
            try:
                display_num = int(f.stem.split('_t')[-1]).__str__()
            except (ValueError, IndexError):
                # Fallback in case of unexpected file names
                display_num = f.name
            timestep_files.append({"text": display_num, "value": f.name})
    
        # Sort the list of dictionaries numerically based on the 'text' value
        timestep_files.sort(key=lambda item: int(item["text"]))
    
        return timestep_files

    def _refresh_batches(self):
        self.state.batch_list = self._list_batches()

    def _refresh_simulations(self, batch_name):
        sims = self._list_simulations(batch_name)
        self.state.simulation_hash_list = sims
        self.state.simulation_hash = None

    def _refresh_timesteps(self, batch_name, simulation_hash):
        timesteps = self._list_timesteps(batch_name, simulation_hash)
        self.state.timestep_list = timesteps
        self.state.timestep_data = None

    def sync_views(self, **kwargs):
        """Sync plotter_pred camera to plotter_wave."""
        if self.plotter_wave and self.plotter_pred:
            self.plotter_pred.camera_position = self.plotter_wave.camera_position
            self.plotter_pred.render()
            self.logger.info("Synchronized plotter_pred camera to plotter_wave.")

    def _build_ui(self):
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.set_text("SBI")
            layout.drawer_width = 300
    
            # ----------------------
            # Drawer content
            # ----------------------
            with layout.drawer:
                v3.VSwitch(
                    label="Show only simulations with k-space predictions",
                    v_model=("filter_kspace", False),
                )
                v3.VSelect(
                    label="Select Batch",
                    items=("batch_list",),
                    v_model=("batch_name", None),
                    clearable=True,
                )
                v3.VSelect(
                    label="Select Simulation",
                    items=("simulation_hash_list",),
                    v_model=("simulation_hash", None),
                    clearable=True,
                )
                v3.VSelect(
                    label="Select Timestep",
                    items=("timestep_list",),
                    v_model=("timestep_data", None),
                    item_title="text",
                    item_value="value",
                    clearable=True,
                )

                v3.VDivider()
                
                with v3.VCard(flat=True, style="max-height: 40vh; overflow-y: auto;"):
                    v3.VCardTitle("Simulation Parameters")
                
                    with v3.VExpansionPanels(multiple=True):
                        # iterate over sections: [section, params] pairs from Object.entries(...)
                        with v3.VExpansionPanel(
                            v_for="([section, params]) in Object.entries(simulation_parameters_dict)",
                            key=("section",),
                        ):
                            v3.VExpansionPanelTitle("{{ section }}")
                            with v3.VExpansionPanelText():
                                with v3.VList(dense=True):
                                    # iterate over key/value pairs inside each section
                                    with v3.VListItem(
                                        v_for="([key, value]) in Object.entries(params)",
                                        key=("key",),
                                    ):
                                        v3.VListItemTitle("{{ key }}")
                                        v3.VListItemSubtitle(
                                            "{{ value }}",
                                            style="white-space: pre-wrap; font-family: monospace;",
                                        )



            # Content with tabs
            # ----------------------
            with layout.content:
                with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                    # Track active tab
                    v_model_tab = "active_tab"
                    self.server.state[v_model_tab] = "image"
    
                    # ----------------------
                    # Tabs header
                    # ----------------------
                    with v3.VTabs(
                        v_model=(v_model_tab, "image"),
                        centered=True,
                        style="max-width: 600px; margin: 0 auto;"
                    ):
                        v3.VTab("Image", value="image")
                        v3.VTab("Simulation", value="simulation")
                        v3.VTab("Data", value="data")                    # ----------------------
                    # Tabs content
                    # ----------------------
                    with v3.VWindow(
                        v_model=(v_model_tab,),
                        style="""
                            height: calc(100vh - 64px);
                            width: 100%;
                        """,
                                       ):

                        with v3.VWindowItem(value="image", style="height: 100%"):
                            # ... (inside v3.VWindowItem(value="image"))
                            with v3.VCol(classes="fill-height", style="display: flex; flex-direction: column;"):
                                # Plots row stretches to fill available space
                                with v3.VRow(classes="flex-grow-1"):
                                    with v3.VCol(classes="fill-height"):
                                        plotter_ui(
                                            self.plotter_wave,
                                            server=self.server,
                                            add_menu=False,
                                            height="100%",
                                        )
                                    with v3.VCol(classes="fill-height", v_if=("show_prediction",)):
                                        plotter_ui(
                                            self.plotter_pred,
                                            server=self.server,
                                            add_menu=False,
                                            height="100%",
                                        )
                                
                                # Wrap the button in a VRow to constrain its height and width
                                with v3.VRow(dense=True, classes="justify-center mt-2", style="max-height: 10vh"): # Aligns the button to the right
                                    v3.VBtn(
                                        "Sync Views",
                                        click=self.sync_views,
                                        # block=True,  <-- REMOVE THIS
                                        color="primary",
                                    )    
                        # ---------- SIMULATION TAB ----------
                        with v3.VWindowItem(value="simulation", style="height: 100%"):
                            plotter_ui(
                                self.plotter,
                                server=self.server,
                                add_menu=False,
                                height="100%",
                            )
    
                        # ---------- DATA TAB ----------
                        with v3.VWindowItem(value="data", style="height: 100%"):
                            with v3.VRow(classes="fill-height align-center justify-center"):
                                with v3.VCol(classes="d-flex align-center justify-center"):
                                    self.sensor_matrix_widget = matplotlib.Figure(figure=None)
                                    self.sensor_matrix_widget.update(plt.figure())
                                with v3.VCol(classes="d-flex align-center justify-center"):
                                    self.energy_widget = matplotlib.Figure(figure=None)
                                    self.energy_widget.update(plt.figure())
