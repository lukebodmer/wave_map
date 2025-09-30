import os
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import TrameApp
from trame.decorators import change
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
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

        # get logger
        self.logger = Logger(log_path=Path(LOGGER_DIR) / "log.txt", name="guilog")

        # --- Browser tab title ---
        self.server.state.trame__title = "SBI"

        # --- PyVista plotter ---
        self.plotter = pv.Plotter()

        # --- State variables ---
        self.state.batch_name = None
        self.state.batch_list = []
        self.state.simulation_hash = None
        self.state.simulation_hash_list = []
        self.state.timestep_data = None
        self.state.timestep_list = []

        # Store loaded mesh
        self.mesh_data = None
        self.visualizer = None

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
        """Update timestep list when simulation changes."""
        self._refresh_timesteps(self.state.batch_name, simulation_hash)

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

        # Find mesh hash for this simulation
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

        # --- Initialize visualizer ---
        self.visualizer = Visualizer(self.mesh_data, timestep_data)
        #self.visualizer.add_nodes_3d("p")  # Replace "field" with your actual field name
        self.visualizer.add_wave_speed()  # Replace "field" with your actual field name

        # Replace plotter content
        self.plotter.clear()
        for actor in self.visualizer.plotter.actors.values():
            self.plotter.add_actor(actor)

    # --- Helpers ---
    def _list_batches(self):
        if not os.path.exists(BATCH_DATA_DIR):
            return []
        return [name for name in os.listdir(BATCH_DATA_DIR) if os.path.isdir(os.path.join(BATCH_DATA_DIR, name))]

    def _list_simulations(self, batch_name):
        if batch_name is None:
            return []
        sim_base = os.path.join(BATCH_DATA_DIR, batch_name, "simulations")
        if not os.path.exists(sim_base):
            return []
        return [name for name in os.listdir(sim_base) if os.path.isdir(os.path.join(sim_base, name))]

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

    # --- UI ---
    def _build_ui(self):
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.set_text("SBI")
            layout.drawer_width = 300

            # Drawer content
            with layout.drawer:
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
                    item_title="text",  # Specify the key for display text
                    item_value="value", # Specify the key for the item's actual value
                    clearable=True,
                )

            # Main content: PyVista plotter
            with layout.content:
                plotter_ui(
                    self.plotter,
                    server=self.server,
                    add_menu=False
                )
