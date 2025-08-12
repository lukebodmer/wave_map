import glob
import pickle
import tomli
import panel as pn
import matplotlib.pyplot as plt
import pyvista as pv
import gmsh

from pathlib import Path
from wave_simulator.visualizer import Visualizer

pn.extension('vtk')

class UserInterface:
    def __init__(self, outputs_dir='data/outputs'):
        self.outputs_dir = Path(outputs_dir)

        self.sim_families = self._get_sim_families()
        family_options = [""] + self.sim_families

        self.selected_family = None
        self.family_selector = pn.widgets.Select(name='Simulation Family', options=family_options, value="")
        self.family_selector.param.watch(self._update_family, 'value')

        self.sim_folders = []  # 👈 Initialize empty, will be set when family is selected
        sim_options = [""]  # 👈 No folders yet

        self.selected_folder = None
        self.data_files = []
        self.visualizer = None

        self.sim_selector = pn.widgets.Select(name='Simulation Run', options=sim_options, value="")
        self.frame_selector = pn.widgets.Select(name='Timestep', options=[])
        self.refresh_button = pn.widgets.Button(name='Load Data', button_type='primary')
        self.status_text = pn.pane.HTML("", height=20)

        self.sim_selector.param.watch(self._update_folder, 'value')
        self.refresh_button.on_click(self._load_frame)

        self.parameters_pane = pn.pane.HTML("<i>No parameters loaded.</i>", width=300)
        self.runtime_pane = pn.pane.HTML("", width=300)

        self.show_3d_button = pn.widgets.Button(name='Show 3D', button_type='success')
        self.show_3d_button.on_click(self._show_3d)

        self.content = pn.Row()
        self.panel = pn.Row(
            pn.Column(
                pn.pane.HTML("<h2 style='margin-top: 4px; margin-bottom: 0px;'>Simulation Viewer</h2>"),
                self.family_selector,  # <- Add family selector here too!
                self.sim_selector,
                self.frame_selector,
                self.refresh_button,
                self.show_3d_button,
                self.status_text,
                pn.pane.HTML("<h3 style='margin-bottom: 4px; margin-top: 12px;'>Simulation Parameters</h3>", height=20),
                self.parameters_pane,
                pn.pane.HTML("<h3 style='margin-bottom: 4px; margin-top: 12px;'>Runtime</h3>", height=20),
                self.runtime_pane,
            ),
            pn.Column(
                self.content
            )
        )

    def _get_sim_families(self):
        return sorted([
            f.name for f in self.outputs_dir.iterdir()
            if f.is_dir()
        ])
    
    def _get_sim_folders(self, family_name):
        family_path = self.outputs_dir / family_name
        if not family_path.exists():
            return []
        return sorted([
            f.name for f in family_path.iterdir()
            if f.is_dir()
        ])

    def _update_family(self, event):
        if not event.new:
            self.sim_selector.options = [""]
            self.sim_selector.value = ""
            self.selected_family = None
            return
    
        self.selected_family = event.new
        folders = self._get_sim_folders(event.new)
        self.sim_selector.options = [""] + folders
        self.sim_selector.value = ""

    def _update_folder(self, event):
        if not event.new:
            self.selected_folder = None
            self.data_files = []
            self.status_text.object = "<span style='color:gray'>Select a simulation run.</span>"
            self.frame_selector.options = []
            self.frame_selector.value = None
            self.refresh_button.disabled = True
            return

        if not self.selected_family:
            self.status_text.object = "<span style='color:red'>⚠️ No simulation family selected.</span>"
            return
        
        self.selected_folder = self.outputs_dir / self.selected_family / event.new

        data_dir = self.selected_folder / "data"
        self.data_files = sorted(data_dir.glob("*.pkl"))

        if not self.data_files:
            self.status_text.object = f"<span style='color:red'>⚠️ No .pkl files found in: {data_dir}</span>"
            self.frame_selector.options = []
            self.frame_selector.value = None
            self.refresh_button.disabled = True
        else:
            self.status_text.object = f"<span style='color:green'>✅ Found {len(self.data_files)} data files.</span>"
            self.timestep_map = {}
            options = []
            for path in self.data_files:
                time_str = path.name.split("_t")[-1].split(".")[0]
                label = f"t = {time_str}"
                self.timestep_map[label] = path
                options.append(label)

            self.frame_selector.options = options
            self.frame_selector.value = options[-1]
            self.refresh_button.disabled = False

        param_file = self.selected_folder / "parameters.toml"
        if param_file.exists():
            html = self._format_parameters(param_file)
            self.parameters_pane.object = html
        else:
            self.parameters_pane.object = "<i>⚠️ No parameters.toml found.</i>"

    def _format_parameters(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                parameters = tomli.load(f)
        except Exception as e:
            return f"<b style='color: red;'>❌ Failed to load parameters.toml:</b> {e}"
    
        html = [
            """
            <div style="
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 13px;
                line-height: 1.3;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px 14px;
                max-width: 600px;
            ">
            """
        ]
    
        for section, values in parameters.items():
            html.append(
                f"<h4 style='color: #2c3e50; border-bottom: 1px solid #ccc; margin: 6px 0 4px; font-size: 14px;'>[{section}]</h4>"
            )
            html.append("<ul style='list-style: none; padding-left: 1em; margin: 0;'>")
            for key, value in values.items():
                html.append(
                    f"<li style='margin: 1px 0;'><span style='color: #34495e; font-weight: bold;'>{key}</span>: "
                    f"<span style='color: #2d3436;'>{value}</span></li>"
                )
            html.append("</ul>")
    
        html.append("</div>")
        return "\n".join(html)

    def _show_3d(self, event=None):
        if self.visualizer is None:
            self.status_text.object = "<span style='color:red'>⚠️ Load data before showing 3D.</span>"
            return
        try:
            self.visualizer.add_nodes_3d("p")
            self.visualizer._show_grid()
            self.visualizer.add_inclusion_boundary()
            self.visualizer.add_sensors()
            self.visualizer.show()
            self.status_text.object = "<span style='color:green'>✅ 3D view launched.</span>"
        except Exception as e:
            self.status_text.object = f"<span style='color:red'>❌ Error in 3D view: {e}</span>"

    def _load_frame(self, event=None):
        if not self.data_files:
            self.status_text.object = "<span style='color:red'>⚠️ No data loaded.</span>"
            return

        try:
            data_path = self.timestep_map[self.frame_selector.value]

            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            mesh_path = data['mesh_directory'] / "mesh.pkl"
            if mesh_path.exists():
                with open(mesh_path, 'rb') as mf:
                    mesh_data = pickle.load(mf)
            else:
                self.status_text.object = "<span style='color:red'>❌ mesh.pkl not found.</span>"
                return

            self.visualizer = Visualizer(mesh_data, data)
            tracked_fig = self.visualizer.plot_sensor_data_as_matrix()

            if data['save_energy_interval'] > 0:
                energy_fig = self.visualizer.plot_energy()
                energy_column = pn.Column(
                    pn.pane.HTML("<b>Energy Plot</b>"),
                    pn.pane.Matplotlib(energy_fig, tight=True, width=550),
                )
            else:
                print("no energy plot available")
                energy_column = pn.Column(
                    pn.pane.HTML("<b>Energy Plot</b>"),
                    pn.pane.HTML("<b>⚠️ No energy plot available.</b>"),
                )

            runtime_sec = data.get("runtime", None)
            if runtime_sec is not None:
                hours = int(runtime_sec // 3600)
                minutes = int((runtime_sec % 3600) // 60)
                seconds = int(runtime_sec % 60)
                runtime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                self.runtime_pane.object = f"""
                <div style='font-family: monospace; font-size: 12px; padding-top:0;'>
                <b>Elapsed Time</b>: {runtime_str} (hh:mm:ss)
                </div>
                """
            else:
                self.runtime_pane.object = "<i>⚠️ Runtime not recorded.</i>"

            time_step = int(data_path.name.split("_t")[-1].split(".")[0])
            image_path = self.selected_folder / "images" / f"t_{time_step:08d}.png"

            image_pane = (
                pn.pane.PNG(str(image_path), width=550)
                if image_path.exists()
                else pn.pane.HTML("<b>⚠️ No image found for this timestep.</b>")
            )

            layout = pn.Row(
                pn.Column(
                    pn.pane.HTML("<b>Sensors</b>"),
                    pn.pane.Matplotlib(tracked_fig, tight=True, height=700, width=700),
                ),
                pn.Column(
                    energy_column,
                    pn.pane.HTML("<b>Snapshot</b>"),
                    image_pane,
                )
            )

            self.content.objects = [layout]
            self.status_text.object = f"<span style='color:green'>✅ Loaded frame: {data_path.name}</span>"
        except Exception as e:
            self.status_text.object = f"<span style='color:red'>❌ Error loading frame: {e}</span>"

    def show(self):
        return self.panel

