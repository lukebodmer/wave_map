import sys
import math
import pickle
import time
import numpy as np
from logging import getLogger
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer
from wave_simulator.spatial_evaluator import SpatialEvaluator


class Simulator:
    def __init__(self,
                 time_stepper: LowStorageRungeKutta,
                 output_path,
                 save_image_interval,
                 save_points_interval,
                 save_data_interval,
                 save_energy_interval,
                 pressure_reciever_locations,
                 u_velocity_reciever_locations,
                 v_velocity_reciever_locations,
                 w_velocity_reciever_locations,
                 mesh_directory):
        self.output_path = output_path
        self.time_stepper = time_stepper
        self.physics = self.time_stepper.physics
        self.mesh = self.physics.mesh
        self.mesh_directory = mesh_directory
        self.spatial_evaluator = SpatialEvaluator(self.mesh)
        self.t_final = self.time_stepper.t_final

        self.save_image_interval = save_image_interval
        self.save_data_interval = save_data_interval
        self.save_points_interval = save_points_interval
        self.save_energy_interval = save_energy_interval
        self.start_time = 0

        self.pressure_reciever_locations = pressure_reciever_locations
        self._get_sensor_information(
            pressure_reciever_locations,
            u_velocity_reciever_locations,
            v_velocity_reciever_locations,
            w_velocity_reciever_locations
        )

        self.energy_index = 0
        self._get_source_data()
        data = self._get_data()
        self.mesh_data = self._get_mesh_data(data)
        self.visualizer = Visualizer(self.mesh_data, data)
        if self.save_energy_interval:
            self.initialize_energy_array()
        else:
            self.energy_data = None
            self.kinetic_data = None
            self.potential_data = None

    def initialize_energy_array(self):
        if self.save_energy_interval == 0:
            return  # Don't initialize anything
        # add one to evaulate energy at zero
        self.num_readings = math.ceil(self.time_stepper.num_time_steps /
                                      self.save_energy_interval) + 1
        self.energy_data = np.zeros(self.num_readings)
        self.kinetic_data = np.zeros(self.num_readings)
        self.potential_data = np.zeros(self.num_readings)

    def _get_sensor_information(self, pressure=None, x=None, y=None, z=None):
        # subtrack one because we don't want 0th reading
        self.num_readings = math.ceil(self.time_stepper.num_time_steps /
                                      self.save_points_interval)
        self.column_index = 0
        self.tracked_fields = {}
        if pressure:
            self.tracked_fields["pressure"] = {
                "points": pressure,
                "data": np.zeros((len(pressure), self.num_readings)),
                "field_name": "p"
            }
        if x:
            self.tracked_fields["x"] = {
                "points": x,
                "data": np.zeros((len(x), self.num_readings)),
                "field_name": "u"
            }
        if y:
            self.tracked_fields["y"] = {
                "points": y,
                "data": np.zeros((len(y), self.num_readings)),
                "field_name": "v"
            }
        if z:
            self.tracked_fields["z"] = {
                "points": z,
                "data": np.zeros((len(z), self.num_readings)),
                "field_name": "w"
            }

    def _save_scheduled_outputs(self):
        t_step = self.time_stepper.current_time_step

        if self.save_image_interval and t_step % self.save_image_interval == 0:
            self._save_image()
        if self.save_points_interval and t_step % self.save_points_interval == 0:
            self._evaluate_sensor_data()
        if self.save_energy_interval and t_step % self.save_energy_interval == 0:
            self._save_energy()
        if self.save_data_interval and t_step % self.save_data_interval == 0:
            self._save_data()

    def run(self):
        # log run information
        logger = getLogger("simlog")
        run_hash = str(self.output_path).rsplit('/', 1)[-1].split('_', 1)[0]
        logger.info(f"......... Running simulation {run_hash} .........")

        # start run timer
        self.start_time = time.time()

        # Save energy timestep 0 (initial condition)
        self._save_energy()

        # main loop
        while self.time_stepper.current_time_step < self.time_stepper.num_time_steps:
            self.time_stepper.advance_time_step()
            self._save_scheduled_outputs()
            self._log_info()

        # Save final sensor data only if last timestep aligned with save_points_interval
        self._save_final_sensor_data()

        logger.info("\n..... Simulation completed with no errors .....")

    def _get_source_data(self):

        num_time_steps = self.time_stepper.num_time_steps
        self.source_data = np.zeros(num_time_steps)
        for timestep in range(num_time_steps):
            t = timestep*self.time_stepper.dt
            self.source_data[timestep] = self.physics._get_source_pressure(t)

    def _get_mesh_data(self, data):
        # Create minimal mesh data for visualization
        # Load mesh data
        mesh_data = data['mesh_directory'] / "mesh.pkl"
        if mesh_data.exists():
            with open(mesh_data, 'rb') as f:
                mesh_data = pickle.load(f)
        else:
            raise Exception("Error: Pickled mesh data (mesh.pkl) doesn't exist.") 
        return mesh_data

    def _get_data(self):
        # Include simulator tracking data if available
        simulator_data = {}
        if hasattr(self, 'tracked_fields') and self.tracked_fields:
            simulator_data['tracked_fields'] = self.tracked_fields
        if hasattr(self, 'energy_data') and self.energy_data is not None:
            simulator_data['energy_data'] = self.energy_data
        if hasattr(self, 'kinetic_data') and self.kinetic_data is not None:
            simulator_data['kinetic_data'] = self.kinetic_data
        if hasattr(self, 'potential_data') and self.potential_data is not None:
            simulator_data['potential_data'] = self.potential_data
        if hasattr(self, 'source_data') and self.source_data is not None:
            simulator_data['source_data'] = self.source_data

        data = {
            'current_time_step': self.time_stepper.current_time_step,
            'current_time': self.time_stepper.t,
            'dt': self.time_stepper.dt,
            't_final': self.t_final,
            'fields': {
                'p': self.physics.p,
                'u': self.physics.u,
                'v': self.physics.v,
                'w': self.physics.w
            },
            'simulator': simulator_data,
            'output_path': self.output_path,
            'save_image_interval': self.save_image_interval,
            'save_data_interval': self.save_data_interval,
            'save_points_interval': self.save_points_interval,
            'save_energy_interval': self.save_energy_interval,
            'mesh_directory': self.mesh_directory,
            'runtime': time.time() - self.start_time,
            'sensor_coordinates': self.pressure_reciever_locations
        }
        return data

    def _save_data(self):
        # initialize the save index for file naming
        if not hasattr(self, '_save_index'):
            self._save_index = 0

        data = self._get_data()
        timestep_str = f'{self.time_stepper.current_time_step:0>8}'
        save_index_str = f'{self._save_index:0>8}'
        file_name = f'{save_index_str}_t{timestep_str}.pkl'
        file_path = f'{self.output_path}/data/{file_name}'

        # save to file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # increment the save index after saving
        self._save_index += 1

    def _save_image(self):
        data = self._get_data()
        self.visualizer.set_data(self.mesh_data, data)
        self.visualizer.save()

    def _save_final_sensor_data(self):
        if "pressure" not in self.tracked_fields:
            return

        pressure_data = self.tracked_fields["pressure"]["data"]

        final_path = f"{self.output_path}/final_sensor_data.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(pressure_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _evaluate_sensor_data(self):
        for name, field in self.tracked_fields.items():
            values = field["data"]
            points = field["points"]
            field_array = getattr(self.physics, field["field_name"])

            for i, (x, y, z) in enumerate(points):
                values[i, self.column_index] = self.spatial_evaluator.eval_at_point(x, y, z, field_array)

        self.column_index += 1

    def _save_energy(self):
        # on page 37 in Hesthaven and warburton we see how the mass matrix can be
        # used to recover the energy (l2 norm) of the system
        # uT M u = || u ||^2
        # get nodal values
        j = self.mesh.jacobians[0,:]  # shape (K,)
        p = self.physics.p
        u = self.physics.u
        v = self.physics.v
        w = self.physics.w
        mass = self.mesh.reference_element_operators.mass_matrix
        num_cells = self.mesh.num_cells
        rho = self.mesh.density[0,:]  # shape (Np, K)
        c = self.mesh.speed[0,:]      # shape (Np, K)
        inv_bulk = 1.0 / (rho * (c**2))  # shape (Np, K

        # potential energy: (1/2) * p^2 / (rho * c^2)
        potential = np.array([p[:, i].T @ mass @ p[:, i] for i in range(num_cells)])
        potential = 0.5 * inv_bulk * j * potential
        self.potential_data[self.energy_index] = np.sum(potential)

        # kinetic energy: (1/2) * rho * (u^2 + v^2 + w^2)
        kinetic_u = np.array([u[:, i].T @ mass @ u[:, i] for i in range(num_cells)])
        kinetic_v = np.array([v[:, i].T @ mass @ v[:, i] for i in range(num_cells)])
        kinetic_w = np.array([w[:, i].T @ mass @ w[:, i] for i in range(num_cells)])
        kinetic = (0.5 * rho * j * (kinetic_u + kinetic_v + kinetic_w))
        self.kinetic_data[self.energy_index] = np.sum(kinetic)

        # total energy
        energy = np.sum(potential + kinetic)
        self.energy_data[self.energy_index] = energy

        self.energy_index += 1

    def _log_info(self):
        runtime = time.time() - self.start_time
        sys.stdout.write(f"\rTimestep: {self.time_stepper.current_time_step}, Time: {self.time_stepper.t:.6f}, Runtime: {runtime:.2f}s")
        sys.stdout.flush()
