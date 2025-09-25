import numpy as np


class SensorPlacer:
    """
    Places sensors on the boundary of a cubic simulation box while ensuring
    they are not too close to any acoustic source.

    Parameters
    ----------
    box_size : float, default=0.25
        The size of the cubic simulation domain (edge length).
    top_sensors : int, optional
        Number of sensors on the top face (used with side_sensors).
    side_sensors : int, optional
        Number of sensors on one side face (used with top_sensors).
    sensors_per_face : int, optional
        Total number of sensors per face (must be a perfect square). If
        specified, overrides top_sensors/side_sensors.
    random_seed : int, default=42
        Random seed for reproducibility of Latin Hypercube Sampling.
    additional_sensors : list of [x, y, z], optional
        Additional fixed sensor positions to include.
    source_centers : array-like of shape (N, 3), optional
        List of source center coordinates in the domain.
    source_radii : array-like of shape (N,), optional
        List of source radii, one for each source center.
    """

    def __init__(self,
                 box_size=0.25,
                 top_sensors=None,
                 side_sensors=None,
                 sensors_per_face=None,
                 random_seed=42,
                 additional_sensors=None,
                 source_centers=None,
                 source_radii=None):
        self.box_size = box_size
        self.top_sensors = top_sensors
        self.side_sensors = side_sensors
        self.sensors_per_face = sensors_per_face
        self.additional_sensors = additional_sensors if additional_sensors is not None else []
        self.source_centers = np.array(source_centers) if source_centers is not None else np.empty((0, 3))
        self.source_radii = np.array(source_radii) if source_radii is not None else np.empty((0,))
        self.sensor_positions = []
        self.margin = 0.2

        if (sensors_per_face is not None) and (top_sensors is not None or side_sensors is not None):
            raise ValueError("Specify either (top_sensors and side_sensors) OR sensors_per_face, not both.")
        if (sensors_per_face is None) and (top_sensors is None or side_sensors is None):
            raise ValueError("You must specify either (top_sensors and side_sensors) OR sensors_per_face.")

        np.random.seed(random_seed)

    def latin_hypercube_2d(self, n_samples, x_min, x_max, y_min, y_max):
        x_strata = np.linspace(x_min, x_max, n_samples + 1)
        y_strata = np.linspace(y_min, y_max, n_samples + 1)
        x = np.random.uniform(x_strata[:-1], x_strata[1:])
        y = np.random.uniform(y_strata[:-1], y_strata[1:])
        np.random.shuffle(x)
        np.random.shuffle(y)
        return np.column_stack((x, y))

    def get_sensor_grid(self, sensors_per_face):
        grid_n = int(np.sqrt(sensors_per_face))
        if grid_n ** 2 != sensors_per_face:
            raise ValueError("sensors_per_face must be a perfect square.")

        grid = np.linspace(self.margin, self.box_size - self.margin, grid_n)
        gx, gy = np.meshgrid(grid, grid)
        grid_points = np.column_stack((gx.ravel(), gy.ravel()))

        # All 6 faces
        top = np.column_stack((grid_points[:, 0], grid_points[:, 1], np.full(len(grid_points), self.box_size)))
        bottom = np.column_stack((grid_points[:, 0], grid_points[:, 1], np.zeros(len(grid_points))))
        xp = np.column_stack((np.full(len(grid_points), self.box_size), grid_points[:, 0], grid_points[:, 1]))
        xn = np.column_stack((np.zeros(len(grid_points)), grid_points[:, 0], grid_points[:, 1]))
        yp = np.column_stack((grid_points[:, 0], np.full(len(grid_points), self.box_size), grid_points[:, 1]))
        yn = np.column_stack((grid_points[:, 0], np.zeros(len(grid_points)), grid_points[:, 1]))

        return np.vstack((top, bottom, xp, xn, yp, yn))

    def get_lhs_sensor_coordinates(self):
        top_samples = self.latin_hypercube_2d(self.top_sensors, 0, self.box_size, 0, self.box_size)
        top_positions = np.column_stack((top_samples, np.full(self.top_sensors, self.box_size)))
        top_center = np.array([[self.box_size / 2, self.box_size / 2, self.box_size]])
        side_samples = self.latin_hypercube_2d(self.side_sensors, 0, self.box_size, 0, self.box_size)
        side_positions = np.column_stack((np.full(self.side_sensors, self.box_size), side_samples))
        return np.vstack((top_positions, top_center, side_positions))

    def _filter_near_source(self, sensors):
        """
        Removes sensors within 2 * radius of any source center.
        """
        mask = np.ones(len(sensors), dtype=bool)
        for center, radius in zip(self.source_centers, self.source_radii):
            distances = np.linalg.norm(sensors - center, axis=1)
            mask &= distances >= 2 * radius
        return sensors[mask]

    def get_sensor_coordinates(self):
        if self.sensors_per_face is not None:
            base_sensors = self.get_sensor_grid(self.sensors_per_face)
        else:
            base_sensors = self.get_lhs_sensor_coordinates()

        # Remove sensors near sources (only for base sensors)
        filtered_sensors = self._filter_near_source(base_sensors)

        # Add additional sensors after filtering
        if len(self.additional_sensors) > 0:
            filtered_sensors = np.vstack((filtered_sensors, np.array(self.additional_sensors)))

        self.sensor_positions = filtered_sensors
        return [list(pos) for pos in filtered_sensors]
