import numpy as np
import matplotlib.pyplot as plt


class SensorPlacer:
    def __init__(self,
                 box_size=0.25,
                 top_sensors=0,
                 side_sensors=0,
                 random_seed=42,
                 additional_sensors=None):
        """
        Initialize the SensorPlacer with cube dimensions and sensor counts.

        Parameters:
        - box_size (float): Side length of the cube (default: 0.25m).
        - top_sensors (int): Number of sensors on the top face (default: 15)
        - side_sensors (int): Number of sensors on the side face (default: 10).
        """
        self.box_size = box_size
        self.top_sensors = top_sensors
        self.side_sensors = side_sensors
        self.additional_sensors = additional_sensors if additional_sensors is not None else []
        self.sensor_positions = []
        np.random.seed(random_seed)  # Ensure reproducibility

    def latin_hypercube_2d(self, n_samples, x_min, x_max, y_min, y_max):
        """
        Generate 2D Latin Hypercube Samples within [x_min, x_max] × [y_min, y_max].

        Parameters:
        - n_samples (int): Number of samples.
        - x_min, x_max (float): Range for x-coordinate.
        - y_min, y_max (float): Range for y-coordinate.

        Returns:
        - np.ndarray: Array of shape (n_samples, 2) with sampled (x, y) points.
        """
        # Create strata
        x_strata = np.linspace(x_min, x_max, n_samples + 1)
        y_strata = np.linspace(y_min, y_max, n_samples + 1)

        # Sample one point per stratum
        x = np.random.uniform(x_strata[:-1], x_strata[1:])
        y = np.random.uniform(y_strata[:-1], y_strata[1:])

        # Shuffle to avoid correlation between x and y
        np.random.shuffle(x)
        np.random.shuffle(y)

        return np.column_stack((x, y))

    def get_sensor_coordinates(self):
        """
        Place sensors on the top and side faces using LHS.

        Returns:
        - list: List of sensor positions as [x, y, z] coordinates.
        """
        # top face (z = box_size)
        top_samples = self.latin_hypercube_2d(
            self.top_sensors, 0, self.box_size, 0, self.box_size
        )
        top_positions = np.column_stack((top_samples, np.full(self.top_sensors, self.box_size)))
        # Top face (y = box_size): Add center sensor
        top_center = np.array([[self.box_size / 2, self.box_size/2, self.box_size]])


        # Side face (x = box_size)
        side_samples = self.latin_hypercube_2d(
            self.side_sensors, 0, self.box_size, 0, self.box_size
        )
        side_positions = np.column_stack((np.full(self.side_sensors, self.box_size), side_samples))

        # Combine base sensors
        base_sensors = np.vstack((top_positions, side_positions, top_center))

        # Add additional sensors only if the list is not empty
        if len(self.additional_sensors) > 0:
            self.sensor_positions = np.vstack((base_sensors, np.array(self.additional_sensors)))
        else:
            self.sensor_positions = base_sensors

        return [list(pos) for pos in self.sensor_positions]


# Example usage
if __name__ == "__main__":
    # Initialize with default values (0.25m box, 15 top sensors, 10 side sensors)
    placer = SensorPlacer(box_size=0.25, top_sensors=15, side_sensors=10)

    # Place sensors and get coordinates
    sensor_coords = placer.place_sensors()

    # Print sensor positions
    print("Sensor positions (x, y, z):")
    for i, pos in enumerate(sensor_coords):
        print(f"Sensor {i+1}: {pos}")
