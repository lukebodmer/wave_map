# File: src/wave_map/emulator/ellipsoid.py

import numpy as np
import pyvista as pv

class Ellipsoid:
    """
    Represents a rotated ellipsoid of revolution in 3D space.
    
    Attributes:
        density: float
        wavespeed: float
        center: np.array, coordinates of the center
        scaling: np.array, semi-major and semi-minor axes [a, b, b]
        rotation_matrix: np.array, 3x3 rotation aligning x-axis to direction vector
        transform: np.array, used for fast inside tests
    """

    def __init__(self, density, wavespeed, semi_major, semi_minor, direction, center=(0.5, 0.5, 0.5)):
        self.density = density
        self.wavespeed = wavespeed
        self.center = np.array(center)

        # Scaling in canonical coordinates: major axis along x
        self.scaling = np.array([semi_major, semi_minor, semi_minor])

        # Rotation: align x-axis with given direction
        dir_vec = np.array(direction)
        if np.linalg.norm(dir_vec) == 0:
            raise ValueError("Direction vector cannot be zero.")
        self.rotation_matrix = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), dir_vec)

        # Transform for inside test
        S_inv = np.diag(1.0 / self.scaling)
        self.transform = S_inv @ self.rotation_matrix.T

    @classmethod
    def from_params_list(cls, params, center=(0.5, 0.5, 0.5)):
        """
        Create an Ellipsoid instance from a parameter list:
        [density, wavespeed, semi_major, semi_minor, dir_x, dir_y, dir_z]
        """
        return cls(
            density=params[0],
            wavespeed=params[1],
            semi_major=params[2],
            semi_minor=params[3],
            direction=params[4:7],
            center=center
        )

    def is_inside(self, point):
        """
        Check if a point lies inside the ellipsoid.
        """
        shifted = point - self.center
        transformed = self.transform @ shifted
        return np.sum(transformed**2) <= 1.0

    def sample_point_inside(self):
        """
        Randomly sample a point inside the ellipsoid using rejection sampling.
        """
        while True:
            point = np.random.uniform(-1, 1, size=3)
            if np.linalg.norm(point) <= 1:
                break
        scaled = point * self.scaling
        rotated = self.rotation_matrix @ scaled
        return self.center + rotated

    def get_mesh(self, resolution=50):
        """
        Return a pyvista mesh of the ellipsoid (triangulated) for visualization.
        """
        sphere = pv.ParametricEllipsoid(1, 1, 1).triangulate()
        points = sphere.points.copy()
        points *= self.scaling
        points = points @ self.rotation_matrix.T
        points += self.center
        return pv.PolyData(points, sphere.faces)

    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        """
        Find rotation matrix that aligns vec1 to vec2.
        """
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.isclose(c, -1.0):
            # 180° rotation around perpendicular axis
            perp = np.array([1, 0, 0]) if not np.isclose(a[0], 1.0) else np.array([0, 1, 0])
            v = np.cross(a, perp)
            v /= np.linalg.norm(v)
            H = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            return -np.eye(3) + 2 * np.outer(v, v)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
        return R
