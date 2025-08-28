import numpy as np
import pyvista as pv
from wave_map.emulator.ellipsoid import Ellipsoid


class EllipsoidSimilarityMeasurer:
    """
    Measures similarity between two ellipsoids of revolution.

    Provides methods to compute intersection over union (IoU) between
    a predicted and true ellipsoid, as well as optionally other similarity metrics.
    """

    def __init__(self, num_samples: int = 50000, center=(0.5, 0.5, 0.5)):
        """
        Args:
            num_samples: Number of points to sample for Monte Carlo IoU estimation
            center: Center of the ellipsoids (default: cube center)
        """
        self.num_samples = num_samples
        self.center = center

    def compute_iou(self, true_params, pred_params):
        """
        Compute the IoU similarity between two ellipsoids given their parameter vectors.

        Args:
            true_params: array-like [density, wavespeed, semi_major, semi_minor, dir_x, dir_y, dir_z]
            pred_params: array-like, same format

        Returns:
            iou: float between 0 and 1
        """
        true_ellipsoid = Ellipsoid.from_params_list(true_params, center=self.center)
        pred_ellipsoid = Ellipsoid.from_params_list(pred_params, center=self.center)

        return self._estimate_iou(true_ellipsoid, pred_ellipsoid)

    def _estimate_iou(self, true_ellipsoid: Ellipsoid, pred_ellipsoid: Ellipsoid) -> float:
        """
        Estimate intersection over union between two ellipsoids by sampling points.

        Args:
            true_ellipsoid: Ellipsoid instance (ground truth)
            pred_ellipsoid: Ellipsoid instance (prediction)

        Returns:
            iou: float between 0 and 1
        """
        # Intersection fractions
        inter_in_pred = sum(true_ellipsoid.is_inside(pred_ellipsoid.sample_point_inside())
                            for _ in range(self.num_samples))
        inter_in_true = sum(pred_ellipsoid.is_inside(true_ellipsoid.sample_point_inside())
                            for _ in range(self.num_samples))

        frac_pred_in_true = inter_in_pred / self.num_samples
        frac_true_in_pred = inter_in_true / self.num_samples

        # Volumes
        def volume(ellipsoid):
            a, b, c = ellipsoid.scaling
            return 4 / 3 * np.pi * a * b * c

        V_true = volume(true_ellipsoid)
        V_pred = volume(pred_ellipsoid)
        V_inter_pred = frac_pred_in_true * V_pred
        V_inter_true = frac_true_in_pred * V_true
        V_inter = 0.5 * (V_inter_pred + V_inter_true)

        # IoU
        return V_inter / (V_true + V_pred - V_inter)
