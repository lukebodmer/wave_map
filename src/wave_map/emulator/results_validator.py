import sys
import os
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from sklearn.model_selection import KFold
from wave_map.PyRobustGaSP import PyRobustGaSP
import logging

from wave_map.emulator.ellipsoid_similarity_measurer import EllipsoidSimilarityMeasurer


class ResultsValidator:
    """
    Perform k-fold cross-validation on a batch of emulator data and log results.
    Success is measured using material properties and ellipsoid IoU.
    """

    def __init__(self, inputs, outputs, simulation_ids, n_splits=5, random_state=42, logger_name="emulatorlog"):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.simulation_ids = np.array(simulation_ids)
        self.n_splits = n_splits
        self.random_state = random_state
        self.similarity_measurer = EllipsoidSimilarityMeasurer(num_samples=10000)
        self.fold_results = []

        # Use the globally configured logger by name
        self.logger = logging.getLogger(logger_name)

    def run_k_fold_validation(self):
        kf = KFold(n_splits=self.n_splits, shuffle=False)#, random_state=self.random_state)
        self.logger.info(f"Running {self.n_splits}-fold cross-validation on emulator data...")

        for fold_index, (train_idx, test_idx) in enumerate(kf.split(self.outputs), start=1):
            self.logger.info(f"\nFold {fold_index}/{self.n_splits}")

            X_train, X_test = self.outputs[train_idx], self.outputs[test_idx]
            y_train, y_test = self.inputs[train_idx], self.inputs[test_idx]

            model = self._train_ppe_model(X_train, y_train)
            predictions = PyRobustGaSP().predict_ppgasp(model, X_test)['mean']

            # Evaluate success metrics
            fold_success = self._evaluate_success(y_test, predictions)

            self.logger.info(f"  Density success:         {fold_success['density_success']:.2f}")
            self.logger.info(f"  Wavespeed success:       {fold_success['wavespeed_success']:.2f}")
            #self.logger.info(f"  Material success:        {fold_success['material_success']:.2f}")
            self.logger.info(f"  Shape success:           {fold_success['shape_success']:.2f}")
            #self.logger.info(f"  Overall success:         {fold_success['overall_success']:.2f}")


            self.fold_results.append({
                "fold": fold_index,
                "predictions": predictions,
                "actual": y_test,
                "fold_success": fold_success,
                "X_test": X_test
            })

        self._log_summary()

    def _train_ppe_model(self, X_train, y_train):
        P_rgasp = PyRobustGaSP()
        task = P_rgasp.create_task(X_train, y_train)
    
        # Suppress console output
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                model = P_rgasp.train_ppgasp(task)
    
        return model

    def _evaluate_success(self, y_true, y_pred):
        density_success_count = 0
        wavespeed_success_count = 0
        material_success_count = 0
        shape_success_count = 0
        overall_success_count = 0
    
        for true_params, pred_params in zip(y_true, y_pred):
            density_success = abs(pred_params[0] - true_params[0]) < 0.2
            wavespeed_success = abs(pred_params[1] - true_params[1]) < 0.2
            material_success = density_success and wavespeed_success
    
            iou = self.similarity_measurer.compute_iou(true_params, pred_params)
            shape_success = iou > 0.8
    
            density_success_count += density_success
            wavespeed_success_count += wavespeed_success
            material_success_count += material_success
            shape_success_count += shape_success
            overall_success_count += material_success and shape_success
    
        n = len(y_true)
        return {
            "density_success": density_success_count / n,
            "wavespeed_success": wavespeed_success_count / n,
            "material_success": material_success_count / n,
            "shape_success": shape_success_count / n,
            "overall_success": overall_success_count / n
        }

    def _log_summary(self):
        density_list = [fold["fold_success"]["density_success"] for fold in self.fold_results]
        wavespeed_list = [fold["fold_success"]["wavespeed_success"] for fold in self.fold_results]
        material_list = [fold["fold_success"]["material_success"] for fold in self.fold_results]
        shape_list = [fold["fold_success"]["shape_success"] for fold in self.fold_results]
        overall_list = [fold["fold_success"]["overall_success"] for fold in self.fold_results]
    
        self.logger.info("\n=== Cross-Validation Success Summary ===")
        self.logger.info(
            f"Density success:   avg={np.mean(density_list):.2f}, min={np.min(density_list):.2f}, max={np.max(density_list):.2f}"
        )
        self.logger.info(
            f"Wavespeed success: avg={np.mean(wavespeed_list):.2f}, min={np.min(wavespeed_list):.2f}, max={np.max(wavespeed_list):.2f}"
        )
        #self.logger.info(
        #    f"Material success:  avg={np.mean(material_list):.2f}, min={np.min(material_list):.2f}, max={np.max(material_list):.2f}"
        #)
        self.logger.info(
            f"Shape success:     avg={np.mean(shape_list):.2f}, min={np.min(shape_list):.2f}, max={np.max(shape_list):.2f}"
        )
        #self.logger.info(
        #    f"Overall success:   avg={np.mean(overall_list):.2f}, min={np.min(overall_list):.2f}, max={np.max(overall_list):.2f}"
        #)
