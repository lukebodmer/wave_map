import os
import json
from contextlib import redirect_stdout, redirect_stderr
import tomli

import numpy as np
from sklearn.model_selection import KFold
from wave_map.PyRobustGaSP import PyRobustGaSP
from logging import getLogger
from importlib import resources

from wave_map.emulator.ellipsoid_similarity_measurer import EllipsoidSimilarityMeasurer
from wave_map.emulator.parallel_partial_emulator import ParallelPartialEmulator


class ResultsValidator:
    """
    Perform k-fold cross-validation on a batch of emulator data and log results.
    Success is measured using material properties and ellipsoid IoU.
    """

    def __init__(self, inputs, outputs, simulation_ids, n_splits=10, random_state=42):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.simulation_ids = np.array(simulation_ids)
        self.n_splits = n_splits
        self.random_state = random_state

        self.material_list_file = resources.files("wave_map.config") / "material_properties.toml"
        self._get_materials_list()
        self.similarity_measurer = EllipsoidSimilarityMeasurer(num_samples=10000)
        self.fold_results = []
        self.prediction_data = {}  # Store all prediction data for file logging

        # Use the globally configured logger by name
        self.logger = getLogger("emulatorlog")

    def _get_materials_list(self):
        with open(self.material_list_file, "rb") as f:
            materials_data = tomli.load(f)

        self.materials = []
        for category, materials in materials_data.items():
            self.materials.append({
                "name": f"{category}",
                "density": materials["density"],
                "wavespeed": materials["wavespeed"]
            })

    def _store_predictions(self, fold_index, test_idx, predictions, y_test, X_test):
        """
        Store prediction data (including IoU) for each test sample in the current fold.
        Returns a list of per-sample results (true params, pred params, iou).
        """
        per_sample_results = []

        for idx, pred, actual, x_input in zip(test_idx, predictions, y_test, X_test):
            sim_id = self.simulation_ids[idx]

            # Compute IoU
            iou = self.similarity_measurer.compute_iou(actual, pred)

            if sim_id not in self.prediction_data:
                self.prediction_data[sim_id] = {
                    'true_params': actual.tolist(),
                    'predictions': []
                }

            self.prediction_data[sim_id]['predictions'].append({
                'fold': fold_index,
                'predicted_params': pred.tolist(),
                'iou': float(iou)
            })

            per_sample_results.append((actual, pred, iou))

        return per_sample_results

    def run_k_fold_validation(self):
        #kf = KFold(n_splits=self.n_splits, shuffle=False)  # , random_state=self.random_state)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        self.logger.info(f"Running {self.n_splits}-fold cross-validation on emulator data...")

        for fold_index, (train_idx, test_idx) in enumerate(kf.split(self.outputs), start=1):
            self.logger.info(f"\nFold {fold_index}/{self.n_splits}")

            X_train, X_test = self.outputs[train_idx], self.outputs[test_idx]
            y_train, y_test = self.inputs[train_idx], self.inputs[test_idx]

            self.logger.info("...training model")
            model = self._train_ppe_model(X_train, y_train)

            #emulator = ParallelPartialEmulator(X_train, y_train)
            #emulator.train()
            #model.fit()

            self.logger.info("...making predictions")
            predictions = PyRobustGaSP().predict_ppgasp(model, X_test)['mean']
            #results = emulator.predict(X_test)["mean"]

            self.logger.info("...evaluaing success")
            # Store prediction data and get per-sample results
            per_sample_results = self._store_predictions(fold_index, test_idx, predictions, y_test, X_test)
            # Evaluate success using stored IoU values
            fold_success = self._evaluate_success(per_sample_results)

            self.logger.info(f"  Density success:         {fold_success['density_success']:.2f}")
            self.logger.info(f"  Wavespeed success:       {fold_success['wavespeed_success']:.2f}")
            self.logger.info(f"  Bulk Modulus success:    {fold_success['bulk_modulus_success']:.2f}")
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
        self._save_prediction_data()

    def _train_ppe_model(self, X_train, y_train):
        P_rgasp = PyRobustGaSP()
        #task = P_rgasp.create_task(X_train, y_train)

        task = P_rgasp.create_task(X_train,
                                   y_train,
                                   isotropic=True,
                                   #prior_choice='ref_xi',
                                   #optimization="nelder-mead",
                                   #max_eval=max(30, 20 + 5 * X_train.shape[1]),
                                   num_initial_values=10,
                                   #kernel_type=["matern_3_2"],
                                   #nugget=1e-6,)
                                   nugget_est=True)

        # Suppress console output
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                model = P_rgasp.train_ppgasp(task)
        #model = P_rgasp.train_ppgasp(task)

        return model

    def _closest_material(self, density, wavespeed):
        """Find the closest material by Euclidean distance in (density, wavespeed)."""
        best_match = None
        best_dist = float("inf")
        for mat in self.materials:
            dist = np.sqrt((mat["density"] - density)**2 + (mat["wavespeed"] - wavespeed)**2)
            if dist < best_dist:
                best_dist = dist
                best_match = mat["name"]
        return best_match

    def _evaluate_success(self, per_sample_results):
        density_success_count = 0
        wavespeed_success_count = 0
        bulk_modulus_success_count = 0
        #material_success_count = 0
        shape_success_count = 0
        overall_success_count = 0

        material_success_tolerance = 0.2
        bulk_modulus_success_tolerance = 5.6
        shape_success_tolerance = 0.9

        for true_params, pred_params, iou in per_sample_results:
            # predicted and actual values
            pred_density, pred_wavespeed = pred_params[:2]
            true_density, true_wavespeed = true_params[:2]

            true_bulk_modulus = true_density * true_wavespeed**2
            pred_bulk_modulus = pred_density * pred_wavespeed**2

            # density & wavespeed tolerance checks
            density_success = abs(pred_density - true_density) < material_success_tolerance
            wavespeed_success = abs(pred_wavespeed - true_wavespeed) < material_success_tolerance
            bulk_modulus_success = abs(pred_bulk_modulus - true_bulk_modulus) < bulk_modulus_success_tolerance

            # find closest materials
            #density_weight = 10
            #pred_material = self._closest_material(density_weight*pred_density, pred_wavespeed)
            #true_material = self._closest_material(density_weight*true_density, true_wavespeed)

            #material_success = (pred_material == true_material)

            shape_success = iou > shape_success_tolerance

            # update counters
            density_success_count += density_success
            wavespeed_success_count += wavespeed_success
            bulk_modulus_success_count += bulk_modulus_success
            #material_success_count += material_success
            shape_success_count += shape_success
            #overall_success_count += material_success and shape_success

            self.logger.debug(
                f"Pred density={pred_density:.1f}, true={true_density:.1f}, "
                f"Pred wavespeed={pred_wavespeed:.1f}, true={true_wavespeed:.1f}, "
                f"Pred bulk modulus ={pred_bulk_modulus:.1f}, true={true_bulk_modulus:.1f}, "
            #    f"Pred material={pred_material}, true material={true_material}"
            )

        n = len(per_sample_results)
        return {
            "density_success": density_success_count / n,
            "wavespeed_success": wavespeed_success_count / n,
            "bulk_modulus_success": bulk_modulus_success_count / n,
            #"material_success": material_success_count / n,
            "shape_success": shape_success_count / n,
            "overall_success": overall_success_count / n
        }

    def _save_prediction_data(self):
        """Save all prediction data to file"""
        with open('emulator_prediction_results.json', 'w') as f:
            json.dump(self.prediction_data, f, indent=2)

    def _log_summary(self):
        density_list = [fold["fold_success"]["density_success"] for fold in self.fold_results]
        wavespeed_list = [fold["fold_success"]["wavespeed_success"] for fold in self.fold_results]
        bulk_modulus_list = [fold["fold_success"]["bulk_modulus_success"] for fold in self.fold_results]
        #material_list = [fold["fold_success"]["material_success"] for fold in self.fold_results]
        shape_list = [fold["fold_success"]["shape_success"] for fold in self.fold_results]
        #overall_list = [fold["fold_success"]["overall_success"] for fold in self.fold_results]

        self.logger.info("\n=== Cross-Validation Success Summary ===")
        self.logger.info(
            f"Density success:   avg={np.mean(density_list):.2f}, min={np.min(density_list):.2f}, max={np.max(density_list):.2f}"
        )
        self.logger.info(
            f"Wavespeed success: avg={np.mean(wavespeed_list):.2f}, min={np.min(wavespeed_list):.2f}, max={np.max(wavespeed_list):.2f}"
        )
        self.logger.info(
            f"Bulk Modulus success: avg={np.mean(bulk_modulus_list):.2f}, min={np.min(bulk_modulus_list):.2f}, max={np.max(bulk_modulus_list):.2f}"
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
