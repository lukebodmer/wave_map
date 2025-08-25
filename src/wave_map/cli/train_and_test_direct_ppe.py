import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from wave_map.PyRobustGaSP import PyRobustGaSP

from wave_map.emulator.final_data_extractor import FinalDataExtractor


def load_simulation_data(batch_name):
    """Use FinalDataExtractor to load all simulation data for the given batch."""
    extractor = FinalDataExtractor(batch_name=batch_name)
    inputs, outputs, simulation_ids = extractor.load()
    return inputs, outputs, simulation_ids


def main(batch_name="single_moving_sphere_variable_radius", test_split=0.05, random_state=42):
    """Train PPE model or load, then test its ability to directly solve the inverse problem."""
    batch_name="rotated_ellipsoid_of_revolution"
    print(f"Preparing PPE model for batch: {batch_name}")
    
    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / f"data/ppe_models/{batch_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_file = results_dir / "trained_ppe_model.pkl"
    test_results_file = results_dir / "test_results.pkl"
    
    # Load or train model
    if model_file.exists() and test_results_file.exists():
        print("Found saved model. Loading...")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(test_results_file, 'rb') as f:
            test_results = pickle.load(f)
    else:
        print("Training model from scratch...")
        # Note: here we flip roles: sensor_data (outputs) → inputs (parameters.toml)
        inputs, outputs, sim_ids = load_simulation_data(batch_name)

        # Swap roles for inverse problem
        X = outputs   # features are sensor data
        y = inputs    # labels are the parameters
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, sim_ids, test_size=test_split, random_state=random_state
        )

        P_rgasp = PyRobustGaSP()
        task = P_rgasp.create_task(X_train, y_train, nugget_est=True, num_initial_values=10)
        model = P_rgasp.train_ppgasp(task)

        predictions = P_rgasp.predict_ppgasp(model, X_test)['mean']
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)

        test_results = {
            'test_ids': ids_test,
            'predictions': predictions,
            'actual': y_test,
            'rmse': rmse,
            'mse': mse,
            'X_test': X_test
        }

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        with open(test_results_file, 'wb') as f:
            pickle.dump(test_results, f)

    # --- Direct inverse testing ---
    print("\nEvaluating inverse problem performance (PPE direct mapping)...")
    for pred, true in zip(test_results['predictions'][:5], test_results['actual'][:5]):
        pred_str = np.array2string(pred, precision=4, separator=',', suppress_small=True)
        true_str = np.array2string(true, precision=4, separator=',', suppress_small=True)
        print(f"\nPredicted input: {pred_str}")
        print(f"True input:      {true_str}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_and_test_ppe.py <batch_name>")
        sys.exit(1)
    #batch_name=sys.argv[1]
    batch_name="rotated_ellipsoid_of_revolution"
    main(batch_name=sys.argv[1])
