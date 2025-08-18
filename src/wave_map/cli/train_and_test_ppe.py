import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import toml
import sys
from sklearn.model_selection import train_test_split

from wave_map.PyRobustGaSP import PyRobustGaSP
from wave_map.input_space_samplers.mcmc import McmcSampler


def load_simulation_data(batch_name):
    """Load all simulation data for the given batch."""
    project_root = Path(__file__).parent.parent.parent.parent
    batch_dir = project_root / f"data/simulation_batch_data/{batch_name}/simulations"
    
    if not batch_dir.exists():
        raise ValueError(f"Batch directory not found: {batch_dir}")
    
    inputs, outputs, simulation_ids = [], [], []
    
    for sim_dir in batch_dir.iterdir():
        if sim_dir.is_dir():
            param_file = sim_dir / "parameters.toml"
            sensor_file = sim_dir / "final_sensor_data.pkl"
            
            if param_file.exists() and sensor_file.exists():
                # Load parameters
                params = toml.load(param_file)
                
                input_features = [
                    params["material"]["inclusion_density"],      # [0.5, 2]
                    params["material"]["inclusion_wave_speed"],  # [0.5, 2]
                    params["mesh"]["inclusion_scaling"][0],      # [0.125, 0.375]
                    params["mesh"]["inclusion_scaling"][1],
                    params["mesh"]["inclusion_scaling"][2],
                    params["mesh"]["inclusion_rotation"][0],     # rotation vector (axis*angle)
                    params["mesh"]["inclusion_rotation"][1],
                    params["mesh"]["inclusion_rotation"][2]
                ]
                
                with open(sensor_file, 'rb') as f:
                    sensor_data = pickle.load(f)
                
                inputs.append(input_features)
                outputs.append(sensor_data.flatten())
                simulation_ids.append(sim_dir.name)
    
    return np.array(inputs), np.array(outputs), simulation_ids


def main(batch_name="testers", test_split=0.05, random_state=42):
    """Train PPE model or load, then test with MCMC inverse search on test samples."""
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
        inputs, outputs, sim_ids = load_simulation_data(batch_name)
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            inputs, outputs, sim_ids, test_size=test_split, random_state=random_state
        )
        P_rgasp = PyRobustGaSP()
        task = P_rgasp.create_task(X_train, y_train, nugget_est=True, num_initial_values=3)
        model = P_rgasp.train_ppgasp(task)
        predictions = P_rgasp.predict_ppgasp(model, X_test)['mean']
        mse = np.mean((predictions - y_test)**2)
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
    
    # --- Inverse test with MCMC on all test cases ---
    print("\nRunning MCMC inverse search on all test cases...")

    # Build a prediction function that maps x -> predicted output (1D array)
    P_rgasp = PyRobustGaSP()
    def predict_fn(x: np.ndarray) -> np.ndarray:
        return P_rgasp.predict_ppgasp(model, x.reshape(1, -1))['mean'][0]

    # Create sampler (n_steps is passed to .sample(), not __init__)
    sampler = McmcSampler(
        model=predict_fn,
        #proposal_scale=np.array([0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
        proposal_scale=np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
        burn_in=200,
        thin=5,
        random_state=42
    )

    # Parameter bounds: density & wave_speed [0.5,2], scalings [0.125,0.375],
    # rotation components (axis*angle) bounded by [-pi, pi] and z in [0, pi]
    bounds = np.array([
        [0.5, 2.0],      # density
        [0.5, 2.0],      # wave speed
        [0.125, 0.375],  # scaling x
        [0.125, 0.375],  # scaling y
        [0.125, 0.375],  # scaling z
        [-np.pi, np.pi], # rotation x
        [-np.pi, np.pi], # rotation y
        [0.0, np.pi],    # rotation z (encourages +z hemisphere)
    ], dtype=float)


    for i, (true_input, target_output) in enumerate(zip(test_results['X_test'], test_results['actual'])):
        result = sampler.sample(
            target_output=target_output,
            init_input=true_input,   # or None for random start
            n_steps=5000,
            sigma=0.001,
            bounds=bounds
        )

        best_input = result["best_input"]
        # compute L2 error explicitly (easier to interpret than best_likelihood)
        pred_best = predict_fn(best_input)
        best_error = np.linalg.norm(pred_best - target_output)

        print(f"\nTest case {i+1}:")
        print(f"  True input:    {true_input}")
        print(f"  Best inferred: {best_input}")
        print(f"  L2 error vs. target output: {best_error:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_and_test_ppe.py <batch_name>")
        sys.exi
