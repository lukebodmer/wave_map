# --- imports remain the same ---
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
                    params["material"]["inclusion_density"],
                    params["material"]["inclusion_wave_speed"],
                    params["mesh"]["inclusion_scaling"][0],
                    params["mesh"]["inclusion_scaling"][1],
                    params["mesh"]["inclusion_scaling"][2],
                    params["mesh"]["inclusion_rotation"][0],
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
        task = P_rgasp.create_task(X_train, y_train, nugget_est=True, num_initial_values=5)
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

    P_rgasp = PyRobustGaSP()

    bounds = np.array([
        [0.5, 2.0], [0.5, 2.0], [0.125, 0.375], [0.125, 0.375], [0.125, 0.375],
        [-np.pi, np.pi], [-np.pi, np.pi], [0.0, np.pi]
    ], dtype=float)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return P_rgasp.predict_ppgasp(model, x.reshape(1, -1))['mean'][0]

    param_ranges = bounds[:, 1] - bounds[:, 0]
    proposal_scale = 0.10 * param_ranges  # 5% of range as starting point

    sampler = McmcSampler(
        model=predict_fn,
        proposal_scale=proposal_scale,
        burn_in=10,
        thin=1,
        random_state=42,
        use_simulated_annealing=True,
        initial_temp=100.0,  # Higher initial temp for exploration
        cooling_rate=0.9998, # slower cooling
        l2_threshold=0.01,
        likelihood_method="gaussian"
       )

    for i, (true_input, target_output) in enumerate(zip(test_results['X_test'], test_results['actual'])):
        if i == 0:
            continue
    
        # Print true input once at the start
        true_input_str = np.array2string(true_input, precision=4, separator=',', suppress_small=True)
        print(f"True input: {true_input_str}")

        # Multi-restart MCMC implementation
        n_restarts = 5
        max_steps_per_restart = 4000  # Reduced per restart but more total exploration
        restart_results = []
        rng = np.random.default_rng(42 + i)  # Deterministic but different per test case
        
        print(f"Running {n_restarts} MCMC restarts...")
        
        for restart in range(n_restarts):
            # Diversified initialization strategies
            if restart == 0:
                # Start from parameter midpoints
                init_guess = (bounds[:, 0] + bounds[:, 1]) / 2
            else:
                # Random initialization within bounds
                init_guess = bounds[:, 0] + rng.random(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
            
            # Track best for this restart
            restart_best_l2 = np.inf
            restart_best_input = None
            
            def on_step(step_idx, current_input, current_like):
                nonlocal restart_best_l2, restart_best_input
                pred_output = predict_fn(current_input)
                l2_error = np.linalg.norm(pred_output - target_output)
                
                if l2_error < restart_best_l2:
                    restart_best_l2 = l2_error
                    restart_best_input = current_input.copy()
                
                # Less verbose progress output
                if (step_idx + 1) % 500 == 0:
                    print(f"\rRestart {restart+1}/{n_restarts}, Step {step_idx+1:4d}, Best L2: {restart_best_l2:.6f}", end='', flush=True)
            
            # Run this restart
            result = sampler.sample(
                target_output=target_output,
                init_input=init_guess,
                n_steps=max_steps_per_restart,
                sigma=0.001,
                bounds=bounds,
                on_step=on_step
            )
            
            # Evaluate final result for this restart
            final_pred = predict_fn(result.best_input)
            final_l2 = np.linalg.norm(final_pred - target_output)
            
            restart_results.append({
                'result': result,
                'l2_error': final_l2,
                'init_guess': init_guess
            })
            
            print(f" -> Final L2: {final_l2:.6f}")
        
        # Select best result across all restarts
        best_restart = min(restart_results, key=lambda r: r['l2_error'])
        result = best_restart['result']
        best_input = result.best_input
        best_error = best_restart['l2_error']
        acceptance_rate = result.acceptance_rate
        
        print(f"\nTest case {i+1}:")
        print(f"Best input recovered (multi-restart MCMC with annealing):")
        print(f"Best restart L2 errors: {[f'{r['l2_error']:.6f}' for r in restart_results]}")
        print(f"Final acceptance rate: {acceptance_rate:.3f}")
        print(f"Wave speed: {true_input[1]:.8f} (true), {best_input[1]:.8f} (predicted)")
        print(f"Density: {true_input[0]:.8f} (true), {best_input[0]:.8f} (predicted)")
        print(f"\ntrue_ellipsoid = Ellipsoid(")
        print(f"    center=[0.5, 0.5, 0.5],")
        print(f"    scaling=[{true_input[2]:.8f}, {true_input[3]:.8f}, {true_input[4]:.8f}],")
        print(f"    rotation_vector=[{true_input[5]:.8f}, {true_input[6]:.8f}, {true_input[7]:.8f}]")
        print(f")")
        print(f"\npredicted_ellipsoid = Ellipsoid(")
        print(f"    center=[0.5, 0.5, 0.5],")
        print(f"    scaling=[{best_input[2]:.8f}, {best_input[3]:.8f}, {best_input[4]:.8f}],")
        print(f"    rotation_vector=[{best_input[5]:.8f}, {best_input[6]:.8f}, {best_input[7]:.8f}]")
        print(f")")
        print(f"\nL2 error vs. target output: {best_error:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_and_test_ppe.py <batch_name>")
        sys.exit(1)
    main(batch_name=sys.argv[1])
