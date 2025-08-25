# train_and_test_ppe_ga.py

# --- imports remain the same ---
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import toml
import sys
from sklearn.model_selection import train_test_split

from wave_map.PyRobustGaSP import PyRobustGaSP
from wave_map.input_space_samplers.genetic_sampler import GeneticSampler  # <-- we’ll implement later


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
                params = toml.load(param_file)
                
                input_features = [
                    params["material"]["inclusion_density"],
                    params["material"]["inclusion_wave_speed"],
                    #*params["mesh"]["inclusion_scaling"],
                    params["mesh"]["inclusion_scaling"][0],
                    *params["mesh"]["inclusion_center"]
                    #*params["mesh"]["inclusion_rotation"]
                    #*params["mesh"]["inclusion_semi_major_axis_direction"]
                ]
                
                with open(sensor_file, 'rb') as f:
                    sensor_data = pickle.load(f)
                
                inputs.append(input_features)
                outputs.append(sensor_data.flatten())
                simulation_ids.append(sim_dir.name)
 
    return np.array(inputs), np.array(outputs), simulation_ids


def main(batch_name="single_moving_sphere_variable_radius", test_split=0.05, random_state=42):
    """Train PPE model or load, then test with GA inverse search on test samples."""
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
    
    # --- Inverse test with GA on all test cases ---
    print("\nRunning Genetic Algorithm inverse search on all test cases...")

    P_rgasp = PyRobustGaSP()

    bounds = np.array([
        [2.0, 4.0],
        [2.0, 4.0],
        [0.125, 0.375],
        [0, 1],
        [0, 1],
        [0, 1],
        #[0.125, 0.375],
        #[0.125, 0.375],
        #[-np.pi, np.pi],
        #[-np.pi, np.pi],
        #[0.0, np.pi]
    ], dtype=float)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return P_rgasp.predict_ppgasp(model, x.reshape(1, -1))['mean'][0]

    sampler = GeneticSampler(
        model=predict_fn,
        bounds=bounds,
        population_size=1000,
        n_generations=200,
        #mutation_rate=0.20,
        #crossover_rate=0.8,
        random_state=42,
        fitness_type="l2"
    )

    for i, (true_input, target_output) in enumerate(zip(test_results['X_test'], test_results['actual'])):
        if i < 2:
            continue
    
        true_input_str = np.array2string(true_input, precision=4, separator=',', suppress_small=True)
        print(f"True input: {true_input_str}")

        result = sampler.evolve(
            target_output,
            #n_elite=10
        )

        best_input = result.best_input
        best_error = np.linalg.norm(predict_fn(best_input) - target_output)

        print(f"\nTest case {i+1}:")
        print(f"Best input recovered (GA):")
        print(f"Wave speed: {true_input[1]:.8f} (true), {best_input[1]:.8f} (predicted)")
        print(f"Density: {true_input[0]:.8f} (true), {best_input[0]:.8f} (predicted)")
        print(f"\ntrue_ellipsoid = Ellipsoid(")
        print(f"    center=[0.5, 0.5, 0.5],")
        print(f"    scaling=[{true_input[2]:.8f}, {true_input[3]:.8f}, {true_input[4]:.8f}],")
        #print(f"    rotation_vector=[{true_input[5]:.8f}, {true_input[6]:.8f}, {true_input[7]:.8f}]")
        print(f")")
        print(f"\npredicted_ellipsoid = Ellipsoid(")
        print(f"    center=[0.5, 0.5, 0.5],")
        print(f"    scaling=[{best_input[2]:.8f}, {best_input[3]:.8f}, {best_input[4]:.8f}],")
        print(f"    rotation_vector=[{best_input[5]:.8f}, {best_input[6]:.8f}, {best_input[7]:.8f}]")
        print(f")")
        print(f"\nL2 error vs. target output: {best_error:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_and_test_ppe_ga.py <batch_name>")
        sys.exit(1)
    main(batch_name=sys.argv[1])
