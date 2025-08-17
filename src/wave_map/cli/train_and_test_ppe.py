import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import toml
import sys
from sklearn.model_selection import train_test_split

from wave_map.PyRobustGaSP import PyRobustGaSP

def load_simulation_data(batch_name):
    """Load all simulation data for the given batch."""
    # Get the project root directory (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent
    batch_dir = project_root / f"data/simulation_batch_data/{batch_name}/simulations"
    
    if not batch_dir.exists():
        raise ValueError(f"Batch directory not found: {batch_dir}")
    
    inputs = []
    outputs = []
    simulation_ids = []
    
    for sim_dir in batch_dir.iterdir():
        if sim_dir.is_dir():
            param_file = sim_dir / "parameters.toml"
            sensor_file = sim_dir / "final_sensor_data.pkl"
            
            if param_file.exists() and sensor_file.exists():
                # Load parameters
                params = toml.load(param_file)
                
                # Extract the 8 key parameters for PPE input
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
                
                # Load sensor data
                with open(sensor_file, 'rb') as f:
                    sensor_data = pickle.load(f)
                
                inputs.append(input_features)
                outputs.append(sensor_data.flatten())  # Flatten (53, 200) to 1D
                simulation_ids.append(sim_dir.name)
    
    return np.array(inputs), np.array(outputs), simulation_ids

def main(batch_name="testers", test_split=0.2, random_state=42):
    """Train and test PPE model on simulation data."""
    print(f"Loading data for batch: {batch_name}")
    
    # Load all simulation data
    inputs, outputs, sim_ids = load_simulation_data(batch_name)
    
    print(f"Loaded {len(inputs)} simulations")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        inputs, outputs, sim_ids, test_size=test_split, random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} simulations")
    print(f"Testing set: {len(X_test)} simulations")
    
    # Initialize and train PyRobustGaSP model
    print("Training PPE model...")
    P_rgasp = PyRobustGaSP()
    
    # Create task for the model
    task = P_rgasp.create_task(
        X_train,
        y_train,
        nugget_est=True,
        num_initial_values=3
    )
    
    # Train the model
    print("Training PPGaSP model...")
    model = P_rgasp.train_ppgasp(task)
    
    print("Model training completed!")
    
    # Test the model
    print("Testing model on held-out data...")
    predictions_dict = P_rgasp.predict_ppgasp(model, X_test)
    predictions = predictions_dict['mean']
    
    # Calculate some basic metrics
    mse = np.mean((predictions - y_test)**2)
    rmse = np.sqrt(mse)
    
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MSE: {mse:.6f}")
    
    # Save model and results  
    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / f"results/{batch_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the trained model
    with open(results_dir / "trained_ppe_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save test results
    test_results = {
        'test_ids': ids_test,
        'predictions': predictions,
        'actual': y_test,
        'rmse': rmse,
        'mse': mse
    }
    
    with open(results_dir / "test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)
    
    print(f"Results saved to {results_dir}")
    
    return model, test_results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_and_test_ppe.py <batch_name>")
        sys.exit(1)
    
    batch_name = "testers"
    main(batch_name)

