import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from PyRobustGaSP import PyRobustGaSP

P_rgasp = PyRobustGaSP()

# Load your data
X = np.load("ppe_inputs_train.npy")  # shape: (n_samples, 2)
Y = np.load("ppe_outputs_train.npy")  # shape: (n_samples, n_sensors, n_timesteps)

# Flatten pressure field to shape: (n_samples, n_outputs)
Y = Y[:, :, 1:]  # delete columns that are all the same
Y_flat = Y.reshape(Y.shape[0], -1)

# Identify and remove constant columns
non_constant_cols = np.any(Y_flat != Y_flat[0, :], axis=0)
Y_filtered = Y_flat[:, non_constant_cols]

# Create task with filtered outputs
task = P_rgasp.create_task(
    X, Y_flat,
    nugget_est=True,
    num_initial_values=3
)

# Train the PPGaSP model
model = P_rgasp.train_ppgasp(task)

# Load test point
test_point_path = Path("00000000_t00003000.pkl")
with open(test_point_path, "rb") as f:
    data = pickle.load(f)
    test_matrix = data['simulator']['tracked_fields']['pressure']['data']
    test_matrix = test_matrix[:, 1:]

# Store all metrics
mse_results = []
mae_results = []
frobenius_results = []

# Predict and compute metrics
for i in np.arange(0.5, 2.0, 0.05):
    for j in np.arange(0.5, 2.0, 0.05):
        breakpoint()
        predict = P_rgasp.predict_ppgasp(model, np.array([[i, j]]))
        #predicted_value = predict['mean'].reshape(26, 248)
        predicted_value = predict['mean'].reshape(26, 100)
        diff = predicted_value - test_matrix

        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        frob = np.linalg.norm(diff, ord='fro')

        mse_results.append((mse, i, j))
        mae_results.append((mae, i, j))
        frobenius_results.append((frob, i, j))

# Sort and print top 10 for each metric
def print_top(metric_name, results):
    print(f"\nTop 10 lowest {metric_name}:")
    for rank, (value, i, j) in enumerate(sorted(results, key=lambda x: x[0])[:10], 1):
        print(f"{rank:2d}. density={i:.2f}, wavespeed={j:.2f}, {metric_name}={value:.6f}")

print_top("MSE", mse_results)
print_top("MAE", mae_results)
print_top("Frobenius Norm", frobenius_results)
