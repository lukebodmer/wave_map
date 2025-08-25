import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_all_sensor_data(batch_name):
    """Load all final_sensor_data.pkl arrays for the given batch (with inclusions)."""
    project_root = Path(__file__).parent.parent.parent.parent
    batch_dir = project_root / f"data/simulation_batch_data/{batch_name}/simulations"

    if not batch_dir.exists():
        raise ValueError(f"Batch directory not found: {batch_dir}")

    sensor_arrays = []
    sim_dirs = []
    for sim_dir in sorted(batch_dir.iterdir()):
        if sim_dir.is_dir():
            sensor_file = sim_dir / "final_sensor_data.pkl"
            if sensor_file.exists():
                with open(sensor_file, "rb") as f:
                    arr = pickle.load(f)
                    arr = np.asarray(arr)
                    sensor_arrays.append(arr)
                    sim_dirs.append(sim_dir.name)
    return sensor_arrays, sim_dirs


def load_no_inclusion_data(batch_name):
    """Load the no-inclusion reference simulation data."""
    project_root = Path(__file__).parent.parent.parent.parent
    no_inclusion_file = project_root / f"data/simulation_batch_data/{batch_name}/no_inclusion_simulation/final_sensor_data.pkl"
    if not no_inclusion_file.exists():
        raise ValueError(f"No-inclusion simulation not found: {no_inclusion_file}")
    with open(no_inclusion_file, "rb") as f:
        return np.asarray(pickle.load(f))


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <batch_name>")
        sys.exit(1)

    batch_name = sys.argv[1]

    # Load data
    inclusion_data_list, sim_names = load_all_sensor_data(batch_name)
    no_inclusion_data = load_no_inclusion_data(batch_name)

    vmin, vmax = -0.003, 0.003  # fixed color scale

    # Plot for each inclusion simulation
    for arr, name in zip(inclusion_data_list, sim_names):
        diff = arr - no_inclusion_data

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(arr, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Original: {name}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(diff, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)
        axes[1].set_title("Difference (Inclusion - No-Inclusion)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(no_inclusion_data, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)
        axes[2].set_title("No-Inclusion Reference")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
