import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from wave_map.inverse_models.PyRobustGaSP import PyRobustGaSP
from wave_map.loggers.logger import Logger
from wave_map.hash_functions.parameter_hashing import ParameterHashFunctions

BATCH_DATA_DIR = "data/simulation_batch_data"
LOG_FILENAME = "data_processor_log.txt"
LOG_NAME = "dataprocessorlog"


class ParallelPartialEmulator:
    """Parallel Partial inverse model using Robust GaSP."""

    def __init__(self, batch_name: str, model=None):
        self.batch_name = batch_name
        self.batch_dir = Path(f"{BATCH_DATA_DIR}/{batch_name}/simulations")
        self.save_dir = Path(f"{BATCH_DATA_DIR}/{batch_name}/saved_inversion_models")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.P_rgasp = PyRobustGaSP()
        self.train_hashes = []
        self.test_hashes = []

        self.logger = Logger(Path(f"{self.batch_dir} / {LOG_FILENAME}"), LOG_NAME)

    def _load_all_data(self):
        """Load model_input.pkl and model_output.pkl from each simulation directory."""
        X, y, sim_hashes = [], [], []

        count = 0
        for sim_dir in sorted(self.batch_dir.iterdir()):
            if count > 5:
                break
            if not sim_dir.is_dir():
                continue

            model_input_path = sim_dir / "model_input.pkl"
            model_output_path = sim_dir / "model_output.pkl"

            if not (model_input_path.exists() and model_output_path.exists()):
                continue

            try:
                with open(model_input_path, "rb") as f_in, open(model_output_path, "rb") as f_out:
                    x = pickle.load(f_in)
                    y_ = pickle.load(f_out)

                # --- Convert from CuPy to NumPy if necessary ---
                if hasattr(x, "get"):  # CuPy arrays have .get()
                    x = x.get()
                if hasattr(y_, "get"):
                    y_ = y_.get()

                X.append(np.ravel(x))
                y.append(np.ravel(y_))
                sim_hashes.append(sim_dir.name)

            except Exception as e:
                print(f"Warning: could not load {sim_dir.name}: {e}")

            count += 1

        if len(X) == 0 or len(y) == 0:
            raise RuntimeError("No valid simulation data found.")

        return np.array(X), np.array(y), sim_hashes

    def _remove_constant_columns(self, Y, ref_mask=None, log_prefix=""):
        """
        Remove columns in Y that are constant across all rows.
        If ref_mask is provided, apply the same mask (for test data).
        Returns (Y_new, mask)
        """
        if ref_mask is None:
            stds = np.std(Y, axis=0)
            mask = stds > 1e-12
            dropped = np.where(~mask)[0]
            self.logger.info(f"{log_prefix}Dropped {len(dropped)} constant columns out of {Y.shape[1]} total.")
            if len(dropped) > 0:
                self.logger.info(f"{log_prefix}Constant column indices: {dropped.tolist()}")
        else:
            mask = ref_mask

        return Y[:, mask], mask

    def _ppe_prediction_to_kspace(self, pred_vec: np.ndarray) -> np.ndarray:
        """
        Convert a predicted flattened vector (after dropped columns) into complex k-space
        using the saved mask to reinsert zeroed positions.
        """
        pred_vec = np.asarray(pred_vec).ravel()

        if self.mask is None:
            raise RuntimeError("Mask not loaded — cannot reconstruct full prediction vector.")

        # Validate shape
        kept_count = int(self.mask.sum())
        if pred_vec.size != kept_count:
            self.logger.warning(
                f"Predicted vector size {pred_vec.size} does not match mask kept count {kept_count}; adjusting."
            )
            pred_vec = np.pad(pred_vec, (0, max(0, kept_count - pred_vec.size)))[:kept_count]

        # Reconstruct full flattened vector
        full_pred = np.zeros(self.mask.size, dtype=float)
        full_pred[self.mask] = pred_vec

        # Split into cosine/sine halves
        n_half = full_pred.size // 2
        cos_coeffs = full_pred[:n_half]
        sin_coeffs = full_pred[n_half:]

        # Infer cube dimension
        cube_root = round((n_half) ** (1 / 3))
        if cube_root**3 != n_half:
            raise RuntimeError(f"Cannot infer cubic grid from {n_half} voxels (not a perfect cube).")

        # Reshape and combine
        cos_coeffs = cos_coeffs.reshape((cube_root,) * 3)
        sin_coeffs = sin_coeffs.reshape((cube_root,) * 3)

        return cos_coeffs + 1j * sin_coeffs

    def train(self, test_set_size: float = 0.2):
        """Train the Robust GaSP Parallel Partial Emulator."""
        X, y, sim_hashes = self._load_all_data()

        # --- Split into train/test ---
        X_train, X_test, y_train, y_test, train_hashes, test_hashes = train_test_split(
            X, y, sim_hashes, test_size=test_set_size, random_state=42
        )

        # --- Remove constant columns in response (y) ---
        y_train, mask = self._remove_constant_columns(y_train, log_prefix="[Train] ")
        y_test, _ = self._remove_constant_columns(y_test, ref_mask=mask, log_prefix="[Test] ")

        self.train_hashes = train_hashes
        self.test_hashes = test_hashes
        self.mask = mask

        # --- Train PPE quietly ---
        task = self.P_rgasp.create_task(
            X_train, y_train,
            isotropic=True,
            #num_initial_values=10,
            nugget_est=True
        )

        #with open(os.devnull, "w") as fnull:
        #    with redirect_stdout(fnull), redirect_stderr(fnull):
        self.model = self.P_rgasp.train_ppgasp(task)

        # Automatically save after training
        self.save()

        return self.model

    def predict(self, X_test):
        """Predict mean output for a test input set."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        prediction = self.P_rgasp.predict_ppgasp(self.model, X_test)["mean"]
        kspace_prediction = self._ppe_prediction_to_kspace(prediction)
        return kspace_prediction

    def save(self):
        """Save the trained model and hash information."""
        hash_function = ParameterHashFunctions()
        model_hash = hash_function.get_inversion_model_hash(self.model)

        model_path = self.save_dir / f"{model_hash}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_type": "parallel_partial_emulator",
                "train_hashes": self.train_hashes,
                "test_hashes": self.test_hashes,
                "mask": self.mask
            }, f)

        print(f"PPE model saved to {model_path}")

    def load(self, path: Path = None):
        """Load the trained model and hash information."""
        if path is None:
            path = self.save_dir / "ppe_model.pkl"

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data.get("model", None)
        self.train_hashes = data.get("train_hashes", [])
        self.test_hashes = data.get("test_hashes", [])
        self.mask = np.array(data.get("mask"), dtype=bool)

        print(f"Loaded PPE model from {path}")
