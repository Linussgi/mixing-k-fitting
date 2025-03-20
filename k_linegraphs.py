import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def model(t: np.ndarray, k: float, A: float) -> np.ndarray:
    """Computes A * (1 - exp(-k * t)) for given t values."""
    return A * (1 - np.exp(-k * t))


def get_lacey_predictions(df: pd.DataFrame, study_name: str, t_range: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieves k and A values from the DataFrame and generates predicted values for t=0 to t=t_max.

    Args:
        df (pd.DataFrame): DataFrame containing fitted k-values.
        study_name (str): The study to extract data for.
        t_max (int): Maximum time value.

    Returns:
        Tuple of NumPy arrays: (x_vals, y_vals, z_vals, r_vals)
    """
    row = df[df["study name"] == study_name]
    if row.empty:
        raise ValueError(f"Study '{study_name}' not found.")

    # Extract k and A values for each dimension
    k_values = [row[f"{dim} lacey k"].values[0] for dim in ["x", "y", "z", "r"]]
    A_values = [row[f"{dim} lacey A"].values[0] for dim in ["x", "y", "z", "r"]]

    # Compute predicted values for each dimension
    predictions = [model(t_range, k, A) for k, A in zip(k_values, A_values)]

    return tuple(predictions)  # (x_pred, y_pred, z_pred, r_pred)


# ========================================================================================================

sweep = "amp-particles"
tag = "r3"

k_path = f"k_values_csvs/{sweep}/fitted_k_values_{tag}.csv"
k_df = pd.read_csv(k_path)

STUDY_NAME = "amp_0.00617_particles_131570.0"

t_data = np.linspace(0, 13, 100)

x_pred, y_pred, z_pred, r_pred = get_lacey_predictions(k_df, STUDY_NAME, t_data)

lacey_path = f"raw_lacey_csvs/{sweep}/lrv_{sweep}_{tag}.csv"
lacey_df = pd.read_csv(lacey_path)

study_columns = [col for col in lacey_df.columns if col.startswith(STUDY_NAME)]

lacey_study = lacey_df[lacey_df["time"] >= 2][study_columns]
lacey_time = lacey_df[lacey_df["time"] >= 2]["time"]

lacey_x = [col for col in lacey_study.columns if col.endswith("x lacey")]
lacey_y = [col for col in lacey_study.columns if col.endswith("y lacey")]
lacey_z = [col for col in lacey_study.columns if col.endswith("z lacey")]
lacey_r = [col for col in lacey_study.columns if col.endswith("r lacey")]

# Create plots
fig, ax = plt.subplots(2, 1, figsize=[12, 12])  # Two subplots (original data & fitted curves)

# --- Original Data ---
ax[0].errorbar(lacey_time, lacey_study[lacey_x], yerr=None, fmt="-r", label="x mixing", linewidth=2, capsize=2)
ax[0].errorbar(lacey_time, lacey_study[lacey_y], yerr=None, fmt="-b", label="y mixing", linewidth=2, capsize=2)
ax[0].errorbar(lacey_time, lacey_study[lacey_z], yerr=None, fmt="-g", label="z mixing", linewidth=2, capsize=2)
ax[0].errorbar(lacey_time, lacey_study[lacey_r], yerr=None, fmt="-k", label="r mixing", linewidth=2, capsize=2)

ax[0].set_xlabel("Time (s)", fontsize=14)
ax[0].set_ylabel("Mixing Index", fontsize=14)
ax[0].set_title(f"Lacey Data {STUDY_NAME}", fontsize=14)
ax[0].legend(fontsize=12)
ax[0].grid()

ax[0].tick_params(axis="both", labelsize=14)

# --- Fitted Models ---

ax[1].plot(t_data, x_pred, "-r", label=f"Fitted x", linewidth=2)
ax[1].plot(t_data, y_pred, "-b", label=f"Fitted y", linewidth=2)
ax[1].plot(t_data, z_pred, "-g", label=f"Fitted z", linewidth=2)
ax[1].plot(t_data, r_pred, "-k", label=f"Fitted r", linewidth=2)

ax[1].set_xlabel("Time (s)", fontsize=14)
ax[1].set_ylabel("Mixing Index (Fitted)", fontsize=14)
ax[1].set_title(f"Fitted Data {STUDY_NAME}", fontsize=14)
ax[1].legend(fontsize=12)
ax[1].grid()

ax[1].tick_params(axis="both", labelsize=14)

plt.tight_layout()
plt.show()
