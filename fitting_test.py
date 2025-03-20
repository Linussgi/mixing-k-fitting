import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from utils.general_utils import get_param_values

# d = default run, c = cell testing run
tag = "r3"
sweep = "amp-particles"
OFFSET = 2

def filter_df(df: pd.DataFrame, increase_point) -> pd.DataFrame:
    """Filters the DataFrame based on a fixed starting time."""
    new_df = df[["time"]]
    
    lacey_columns = [col for col in df.columns[2:] if "lacey" in col]
    new_df = pd.concat([new_df, df[lacey_columns]], axis=1)
    
    new_df.dropna(inplace=True)

    # Apply filtering with the new threshold
    new_df = new_df[new_df["time"] >= increase_point]
    new_df["time"] = new_df["time"] - increase_point  # Shift time so it starts from 0

    print(new_df.head())
    
    return new_df

def model(t: np.ndarray, k: float, A: float) -> np.ndarray:
    return A * (1 - np.exp(-k * t))

def calculate_fit_quality(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    correlation_matrix = np.corrcoef(y_true, y_pred)
    r_squared = correlation_matrix[0, 1] ** 2
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return r_squared, rmse

def slice_series(time: np.ndarray, data: np.ndarray, threshold: float = 0.05, window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterates over the series starting at the first data point (after offset) and finds the first
    window of `window` consecutive points where the overall increase is less than `threshold`.
    If such a window is found, the function returns the arrays sliced up to (and including) that window.
    Otherwise, it returns the full arrays.
    """
    n = len(data)
    if n < window:
        return time, data
    # Loop from the beginning until a qualifying window is found
    for i in range(0, n - window + 1):
        # Check if the increase from the start to the end of the window is less than threshold
        if data[i + window - 1] - data[i] < threshold:
            slice_index = i + window  # slice_index is exclusive, so it includes the window end
            return time[:slice_index], data[:slice_index]
    return time, data

def fit_lacey_data(lacey_data: np.ndarray, time: np.ndarray) -> tuple[float, float, float, float]:
    """
    Fits the lacey data using the model y = A*(1 - exp(-kt)), but only using data up until the
    first instance (after the offset) where the data increases by less than 0.05 over 4 consecutive points.
    """
    # Slice the series based on the first instance where the change is less than 0.05 over 4 points
    time_slice, data_slice = slice_series(time, lacey_data, threshold=0.05, window=4)
    
    # Compute A as the maximum value in the (sliced) data
    A = np.max(data_slice)

    def fitting_function(t, k):
        return model(t, k, A)

    popt, _ = curve_fit(fitting_function, time_slice, data_slice, p0=[0.1])
    k = popt[0]

    y_pred = fitting_function(time_slice, k)
    r_squared, rmse = calculate_fit_quality(data_slice, y_pred)

    return k, r_squared, rmse, A  # Returning A as well

def build_k_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    new_data = []
    lacey_columns = [col for col in filtered_df.columns[1:] if "lacey" in col]

    for i in range(0, len(lacey_columns), 4):
        row = []
        study_name = lacey_columns[i][:-8]  # Extract study name
        row.append(study_name)

        # Extract parameter values
        (p1_name, p1_value), (p2_name, p2_value) = get_param_values(study_name)
        row.extend([p1_value, p2_value])  # Add extracted values to the row

        k_values, A_values, r_squared_values, rmse_values = [], [], [], []

        for j in range(4):
            variable_column = lacey_columns[i + j]
            lacey_data = filtered_df[variable_column].values
            time = filtered_df["time"].values

            # Fit the data (using the sliced series if the plateau condition is met)
            k_value, r_squared, rmse, A_value = fit_lacey_data(lacey_data, time)
            
            # Append k, A, R-squared, RMSE for each dimension (x, y, z, r)
            k_values.append(k_value)
            A_values.append(A_value)
            r_squared_values.append(r_squared)
            rmse_values.append(rmse)

        # Extend the row with k_values, A_values, r_squared_values, and rmse_values
        row.extend(k_values + A_values + r_squared_values + rmse_values)
        new_data.append(row)

    dimensions = ["x", "y", "z", "r"]
    
    # Create the column names correctly
    columns = ["study name", p1_name, p2_name] + \
              [f"{dim} lacey k" for dim in dimensions] + \
              [f"{dim} lacey A" for dim in dimensions] + \
              [f"k{dim} Rsquared" for dim in dimensions] + \
              [f"k{dim} RMSE" for dim in dimensions]

    return pd.DataFrame(new_data, columns=columns)

# Read CSV
df = pd.read_csv(f"raw_lacey_csvs/{sweep}/lrv_{sweep}_{tag}.csv")

filtered_df = filter_df(df, OFFSET)
new_df = build_k_df(filtered_df)

print(f"Fitted k-values DataFrame created with shape {new_df.shape}")

out_path = f"k_values_csvs/{sweep}/fitted_reattempt_{tag}.csv"

new_df.to_csv(out_path, index=False)
print(f"Fitted k-values saved to '{out_path}'")
