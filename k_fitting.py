import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from utils.general_utils import get_param_values

# d = default run, c = cell testing run
tag = "r3"
sweep = "amp-cor"
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


def fit_lacey_data(lacey_data: np.ndarray, time: np.ndarray) -> tuple[float, float, float, float]:
    """Fits the lacey data using the model, starting at t = 2."""
    A = np.max(lacey_data)  # Compute A as the max value of the lacey column

    def fitting_function(t, k):
        return model(t, k, A)

    popt, _ = curve_fit(fitting_function, time, lacey_data, p0=[0.1])
    k = popt[0]

    y_pred = fitting_function(time, k)
    r_squared, rmse = calculate_fit_quality(lacey_data, y_pred)

    return k, r_squared, rmse, A  # Returning A


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

            # Fit the data
            k_value, r_squared, rmse, A_value = fit_lacey_data(lacey_data, time)
            
            # Append k, A, r_squared, rmse for each dimension (x, y, z, r)
            k_values.append(k_value)
            A_values.append(A_value)
            r_squared_values.append(r_squared)
            rmse_values.append(rmse)

        # Now, we extend the row with k_values, r_squared_values, rmse_values, and A_values for each dimension
        row.extend(k_values + A_values + r_squared_values + rmse_values)
        new_data.append(row)

    dimensions = ["x", "y", "z", "r"]
    
    # Create the column names correctly
    columns = ["study name", p1_name, p2_name] + \
              [f"{dim} lacey k" for dim in dimensions] + \
              [f"{dim} lacey A" for dim in dimensions] + \
              [f"k{dim} Rsquared" for dim in dimensions] + \
              [f"k{dim} RMSE" for dim in dimensions]

    # Create the DataFrame
    return pd.DataFrame(new_data, columns=columns)


df = pd.read_csv(f"raw_lacey_csvs/{sweep}/lrv_{sweep}_{tag}.csv")

filtered_df = filter_df(df, OFFSET)
new_df = build_k_df(filtered_df)

print(f"Fitted k-values DataFrame created with shape {new_df.shape}")

out_path = f"k_values_csvs/{sweep}/fitted_k_values_{tag}.csv"

new_df.to_csv(out_path, index=False)

print(f"Fitted k-values saved to '{out_path}'")
