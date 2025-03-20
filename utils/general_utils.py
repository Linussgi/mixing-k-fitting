import pandas as pd
import numpy as np


def get_param_values(text: str) -> list[tuple[str, float]]:
    """Extracts parameter names and values from a given text."""
    p1_name, p1_value, p2_name, p2_value = text.split("_")
    return (p1_name, float(p1_value)), (p2_name, float(p2_value))


def get_study_names(df: pd.DataFrame, frag=None) -> list[str]:
    studies = set()

    for col in df.columns[2:]:
        name = col.split()[0]   
        
        if (frag and frag in name) or (not frag):
            studies.add(name)

    return list(studies)


def calculate_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if not df1["study name"].equals(df2["study name"]):
        raise ValueError("The 'study name' columns in both DataFrames do not match")

    diff_df = pd.DataFrame({
        "study name": df1["study name"],
        df1.columns[1]: df1[df1.columns[1]],
        df1.columns[2]: df1[df1.columns[2]],
        "x lacey k pdiff": np.abs(df1["x lacey k"] - df2["x lacey k"]) / df1["x lacey k"],
        "y lacey k pdiff": np.abs(df1["y lacey k"] - df2["y lacey k"]) / df1["y lacey k"],
        "z lacey k pdiff": np.abs(df1["z lacey k"] - df2["z lacey k"]) / df1["z lacey k"],
        "r lacey k pdiff": np.abs(df1["r lacey k"] - df2["r lacey k"]) / df1["r lacey k"]
    })

    return diff_df


def calculate_avg(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    # Ensure that all DataFrames have the same "study name" column
    for i in range(1, len(df_list)):
        if not df_list[i]["study name"].equals(df_list[0]["study name"]):
            raise ValueError("The 'study name' columns in all DataFrames do not match")
    
    # Create a DataFrame to hold the averages
    avg_df = pd.DataFrame({
        "study name": df_list[0]["study name"],
        df_list[0].columns[1]: df_list[0].iloc[:, 1],  # Keep column [1]
        df_list[0].columns[2]: df_list[0].iloc[:, 2],  # Keep column [2]
    })
    
    # Iterate over each DataFrame and calculate the averages for key columns
    for key in df_list[0].columns[3:]:  # Start from the 4th column (index 3)
        if "lacey k" not in key:
            continue

        # For each key column, calculate the average across all DataFrames
        avg_df[key] = np.mean([df[key].values for df in df_list], axis=0)

    return avg_df