import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def create_heatmap(df: pd.DataFrame, ax, x_axis: str, y_axis: str, color_variable: str, colmin=None, colmax=None, **kwargs):
    """
    Creates a heatmap from a DataFrame using specified x and y axes and a color variable.
    
    Parameters:
    - df: DataFrame containing the data
    - ax: Matplotlib axis to plot on
    - x_axis: Column name for x-axis categories
    - y_axis: Column name for y-axis categories
    - color_variable: Column name for the values to be represented by colors
    
    Keyword arguments (passed as **kwargs):
    - title: Title for the heatmap (default is f"Heatmap of {color_variable}")
    - xlabel: Label for the x-axis (default is x_axis)
    - ylabel: Label for the y-axis (default is y_axis)
    - cbar_label: Label for the colorbar (default is color_variable)
    - fontsize: Font size for labels and ticks (default is 12)
    - show_colorbar: Whether to show a colorbar (default is True)
    
    Returns:
    - The matplotlib axis with the heatmap
    """
    df[color_variable] = pd.to_numeric(df[color_variable], errors="coerce")

    font_size = kwargs.get("fontsize", 12)
    title = kwargs.get("title", f"Heatmap of {color_variable}")
    xlabel = kwargs.get("xlabel", x_axis)
    ylabel = kwargs.get("ylabel", y_axis)
    cbar_label = kwargs.get("cbar_label", color_variable)
    show_colorbar = kwargs.get("show_colorbar", True)

    heatmap_data = df.pivot_table(
        values=color_variable, 
        index=y_axis, 
        columns=x_axis, 
        aggfunc="mean" 
    )

    heatmap = sns.heatmap(
        heatmap_data, 
        ax=ax,
        annot=True,
        vmin=colmin,
        vmax=colmax,
        cmap="inferno",  
        cbar=show_colorbar,
        cbar_kws={"label": cbar_label} if show_colorbar else None
    )

    if show_colorbar:
        cbar = heatmap.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=font_size, rotation=90)
        cbar.ax.tick_params(labelsize=font_size)

    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)

    return ax


def create_contour_plot(df: pd.DataFrame, x_axis: str, y_axis: str, color_variable: str, **kwargs):
    """
    Creates a contour plot from a DataFrame using specified x and y axes and a color variable,
    with black circles representing the datapoint locations.
    
    Keyword arguments (passed as **kwargs):
    - title: Title for the plot (default is f"Contour Plot of {color_variable}")
    - xlabel: Label for the x-axis (default is x_axis)
    - ylabel: Label for the y-axis (default is y_axis)
    - cbar_label: Label for the colorbar (default is color_variable)
    - fontsize: Font size for labels and ticks (default is 12)
    - levels: Number of contour levels (default is 20)
    - point_size: Size of the black circles representing data points (default is 20)
    - point_alpha: Transparency of the black circles (default is 0.6)
    """
    # Convert color variable to numeric
    df[color_variable] = pd.to_numeric(df[color_variable], errors="coerce")
    
    # Get parameters from kwargs with defaults
    font_size = kwargs.get("fontsize", 12)
    title = kwargs.get("title", f"Contour Plot of {color_variable}")
    xlabel = kwargs.get("xlabel", x_axis)
    ylabel = kwargs.get("ylabel", y_axis)
    cbar_label = kwargs.get("cbar_label", color_variable)
    levels = kwargs.get("levels", 20)
    point_size = kwargs.get("point_size", 20)
    point_alpha = kwargs.get("point_alpha", 0.6)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a meshgrid for contour plotting
    x_data = df[x_axis].values
    y_data = df[y_axis].values
    z_data = df[color_variable].values
    
    # Get exact min/max from data for plotting without padding
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    
    # Create the grid
    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate the scattered data onto the regular grid using griddata
    from scipy.interpolate import griddata
    zi_grid = griddata((x_data, y_data), z_data, (xi_grid, yi_grid), method='cubic')
    
    # Create the contour plot
    contour = ax.contourf(xi_grid, yi_grid, zi_grid, levels=levels, cmap='inferno',
                         extent=[x_min, x_max, y_min, y_max])
    
    # Add black circles to represent the datapoint locations
    ax.scatter(x_data, y_data, color='black', s=point_size, alpha=point_alpha, 
              edgecolors='white', linewidths=0.5)
    
    # Set axis limits to exactly match data range
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label, fontsize=font_size, rotation=90)
    cbar.ax.tick_params(labelsize=font_size)
    
    # Set labels and title
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    # Adjust layout
    fig.tight_layout()
    
    return plt


def heatmap_to_csv(df: pd.DataFrame, x_axis: str, y_axis: str, color_variable: str):
    """
    Saves a heatmap representation of the DataFrame in CSV format.
    
    Parameters:
    - df: Pandas DataFrame
    - x_axis: Column name for the x-axis (columns in CSV)
    - y_axis: Column name for the y-axis (rows in CSV)
    - color_variable: Column name for the heatmap values (cell values in CSV)
    - filename: Name of the CSV file to save the data
    
    The resulting CSV will have:
    - First row: x_axis values as headers
    - First column: y_axis values as row indices
    - Remaining cells: color_variable values
    """
    # Ensure numeric values for the heatmap
    df[color_variable] = pd.to_numeric(df[color_variable], errors="coerce")

    # Create pivot table
    heatmap_data = df.pivot_table(
        values=color_variable,
        index=y_axis,
        columns=x_axis,
        aggfunc="mean"
    )

    return heatmap_data


def exponential_model(t: np.ndarray, k: float, A: float) -> float:
    """Exponential function y = A(1 - e^(-kt))"""
    return A * (1 - np.exp(-k * t))


def fit_exponential(x_data: np.ndarray, y_data: np.ndarray) -> tuple[float]:
    """Fit an exponential model to the data."""
    A = max(y_data)  # Fixed A as max of column

    try:
        popt, _ = curve_fit(lambda t, k: exponential_model(t, k, A), x_data, y_data, p0=[0.1])
        k_fitted = popt[0]
        y_fitted = exponential_model(x_data, k_fitted, A)

        return k_fitted, y_fitted
    
    except RuntimeError:
        return None, None  # Return None if fitting fails


def filter_columns_by_header(df: pd.DataFrame, header_string: str) -> pd.Series:
    """Filter columns containing a specific string in their headers, keeping 'time'."""
    filtered_columns = [col for col in df.columns if header_string in col]
    if "time" in df.columns:
        filtered_columns.append("time")

    return df[filtered_columns]


def plot_fitted_exponential(study_name: str, tag: str, sweep: str, offset) -> plt:
    """
    Plots original data and fitted exponential models for a given study.
    
    Parameters:
    - study_name: Name of the study (used in the file path).
    
    Returns:
    - The plot object.
    """
    df_whole = pd.read_csv(f"raw_lacey_csvs/{sweep}/lrv_{sweep}_{tag}.csv").dropna()
    df = df_whole[df_whole["time"] > offset]

    filtered_df = filter_columns_by_header(df, study_name)

    x_data = filtered_df["time"] - offset
    y_data = {
        "x": filtered_df[study_name + " x lacey"],
        "y": filtered_df[study_name + " y lacey"],
        "z": filtered_df[study_name + " z lacey"],
        "r": filtered_df[study_name + " r lacey"]
    }

    # Fit models
    fitted_results = {key: fit_exponential(x_data, y) for key, y in y_data.items()}

    # Create plot
    fig, ax = plt.subplots(2, 1, figsize=[12, 12])

    # --- Original Data ---
    colors = {"x": "r", "y": "b", "z": "g", "r": "k"}
    for key, y in y_data.items():
        ax[0].errorbar(x_data, y, yerr=None, fmt=f"-{colors[key]}", label=f"{key} mixing", linewidth=2, capsize=2)

    ax[0].set_xlabel("Time", fontsize=10)
    ax[0].set_ylabel("Mixing Index", fontsize=10)
    ax[0].set_title(f"Original Mixing Data ({study_name})", fontsize=12)
    ax[0].legend()
    ax[0].grid()

    # --- Fitted Models ---
    for key, (k_fitted, y_fit) in fitted_results.items():
        if y_fit is not None:
            ax[1].plot(x_data, y_fit, f"-{colors[key]}", label=f"Fitted {key} (k={k_fitted:.4f})", linewidth=2)

    ax[1].set_xlabel("Time", fontsize=10)
    ax[1].set_ylabel("Mixing Index (Fitted)", fontsize=10)
    ax[1].set_title(f"Fitted Exponential Models ({study_name})", fontsize=12)
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()

    return plt


def load_data(study_name: str, tag: str, sweep: str, offset=1.8) -> pd.Series:
    """
    Loads the data for a given study.

    Parameters:
    - study_name: Name of the study (column header).
    - sweep: Identifier for the parameter sweep of the study in question.
    - tag: Identifier for the dataset.
    - offset: offset the time to onset of process.

    Returns:
    - x_data: Time values adjusted by offset.
    - y_data: Dictionary containing x, y, z, r mixing data.
    """
    df_whole = pd.read_csv(f"raw_lacey_csvs/{sweep}/lrv_{sweep}_{tag}.csv").dropna()
    df = df_whole[df_whole["time"] > offset]
    filtered_df = filter_columns_by_header(df, study_name)  

    x_data = filtered_df["time"] - offset
    y_data = {axis: filtered_df[study_name + f" {axis} lacey"] for axis in ["x", "y", "z", "r"]}

    return x_data, y_data


def plot_comparison(study_1_name: str, study_2_name: str, sweep1: str, tag1: str, sweep2=None, tag2=None) -> plt:
    """
    Compares original mixing data from two studies in a two-subplot plot.

    Parameters:
    - study_1_name: Name of the first study.
    - study_2_name: Name of the second study.
    - tag: Identifier for the dataset.

    Returns:
    - The plot object.
    """
    sweep2 = sweep2 or sweep1
    tag2 = tag2 or tag1

    x_data_1, y_data_1 = load_data(study_1_name, tag1, sweep1)
    x_data_2, y_data_2 = load_data(study_2_name, tag2, sweep2)

    fig, axes = plt.subplots(2, 1, figsize=[12, 12])
    colors = {"x": "r", "y": "b", "z": "g", "r": "k"}

    for ax, (study_name, y_data) in zip(axes, [(study_1_name, y_data_1), (study_2_name, y_data_2)]):
        for axis, y in y_data.items():
            ax.errorbar(x_data_1, y, yerr=None, fmt=f"-{colors[axis]}", label=f"{axis} mixing", linewidth=2, capsize=2)

        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Mixing Index", fontsize=10)
        ax.set_title(f"Original Mixing Data ({study_name})", fontsize=12)
        ax.legend()
        ax.grid()

    plt.tight_layout()

    return plt


def plot_difference(study_1_name: str, study_2_name: str, sweep1: str, tag1: str, sweep2=None, tag2=None) -> plt:
    """
    Plots the difference between two studies on a single graph.

    Parameters:
    - study_1_name: Name of the first study.
    - study_2_name: Name of the second study.
    - sweep1: Identifier for the study_1 parameter sweep in question.
    - tag: Identifier for the study_1 dataset.
    - sweep2: Identifier for the study_2 parameter sweep in question. Deafults to sweep1 if left as None.
    - tag2: Identifier for the study_2 dataset. Defaults to tag1 if left as None.

    Returns:
    - The plot object.
    """
    sweep2 = sweep2 or sweep1
    tag2 = tag2 or tag1

    x_data_1, y_data_1 = load_data(study_1_name, tag1, sweep1)
    x_data_2, y_data_2 = load_data(study_2_name, tag2, sweep2)

    y_diff = {axis: y_data_1[axis] - y_data_2[axis] for axis in ["x", "y", "z", "r"]}

    fig, ax = plt.subplots(figsize=[12, 6])
    colors = {"x": "r", "y": "b", "z": "g", "r": "k"}

    for axis, y in y_diff.items():
        ax.plot(x_data_1, y, f"-{colors[axis]}", label=f"Δ {axis} (Study1 - Study2)", linewidth=2)

    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Difference in Mixing Index", fontsize=10)
    ax.set_title(f"Difference in Mixing Data ({study_1_name}_{tag1} - {study_2_name}_{tag2})", fontsize=12)
    ax.legend()
    ax.grid()

    plt.tight_layout()

    return plt
