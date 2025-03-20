import pandas as pd


sweep = "amp-particles"
tag = "r1"

path = f"k_values_csvs/{sweep}/fitted_reattempt_{tag}.csv"

df = pd.read_csv(path)

# Identify RMSE columns
rmse_columns = [col for col in df.columns if col.endswith("Rsquared")]

# Create a list to store (study_name, RMSE value, column name)
rs_values = []

for col in rmse_columns:
    for _, row in df.iterrows():
        rs_values.append((row["study name"], row[col], col))

# Convert to DataFrame and sort by RMSE values
rmse_df = pd.DataFrame(rs_values, columns=["Study Name", "Rsquared Value", "Rsquared Column"])
rmse_df = rmse_df.nsmallest(20, "Rsquared Value")  # Get the 10 lowest RMSE values

# Print results
print(rmse_df)