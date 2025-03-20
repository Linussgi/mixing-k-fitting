from utils.general_utils import get_study_names, calculate_diff, calculate_avg
from utils.plotting_utils import plot_comparison, plot_fitted_exponential, plot_difference, create_heatmap

import pandas as pd
from matplotlib import pyplot as plt

sweep = "amp-cor"
p1_name, p2_name = sweep.split("-")

main_tag = "r"

tag1 = "r1"
tag2 = "r2"
tag3 = "r3"

dim = "r"

path1 = f"k_values_csvs/{sweep}/fitted_k_values_{tag1}.csv"
path2 = f"k_values_csvs/{sweep}/fitted_k_values_{tag2}.csv"
path3 = f"k_values_csvs/{sweep}/fitted_k_values_{tag3}.csv"


df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)

# =============================================================================

kdf_cols = ["study name", p1_name, p2_name, "x lacey k", "y lacey k", "z lacey k", "r lacey k"]

kdf_1 = df1[kdf_cols]
kdf_2 = df2[kdf_cols]
kdf_3 = df2[kdf_cols]


# diff_df = calculate_diff(kdf_1, kdf_2)
# diff_df[p1_name] = (diff_df[p1_name] / 6.37483e-5).astype(int)

df_list = [kdf_1, kdf_2, kdf_3]
avg_df = calculate_avg(df_list)

out_path = f"avg_k_{sweep}_{main_tag}.csv"
avg_df.to_csv(out_path, index=False)

print(f"Average k values caved to {out_path}")

avg_df[p1_name] = (avg_df[p1_name] / 6.37483e-5).astype(int)
# avg_df[p2_name] = (avg_df[p2_name]).astype(int)

fig, ax = plt.subplots(1, 1)

# plot1 = create_heatmap(df1, ax[0], p1_name, p2_name, f"{dim} lacey k")
# plot2 = create_heatmap(df2, ax[1], p1_name, p2_name, f"{dim} lacey k")

plot3 = create_heatmap(avg_df, ax, p1_name, p2_name, f"{dim} lacey k")

plt.show()

# =============================================================================

