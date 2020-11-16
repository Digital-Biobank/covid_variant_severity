# %% Imports
import re
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# %% Read in cleaned data
df = pd.read_parquet("data/2020-10-21_vcf-join.parquet")
df = df.reset_index()
df['covv_collection_date'] = pd.to_datetime(df['covv_collection_date'], format = '%Y-%m-%d')
df["day"] = df["covv_collection_date"] - df["covv_collection_date"].min()
df["is_green"] = ~df["is_red"]
df = df[df["is_red"].notna()]
df["is_green"] = np.logical_not(df["is_red"]).astype(int)

plot_df = df.groupby(['covv_collection_date', "continent"]).agg({"is_green": "sum","is_red": "sum", "covv_accession_id": "count"})
# sns.lineplot(x="covv_collection_date", y="is_red", hue="region", data=plot_df.reset_index())
plot_df = plot_df.reset_index()
plot_df.columns = ["Date", "Region", "Green outcome", "Red outcome", "Count"]
plot_df
# Figure S2
legend_elements = [
    Line2D([0], [0], color='red', label='Red outcome', lw=3, linestyle="-"),
    Line2D([0], [0], color='green', label='Green outcome', lw=3, linestyle=":"),
]
for name, group in plot_df.groupby('Region'):
    group.plot.area(x="Date", y=["Red outcome", "Green outcome"], title=name, color=["red", "green"], style=['-', '.'], lw=5, alpha=0.2)
    plt.legend(handles=legend_elements)
    plt.ylabel("Count")
    plt.xlabel("")
    plt.savefig(f"{pd.Timestamp.today().date()}_{name}_stacked-area-chart.png", dpi=300)
plot_df["Region"] = plot_df.Region.str.strip()
# Figure S2
legend_elements = [
    Line2D([0], [0], marker='^', markerfacecolor='red', markersize=5, color='red', label='Red', lw=1, linestyle="-"),
    Line2D([0], [0], marker='s', markerfacecolor='green', markersize=5,  color='green', label='Green', lw=1, linestyle="--"),
]
for name, group in plot_df.groupby('Region'):
    group.plot.line(
        x="Date",
        y=["Red outcome", "Green outcome"],
        title=name,
        style=['r^-', 'gs--'],
        lw=1,
        markersize=6,
        alpha=0.6
        )
    plt.legend(handles=legend_elements)
    plt.ylabel("Count")
    plt.xlabel("")
    plt.savefig(f"{pd.Timestamp.today().date()}_{name}_fig-s2.png", dpi=300)
