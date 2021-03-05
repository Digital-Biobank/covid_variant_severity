# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
plot_df.columns = ["Date", "Region", "Mild outcome", "Severe outcome", "Count"]
plot_df
# Figure S2
legend_elements = [
    Line2D([0], [0], color='red', label='Severe outcome', lw=3, linestyle="-"),
    Line2D([0], [0], color='blue', label='Mild outcome', lw=3, linestyle=":"),
]
for name, group in plot_df.groupby('Region'):
    group.plot.area(x="Date", y=["Severe outcome", "Mild outcome"], title=name, color=["red", "blue"], style=['-', '.'], lw=5, alpha=0.2)
    plt.legend(handles=legend_elements)
    plt.ylabel("Count")
    plt.xlabel("")
    plt.savefig(f"plots/{pd.Timestamp.today().date()}_{name}_stacked-area-chart.png", dpi=300)
plot_df["Region"] = plot_df.Region.str.strip()
# Figure S2
legend_elements = [
    Line2D([0], [0], marker='^', markerfacecolor='red', markersize=5, color='red', label='Severe', lw=1, linestyle="-"),
    Line2D([0], [0], marker='s', markerfacecolor='blue', markersize=5,  color='blue', label='Mild', lw=1, linestyle="--"),
]
for name, group in plot_df.groupby('Region'):
    group.plot.line(
        x="Date",
        y=["Severe outcome", "Mild outcome"],
        title=name,
        style=['r^-', 'bs--'],
        lw=1,
        markersize=6,
        alpha=0.6
        )
    print(name)
    if name == "Oceania":
        tiks, labs = plt.xticks()
        plt.xticks(tiks[1:])
        plt.yticks([0, 1, 2])
    plt.legend(handles=legend_elements)
    plt.ylabel("Count")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(f"plots/2020-10-21_{name}_fig-s2.png", dpi=300)
df["continent"].str.lower().str.strip().value_counts()