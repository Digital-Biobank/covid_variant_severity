import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/01_77142-vcf_wide_join.parquet")

variants_by_region = df.groupby("region").sum()
variants_by_region

variants_by_region.to_csv("../../data/variants_by_region.csv")
variants_by_region.T.value_counts().head()
variants_by_region.head()
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).head().plot.barh(stacked=True)
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)

plt.savefig("variants_by_region.png")
plt.tight_layout()
plt.gca().get_legend().remove()
plt.tight_layout()
plt.savefig("variants_by_region_no-legend.png")
norm = variants_by_region.div(variants_by_region.sum(axis=1), axis=0)
normt = norm.T
normt.assign(sum=normt.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
norm.assign(sum=norm.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
norm.assign(sum=norm.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
plt.gca().get_legend().remove()
plt.tight_layout()
plt.savefig("variants_by_region_no-legend_normalized.png")

# %% Combine cleaned data with variant data
df = df.assign(
    region=df.region,
    cat_region=df.cat_region,
    is_red=df.is_red,
    clade=df.clade,
    covv_clade=df.covv_clade,
)

# %% save variants by region, is_red, etc.
df.to_parquet("../../data/01_77142-vcf_wide_region-out-clade.parquet")

# %% plot top variants by red/green status
variants_by_red = df.drop([
    "region",
    "clade",
    "covv_clade"
    ], axis=1).groupby("is_red").sum()
variants_by_red.index =  ["Green", "Red"]
variants_by_red.T.assign(
    sum=variants_by_red.T.sum(axis=1)
    ).sort_values(by="sum").drop(
        "sum",
        axis=1
        ).tail(20).plot.barh(stacked=True, color=["green", "red"])
plt.legend(title="Red/Green Status", loc=4)
plt.tight_layout()
plt.savefig("top_variants_by_red.png", dpi=300)

# %% plot top variants by red/green status
variants_by_red = df.drop([
    "region",
    "clade",
    "covv_clade"
    ], axis=1).groupby("is_red").sum()
variants_by_red.index = ["Green", "Red"]
variants_by_red.assign(
    sum=variants_by_red.sum(axis=1)
    ).sort_values(by="sum").drop(
        "sum",
        axis=1
        ).plot.barh(stacked=True, color=["green", "red"])
plt.legend(title="Red/Green Status", loc=4)
plt.tight_layout()
plt.figure(figsize=(4, 4))
plt.savefig("variants_by_red.png", dpi=300)
