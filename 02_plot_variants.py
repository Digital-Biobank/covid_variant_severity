import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

variants = pd.read_csv("data/01_77142-vcf_wide_variants.csv")
nz_coefs = pd.read_csv(
    "03_red-target_logistic-regression-coefs_non-zero-coefficients.csv"
)
nz_coefs.columns = ["ref_pos_alt", "coefficient"]
nz_coefs = nz_coefs.set_index("ref_pos_alt")
df = nz_coefs.join(variants.set_index("ref_pos_alt")).dropna()
df2 = pd.read_csv("https://github.com/galaxyproject/SARS-CoV-2/blob/master/genomics/4-Variation/variant_list.tsv.gz?raw=true", sep="\t", compression="gzip")

df2 = df2.assign(
    ref_pos_alt=df2["REF"] + df2["POS"].astype(str) + df2["ALT"],
)

df3 = df.join(df2.set_index("ref_pos_alt"))
df3
df3.iloc[:, -4]
df.columns = ["coefficient", "variant_count"]
df
# TODO don't log transform freq, instead use matplotlib to log the axis
# TODO label colors from joined galaxy dataset
# TODO label variant names for top variants by odds ratio
df3["variant_freq"] = df3["variant_count"] / 77142
df3["log10_variant_freq"] = df3["variant_freq"].transform("log10")
df3["odds_ratio"] = df3["coefficient"].transform("exp")
df3.to_csv("variant-or-plot.csv")
df3
sns.scatterplot(x="AF", y="odds_ratio", hue="EFF[*].FUNCLASS", data=df3, size="odds_ratio", legend=None)
plt.xscale('log')
plt.show()
sns.scatterplot(x="AF", y="variant_freq", hue="EFF[*].FUNCLASS", data=df3, size="odds_ratio", legend=None)
plt.show()

df3["variant_freq"].isna().sum()
df3["odds_ratio"].gt(5).sum()
df3["odds_ratio"].drop_duplicates().sort_values(ascending=False)
df3["mutation_type"] = df3["EFF[*].FUNCLASS"]
df3.loc[df3["odds_ratio"] < 2, "mutation_type"] = "Odds ratio below 2"

palette ={
    "Odds ratio": "gray", "B": "orange", "C": "C2", "Total": "k"}


legend_elements = [
    Line2D([0], [0], color='b', lw=3, label='Region of interest', linestyle=":"),
    Line2D([0], [0], marker='o', color='w', label='Missense', markerfacecolor='r', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Odds ratio below 2', markerfacecolor='b', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Silent', markerfacecolor='orange', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='.', markerfacecolor='g', markersize=5),
]
sns.scatterplot(x="POS", y="odds_ratio", hue="mutation_type", data=df3, size="odds_ratio")
plt.axvline(10000, ls=":")
plt.axvline(12000, ls=":")
# plt.legend(handles=legend_elements)
plt.show()

df["variant_freq"].to_csv("data/variant-freq.csv")

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
