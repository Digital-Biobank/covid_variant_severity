import ssl

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

ssl._create_default_https_context = ssl._create_unverified_context

TARGET_NAME = "is_red"
TARGET_PART = "red"
TARGET_FULL = "Green/Red"

variants = pd.read_csv("data/01_77142-vcf_wide_variants.csv")
nz_coefs = pd.read_csv(
    "03_red-target_logistic-regression-coefs_non-zero-coefficients.csv"
)
nz_coefs.columns = ["ref_pos_alt", "coefficient"]
nz_coefs = nz_coefs.set_index("ref_pos_alt")
df = nz_coefs.join(variants.set_index("ref_pos_alt")).dropna()
df2 = pd.read_csv(
    "https://github.com/galaxyproject/SARS-CoV-2/blob/master/genomics/4-Variation/variant_list.tsv.gz?raw=true",
    sep="\t", compression="gzip"
    )

df2 = df2.assign(
    ref_pos_alt=df2["REF"] + df2["POS"].astype(str) + df2["ALT"],
)

df3 = df.join(df2.set_index("ref_pos_alt"))
df3["variant_freq"] = df3["variant_count"] / 77142
df3["log10_variant_freq"] = df3["variant_freq"].transform("log10")
df3["odds_ratio"] = df3["coefficient"].transform("exp")
df3.to_csv("variant-or-plot.csv")

df3 = pd.read_csv("variant-or-plot.csv", index_col=0)
df3 = df3[~df3.index.duplicated()]

df3["mutation_type"] = df3["EFF[*].FUNCLASS"].astype(str)
df3["mutation_type"] = df3["mutation_type"].str.replace(".", "Non-coding")
df3["or_lt_two"] = df3["odds_ratio"] < 2
df3["or_gt_half"] = df3["odds_ratio"] > .5

top_btm = df3[(df3["odds_ratio"] > 2) | (df3["odds_ratio"] < .5)].index
len(top_btm)
pd.Series(top_btm).to_csv(
    "03_red-target_logistic-regression_top-and-bottom-odds-ratios.csv"
)
# df3.loc[df3["odds_ratio"] > .5, "mutation_type"] = "Odds ratio greater than .5"

# df3["or_lt_2"] = df3["mutation_type"] == "Odds ratio below 2"
df3["is_ct"] = df3.index.str.replace(r'\d', '') == "CT"
df3["transition"] = df3.index.astype(str).str.replace(r'\d', '').isin(["CT", "TC", "AG", "GA"])

# Get Ratios
df3["btm"] = ~df3["or_gt_half"]
df3["top"] = ~df3["or_lt_two"]
df3["mid"] = df3["or_gt_half"] & df3["or_lt_two"]

df3.loc[df3["mutation_type"] == "nan", "POS"] = df3.loc[df3["mutation_type"] == "nan"].index.str.replace(r'\D', "")
df3.loc[(df3["mutation_type"] == "nan") & ~df3["mid"], "mutation_type"] = ["MISSENSE", "DELETION", "DELETION", "DELETION", "DELETION", "MISSENSE", "DELETION"]

round(df3["is_ct"].sum() / len(df3["is_ct"]), 2)
round(df3["transition"].sum() / len(df3["is_ct"]), 2)
round(sum(df3["mutation_type"] == "MISSENSE") / len(df3), 2)
round(sum(df3["mutation_type"] == "SILENT") / len(df3), 2)
round(sum(df3["mutation_type"] == "Non-coding") / len(df3), 2)
sum(df3["mutation_type"] == "DELETION") / len(df3)
sum(df3["mutation_type"] == "nan") / len(df3)

btm_df = df3[~df3["or_gt_half"]]
top_df = df3[~df3["or_lt_two"]]
mid_df = df3[df3["or_gt_half"] & df3["or_lt_two"]]

sum(top_df["is_ct"] == True) / len(top_df)
sum(mid_df["is_ct"] == True) / len(mid_df)
sum(btm_df["is_ct"] == True) / len(btm_df)
sum(top_df["transition"] == True) / len(top_df)
sum(mid_df["transition"] == True) / len(mid_df)
sum(btm_df["transition"] == True) / len(btm_df)

sum(top_df["mutation_type"] == "MISSENSE") / len(top_df)
sum(top_df["mutation_type"] == "SILENT") / len(top_df)
sum(top_df["mutation_type"] == "Non-coding") / len(top_df)
sum(top_df["mutation_type"] == "DELETION") / len(top_df)

sum(mid_df["mutation_type"] == "MISSENSE") / len(mid_df)
sum(mid_df["mutation_type"] == "SILENT") / len(mid_df)
sum(mid_df["mutation_type"] == "Non-coding") / len(mid_df)
sum(mid_df["mutation_type"] == "DELETION") / len(mid_df)

sum(btm_df["mutation_type"] == "MISSENSE") / len(btm_df)
sum(btm_df["mutation_type"] == "SILENT") / len(btm_df)
sum(btm_df["mutation_type"] == "Non-coding") / len(btm_df)
sum(btm_df["mutation_type"] == "DELETION") / len(btm_df)

mid_df
df_lt_2["transition"].sum() / len(df_lt_2)
sum(df_lt_2["transition"] == False) / len(df_lt_2)

sum(df_lt_2["EFF[*].FUNCLASS"] == "MISSENSE") / len(df_lt_2)
sum(df_lt_2["EFF[*].FUNCLASS"] == "SILENT") / len(df_lt_2)

df_gt_2.index.duplicated()
df_gt_2["is_ct"].sum()
sum(df_gt_2["is_ct"] == False)
df_gt_2["is_ct"].sum() / len(df_gt_2)
sum(df_gt_2["is_ct"] == False) / len(df_gt_2)

df_gt_2["transition"].sum() / len(df_gt_2)
sum(df_gt_2["transition"] == False) / len(df_gt_2)

sum(df_gt_2["EFF[*].FUNCLASS"] == "MISSENSE") / len(df_gt_2)
sum(df_gt_2["EFF[*].FUNCLASS"] == "SILENT") / len(df_gt_2)

df3["EFF[*].FUNCLASS"].sum()

plt.rcParams['figure.figsize'] = 7, 5.25
plt.rcParams.update(plt.rcParamsDefault)

import numpy as np
df3.loc[df3["mid"], "mutation_type"] = "Low-priority"
df3.loc[df3["mid"], "is_ct"] = "Low-priority"
df3.loc[df3["mid"], "transition"] = "Low-priority"
df3["trans"] = np.where(
    df3["is_ct"],
    "C->T",
    np.where(df3["transition"], "transition", "transversion"
    ))
df3.loc[, "transition"] = "Low-priority"

sum(df3["mutation_type"] == "nan")
df3[df3["mutation_type"] == "nan"].index.tolist()

df3["log_or"] = df3["odds_ratio"].transform("log2")
df4 = df3[df3["odds_ratio"] > 2]

palette = {
    "SILENT": "orange",
    "MISSENSE": "r",
    "Non-coding": "g",
    "Low-priority": 'gray',
    "DELETION": 'purple',
}

import matplotlib.ticker as mticker

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Missense', markerfacecolor='r', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Silent', markerfacecolor='orange', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Non-coding', markerfacecolor='g', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Deletion', markerfacecolor='purple', markersize=5),
]
sns.scatterplot(x="variant_freq", y="odds_ratio", hue="mutation_type", data=df3, palette=palette)
plt.legend(handles=legend_elements)
plt.xlabel('Variant Frequency')
plt.ylabel('Odds ratio')
plt.xscale('log')
plt.yscale('log', basey=2)
# plt.axhline(1, linestyle="--")
plt.gcf().axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("plots/all-variants_or-vs-var-freq_without-line.png", dpi=300)
plt.show()

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Missense', markerfacecolor='r', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Silent', markerfacecolor='orange', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Non-coding', markerfacecolor='g', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Deletion', markerfacecolor='purple', markersize=5),
]

sns.scatterplot(x="POS", y="odds_ratio", hue="mutation_type", data=df3, palette=palette)
plt.axvline(21563, ls=":")
plt.axvline(25384, ls=":")
plt.legend(handles=legend_elements, fancybox=True, framealpha=0, loc="upper left", prop={'size': 9})
plt.xlabel('Position')
plt.ylabel('Odds ratio')
plt.yscale('log', basey=2)
plt.gcf().axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("plots/all-variants_or-vs-var-pos.png", dpi=300)
plt.show()

palette = {"C->T": "orange", "transition": "b", "transversion": 'green'}
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='C->T transition', markerfacecolor='orange', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Other transition', markerfacecolor='b', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Transversion', markerfacecolor='g', markersize=5),
]
sns.scatterplot(x="variant_freq", y="odds_ratio", hue="trans", data=df3, alpha=.6)
plt.legend(handles=legend_elements)
plt.xlabel('Variant Frequency')
plt.ylabel('Odds ratio')
plt.xscale('log')
plt.yscale('log', basey=2)
# plt.axhline(1, linestyle="--")
plt.gcf().axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("plots/all-variants_or-vs-var-freq_without-line.png", dpi=300)
plt.show()

sns.scatterplot(x="POS", y="odds_ratio", hue="trans", data=df3, palette=palette)
plt.legend(handles=legend_elements, loc=0, ncol=len(legend_elements))
plt.xlabel('Position')
plt.ylabel('Odds ratio')
plt.yscale('log', basey=2)
plt.gcf().axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.tight_layout()
plt.ylim((-1, 9))
plt.savefig("plots/all-variants_or-vs-var-pos_ct.png", dpi=300)
plt.show()


palette = {True: "b", False: "orange", "Low-priority": 'gray'}

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Transversion', markerfacecolor='orange', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Transition', markerfacecolor='b', markersize=5),
]
sns.scatterplot(x="variant_freq", y="odds_ratio", hue="transition", data=df3, size="odds_ratio", palette=palette)
plt.legend(handles=legend_elements)
plt.xlabel('Variant Frequency')
plt.ylabel('Odds ratio')
plt.xscale('log')
plt.tight_layout()
plt.savefig("plots/top-variants_or-vs-var-freq_trans.png", dpi=300)
plt.show()

palette = {True: "orange", False: "b", "Low-priority": 'gray'}

legend_elements = [
    Line2D([0], [0], color='b', lw=3, label='Region of interest', linestyle=":"),
    Line2D([0], [0], marker='o', color='w', label='C->T', markerfacecolor='orange', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='b', markersize=5),
]
sns.scatterplot(x="POS", y="odds_ratio", hue="is_ct", data=df3, size="odds_ratio", palette=palette)
plt.axvline(21563, ls=":")
plt.axvline(25384, ls=":")
plt.legend(handles=legend_elements)
plt.xlabel('Position')
plt.ylabel('Odds ratio')
plt.tight_layout()
plt.savefig("plots/all-variants_or-vs-var-pos_ct.png", dpi=300)
plt.show()

palette = {True: "b", False: "orange", "Low-priority": 'gray'}

legend_elements = [
    Line2D([0], [0], color='b', lw=3, label='Region of interest', linestyle=":"),
    Line2D([0], [0], marker='o', color='w', label='Transversion', markerfacecolor='orange', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Transition', markerfacecolor='b', markersize=5),
]
sns.scatterplot(x="POS", y="odds_ratio", hue="transition", data=df3, size="odds_ratio", palette=palette)
plt.axvline(21563, ls=":")
plt.axvline(25384, ls=":")
plt.legend(handles=legend_elements)
plt.xlabel('Position')
plt.ylabel('Odds ratio')
plt.tight_layout()
plt.savefig("plots/top-variants_or-vs-var-pos_trans.png", dpi=300)
plt.show()
df3.columns
sorted_or = df3["odds_ratio"].sort_values()
N = 20
cmap = plt.cm.get_cmap('coolwarm', N)
sorted_or = sorted_or.dropna().drop_duplicates()
sorted_or.index[0]
sorted_or.index = ["33bp del at 499"] + list(sorted_or.index[1:])
top_or = sorted_or[-10:]
top_or
btm_or = sorted_or[:10]
btm_or
all_or = pd.concat([top_or, btm_or]).sort_values()
varlist = all_or.index.tolist()
varlist[0] = 'TGCACCTCATGGTCATGTTATGGTTGAGCTGGTA499T'
varlist
df3 = df3[~df3.index.duplicated(keep='first')]

or_df = df3.loc[varlist, "variant_freq"]
or_df[::-1]
my_norm = Normalize(vmin=all_or.min(), vmax=all_or.max())
all_or.plot.barh(
    legend=False,
    color=cmap(my_norm(all_or)),
    edgecolor="k"
)
plt.title(f"Variants By Odds Ratio")
plt.xlabel("Odds Ratio")
plt.ylabel("Variant")
plt.axvline(1, ls=":")
plt.xscale('log', basex=2)
plt.gcf().axes[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.tight_layout()
plt.savefig(f"plots/top_variants_odds-ratio.png", dpi=300)
plt.show()

sorted_or = df3["odds_ratio"].sort_values()
N = 20
cmap = plt.cm.get_cmap('coolwarm', N)
sorted_or = sorted_or.dropna()
top_or = sorted_or.drop(sorted_or.index[N // 2:-N // 2])
top_or
my_norm = Normalize(vmin=top_or.min(), vmax=top_or.max())
top_or.plot.barh(
    legend=False,
    color=cmap(my_norm(top_or)),
    edgecolor="k"
)
plt.tight_layout()
plt.title(f"Top {TARGET_FULL} Logistic Regression Variables")
plt.xlabel("Coefficient")
plt.ylabel("Variant")
plt.savefig(f"top_{TARGET_PART}_variable_coefs.png")
plt.show()

df = pd.read_parquet("data/01_77142-vcf_wide_join.parquet")
variants_by_region = df.groupby("region").sum()
variants_by_region

variants_by_region.to_csv("../../data/variants_by_region.csv")
variants_by_region.T.value_counts().head()
variants_by_region.head()
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(
    stacked=True)
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(
    stacked=True)
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum",
                                                                                         axis=1).head().plot.barh(
    stacked=True)
variants_by_region.assign(sum=variants_by_region.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(
    stacked=True)

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
variants_by_red.index = ["Green", "Red"]
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
