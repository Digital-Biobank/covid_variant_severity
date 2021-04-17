import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from dna_features_viewer import BiopythonTranslator


ann = pd.read_csv("data/2020-11-10_SnpEffSnpSift_3612pats.vcf", sep="\t")
ann["aa"] = ann["EFF[*].AA"].str.slice(2,).replace({
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Glu": "E",
    "Gln": "Q",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    },
    regex=True
).str.strip()
ann["ref_pos_alt"] = ann["REF"] + ann["POS"].astype(str) + ann["ALT"]
ann = ann.set_index("ref_pos_alt")
ann = ann.drop_duplicates()

pval_df = pd.read_csv(f"data/2020-10-21_p-values.csv", index_col=0)

pval_df = pval_df.assign(
    neg_log10_trend_test_pvalue=-np.log10(pval_df["trend_test_pvalue"]),
    neg_log10_chi_square_pvalue=-np.log10(pval_df["chi_square_pvalue"]),
    neg_log10_odds_ratio_pvalue=-np.log10(pval_df["odds_ratio_pvalue"]),
)

pval_ann = pval_df.join(ann)
pval_ann["mutation_type"] = pval_ann["EFF[*].EFFECT"].map({
    "synonymous_variant": "Silent",
    "upstream_gene_variant": np.nan,
    "downstream_gene_variant": np.nan,
    "missense_variant": "Missense",
    "stop_gained": "Nonsense",
    "frameshift_variant": "Frameshift",
    "intergenic_region": "Non-coding",
    "disruptive_inframe_deletion": "Deletion"
 })

df = pd.read_parquet(f"data/2020-10-21_vcf-clean.parquet")

df = df.reset_index().drop_duplicates(subset="pid")
df.drop(["is_red", "GH", "GR", "L", "O", "S", "V"], axis="columns")
df.iloc[:, 2:-13]
df
voi = df[pval_ann.index.tolist() + ["is_red"]]
pval_ann = pval_ann.reset_index().drop_duplicates(subset="index")
sig_pval_ann = pval_ann[pval_ann["odds_ratio_pvalue"].lt(0.05)]
var = pd.read_csv("data/2020-10-21_variant-freq.csv", index_col=0)
var
sig_pval_ann_var = sig_pval_ann.set_index("index").join(var)

# Supplemental Table 1
sig_pval_ann_var[[
    "aa",
    "mutation_type",
    "variant_frequency",
    "lower",
    "odds_ratio",
    "upper",
    "odds_ratio_pvalue"
]].sort_values("odds_ratio")[:20].append(sig_pval_ann_var[[
    "aa",
    "lower",
    "mutation_type",
    "variant_frequency",
    "odds_ratio",
    "upper",
    "odds_ratio_pvalue"
]].sort_values("odds_ratio")[-20:]).to_excel("top_btm_var.xlsx")

# Some variants are both mild and severe
voi = voi.groupby("is_red").sum().T
# The difference will be the color intensity of red
voi["diff"] = voi[1] - voi[0]

ors = pd.read_csv("data/2020-10-21_odds-ratios.csv", index_col=0)
ors = ors.drop([
    "pid",
    'age',
    'Asia',
    "Europe",
    # "male",
    'North America',
    'South America',
]).rename(
    {"0": "odds_ratio"},
    axis=1
)

logreg_ors = pd.read_csv("data/2020-10-21_top-and-btm-ors.csv", index_col=0)

logreg_ors.shape
loav = logreg_ors.join(ann).join(var)
loav = loav.reset_index().drop_duplicates(keep="first", subset=["index"])
loav["mutation_type"] = loav["EFF[*].EFFECT"].map({
    "synonymous_variant": "silent",
    "upstream_gene_variant": np.nan,
    "downstream_gene_variant": np.nan,
    "missense_variant": "missense",
    "stop_gained": "nonsense",
    "frameshift_variant": "frameshift",
    "intergenic_region": "non-coding",
    "disruptive_inframe_deletion": "deletion"
})
loav
loav.to_csv("data/logreg_ors.csv")

ann.loc["G29711T", "EFF[*].EFFECT":]

ors_ann = ors.join(ann)

ors_ann["mutation_type"] = ors_ann["eff[*].effect"].map({
    "synonymous_variant": "silent",
    "upstream_gene_variant": np.nan,
    "downstream_gene_variant": np.nan,
    "missense_variant": "missense",
    "stop_gained": "nonsense",
    "frameshift_variant": "frameshift",
    "intergenic_region": "non-coding",
    "disruptive_inframe_deletion": "deletion"
 })


df = ors_ann.dropna(subset=["mutation_type"]).join(var)
df = df[~df.index.duplicated(keep='first')]
df.to_csv("data/2020-10-21_variants.csv")
df["or_lt_two"] = df["odds_ratio"] < 2
df["or_gt_half"] = df["odds_ratio"] > .5

df["is_ct"] = df.index.str.replace(r'\d', '', regex=True) == "CT"
df["transition"] = df.index.astype(str).str.replace(
    r'\d',
    '',
    regex=True
).isin(["CT", "TC", "AG", "GA"])

df["btm"] = ~df["or_gt_half"]
df["top"] = ~df["or_lt_two"]
df["mid"] = df["or_gt_half"] & df["or_lt_two"]
# df.loc[(df["mutation_type"] == "nan") & ~df["mid"], "mutation_type"] = ["MISSENSE", "DELETION", "DELETION", "DELETION", "DELETION", "MISSENSE", "DELETION"]
df["trans"] = np.where(
    df["is_ct"],
    "C->T",
    np.where(df["transition"], "transition", "transversion"
    ))

# Figure 1A

markers = {
    "Deletion": 'D',
    "Missense": "^",
    "Frameshift": ">",
    "Non-coding": "p",
    "Nonsense": 'X',
    "Silent": "s",
    "Low-priority": 'o',
    "nan": 'o',
    # "MISSENSE,MISSENSE": 'r',
    # "MISSENSE,SILENT": 'yellow',
}

legend_elements = [
    Line2D([0], [0], marker='D', color='w', markeredgecolor="k", label='Deletion', markerfacecolor='w', markersize=7),
    Line2D([0], [0], marker='^', color='w', markeredgecolor="k", label='Missense', markerfacecolor='w', markersize=7),
    Line2D([0], [0], marker='>', color='w', markeredgecolor="k", label='Frameshift', markerfacecolor='w', markersize=7),
    Line2D([0], [0], marker='p', color='w', markeredgecolor="k", label='Non-coding', markerfacecolor='w', markersize=7),
    Line2D([0], [0], marker='X', color='w', markeredgecolor="k", label='Nonsense', markerfacecolor='w', markersize=7),
    Line2D([0], [0], marker='s', color='w', markeredgecolor="k", label='Silent', markerfacecolor='w', markersize=7),
]

record = BiopythonTranslator().translate_record("NC_045512.2.gb")
for f in record.features:
    print(f)
for f in record.features[3:34]:
    record.features.remove(f)
for f in record.features[5:24:2]:
    record.features.remove(f)
for f in record.features[13:16:2]:
    record.features.remove(f)
record.features.remove(record.features[0])
record.features[8] = record.features[8].crop((27759, 27887))
dict(enumerate(record.features))
for i in (0, 3, 4, 5, 6, 7, 8, 9, 11, -1):
    record.features[i].label = None

voi = voi.reset_index().drop_duplicates().set_index("index")
pval_ann = pval_ann.set_index("index")
pval_ann = pval_ann.join(voi["diff"].rename("color"))
df = df.join(voi["diff"].rename("color"))

color_min = pval_ann["color"].min()
color_max = pval_ann["color"].max()
color_mid = 0
tsn = colors.TwoSlopeNorm(vcenter=color_mid, vmin=color_min, vmax=color_max)
color_data = tsn(pval_ann["color"])

color_min = df["color"].min()
color_max = df["color"].max()
color_mid = 0
tsn = colors.TwoSlopeNorm(vcenter=color_mid, vmin=color_min, vmax=color_max)
color_df = tsn(df["color"])
rdgn = sns.diverging_palette(h_neg=250, h_pos=15, s=99, l=55, sep=3, as_cmap=True)

fig, (ax0, ax1, ax2) = plt.subplots(
    3,
    1,
    figsize=(11, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 7, 2.5]}
)
plt.subplots_adjust(top=0.9, bottom=.2, wspace=-0.1, hspace=0)
sns.scatterplot(
    x="POS",
    y="neg_log10_chi_square_pvalue",
    hue=color_data,
    data=pval_ann,
    style="mutation_type",
    markers=markers,
    palette=rdgn,
    edgecolor='k',
    ax=ax0
    )
sns.scatterplot(
    x="POS",
    y="neg_log10_chi_square_pvalue",
    hue=color_data,
    data=pval_ann,
    style="mutation_type",
    markers=markers,
    palette=rdgn,
    edgecolor='k',
    ax=ax1
    )
record.plot(ax=ax2)
plt.xlabel('Position')
ax1.set_ylabel('Negative log10 p-value')
ax0.set_ylabel('')
box = ax2.get_position()
box.y0 += .08
box.y1 += .08
ax2.set_position(box)
ax1.set_ylim(10**-4.9, 19.99)
ax0.set_ylim(35.01, 48)
# ax1.set_yscale('log', base=10)
sm = plt.cm.ScalarMappable(cmap=rdgn, norm=tsn)
sm.set_array([])
plt.colorbar(sm)
ax1.legend(
    handles=legend_elements,
    loc="upper left",
    ncol=3,
    framealpha=0.5,
    fancybox=True,
    prop={'size': 11}
    )
ax0.legend().remove()
plt.savefig(
    f"plots/2020-10-21_all-variants_pval-vs-var-pos.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1
    )
plt.show()

pval_ann.sort_values(by="neg_log10_odds_ratio_pvalue").head(10)[["neg_log10_odds_ratio_pvalue","color", "mutation_type", "POS"]]
pval_ann.sort_values(by="neg_log10_chi_square_pvalue").head(10)[["neg_log10_chi_square_pvalue","color", "mutation_type", "POS"]]
pval_ann
# Figure 1B
df.loc[df["mid"], "mutation_type"] = "Low-priority"
df.loc[df["mid"], "is_ct"] = "Low-priority"
df.loc[df["mid"], "transition"] = "Low-priority"
sns.scatterplot(
    x="variant_frequency",
    y="odds_ratio",
    hue=color_df,
    data=df,
    style="mutation_type",
    markers=markers,
    palette=rdgn,
    edgecolor='k',
    )
plt.xlabel('Variant Frequency')
plt.ylabel('Odds ratio')
plt.ylim(2**-7, 2**6)
plt.hlines(y=2, ls="--", xmin=0, xmax=df["variant_frequency"].max())
plt.hlines(y=0.5, ls="--", xmin=0, xmax=df["variant_frequency"].max())
plt.xscale('log')
plt.yscale('log', base=2)
plt.legend(
    handles=legend_elements,
    loc="upper left",
    ncol=2,
    framealpha=0.5,
    fancybox=True,
    prop={'size': 10}
    )
plt.yticks(
    (0.015625, 0.0625, 0.25, 1, 4, 16, 32, 64),
    ("0.0156", "0.0625", "0.25", "1", "4", "16", "32", "64")
    )
plt.tight_layout()
plt.savefig(f"plots/2020-10-21_all-variants_or-vs-var-freq.png", dpi=300)
plt.show()

# Figure 1B
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(11, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": [5, 1]}
)
plt.subplots_adjust(wspace=-0.1, hspace=-0.18)
g = sns.scatterplot(
    x="POS",
    y="odds_ratio",
    hue=color_df,
    data=df,
    style="mutation_type",
    markers=markers,
    palette=rdgn,
    edgecolor='k',
    ax=ax1
    )
record.plot(ax=ax2)
plt.xlabel('Position')
ax1.set_ylabel('Odds ratio')
ax1.set_ylim(2**-7, 2**6)
ax1.set_yscale('log', base=2)
g.set_yticks(
    [0.015625, 0.0625, 0.25, 1, 4, 16, 32, 64]
    )
g.set_yticklabels(
    ["0.0156", "0.0625", "0.25", "1", "4", "16", "32", "64"]
    )
print(ax1.get_yticks())
print(ax1.get_yticklabels())
ax1.legend(
    handles=legend_elements,
    loc="upper left",
    ncol=2,
    framealpha=0.5,
    fancybox=True,
    prop={'size': 10}
    )
ax1.axhline(y=2, ls="--")
ax1.axhline(y=0.5, ls="--")
ax1.minorticks_off()
box = ax2.get_position()
box.y0 += .005
box.y1 += .005
ax2.set_position(box)
plt.savefig(
    f"plots/2020-10-21_all-variants_or-vs-var-pos.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1
    )
plt.show()

# Figure S3a
df["C->T"] = df.index.str.replace(r'\d', '', regex=True) == "CT"
df["transition"] = df.index.astype(str).str.replace(r'\d', '', regex=True).isin(["CT", "TC", "AG", "GA"])
df["Transversion"] = df.index.astype(str).str.replace(r'\d', '', regex=True).isin(["CA", "AC", "TG", "GT", "CG", "GC", "AT", "TA"])
df["trans"] = np.where(
    df["C->T"],
    "C->T",
    np.where(df["transition"], "transition", "Transversion"
    ))
df['POS'] = df['POS'].astype(float)
cut_bins = list(range(0, 30001, 3000))
df['cut_pos'] = pd.cut(df["POS"], bins=cut_bins, right=False)
g = df.groupby("cut_pos")[["C->T", "transition", "Transversion"]].sum()
g.index = [f"{int(i.left/1000)}-{int(i.right/1000)}" for i in g.index]
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
plt.xlabel("Position (kb)")
plt.ylabel("Count")
plt.legend(loc="upper right", ncol=3)
plt.tight_layout()
plt.savefig(f"2020-10-21_fig-s3a.png", dpi=300)
plt.show()


# Figure S3b
cut_bins = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**-0]
df['cut_var_freq'] = pd.cut(df["variant_frequency"], bins=cut_bins, right=False)
g = df.groupby("cut_var_freq")[["C->T", "transition", "Transversion"]].sum()
g.index = [f"{i.left}-{i.right}" if i.right != 0.0001 else f"<{i.right}" for i in g.index]
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
locs, labs = plt.xticks()
plt.xticks(list(locs)[1::2], list(labs)[1::2])
plt.xlabel("Variant Frequency")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"2020-10-21_fig-s3b.png", dpi=300)
plt.show()

# Figure S3c
pval_ann = pval_ann[~pval_ann.index.duplicated(keep='first')]

pval_ann["C->T"] = pval_ann.index.str.replace(r'\d', '', regex=True) == "CT"
pval_ann["transition"] = pval_ann.index.astype(str).str.replace(
    r'\d',
    '',
    regex=True
).isin(["CT", "TC", "AG", "GA"])
pval_ann["Transversion"] = pval_ann.index.astype(str).str.replace(
    r'\d',
    '',
    regex=True
).isin(["CA", "AC", "TG", "GT", "CG", "GC", "AT", "TA"])
pval_ann["trans"] = np.where(
    pval_ann["C->T"],
    "C->T",
    np.where(pval_ann["transition"], "transition", "Transversion"
    ))
pval_ann[["C->T", "trans", "transition"]]
sum(pval_ann["trans"] == "C->T")
sum(pval_ann["trans"] != "C->T")
pval_ann = pval_ann.join(var)
pval_ann["ct_count"] = pval_ann.groupby("variant_frequency")["C->T"].count()
pval_ann["ct_count"]
pval_ann['POS'] = pval_ann['POS'].astype(float)
len(pval_ann)
cut_bins = list(range(0, 30001, 3000))
pval_ann['cut_pos'] = pd.cut(pval_ann["POS"], bins=cut_bins, right=False)
g = pval_ann.groupby("cut_pos")[["C->T", "transition", "Transversion"]].sum()
g.index = [f"{int(i.left/1000)}-{int(i.right/1000)}" for i in g.index]
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
plt.xlabel("Position (kb)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"2020-10-21_fig-s3c.png", dpi=300)
plt.show()

# Figure S3d
cut_bins = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**-0]
pval_ann['cut_var_freq'] = pd.cut(pval_ann["variant_frequency"], bins=cut_bins, right=False)
g = pval_ann.groupby("cut_var_freq")[["C->T", "transition", "Transversion"]].sum()
g.index = [f"{i.left}-{i.right}" if i.right != 0.0001 else f"<{i.right}" for i in g.index]
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
locs, labs = plt.xticks()
plt.xticks(list(locs)[1::2], list(labs)[1::2])
plt.xlabel("Variant Frequency")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"2020-10-21_fig-s3d.png", dpi=300)
plt.show()

df.loc[~df["mid"], "variant_frequency"].quantile(.75)
df.loc[~df["mid"], "variant_frequency"].quantile(.25)
df.loc[~df["mid"], "variant_frequency"].median()
df.loc[~df["mid"], "variant_frequency"].min()
df.loc[~df["mid"]]
pval_ann
df[df["btm"]]
df[df["btm"]]
df["variant_frequency"].max()
df["variant_frequency"].median()
df["variant_frequency"].quantile(.25)
df.loc[:, "variant_frequency"].gt(.005)
df["variant_frequency"].lt(0.05)
df[df["variant_frequency"].ge(0.05)]
df[df["variant_frequency"].ge(0.05)]

# Supplemental Figure 1
today = pd.to_datetime("today").date()
df[df["mid"].ne(True) & df["variant_frequency"].lt(0.05)]
df[df["top"].eq(True) & df["variant_frequency"].lt(0.05)]
df[df["btm"].eq(True) & df["variant_frequency"].lt(0.05)]
df[df["mid"].ne(True) & df["variant_frequency"].lt(0.05)].to_csv(f"{today}_highest-and-lowest-odds-ratios.csv")
df[df["top"].eq(True) & df["variant_frequency"].lt(0.05)].to_csv(f"{today}_highest-odds-ratios.csv")
df[df["btm"].eq(True) & df["variant_frequency"].lt(0.05)].to_csv(f"{today}_lowest-odds-ratios.csv")
df[~df["mid"].eq(True) & df["transition"]].shape
df["transition"].sum() / len(df)
df.loc[df["btm"] == True].shape
pval_ann.odds_ratio.min()
-np.log10(pval_ann.odds_ratio_pvalue.min())
pval_ann["variant_frequency"].min()