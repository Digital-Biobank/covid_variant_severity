import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
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

ors = pd.read_csv("data/2020-10-21_odds-ratios.csv", index_col=0)

ors = ors.drop([
    'male',
    'age',
    'Asia',
    'North America',
    'South America',
]).rename(
    {"0": "odds_ratio"},
    axis=1
)

ors_ann = ors.join(ann)
ors_ann
ors_ann["mutation_type"] = ors_ann["EFF[*].EFFECT"].map({
    "synonymous_variant": "Silent",
    "upstream_gene_variant": np.nan,
    "downstream_gene_variant": np.nan,
    "missense_variant": "Missense",
    "stop_gained": "Nonsense",
    "frameshift_variant": "Frameshift",
    "intergenic_region": "Non-coding",
    "disruptive_inframe_deletion": "Deletion"
 })

var = pd.read_csv("data/variant-freq.csv", index_col=0)
var
df = ors_ann.dropna(subset=["mutation_type"]).join(var)
df = df[~df.index.duplicated(keep='first')]
df.to_csv("data/2020-10-21_variants.csv")
df["or_lt_two"] = df["odds_ratio"] < 2
df["or_gt_half"] = df["odds_ratio"] > .5

df["is_ct"] = df.index.str.replace(r'\d', '') == "CT"
df["transition"] = df.index.astype(str).str.replace(r'\d', '').isin(["CT", "TC", "AG", "GA"])

df["btm"] = ~df["or_gt_half"]
df["top"] = ~df["or_lt_two"]
df["mid"] = df["or_gt_half"] & df["or_lt_two"]
# df.loc[(df["mutation_type"] == "nan") & ~df["mid"], "mutation_type"] = ["MISSENSE", "DELETION", "DELETION", "DELETION", "DELETION", "MISSENSE", "DELETION"]

df["trans"] = np.where(
    df["is_ct"],
    "C->T",
    np.where(df["transition"], "transition", "transversion"
    ))

# Figure 1
palette = {
    "Silent": "orange",
    "Missense": "r",
    "Non-coding": "g",
    "Frameshift": "violet",
    "Low-priority": 'gray',
    "nan": 'black',
    # "MISSENSE,MISSENSE": 'r',
    "Nonsense": 'b',
    # "MISSENSE,SILENT": 'yellow',
    "Deletion": 'purple',
}

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
    Line2D([0], [0], marker='D', color='w', label='Deletion', markerfacecolor='purple', markersize=7),
    Line2D([0], [0], marker='^', color='w', label='Missense', markerfacecolor='r', markersize=7),
    Line2D([0], [0], marker='>', color='w', label='Frameshift', markerfacecolor='violet', markersize=7),
    Line2D([0], [0], marker='p', color='w', label='Non-coding', markerfacecolor='g', markersize=7),
    Line2D([0], [0], marker='X', color='w', label='Nonsense', markerfacecolor='b', markersize=7),
    Line2D([0], [0], marker='s', color='w', label='Silent', markerfacecolor='orange', markersize=7),
]

record = BiopythonTranslator().translate_record("NC_045512.2.gb")
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

fig, (ax0, ax1, ax2) = plt.subplots(
    3,
    1,
    figsize=(11, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 7, 2.5]}
)
plt.subplots_adjust(top=0.9,bottom=.2, wspace=-0.1, hspace=0)
sns.scatterplot(
    x="POS",
    y="neg_log10_chi_square_pvalue",
    hue="mutation_type",
    data=pval_ann,
    style="mutation_type",
    markers=markers,
    palette=palette,
    ax=ax0
    )
sns.scatterplot(
    x="POS",
    y="neg_log10_chi_square_pvalue",
    hue="mutation_type",
    data=pval_ann,
    style="mutation_type",
    markers=markers,
    palette=palette,
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
ax1.set_ylim(10**-4.9, 17.49)
ax0.set_ylim(34.9, 45.01)
# ax1.set_yscale('log', basey=10)
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
    "plots/all-variants_pval-vs-var-pos.png",
    dpi=300,
    bbox_inches = 'tight',
    pad_inches = 0.1
    )
plt.show()

df.loc[df["mid"], "mutation_type"] = "Low-priority"
df.loc[df["mid"], "is_ct"] = "Low-priority"
df.loc[df["mid"], "transition"] = "Low-priority"
sns.scatterplot(
    x="variant_frequency",
    y="odds_ratio",
    hue="mutation_type",
    data=df,
    style="mutation_type",
    markers=markers,
    palette=palette
    )
plt.xlabel('Variant Frequency')
plt.ylabel('Odds ratio')
plt.ylim(2**-6.7, 2**6.5)
plt.xscale('log')
plt.yscale('log', basey=2)
plt.legend(
    handles=legend_elements,
    loc="upper right",
    ncol=1,
    framealpha=0.5,
    fancybox=True,
    prop={'size': 10}
    )
plt.yticks(
    (0.015625, 0.0625, 0.25, 1, 4, 16, 32, 64),
    ("0.0156", "0.0625", "0.25", "1", "4", "16", "32", "64")
    )
plt.tight_layout()
plt.savefig("plots/all-variants_or-vs-var-freq.png", dpi=300)
plt.show()

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
    hue="mutation_type",
    data=df,
    style="mutation_type",
    markers=markers,
    palette=palette,
    ax=ax1
    )
record.plot(ax=ax2)
plt.xlabel('Position')
ax1.set_ylabel('Odds ratio')
ax1.set_ylim(2**-6.7, 2**6.5)
ax1.set_yscale('log', basey=2)
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
ax1.minorticks_off()
box = ax2.get_position()
box.y0 += .005
box.y1 += .005
ax2.set_position(box)
plt.savefig(
    "plots/all-variants_or-vs-var-pos.png",
    dpi=300,
    bbox_inches = 'tight',
    pad_inches = 0.1
    )
plt.show()

# Figure S3
palette = {"C->T": "orange", "transition": "b", "transversion": 'green'}
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='C->T transition', markerfacecolor='orange', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Other transition', markerfacecolor='b', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Transversion', markerfacecolor='g', markersize=6),
]
sns.scatterplot(x="variant_frequency", y="odds_ratio", hue="trans", data=df, alpha=.6)
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

df["C->T"] = df.index.str.replace(r'\d', '') == "CT"
df["transition"] = df.index.astype(str).str.replace(r'\d', '').isin(["CT", "TC", "AG", "GA"])
df["Transversion"] = df.index.astype(str).str.replace(r'\d', '').isin(["CA", "AC", "TG", "GT", "CG", "GC", "AT", "TA"])
df["trans"] = np.where(
    df["C->T"],
    "C->T",
    np.where(df["transition"], "transition", "Transversion"
    ))
df[["C->T", "trans", "transition"]]
sum(df["trans"] == "C->T")
sum(df["trans"] != "C->T")
df["ct_count"] = df.groupby("variant_frequency")["C->T"].count()
df["ct_count"] 
df['POS'] = df['POS'].astype(float)
len(df)
cut_bins = list(range(0, 30001, 3000))
df['cut_pos'] = pd.cut(df["POS"], bins=cut_bins, right=False)
g = df.groupby("cut_pos")[["C->T", "transition", "Transversion"]].sum()
g
g.index = [f"{int(i.left/1000)}-{int(i.right/1000)}" for i in g.index]
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
plt.xlabel("Position (kb)")
plt.ylabel("Count")
plt.legend(loc="upper right", ncol=3)
plt.tight_layout()
plt.savefig(f"{pd.Timestamp.today().date()}_fig-s3a.png", dpi=300)

cut_bins = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**-0]
df['cut_var_freq'] = pd.cut(df["variant_frequency"], bins=cut_bins, right=False)
g = df.groupby("cut_var_freq")[["C->T", "transition", "Transversion"]].sum()
g.index = [f"{i.left}-{i.right}" if i.right != 0.0001 else f"<{i.right}" for i in g.index]
g
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
locs, labs = plt.xticks()
plt.xticks(list(locs)[1::2], list(labs)[1::2])
plt.xlabel("Variant Frequency")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{pd.Timestamp.today().date()}_fig-s3b.png", dpi=300)

pval_ann = pval_ann.join(var)
pval_ann = pval_ann[~pval_ann.index.duplicated(keep='first')]

pval_ann["C->T"] = pval_ann.index.str.replace(r'\d', '') == "CT"
pval_ann["transition"] = pval_ann.index.astype(str).str.replace(r'\d', '').isin(["CT", "TC", "AG", "GA"])
pval_ann["Transversion"] = pval_ann.index.astype(str).str.replace(r'\d', '').isin(["CA", "AC", "TG", "GT", "CG", "GC", "AT", "TA"])
pval_ann["trans"] = np.where(
    pval_ann["C->T"],
    "C->T",
    np.where(pval_ann["transition"], "transition", "Transversion"
    ))
pval_ann[["C->T", "trans", "transition"]]
sum(pval_ann["trans"] == "C->T")
sum(pval_ann["trans"] != "C->T")
pval_ann["ct_count"] = pval_ann.groupby("variant_frequency")["C->T"].count()
pval_ann["ct_count"] 
pval_ann['POS'] = pval_ann['POS'].astype(float)
len(pval_ann)
cut_bins = list(range(0, 30001, 3000))
pval_ann['cut_pos'] = pd.cut(pval_ann["POS"], bins=cut_bins, right=False)
g = pval_ann.groupby("cut_pos")[["C->T", "transition", "Transversion"]].sum()
g
g.index = [f"{int(i.left/1000)}-{int(i.right/1000)}" for i in g.index]
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
plt.xlabel("Position (kb)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{pd.Timestamp.today().date()}_fig-s3c.png", dpi=300)

cut_bins = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**-0]
pval_ann['cut_var_freq'] = pd.cut(pval_ann["variant_frequency"], bins=cut_bins, right=False)
g = pval_ann.groupby("cut_var_freq")[["C->T", "transition", "Transversion"]].sum()
g.index = [f"{i.left}-{i.right}" if i.right != 0.0001 else f"<{i.right}" for i in g.index]
g
g["Other transition"] = g["transition"] - g["C->T"]
g[["C->T", "Other transition", "Transversion"]].plot.line(style=["^-", "o--", "s:"])
locs, labs = plt.xticks()
plt.xticks(list(locs)[1::2], list(labs)[1::2])
plt.xlabel("Variant Frequency")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{pd.Timestamp.today().date()}_fig-s3d.png", dpi=300)
