# %% Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Read in variant frequencies
variants = pd.read_csv("data/00_77142-vcf_wide_variants.csv", index_col=0)

# %% Plot top 20 variants by frequency
variants.sort_values(by="variant_count").tail(20).plot.barh(legend=False)
plt.xlabel("Variant Frequency")
plt.ylabel("Variant")
plt.title("Top 20 variants by frequency")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_variant-bar.png")
plt.show()

# %% Plot distribution of variant frequency
sns.distplot(
    variants, rug=True, axlabel="Variant Frequency",
    rug_kws={"color": "k", "height": .02},
    kde_kws={"color": "b", "lw": 2},
    hist=False
)
plt.title("Distribution of variant frequency")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_variant-kde.png")
plt.show()

# %% Read in total mutations per patient
mutations = pd.read_csv("data/00_77142-vcf_wide_mutations.csv", index_col=0)

# %% Plot distribution of total_mutations
sns.distplot(
    mutations, rug=True, axlabel="Total Mutations",
    rug_kws={"color": "k", "height": .02},
    kde_kws={"color": "b", "lw": 2},
    # hist_kws={"histtype": "step", "linewidth": 3, "alpha": .5, "color": "g"}
    hist=False
)
plt.ylabel("Density")
plt.title("Distribution of total mutation count")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_mutation-kde.png")
plt.show()

# %% Read in outcomes data
df = pd.read_csv("data/01_77142-vcf_wide_mutations_red.csv", index_col=0)
df.shape
sizes = df.groupby(["region", "covv_clade"]).size().unstack()
sizes.assign(sum=sizes.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
plt.xlabel("Patient Count")
plt.ylabel("Region")
plt.title("Clade by region")
plt.legend(title="Clade")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_clade-by-region.png")
plt.show()
df["gender"].value_counts(normalize=True)
df["gender"].value_counts()
df["mutation_count"].quantile(.99)
df.groupby("is_red")["covv_clade"].value_counts(normalize=True)
sizes = df[
    (df.covv_gender != "Kvinna")
    & (df.covv_gender != "Unknown")
    ].groupby([
    "is_red",
    "covv_gender"
]).size().unstack()
sizes.index = ["Green", "Red"]
sizes.assign(sum=sizes.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
plt.xlabel("Patient Count")
plt.ylabel("Status")
plt.title("Gender by green/red status")
plt.legend(title="Gender")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_gender-by-red.png")
plt.show()

sizes = df.groupby(["is_red", "covv_clade"]).size().unstack()
sizes.index = ["Green", "Red"]
sizes.assign(sum=sizes.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
plt.xlabel("Patient Count")
plt.ylabel("Status")
plt.title("Clade by green/red status")
plt.legend(title="Clade", loc=4)
plt.xlim((0, 3200))
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_clade-by-red.png")
plt.show()
reg_list = list(region_key)
regions = [r[1] for r in reg_list]
df["is_red"].value_counts()
sizes = df.groupby("is_red")["cat_region"].value_counts(normalize=True).unstack()
sizes.index = ["Green", "Red"]
sizes.columns = regions
sizes
sizes.assign(sum=sizes.sum(axis=1)).sort_values(by="sum").drop("sum", axis=1).plot.barh(stacked=True)
plt.xlabel("Proportion of Patients")
plt.ylabel("Status")
plt.title("Region by green/red status")
plt.legend(title="Region", loc=4)
plt.xlim((0, 1.49))
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_region-by-red.png")
plt.show()

# Plot mutation count by gender
for val, target in [
    (val, df[df['covv_gender'] == val])
    for val in df['covv_gender'].unique()
]:
    sns.distplot(target[['mutation_count']], hist=False, label=val)
plt.xlabel("Total Mutations")
plt.ylabel("Density")
plt.title("Total mutations by gender")
plt.xlim(0, 21)
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_mutation-kde-by-gender.png")
plt.show()

# Plot mutation count by region
for val, target in [
    (val, df[df['region'] == val])
    for val in df['region'].unique()
]:
    sns.distplot(target[['mutation_count']], hist=False, label=val)
plt.xlabel("Total Mutations")
plt.ylabel("Density")
plt.title("Total mutations by region")
plt.xlim(0, 21)
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_mutation-kde-by-region.png")
plt.show()

# Plot mutation count by clade
for val, target in [
    (val, df[df['covv_clade'] == val])
    for val in df['covv_clade'].unique()
]:
    sns.distplot(target[['mutation_count']], hist=False, label=val)
plt.xlabel("Total Mutations")
plt.ylabel("Density")
plt.title("Total mutations by clade")
plt.xlim(0, 21)
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_mutation-kde-by-clade.png")
plt.show()

# Plot mutation count by is_red
for val, target in [
    (val, df[df['is_red'] == val])
    for val in df['is_red'].unique()
]:
    sns.distplot(
        target[['mutation_count']],
        hist=False,
        label="red" if val else "green",
        color="red" if val else "green"
    )
plt.xlabel("Total Mutations")
plt.ylabel("Density")
plt.title("Total mutations by green/red classification")
plt.xlim(0, 21)
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_mutation-kde-by-red.png")
plt.show()

# Plot age by is_red
for val, target in [
    (val, df[df['is_red'] == val])
    for val in df['is_red'].unique()
]:
    sns.distplot(
        target[['covv_patient_age']],
        label="red" if val else "green",
        color="red" if val else "green",
        kde_kws={"lw": 4},
        # hist_kws={"histtype": "step", "linewidth": 3, "alpha": .5},
        kde=True
    )
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Age by green/red classification")
plt.xlim(-9, 109)
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_age-kde-by-red.png")
plt.show()

# Plot mutation count by is_red
for val, target in [
    (val, df[df['is_red'] == val])
    for val in df['is_red'].unique()
]:
    sns.distplot(
        target[['mutation_count']],
        label="red" if val else "green",
        color="red" if val else "green",
        kde_kws={"lw": 4},
        # hist_kws={"histtype": "step", "linewidth": 3, "alpha": .5},
        kde=True
    )
plt.xlabel("Mutation Count")
plt.ylabel("Density")
plt.title("Mutation count by green/red classification")
plt.xlim(0, 20)
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_mutation-kde-by-red.png")
plt.show()

# Plot mutation count by status
top_statuses = df["covv_patient_status"].value_counts().head(9)
top_statuses = top_statuses[top_statuses.index != "Unknown"]
for val, target in [
    (val, df[df['covv_patient_status'] == val])
    for val in top_statuses.index
]:
    sns.distplot(target[['mutation_count']], hist=False, label=val)
plt.xlabel("Total Mutations")
plt.ylabel("Density")
plt.title("Total mutations by status")
plt.xlim(0, 21)
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_mutation-kde-by-status.png")
plt.show()

red_status = df.reset_index().groupby(
    ["is_red", "covv_patient_status"]
)["pid"].count().reset_index().sort_values(
    ["is_red", "pid"], ascending=False
)
colors = ["red" if r else "green" for r in red_status["is_red"]]
sns.barplot(x="pid", y="covv_patient_status", data=red_status, palette=colors)
plt.xlabel("Patient Count")
plt.ylabel("Status")
plt.title("Patient count by status")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_patient-bar-by-red.png")
plt.show()

df.reset_index().groupby("region")["pid"].count().sort_values().tail(20).plot.barh(legend=False)
plt.xlabel("Number of patients")
plt.ylabel("Region")
plt.title("Region frequency")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_region-bar.png")
plt.show()

df.reset_index().groupby("covv_clade")["pid"].count().sort_values().tail(20).plot.barh(legend=False)
plt.xlabel("Number of patients")
plt.ylabel("Clade")
plt.title("Clade frequency")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_clade-bar.png")
plt.show()

df[(df.covv_gender != "Kvinna") & (df.covv_gender != "Unknown")].reset_index().groupby("covv_gender")[
    "pid"].count().sort_values().tail(20).plot.barh(legend=False)
plt.xlabel("Patient Count")
plt.ylabel("Gender")
plt.title("Gender frequency")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_gender-bar.png")
plt.show()

sns.distplot(
    df["covv_patient_age"], rug=True, axlabel="Patient Age",
    rug_kws={"color": "k", "height": .02},
    kde_kws={"color": "b", "lw": 2},
    hist=False
)
plt.title("Age distribution")
plt.ylabel("Density")
plt.xlim((-9, 109))
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_age-kde.png")
plt.show()


