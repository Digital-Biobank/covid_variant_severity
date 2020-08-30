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
plt.savefig("plots/00_77142-vcf_mutation-kde.png")
plt.show()

# %% Read in outcomes data
df = pd.read_csv("data/01_77142-vcf_wide_mutations_red.csv", index_col=0)

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
plt.savefig("plots/00_77142-vcf_patient-bar-by-red.png")
plt.show()