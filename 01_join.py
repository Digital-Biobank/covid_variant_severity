import pandas as pd
import matplotlib.pyplot as plt

# %% Read in VCF wide data
df = pd.read_parquet("data/00_77142-vcf_wide.parquet")

# %% Read in outcomes data
df2 = pd.read_csv("data/2020-09-01all_cleaned_GISAID0901pull.csv", index_col=0)
df2 = df2.assign(
    pid=df2["covv_accession_id"].str.extract(
        r"(\d+)",
        expand=False
    ).astype(int),
    covv_patient_status=df2["covv_patient_status"].str.strip(),
    # %% Binary encode mortality data and gender
    is_dead=df2["covv_patient_status"].map({"Live": 0, "Deceased": 1}),
    is_symp=df2["covv_patient_status"].map({"Asymptomatic": 0, "Symptomatic": 1}),
    is_sevr=df2["covv_patient_status"].map({"Mild": 0, "Severe": 1}),
    gender=df2["covv_gender"].map({"Female": 0, "Male": 1, "Kvinna": 0}),
    covv_clade=df2["covv_clade"].astype("str")
).set_index("pid")
region_key = enumerate(sorted(df2["region"].unique()))
clade_key = enumerate(sorted(df2["covv_clade"].unique()))
df2 = df2.assign(
    is_red=df2["covv_patient_status"].map(
        {
            # Green
            "Asymptomatic": 0,
            "Released": 0,
            "Recovered": 0,
            "Live": 0,
            "Mild": 0,
            # Red
            "Deceased": 1,
            "Severe": 1,
            "Symptomatic": 1,
        }
    ),
    cat_region=df2["region"].map({v: k for k, v in region_key}),
    clade=df2["covv_clade"].map({v: k for k, v in clade_key}),
)

df2.to_parquet("data/01_77142-vcf_red.parquet")
df2.to_csv("data/01_77142-vcf_wide_mutations_red.csv")

df = df.assign(region=df2.region)


df.to_parquet("data/01_77142-vcf_wide_region.parquet")
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

# %% Merge variants with outcome data
df = df.assign(
    region=df2.region,
    is_red=df2.is_red,
    clade=df2.clade,
    covv_clade=df2.covv_clade,
)

# %% save variants by region, is_red, etc.
df.to_parquet("../../data/01_77142-vcf_wide_region-red-clade.parquet")

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
