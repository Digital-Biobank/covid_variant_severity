import pandas as pd

# %% Read in VCF wide data
df = pd.read_parquet("data/00_77142-vcf_wide.parquet")

# %% Calculate variant frequencies
variants = df.sum().rename("variant_count")
variants.to_csv("data/01_77142-vcf_wide_variants.csv")

# %% Calculate total mutations per patient
mutations = df.sum(axis=1).rename("mutation_count")
mutations.to_csv("data/01_77142-vcf_wide_mutations.csv")

# %% Read in outcomes data

df2 = pd.read_csv("data/2020-08-25_cleaned_GISAID.csv", index_col=0)

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
    cat_region=df2["region"].map(
        {v: k for k, v in enumerate(sorted(df2["region"].unique()))}
    ),
    clade=df2["covv_clade"].map(
        {v: k for k, v in enumerate(sorted(df2["covv_clade"].unique()))}
    ),
)
df = df.assign(
    region=df2.region
)


df.to_parquet("data/01_77142-vcf_wide_region.parquet")
variants_by_region = df.groupby("region").sum()
variants_by_region

df.to_csv("data/01_77142-vcf_wide_mutations_red.csv")
