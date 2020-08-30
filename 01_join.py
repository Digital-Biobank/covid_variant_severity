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
df = pd.read_csv("data/2020-08-25_cleaned_GISAID.csv", index_col=0)

df = df.assign(
    pid=df["covv_accession_id"].str.extract(
        r"(\d+)",
        expand=False
    ).astype(int),
    covv_patient_status=df["covv_patient_status"].str.strip(),
    # %% Binary encode mortality data and gender
    is_dead=df["covv_patient_status"].map({"Live": 0, "Deceased": 1}),
    is_symp=df["covv_patient_status"].map({"Asymptomatic": 0, "Symptomatic": 1}),
    is_sevr=df["covv_patient_status"].map({"Mild": 0, "Severe": 1}),
    gender=df["covv_gender"].map({"Female": 0, "Male": 1, "Kvinna": 0}),
    covv_clade=df["covv_clade"].astype("str")
).set_index("pid")

df = df.assign(
    is_red=df["covv_patient_status"].map(
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
    cat_region=df["region"].map(
        {v: k for k, v in enumerate(sorted(df["region"].unique()))}
    ),
    clade=df["covv_clade"].map(
        {v: k for k, v in enumerate(sorted(df["covv_clade"].unique()))}
    ),
)
df = df.join(mutations)
df.to_csv("data/01_77142-vcf_wide_mutations_red.csv")
