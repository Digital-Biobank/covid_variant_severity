# %% Imports
import pathlib
import pandas as pd

proj_dir = pathlib.Path.home() / "covid" / "vcf"

# %% Read in cleaned data
df = pd.read_csv(proj_dir / "data/2020-09-01all_cleaned_GISAID0901pull.csv", index_col=0)

# %% Create pid and binary variables
df = df.assign(
    pid=df["covv_accession_id"].str.extract(
        r"(\d+)",
        expand=False
    ).astype(int),
    # Strip whitespace from status
    covv_patient_status=df["covv_patient_status"].str.strip(),
    # %% Binary encode mortality data and gender
    is_dead=df["covv_patient_status"].map({"Live": 0, "Deceased": 1}),
    is_symp=df["covv_patient_status"].map({"Asymptomatic": 0, "Symptomatic": 1}),
    is_sevr=df["covv_patient_status"].map({"Mild": 0, "Severe": 1}),
    gender=df["covv_gender"].map({"Female": 0, "Male": 1, "Kvinna": 0}),
    covv_clade=df["covv_clade"].astype("str")
).set_index("pid")

# %% Create keys to encode region and clade
region_key = enumerate(sorted(df["region"].unique()))
clade_key = enumerate(sorted(df["covv_clade"].unique()))

# %% Create is_red status variable and encode region and clade
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
    # %% Encode region and clade
    cat_region=df["region"].map({v: k for k, v in region_key}),
    clade=df["covv_clade"].map({v: k for k, v in clade_key}),
)
df.columns
# %% Read in VCF wide data
var_df = pd.read_parquet(proj_dir / "data/00_77142-vcf_wide.parquet").fillna(0)

# %% Combined cleaned and variant data
df = df.join(var_df)

# %% Save cleaned and variant data
df.to_parquet(proj_dir / "data/01_77142-vcf_wide_join.parquet")
