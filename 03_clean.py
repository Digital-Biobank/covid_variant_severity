import numpy as np
import pandas as pd

df = pd.read_parquet("data/2020-10-21_vcf-join.parquet")

df = df.assign(
    continent=df["continent"].str.strip(),
    gender=df["covv_gender"].str.strip().str.lower().replace({
        "woman": "female",
        "unknown": np.nan
        }),
    age=pd.to_numeric(
        df["covv_patient_age"].replace({
            '30s': 35,
            '>60': 73,
            '44-year old': 44,
            '31 years 6 months': 31.5,
            '59 years 1 months': 59.083,
            '33 years 5 months': 33.5,
            '51 years 3 months': 51.25,
            '4 months': 0.33,
            '75, 4 months': 75.33,
            '18-49': 33.5,
            '50-64': 57,
            '20-30': 25,
            '20 - 30': 25,
            '10-20': 15,
            '2 months': .166,
            '20s': 25,
            '53 years': 53,
            '36, 11 months': 36.916,
            '50-54': 52,
            '55-59': 57,
            '65-69': 67,
            '25-29': 27,
            '30-34': 32,
            '45-49': 47,
            '75-79': 77,
            '1 month': .083,
            '70 ': 70,
            '07': 7,
            '3 months': 0.25,
            '6 months': 0.5,
            '5 months': 0.416
        }),
        errors="coerce"
    )
)

# %% Drop columns not needed for modeling
df = df.drop([
    'covv_gender',
    'covv_patient_age',
    'covv_passage',
    'covv_virus_name',
    'covv_specimen',
    'covv_location',
    'covv_seq_technology',
    'covv_patient_status',
    'covv_lineage',
    'sequence_length',
    'covv_collection_date',
    'covv_accession_id',
    'covv_add_host_info',
    'covv_add_location',
    'sequence',
    'covv_host',
    'covv_subm_date',
    'covv_assembly_method',
    'country',
    'region',
    'city',
    ],
    axis=1
    )

# %% Convert categorical data into indicator (dummy) variables
pd.get_dummies(
    df,
    columns=["continent", "gender", "covv_clade"],
    drop_first=True,
    prefix="",
    prefix_sep="",
    ).to_parquet("data/2020-10-21_vcf-clean.parquet")