# %% Imports
import pathlib
import pandas as pd

proj_dir = pathlib.Path.home() / "covid" / "vcf"

# %% Read in cleaned data
df = pd.read_parquet(proj_dir / "data/01_77142-vcf_wide_join.parquet")

# %% Drop unneeded columns
df = df.drop([
    'covv_gender',
    'covv_passage',
    'covv_virus_name',
    'covv_specimen',
    'covv_location',
    # 'covv_patient_age',
    'covv_seq_technology',
    'covv_patient_status',
    'covv_lineage',
    'sequence_length',
    'covv_collection_date',
    'covv_accession_id',
    'covv_add_host_info',
    'covv_add_location',
    'sequence',
    'covv_clade',
    'covv_host',
    'covv_subm_date',
    'covv_assembly_method',
    'region',
    'is_dead',
    # 'is_symp',
    'is_sevr',
    # 'gender',
    'is_red',
    # 'cat_region',
    # 'clade'
], axis=1).dropna()

# %% Save dataset tailored to a particular outcome
df.to_parquet(proj_dir / "data/01_77142-vcf_wide_join_symp.parquet")

