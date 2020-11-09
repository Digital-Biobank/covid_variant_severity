# %% Imports
import pandas as pd

pd.concat([
    pd.read_parquet(f"data/2020-10-21_vcf-wide1.parquet"),
    pd.read_parquet(f"data/2020-10-21_vcf-wide2.parquet")
]).to_parquet(f"data/2020-10-21_vcf-concat.parquet")
