# %% Imports
import pandas as pd

df1 = pd.read_parquet(f"data/2020-10-21_vcf-wide1.parquet")
df2 = pd.read_parquet(f"data/2020-10-21_vcf-wide2.parquet")
df1.append(df2).to_parquet(f"data/2020-10-21_vcf-append.parquet")