# %% Imports
import numpy as np
import pandas as pd

# %% Combine and save outcome and variant data
df = pd.read_parquet("data/2020-10-21_green-red.parquet")

df.join(
    pd.read_parquet("data/2020-10-21_vcf-wide.parquet")
).to_parquet("data/2020-10-21_vcf-join.parquet")
