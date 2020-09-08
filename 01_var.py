import pandas as pd

df = pd.read_parquet("data/00_77142-vcf_wide.parquet")

# %% Calculate variant frequencies
variants = df.sum().rename("variant_count")
variants.to_csv("data/01_77142-vcf_wide_variants.csv")
