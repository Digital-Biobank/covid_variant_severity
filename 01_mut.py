import pandas as pd

df = pd.read_parquet("data/00_77142-vcf_wide.parquet")

# %% Calculate total mutations per patient
mutations = df.sum(axis=1).rename("mutation_count")
mutations.to_csv("data/01_77142-vcf_wide_mutations.csv")
