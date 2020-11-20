# %% Imports
import numpy as np
import pandas as pd

# %% Split and save data
df = pd.read_parquet("data/2020-10-21_vcf-long.parquet")
gr = pd.read_parquet("data/2020-10-21_green-red.parquet")

df = df[df.pid.isin(gr.index)]
df["POS"] = df.POS.fillna(df["    POS"])

# Combine REF, POS, ALT columns into ref_pos_alt (variant name) column
# Save all vcf files with patient ID (pid) column in wide format
df = df.assign(
    ref_pos_alt=df["REF"] + df["POS"].astype(int).astype(str) + df["ALT"],
    # Create column of ones
    ones=1
).pivot(
    index="pid",
    columns="ref_pos_alt",
    values="ones"
).to_parquet(f"data/2020-10-21_vcf-wide.parquet")