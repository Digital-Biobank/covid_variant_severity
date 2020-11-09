# %% Imports
import numpy as np
import pandas as pd

# %% Split and save data
for i, df in enumerate(
    np.array_split(pd.read_parquet("data/2020-10-21_vcf-long.parquet"), 2)
    ):
    # Combine REF, POS, ALT columns into ref_pos_alt (variant name) column
    df.assign(
        ref_pos_alt=df["REF"] + df["POS"].astype(str) + df["ALT"],
        # Create column of ones
        ones=1
    # Pivot df from long to wide (variants go from rows to columns)
    ).pivot(
        index="pid",
        columns="ref_pos_alt",
        values="ones"
    # Save all vcf files with patient ID (pid) column in wide format
    ).to_parquet(f"data/2020-10-21_vcf-wide{i+1}.parquet")