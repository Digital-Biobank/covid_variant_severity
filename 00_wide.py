# %% Imports
from pathlib import Path

import pandas as pd

# %% Read in all vcf files with patient IDs (pid)
df = pd.concat([
    pd.read_csv(
        f,
        sep="\t",
        skiprows=range(6)
    ).assign(
        pid=int(''.join(c for c in f.name if c.isdigit()))
    )
    for f in Path().glob("all_vcfs/*.vcf")
])

# %% Save all vcf files with patient ID (pid) column in long format
df.to_feather("data/00_77142-vcf_long.feather")
df.to_parquet("data/00_77142-vcf_long.parquet")
df.to_pickle("data/00_77142-vcf_long.pickle")

# %% Combine REF, POS, ALT columns into ref_pos_alt (variant name) column
df = df.assign(
    ref_pos_alt=df["REF"] + df["POS"].astype(str) + df["ALT"],
    # Create column of ones
    ones=1
)

# %% Pivot df from long to wide (variants go from rows to columns)
df_wide = df.pivot(
    index="pid",
    columns="ref_pos_alt",
    values="ones"
)

# %% Save all vcf files with patient ID (pid) column in wide format
df_wide.reset_index().to_feather("data/00_77142-vcf_wide.feather")
df_wide.to_parquet("data/00_77142-vcf_wide.parquet")
df_wide.to_pickle("data/00_77142-vcf_wide.pickle")
