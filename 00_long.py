# %% Imports
from pathlib import Path

import pandas as pd

# %% Read in all vcf files with patient IDs (pid)
pd.concat([
    pd.read_csv(
        f,
        sep="\t",
        skiprows=range(6)
    ).assign(
        pid=int(''.join(c for c in f.name if c.isdigit()))
    )
    for f in Path().glob("2020-10-21_gisaid_vcfs/*.vcf")
    if f.stat().st_size > 300
# Save all vcf files with patient ID (pid) column in long format
]).to_parquet("data/2020-10-21_vcf-long.parquet")
