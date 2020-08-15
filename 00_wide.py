from pathlib import Path

import pandas as pd

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

df.to_csv("00_77142-vcf.csv")

df = df.assign(
    ref_pos_alt=df["REF"] + df["POS"].astype(str) + df["ALT"],
    ones=1
)

df_wide = df.pivot(
    index="pid",
    columns="ref_pos_alt",
    values="ones"
)

df_wide.to_csv("00_77142-vcf_wide.csv")
df_wide.to_parquet("00_77142-vcf_wide.parquet")
