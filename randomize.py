import pandas as pd

# %% Read in data random rows
df = pd.read_parquet("data/00_77142-vcf_wide.parquet")

# %% Randomize rows
random_rows = df.sample(
    frac=1,
    random_state=42,
    axis="index"
).set_axis(df.index, axis="index")

# %% Randomize columns
random_cols = random_rows.sample(
    frac=1,
    random_state=42,
    axis="columns"
).set_axis(df.columns, axis="columns")

# %% Save randomized data
random_cols.to_parquet("00_77142-vcf_wide_random.parquet")
