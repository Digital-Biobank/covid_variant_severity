import pandas as pd

var = pd.read_parquet("data/2020-10-21_vcf-wide.parquet")
var_freq = var.sum() / len(var)

var_freq.to_csv("data/2020-10-21_variant-frequency.csv")