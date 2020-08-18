from pathlib import Path

from make import run_script_if_needed

targets = (
    Path("00_77142-vcf_wide.parquet"),
    Path("01_77142-vcf_2-component-pca-transformed.parquet"),
    Path("02_77142-vcf_2-component-pca-transformed_"
         "mortality_3-cluster-kmeans.parquet"),
    Path("02_77142-vcf_2-component-pca-transformed_"
         "mortality_3-cluster-kmeans_"
         "logisitic-regression_roc-curve.png")
)
scripts = (
    Path("00_wide.py"),
    Path("01_pca.py"),
    Path("02_kmeans.py"),
    Path("03_logit.py"),
)

for t, s in zip(targets, scripts):
    print(f"Running {s} to create {t}")
    run_script_if_needed(t, s)