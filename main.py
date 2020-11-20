from time import time
from runpy import run_path

files = (
    "00_long.py",
    "00_red.py",
    "01_wide.py",
    "02_var-freq.py",
    "02_join.py",
    "03_clean.py",
    "04_logit.py",
    "05_plot-variants.py",
    "05_fig-s1.py",
    "05_fig-s2.py",
)

for f in files:
    print(f"Running file: {f}")
    start = time()
    run_path(f)
    end = time()
    print(f"Run completed after {end - start} seconds.")