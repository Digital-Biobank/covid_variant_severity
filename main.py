from time import time
from runpy import run_path

files = "00_long.py", "01_wide.py", "02_append.py"

for f in files:
    print(f"Running file: {f}")
    start = time()
    run_path(f)
    end = time()
    print(f"Run completed after {end - start} seconds.")