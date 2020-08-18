from pathlib import Path
from runpy import run_path
from typing import Tuple, Dict


def run_script_if_needed(
    target: Path, script: Path, inputs: Tuple[Path, ...] = ()
) -> Dict[str, str]:
    if (
        not target.exists()
        or target.stat().st_mtime < script.stat().st_mtime
        or inputs
        and target.stat().st_mtime < max(i.stat().st_mtime for i in inputs)
    ):
        return run_path(str(script), run_name="__main__")


def run_scripts_if_needed(
    target: Path, scripts: Tuple[Path, ...], inputs: Tuple[Path, ...] = ()
) -> Tuple:
    if (
        not target.exists()
        or target.stat().st_mtime < max(s.stat().st_mtime for s in scripts)
        or inputs
        and target.stat().st_mtime < max(i.stat().st_mtime for i in inputs)
    ):
        return tuple(run_path(str(s), run_name="__main__") for s in scripts)


