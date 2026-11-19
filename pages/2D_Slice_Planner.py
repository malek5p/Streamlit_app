"""Runs the original 2D slice planner (ThermoSEEG_RC_UI.py) as a second page of the multipage app."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "ThermoSEEG_RC_UI.py"), run_name="__main__")