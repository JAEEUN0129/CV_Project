"""Compatibility entry point for the current AI Hair Studio UI."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("vace_app.py")), run_name="__main__")
