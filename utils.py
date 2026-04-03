"""Shared project path helpers.

Author: Jai Sharma
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def data_path(*parts):
    """Build an absolute path under the repository's data directory.

    @param parts: Path components to append under ``data``.
    @return: ``Path`` pointing to the requested location.
    """
    return PROJECT_ROOT / "data" / Path(*parts)

def results_path(*parts):
    """Build an absolute path under the repository's results directory.

    @param parts: Path components to append under ``results``.
    @return: ``Path`` pointing to the requested location.
    """
    return PROJECT_ROOT / "results" / Path(*parts)
