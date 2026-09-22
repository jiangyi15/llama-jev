"""Shared paths for the bench/ scripts.

Running a script directly (``python3 bench/foo.py``) puts ``bench/`` on
``sys.path``; this module also puts the project root there so ``import
jev_server`` works, and exposes the shared data locations.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
JE = os.path.join(DATA, "jevals-data")
SUITE = os.path.join(JE, "suites", "0.1.0")
