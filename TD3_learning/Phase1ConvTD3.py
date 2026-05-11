from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from phase1_convergence_common import make_spec, run_phase1_convergence


if __name__ == "__main__":
    raise SystemExit(run_phase1_convergence(make_spec("td3")))
