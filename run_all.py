#!/usr/bin/env python3
"""
Regenerate every figure used in the paper.

Runs the conceptual-goal script and the six use-case scripts in turn; each one
writes its PDF (and a PNG preview) into ./figures/. Run from this directory:

    python3 run_all.py
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))  # so the scripts can "import pareto"

SCRIPTS = [
    "plot_conceptual_goal",  # Fig. 2  -> figures/conceptual_goal.pdf
    "uc1_sleep_modes",       # Fig. 4  -> figures/uc1_front_power_delay.pdf
    "uc2_streaming_abr",     # Fig. 5  -> figures/uc2_param_study.pdf
    "uc3_edge_cloud",        # Fig. 6  -> figures/uc3_front_power_rtt.pdf
    "uc4_lpwan",             # Fig. 7  -> figures/uc4_aoi_lifetime.pdf
    "uc5_redundancy",        # Fig. 8  -> figures/uc5_front_cost_avail.pdf
    "uc6_autoscaling",       # Fig. 9  -> figures/uc6_front_power_wait.pdf
]

for name in SCRIPTS:
    print(f"[run_all] {name} ...")
    runpy.run_module(name, run_name="__main__")

print(f"[run_all] done -- figures written to {HERE / 'figures'}")
