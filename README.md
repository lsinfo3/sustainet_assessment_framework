# SUSTAINET assessment framework — figure-generation code

Self-contained code that reproduces the evaluation figures of the paper
*"Beyond Isolated Optimization: A Framework for Multi-Dimensional Assessment
of Sustainable and Resilient Communication Networks."* Each of the six use
cases is a small analytical/simulation model; all share one implementation of
the formal assessment model (normalization, Pareto dominance, empirical-front
extraction, hypervolume, distance-to-front, and change classification).

## Requirements

Python 3.9+ with:

```
pip install -r requirements.txt        # numpy, matplotlib
```

## Reproduce all figures

From this directory:

```
python3 run_all.py
```

This regenerates every figure into `figures/` (PDF used in the paper, plus a
`.png` preview). Individual figures can also be produced by running a single
script, e.g. `python3 uc1_sleep_modes.py`.

## Files

| File | Role | Paper figure |
|------|------|--------------|
| `pareto.py` | Shared harness: min–max normalization, dominance, empirical Pareto front, hypervolume, distance-to-front, change classification (strict improvement / trade-off improvement / burden shift), and the common 2-D plotting style. Imported by every use-case script. | — |
| `plot_conceptual_goal.py` | Conceptual illustration (synthetic data). | Fig. 2 |
| `uc1_sleep_modes.py` | UC1 — base-station sleep modes (M/M/1 with setup; EARTH power model, advanced sleep modes). | Fig. 4 |
| `uc2_streaming_abr.py` | UC2 — adaptive video streaming (ABR vs. QoE vs. energy; AR(1) capacity). | Fig. 5 |
| `uc3_edge_cloud.py` | UC3 — edge vs. cloud placement (RTT, power, embodied carbon, availability; localization constraint). | Fig. 6 |
| `uc4_lpwan.py` | UC4 — LPWAN technology selection (AoI vs. battery lifetime; duty-cycle and coverage constraints). | Fig. 7 |
| `uc5_redundancy.py` | UC5 — redundancy provisioning (unavailability vs. fiber cost; availability floor). | Fig. 8 |
| `uc6_autoscaling.py` | UC6 — elastic scaling (M/M/c Erlang-C; wait vs. power vs. scaling churn). | Fig. 9 |

Each use-case script also writes a `*_points.json` snapshot of its evaluated
operating points and the resulting classification, and prints a short summary
(baseline, highlighted candidate, distances) to the console.

## Notes

- Parameters are literature-based, order-of-magnitude values or explicit
  assumptions (see each script's module docstring for sources). The figures
  illustrate the methodology; they are not measurements of a deployed system.
- Random seeds are fixed inside the scripts, so results are reproducible.
