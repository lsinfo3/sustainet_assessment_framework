# SusTRA — evaluation code

Code that reproduces every number and figure in

> **SusTRA: A Trade-off Assessment Framework for Sustainable and Resilient
> Communication Networks.** F. Loh, O. Ayoub, C. Baur, F. Foschum, N. Hautz,
> H. Hegermann, D. Heye, D. Hock, T. Hoßfeld, F. Hussein, C. Natalino,
> T. Paulus, J. Pennekamp, M. Rüb, S. Schardt, C. van Sloun, J. Suomalainen,
> P. Swainson, C. Lipps. Developed in the Celtic-Next flagship project
> **SUSTAINET**, Common Task 3.

SusTRA treats network sustainability as a trade-off among four co-equal
assessment dimensions — network quality, network resilience, trust and
governance, and environmental and economic impact — and formulates their joint
assessment as a constrained multi-objective (Pareto) problem. Each of the six
use cases is a small analytical or simulation model. All share one
implementation of the formal assessment model: direction alignment, min–max
normalisation, Pareto dominance, empirical-front extraction, distance to the
front, and classification of a change as a strict improvement, a trade-off
improvement, burden shifting, or a degradation.

## Running it

```
pip install -r requirements.txt
python3 run_all.py
```

`run_all.py` runs the conceptual illustration and the six use cases in turn.
Each writes its figure as PDF and PNG into `figures/`, and each use case also
writes its full candidate set — every configuration's metric vector, the
baseline, the selected configuration, the outcome class and the normalisation
bounds — to `uc<n>_points.json`. Every script also runs standalone, for example
`python3 uc4_lpwan.py`.

`figures/` ends up with eleven plots. Seven of them are the ones the paper
includes, listed in the table below. The other four are diagnostics that are
produced but not printed: parallel-coordinate views of the full objective
vectors for Use Cases 1, 3 and 6 (`uc*_parallel.pdf`), which show the metrics
the two-axis figures project away, and `uc2_front_energy_mos.pdf`, an
energy-against-quality view of Use Case 2 in the shared front grammar.

Random seeds are fixed, so repeated runs are identical. The seven figures
included in the paper are byte-identical to the output of a clean run.

## Files

| File | Role | Figure |
|---|---|---|
| `pareto.py` | shared harness: direction alignment, min–max normalisation, dominance, empirical front, hypervolume, distance to front, change classification, and the common plotting style. Imported by every use-case script. | — |
| `plot_conceptual_goal.py` | conceptual illustration of the assessment goal (synthetic data) | Fig. 2 |
| `uc1_sleep_modes.py` | base-station sleep modes: single-server queue with setup time, Debaillie et al. (2015) Table V power model, sleep modes SM1–SM4 | Fig. 4 |
| `uc2_streaming_abr.py` | adaptive video streaming: rate-based ABR over one hour of content on a physical download clock | Fig. 5 |
| `uc3_edge_cloud.py` | edge versus cloud placement: demand-weighted RTT, power including PUE, amortised embodied emissions, availability, data-localisation constraint | Fig. 6 |
| `uc4_lpwan.py` | LPWAN technology selection: LoRaWAN, NB-IoT and LTE-M; Age of Information against battery lifetime, with duty-cycle and coverage constraints | Fig. 7 |
| `uc5_redundancy.py` | path redundancy: combinatorial availability against provisioned fibre and worst-case RTT | Fig. 8 |
| `uc6_autoscaling.py` | elastic scaling: hourly Erlang-C over a diurnal load, with reconfiguration churn as a third objective | Fig. 9 |
| `run_all.py` | runs all of the above | — |

Requires numpy and matplotlib only.

## Parameters

**The figures illustrate the assessment methodology. They are not measurements
of a deployed network,** and the operating points are modelled, not observed.

Each script's module docstring records where its parameters come from, and
distinguishes three kinds:

- values taken from a published source (for example the Debaillie et al. (2015)
  macro base-station power and sleep-mode model, or the ETSI sub-band
  duty-cycle limits);
- literature-based order-of-magnitude values, where the published range is wide
  and a representative figure was chosen (for example transport-network energy
  intensity in kWh/GB);
- explicit scenario assumptions made for the use case (for example the 60 %
  data-localisation threshold in UC3, a scenario parameter rather than a
  specific regulatory requirement, swept at 0 %, 60 % and 100 %).

Read the docstring before reusing a number. Several are deliberately
order-of-magnitude and exist to exercise the assessment procedure, not to
predict a real deployment.

## Licence

MIT. See [LICENSE](LICENSE).
