"""
UC1 — RAN energy saving via base-station sleep modes.

Model: one macro base station serving elastic flows as a single-server FCFS
queue (Poisson arrivals, exponential service). Sleep policy: after the system
has been idle for tau seconds, the BS enters a sleep state; the next arrival
triggers a wake-up that takes Ts seconds (setup), during which it consumes
setup power. Simulated with a Lindley-type workload recursion with setup.

Power model (Debaillie et al. 2015, Table V; 2x2 macro, 2020 technology,
whole base station). The active, idle and sleep powers and total transition
durations below are read from that table. Setup power is assumed equal to
full-load power; it is not a separate measured transition-power value:
  P_active = 702.6 W (full load), P_idle = 114.5 W (no load),
  P_setup  = 702.6 W (charged at active power during wake-up),
  SM1 (OFDM symbol)  76.5 W, Ts = 71.4 us
  SM2 (sub-frame)     8.6 W, Ts = 1 ms
  SM3 (radio frame)   6.0 W, Ts = 10 ms
  SM4 (standby)       5.3 W, Ts = 1 s

Assumption on Ts: Debaillie reports each duration as a *total* transition,
deactivation plus reactivation. The model charges that whole duration as an
arrival-triggered wake-up at active power and no cost on entry, which is the
declared delay-model assumption, not a measured split between phases. In
zero-timer sensitivity runs at Ts/2 and Ts/4, the mode ordering, power
non-monotonicity and SM4 infeasibility remain. SM2 power changes from 149.7 W
to 148.6/148.0 W; the saving changes numerically but still rounds to 36 %.
This does not establish actual entry/exit costs or interrupted-entry behavior.

Taxonomy metrics (use case U = "low-load macro cell, elastic traffic"):
  - Network quality / time-based:            mean flow response time [ms]
  - Network quality / service continuity:    p95 response time [ms]
  - Resource demand / energy:                average power [W]
  Governance constraint (QoS obligation):    mean response time <= 200 ms

Configuration space X: {always-on (no sleep)} U {SM1..SM4} x idle timer tau,
i.e. 4 modes x 9 timers + 1 = 37 configurations. Baseline x0: always-on with
no sleep, idling at P_idle = 114.5 W. Having the lowest delay (no wake-up
penalty), it lies on the Pareto front; enabling sleep trades power for delay
(a burden shift along the front), while the deepest mode SM4 violates the
delay obligation and *raises* average power. It is outside formal outcome
classification because its destination is infeasible; raw objective dominance
can still be inspected. Power is not monotone in sleep
depth, because the 1 s wake-up is charged at 702.6 W.
"""
import numpy as np
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)   # so each script also runs standalone

# ------------------------------------------------------------- parameters
# Debaillie et al. 2015, Table V (2x2 macro, 2020 technology), whole base station:
# full load 702.6 W, no load 114.5 W; sleep modes SM1..SM4 at 76.5 / 8.6 / 6.0 / 5.3 W
# with TOTAL transition budgets 71.4 us / 1 ms / 10 ms / 1 s. "light" is SM1 (OFDM symbol),
# "deep" is SM4 (standby). Wake-ups are charged at active power.
P_ACTIVE, P_IDLE, P_SETUP = 702.6, 114.5, 702.6     # W (Debaillie 2x2 macro)
# All four sleep modes of the cited model (OFDM symbol / sub-frame / radio frame /
# standby), so the candidate space is not an arbitrary selection of two of them.
SLEEP_MODES = {"SM1": (76.5, 71.4e-6), "SM2": (8.6, 1e-3),
               "SM3": (6.0, 10e-3), "SM4": (5.3, 1.0)}  # (P_sleep W, Ts s)
LAM, MU = 4.0, 20.0          # arrivals/s, service rate/s  (rho = 0.2, night load)
SLA_MEAN_MS = 200.0          # governance: mean delay obligation
N_ARRIVALS = 100_000
TAUS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def simulate(tau, p_sleep, t_setup, seed=7):
    """Lindley workload recursion with idle-timer sleep + setup.

    Returns (mean_delay_s, p95_delay_s, avg_power_W).
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / LAM, N_ARRIVALS)
    svc = rng.exponential(1.0 / MU, N_ARRIVALS)
    V = 0.0                       # workload (time to empty) after last arrival
    delays = np.empty(N_ARRIVALS)
    e_busy = e_setup = t_idle_on = t_sleep = 0.0
    for n in range(N_ARRIVALS):
        rem = V - gaps[n]
        if rem > 0:               # server still busy when arrival occurs
            wait = rem
        else:
            idle = -rem           # idle period length before this arrival
            if idle > tau:        # BS asleep -> wake-up penalty
                wait = t_setup
                e_setup += t_setup
                t_idle_on += tau
                t_sleep += idle - tau
            else:
                wait = 0.0
                t_idle_on += idle
        V = wait + svc[n]
        delays[n] = V
        e_busy += svc[n]
    busy_t = e_busy
    setup_t = e_setup
    total_t = busy_t + setup_t + t_idle_on + t_sleep
    energy = (busy_t * P_ACTIVE + setup_t * P_SETUP
              + t_idle_on * P_IDLE + t_sleep * p_sleep)
    return delays.mean(), np.percentile(delays, 95), energy / total_t


def main():
    names = ["mean delay [ms]", "p95 delay [ms]", "avg. power [W]"]
    directions = ["min", "min", "min"]
    configs, points = [], []

    # baseline x0: always-on, no sleep, idling at the Debaillie no-load power
    # P_idle = 114.5 W. No idle-state assumption beyond that table.
    md, p95, pw = simulate(tau=np.inf, p_sleep=P_IDLE, t_setup=0.0)
    configs.append({"policy": "always-on (no sleep)", "tau": None})
    points.append([md * 1e3, p95 * 1e3, pw])

    for mode, (p_sleep, t_setup) in SLEEP_MODES.items():
        for tau in TAUS:
            md, p95, pw = simulate(tau, p_sleep, t_setup)
            configs.append({"policy": mode, "tau": tau})
            points.append([md * 1e3, p95 * 1e3, pw])

    points = np.asarray(points)
    feasible = points[:, 0] <= SLA_MEAN_MS

    # --- classification per Section IV-A (normalized space, feasible front)
    norm, bounds = pareto.normalize(points, directions)
    front_norm = pareto.pareto_front(norm[feasible])
    base = 0
    # candidate: feasible config with lowest power (resource priority s.t. SLA)
    cand = int(np.where(feasible, points[:, 2], np.inf).argmin())
    cls = pareto.classify_change(norm[base], norm[cand], front_norm,
                                 destination_feasible=bool(feasible[cand]))
    d0 = pareto.distance_to_front(norm[base], front_norm)
    d1 = pareto.distance_to_front(norm[cand], front_norm)
    hv = pareto.hypervolume(norm[feasible])

    # infeasibility example: the deepest sleep mode (SM4, standby), tau = 0
    ex = configs.index({"policy": "SM4", "tau": 0.0})

    # --- outputs
    pareto.save_points(OUT / "uc1_points.json", names, directions, points,
                       configs, extra={
                           "baseline": base, "improved": cand,
                           "classification": cls,
                           "distance_baseline": d0, "distance_improved": d1,
                           "hypervolume_feasible": hv,
                           "sla_mean_ms": SLA_MEAN_MS})

    # Annotate key configurations to strengthen the visual contribution.
    # Each callout is anchored in an empty region (axes-fraction coords) and
    # linked to its point by a thin leader line, so labels never overflow the
    # axes or overlap each other (cf. pareto.anchor_label).
    ann = [
        (ex, "SM4 (standby):\ninfeasible", (0.98, 0.55)),
        (cand, f"{configs[cand]['policy']}: lowest\nfeasible power", (0.42, 0.26),
         {"color": pareto.C_IMP}),
    ]

    disp = points[:, [2, 0]]
    # The front is the feasible non-dominated set over ALL THREE objectives;
    # the figure shows two of them, so the mask must not be recomputed from
    # the projection (see pareto.feasible_front_mask).
    fmask = pareto.feasible_front_mask(points, directions, feasible)
    fig, ax = pareto.plot_front_2d(disp,
                         ["avg. power [W]", "mean delay [ms]"],
                         baseline_idx=base, highlight_idx=cand,
                         feasible=feasible, annotate=ann,
                         front_mask=fmask,
                         highlight_label=pareto.highlight_tag(cls))
    fig.savefig(FIG / "uc1_front_power_delay.pdf")
    fig.savefig(FIG / "uc1_front_power_delay.png")
    pareto.plot_parallel(points, ["mean delay", "p95 delay", "power"],
                         directions, baseline_idx=base, highlight_idx=cand,
                         fname=FIG / "uc1_parallel.pdf")

    print(f"UC1: {len(points)} configs, {feasible.sum()} feasible "
          f"(SLA {SLA_MEAN_MS:.0f} ms)")
    print(f"baseline  {configs[base]}: {np.round(points[base], 1)}")
    print(f"improved  {configs[cand]}: {np.round(points[cand], 1)}  -> {cls}")
    print(f"distance to front: {d0:.3f} -> {d1:.3f}; HV(feasible) = {hv:.3f}")
    print(f"extreme   {configs[ex]}: {np.round(points[ex], 1)} "
          f"feasible={bool(feasible[ex])}")


if __name__ == "__main__":
    main()
