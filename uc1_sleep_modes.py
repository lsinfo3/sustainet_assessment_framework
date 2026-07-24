"""
UC1 — RAN energy saving via base-station sleep modes.

Model: one macro base station serving elastic flows as a single-server FCFS
queue (Poisson arrivals, exponential service). Sleep policy: after the system
has been idle for tau seconds, the BS enters a sleep state; the next arrival
triggers a wake-up that takes Ts seconds (setup), during which it consumes
setup power. Simulated with a Lindley-type workload recursion with setup.

Power model (EARTH macro BS, Auer et al. 2011): P0 = 130 W (idle / minimum
non-zero output), Pmax = 20 W, dp = 4.7, so P_active = P0 + dp*Pmax = 224 W
(transmitting) and Psleep = 75 W. The deeper sleep level (10 W) and the
per-mode wake-up (setup) times follow the advanced sleep-mode model of
Debaillie et al. 2015 (light/micro-DTX: fast wake-up 0.1 s; deep: slow
wake-up 2.0 s):
  P_active = 224 W, P_idle = P0 = 130 W, P_setup = 224 W (during wake-up),
  sleep modes: light (75 W, Ts = 0.1 s), deep (10 W, Ts = 2.0 s).

Taxonomy metrics (use case U = "low-load macro cell, elastic traffic"):
  - Network quality / time-based:            mean flow response time [ms]
  - Network quality / service continuity:    p95 response time [ms]
  - Resource demand / energy:                average power [W]
  Governance constraint (QoS obligation):    mean response time <= 200 ms

Configuration space X: {always-on (no sleep)} U {light, deep} x idle timer tau.
Baseline x0: always-on with no sleep, idling at the EARTH P0 = 130 W. Having
the lowest delay (no wake-up penalty), it lies on the Pareto front; enabling
sleep trades power for delay (a burden shift along the front), while overly
aggressive deep sleep violates the delay obligation and raises average power.
"""
import numpy as np
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"

# ------------------------------------------------------------- parameters
P_ACTIVE, P_IDLE, P_SETUP = 224.0, 130.0, 224.0     # W (EARTH macro)
SLEEP_MODES = {"light": (75.0, 0.1), "deep": (10.0, 2.0)}  # (P_sleep W, Ts s)
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

    # baseline x0: always-on, no sleep, idling at the EARTH macro P0 = 130 W
    # (Auer et al. 2011). No idle-state assumption beyond the EARTH model.
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
    cls = pareto.classify_change(norm[base], norm[cand], front_norm)
    d0 = pareto.distance_to_front(norm[base], front_norm)
    d1 = pareto.distance_to_front(norm[cand], front_norm)
    hv = pareto.hypervolume(norm[feasible])

    # burden-shift / infeasibility example: deep sleep, tau = 0
    ex = configs.index({"policy": "deep", "tau": 0.0})

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
        (ex, "deep sleep\n(burden shift /\nSLA violation)", (0.46, 0.74)),
        (base, "always-on\n(no sleep, baseline)", (0.46, 0.26),
         {"color": pareto.C_BASE}),
        (cand, "light sleep\n(target trade-off)", (0.03, 0.40),
         {"color": pareto.C_IMP}),
    ]

    disp = points[:, [2, 0]]
    fig, ax = pareto.plot_front_2d(disp,
                         ["avg. power [W]", "mean delay [ms]"],
                         baseline_idx=base, highlight_idx=cand,
                         feasible=feasible, annotate=ann,
                         legend_loc="upper left")
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
