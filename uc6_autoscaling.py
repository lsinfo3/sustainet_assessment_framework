"""
UC6 — Elastic scaling of a core-cloud service (M/M/c with autoscaling).

Model: 24 h diurnal load lambda(t) = lambda_peak * (0.55 + 0.45 sin(...)),
hourly decisions. For each hour the system is an M/M/c queue (Erlang-C).
Autoscaling policy: c(t) = clip(ceil(lambda(t) / mu / u_target), c_min, c_max).

Literature anchors:
  - server power: idle 100 W / full 200 W and linear utilization response
    are assumptions. Barroso/Hoelzle motivates the half-peak idle ratio,
    not these absolute values or a modern hardware-specific calibration.
  - Erlang-C waiting probability / mean wait (standard queueing)

Metrics (use case U = "stateless service, diurnal demand"):
  - quality/time-based:      day-mean E[wait] [ms]                [min]
  - resource/energy:         day-mean power [kW]                  [min]
  - resilience/stability:    scaling events per day (churn)       [min]
  Constraint (SLA): hourly P(wait) <= 0.2 at all hours.

Config space X: static c in {44..60} U autoscale u_target in {0.50..0.95}.
Baseline x0: static c = 56 (peak + safety margin, legacy overprovisioning).
Parameter study: peak load lambda_peak in {30, 40, 50}.

Unstable candidates (lambda >= c*mu) have no finite steady-state M/M/c
mean wait. They are infeasible. For finite normalization across all candidates
only, their affected hourly wait receives OVERLOAD_WAIT_PENALTY_S = 10 s.
This is a synthetic penalty, not a queueing prediction. It affects the peak-50
normalization bounds; the primary peak-40 scenario has no unstable candidate.
Mean waiting time is an unweighted mean of the 24 hourly means, not an
arrival-weighted customer mean. Churn counts the repeated day including midnight.
"""
import math
import numpy as np
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)   # so each script also runs standalone

MU = 1.0                       # service rate per server [jobs/s]
P_IDLE, P_FULL = 100.0, 200.0  # W per active server
C_MIN, C_MAX = 4, 64
OVERLOAD_WAIT_PENALTY_S = 10.0   # synthetic finite score for infeasible overload
PW_SLA = 0.2                   # max hourly Erlang-C waiting probability
HOURS = np.arange(24)


def erlang_c(c, a):
    """P(wait) for M/M/c with offered load a = lambda/mu (a < c)."""
    if a >= c:
        return 1.0
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) * c / (c - a)
    return top / (s + top)


def day_metrics(lams, cs):
    waits, powers, viol = [], [], False
    for lam, c in zip(lams, cs):
        a = lam / MU
        pw = erlang_c(c, a)
        if pw > PW_SLA:
            viol = True
        ew = pw / (c * MU - lam) if lam < c * MU else OVERLOAD_WAIT_PENALTY_S  # s
        util = min(1.0, a / c)
        waits.append(ew * 1e3)
        powers.append(c * (P_IDLE + (P_FULL - P_IDLE) * util) / 1e3)
    # The diurnal load repeats, so the 23:00 -> 00:00 transition is a real
    # reconfiguration: diff over the 24 hourly decisions must wrap around.
    cs_arr = np.asarray(cs)
    churn = int(np.sum(np.abs(np.diff(cs_arr, append=cs_arr[0])) > 0))
    return float(np.mean(waits)), float(np.mean(powers)), churn, not viol


def scenario(lam_peak):
    lams = lam_peak * (0.55 + 0.45 * np.sin((HOURS - 5) / 24 * 2 * np.pi))
    names = ["mean wait [ms]", "mean power [kW]", "scale events/day"]
    directions = ["min", "min", "min"]
    configs, points, feas = [], [], []
    for c in range(44, 61, 4):                     # static provisioning
        w, p, ch, ok = day_metrics(lams, [c] * 24)
        configs.append({"policy": "static", "c": c})
        points.append([w, p, ch]); feas.append(ok)
    for u in np.arange(0.50, 0.96, 0.05):          # autoscaling
        cs = [min(C_MAX, max(C_MIN, math.ceil(l / MU / u))) for l in lams]
        w, p, ch, ok = day_metrics(lams, cs)
        configs.append({"policy": "auto", "u": round(float(u), 2)})
        points.append([w, p, ch]); feas.append(ok)
    return names, directions, configs, np.asarray(points), np.asarray(feas)


def main():
    for lam_peak in (30.0, 40.0, 50.0):
        names, dirs, configs, pts, feas = scenario(lam_peak)
        norm, _ = pareto.normalize(pts, dirs)
        front = pareto.pareto_front(norm[feas])
        base = next(i for i, c in enumerate(configs)
                    if c.get("policy") == "static" and c.get("c") == 56)
        pool = np.where(feas)[0]
        cand = int(pool[int(np.argmin(pts[pool, 1]))])   # min energy feasible
        cls = pareto.classify_change(norm[base], norm[cand], front,
                                     destination_feasible=bool(feas[cand]))
        d0 = pareto.distance_to_front(norm[base], front)
        print(f"peak={lam_peak:.0f}: base static-56 {np.round(pts[base],2)} "
              f"d={d0:.2f} | best {configs[cand]} {np.round(pts[cand],2)} "
              f"-> {cls} | {feas.sum()}/{len(pts)} feasible")
        if lam_peak == 40.0:
            keep = (configs, pts, feas, base, cand)

    configs, pts, feas, base, cand = keep
    pareto.save_points(OUT / "uc6_points.json",
                       ["mean wait [ms]", "mean power [kW]", "scale events/day"],
                       ["min", "min", "min"], pts, configs,
                       extra={"baseline": base, "improved": cand,
                              "overload_wait_penalty_s": OVERLOAD_WAIT_PENALTY_S,
                              "waiting_time_aggregation": "unweighted hourly mean",
                              "churn_periodic_midnight": True})
    ann = [
        (cand, f"autoscaling\nu={configs[cand]['u']}", (0.42, 0.90),
         {"color": pareto.C_IMP}),
    ]
    _norm, _ = pareto.normalize(pts, ["min", "min", "min"])
    _front = pareto.pareto_front(_norm[feas])
    _cls = pareto.classify_change(_norm[base], _norm[cand], _front,
                                  destination_feasible=bool(feas[cand]))
    fig6, ax6 = pareto.plot_front_2d(pts[:, [1, 0]], ["mean power [kW]", "mean wait [ms]"],
                         baseline_idx=base, highlight_idx=cand,
                         feasible=feas, annotate=ann, legend_loc="lower left",
                         front_mask=pareto.feasible_front_mask(
                             pts, ["min", "min", "min"], feas),
                         highlight_label=pareto.highlight_tag(_cls),
                         fname=None)
    # Mean waiting time spans four and a half decades here, so a linear axis
    # collapses every feasible static design onto the x-axis. A log axis keeps
    # them apart and lets each be named.
    ax6.set_yscale("log")
    for i, c in enumerate(configs):
        if c.get("policy") == "static" and feas[i]:
            ax6.annotate("$c{=}" + str(c["c"]) + "$", (pts[i, 1], pts[i, 0]),
                         textcoords="offset points", xytext=(3.5, -1.5),
                         fontsize=5.5, va="center", ha="left",
                         color=pareto.C_BASE if i == base else "#5f6368")
    fig6.savefig(FIG / "uc6_front_power_wait.pdf")
    fig6.savefig(FIG / "uc6_front_power_wait.png")
    pareto.plot_parallel(pts, ["wait", "power", "churn"],
                         ["min", "min", "min"], baseline_idx=base,
                         highlight_idx=cand, fname=FIG / "uc6_parallel.pdf")


if __name__ == "__main__":
    main()
