"""
UC5 — Redundancy provisioning: availability vs. resource demand.

Model: metro/regional connection s->t with up to 3 link-disjoint candidate
paths (ring + chord topology). Configuration: number of provisioned disjoint
paths k in {1,2,3} x link protection class. Service is up if at least one
provisioned path is fully up (independent link failures).

Literature anchors (to cite):
  - link availability: standard long-haul fiber ~0.999 per 100 km-class link
    (availability modeling literature, e.g., Heegaard/Trivedi class);
    protected/premium links 0.9995 at ~1.3x cost
  - resource proxy: provisioned fiber-km (embodied + operational cost scale
    with lit fiber)
  - governance: critical-infrastructure availability floor 99.99 %
    (CER-directive-style obligation)

Metrics (use case U = "connectivity for critical regional service"):
  - resilience/availability:  service unavailability [min/year]   [min]
  - resource demand:          provisioned fiber cost [km-eq]      [min]
  - quality/time-based:       worst-case provisioned path RTT [ms][min]
  Constraint: availability >= 99.99 % (<= 52.6 min/year downtime)

Baseline x0: single unprotected shortest path (typical cost-minimal design).
Parameter study: per-link availability 0.995 .. 0.9999.
"""
import numpy as np
from itertools import product
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"

# three link-disjoint paths s->t: lengths of their links [km]
PATHS = {
    "P1 (short)": [120, 180, 150],          # 450 km
    "P2 (ring)":  [200, 220, 240, 160],     # 820 km
    "P3 (chord)": [300, 350],               # 650 km
}
CLASSES = {"std": (0.999, 1.0), "prot": (0.9995, 1.3)}  # (A_link, cost factor)
US_PER_KM = 5.0
AVAIL_FLOOR = 0.9999
MIN_PER_YEAR = 365.25 * 24 * 60


def evaluate(sel, cls, a_link):
    """sel: tuple of path names; cls: protection class name."""
    _, cost_f = CLASSES[cls]
    # protected links halve the outage probability at 1.3x cost
    a = a_link if cls == "std" else 1 - (1 - a_link) * 0.5
    p_up = []
    cost = 0.0
    rtt = 0.0
    for name in sel:
        links = PATHS[name]
        p_up.append(a ** len(links))
        cost += sum(links) * cost_f
        rtt = max(rtt, 2 * sum(links) * US_PER_KM / 1e3)
    unavail = np.prod([1 - p for p in p_up])
    return unavail * MIN_PER_YEAR, cost, rtt, 1 - unavail


def main():
    names = ["unavailability [min/y]", "fiber cost [km-eq]", "worst RTT [ms]"]
    directions = ["min", "min", "min"]
    path_sets = [("P1 (short)",), ("P1 (short)", "P2 (ring)"),
                 ("P1 (short)", "P3 (chord)"),
                 ("P1 (short)", "P2 (ring)", "P3 (chord)")]

    for a_link in (0.995, 0.999, 0.9999):
        configs, points = [], []
        for sel, cls in product(path_sets, CLASSES):
            u, c, r, av = evaluate(sel, cls, a_link)
            configs.append({"paths": len(sel), "class": cls,
                            "names": [s.split()[0] for s in sel]})
            points.append([u, c, r])
        points = np.asarray(points)
        avail = 1 - points[:, 0] / MIN_PER_YEAR
        feas = avail >= AVAIL_FLOOR
        norm, _ = pareto.normalize(points, directions)
        base = 0  # single std path
        if feas.any():
            front = pareto.pareto_front(norm[feas])
            pool = np.where(feas)[0]
            cand = int(pool[int(np.argmin(points[pool, 1]))])  # cheapest feasible
            cls_ = pareto.classify_change(norm[base], norm[cand], front)
            d0 = pareto.distance_to_front(norm[base], front)
            cc = configs[cand]
            print(f"A_link={a_link}: base 1xstd feas={bool(feas[base])} "
                  f"unavail={points[base,0]:.0f} min/y d={d0:.2f} | cheapest "
                  f"feasible: {cc['paths']} paths ({cc['class']}) "
                  f"{np.round(points[cand],1)} -> {cls_} | {feas.sum()}/{len(points)} ok")
        else:
            print(f"A_link={a_link}: NO feasible config (floor unreachable)")
        if a_link == 0.999:
            keep = (configs, points, feas, base, cand)

    configs, points, feas, base, cand = keep
    pareto.save_points(OUT / "uc5_points.json", names, directions, points,
                       configs, extra={"baseline": base, "improved": cand,
                                       "avail_floor": AVAIL_FLOOR,
                                       "a_link": 0.999})
    ann = [
        (base, "baseline\n(unprotected, infeasible)", (0.30, 0.92),
         {"color": pareto.C_BASE}),
        (cand, "target\n(cost-minimal compliant)", (0.55, 0.30),
         {"color": pareto.C_IMP}),
    ]
    pareto.plot_front_2d(points[:, [1, 0]],
                         ["fiber cost [km-eq]", "unavailability [min/y]"],
                         baseline_idx=base, highlight_idx=cand,
                         feasible=feas, annotate=ann,
                         legend_loc="center right", legend_bbox=(1.0, 0.60),
                         fname=FIG / "uc5_front_cost_avail.pdf")


if __name__ == "__main__":
    main()
