"""
UC5 — Redundancy provisioning: availability vs. resource demand.

Model: metro/regional connection s->t with up to 3 link-disjoint candidate
paths (ring + chord topology). Configuration: number of provisioned disjoint
paths k in {1,2,3} x link protection class. Service is up if at least one
provisioned path is fully up (independent link failures).

Input provenance and boundary:
  - topology, link counts/lengths and link availability sweep .995/.999/.9999
    are scenario assumptions, not values supplied by the cited survivability
    or path-diversity papers;
  - protection halves each link's outage probability and carries a 1.3x
    cost markup: assumed sensitivity model, not a measured protection system;
  - weighted fiber-km is a cost/resource proxy. A protection markup does not
    represent extra physical fiber or a measured environmental footprint;
  - the 99.99 % availability floor is a scenario service requirement, not
    asserted to be a numerical requirement of a specific regulation;
  - component failures are independent and paths link-disjoint; common-mode
    failures and endpoints are outside this simplified reliability boundary.

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
FIG.mkdir(exist_ok=True)   # so each script also runs standalone

# three link-disjoint paths s->t: lengths of their links [km]
PATHS = {
    "P1 (short)": [120, 180, 150],          # 450 km
    "P2 (ring)":  [200, 220, 240, 160],     # 820 km
    "P3 (chord)": [300, 350],               # 650 km
}
# per-link protection class: (outage-probability factor, relative cost factor).
# A protected link halves the outage probability of the scenario's A_link at
# 1.3x the cost; the class does not carry an absolute availability of its own.
CLASSES = {"std": (1.0, 1.0), "prot": (0.5, 1.3)}
US_PER_KM = 5.0
AVAIL_FLOOR = 0.9999
MIN_PER_YEAR = 365.25 * 24 * 60


def evaluate(sel, cls, a_link):
    """sel: tuple of path names; cls: protection class name."""
    out_f, cost_f = CLASSES[cls]
    a = 1 - (1 - a_link) * out_f
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
            cls_ = pareto.classify_change(norm[base], norm[cand], front,
                                     destination_feasible=bool(feas[cand]))
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
        (base, "unprotected:\ninfeasible", (0.33, 0.97), {"color": pareto.C_BASE}),
        (cand, "cheapest\ncompliant", (0.62, 0.40), {"color": pareto.C_IMP}),
    ]
    _norm, _ = pareto.normalize(points, directions)
    _front = pareto.pareto_front(_norm[feas])
    _cls = pareto.classify_change(_norm[base], _norm[cand], _front,
                                  destination_feasible=bool(feas[cand]))
    pareto.plot_front_2d(points[:, [1, 0]],
                         ["fiber cost [km-eq]", "unavailability [min/y]"],
                         baseline_idx=base, highlight_idx=cand,
                         feasible=feas, annotate=ann,
                         front_mask=pareto.feasible_front_mask(
                             points, directions, feas),
                         highlight_label=pareto.highlight_tag(_cls),
                         fname=FIG / "uc5_front_cost_avail.pdf")


if __name__ == "__main__":
    main()
