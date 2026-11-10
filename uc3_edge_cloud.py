"""
UC3 — Edge vs. cloud placement of a latency-sensitive service.

Model: 12 access sites in a 400x400 km region. The scenario defines each
site-to-cloud fiber route as 500 km plus the site's Euclidean distance to
the region center (200, 200); 500 km is a route offset, not a common path
length. Their demand-weighted mean is 632.497 km. Configuration: k edge
sites selected greedily by demand, serving their nearest sites within
150 km; remaining demand goes to the cloud.

Input provenance:
  - propagation: 5 us/km in fiber, a physical approximation;
  - idle/full server power: 100/200 W, scenario assumptions. The half-peak
    idle ratio is motivated by Barroso/Hoelzle, not these absolute values;
  - PUE 1.2 cloud / 1.6 edge: assumptions. ISO/IEC 30134-2 defines PUE,
    not these particular values;
  - 1300 kgCO2e/server amortized over 5 y: an assumed embodied inventory,
    without a specific vendor report; the LCA citation supplies methodology;
  - availability .999 edge / .9999 cloud / .9995 per 100 km segment:
    assumed independent component probabilities, not measured deployment data;
  - the geometry, 2 ms processing time, load/server and localization thresholds
    are scenario assumptions. No financial cost objective is modeled.

Metrics (use case U = "regional latency-sensitive service, e.g., CPS control"):
  - quality/time-based:    demand-weighted mean RTT [ms]        [min]
  - resource/energy:       total power incl. PUE [W]            [min]
  - resource/embodied:     amortized embodied carbon [kgCO2e/y] [min]
  - resilience/availability: demand-weighted service availability [max]
  Governance constraint (data localization): >= 60 % of demand served
  in-region (edge) -- sovereignty scenario; parameter study relaxes this.

Config space X: k in {0..12} edge sites. Baseline x0: k = 0 (cloud-only).
Parameter study: localization requirement in {0, 60, 100 %} and load level.
"""
import numpy as np
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)   # so each script also runs standalone

RNG = np.random.default_rng(3)
N_SITES = 12
POS = RNG.uniform(0, 400, (N_SITES, 2))            # km
DEMAND = RNG.uniform(0.5, 2.0, N_SITES)            # normalized load units
CLOUD_DIST = 500.0 + np.linalg.norm(POS - 200.0, axis=1)  # km site->cloud

US_PER_KM = 5.0
P_IDLE, P_FULL = 100.0, 200.0                      # W per server
PUE_CLOUD, PUE_EDGE = 1.2, 1.6
EMBODIED_PER_SERVER = 1300.0 / 5.0                 # kgCO2e per year
A_EDGE, A_CLOUD = 0.999, 0.9999
A_SEG = 0.9995                                     # per 100 km fiber segment
PROC_MS = 2.0                                      # processing latency
LOAD_PER_SERVER = 4.0                              # demand units at full load


def path_availability(dist_km):
    segs = int(np.ceil(dist_km / 100.0))
    return A_SEG ** segs


def evaluate(k):
    """k edge sites at the k highest-demand locations."""
    edge_idx = np.argsort(-DEMAND)[:k]
    is_edge_site = np.zeros(N_SITES, bool)
    is_edge_site[edge_idx] = True
    rtt = np.empty(N_SITES)
    avail = np.empty(N_SITES)
    served_local = 0.0
    edge_load = np.zeros(N_SITES)
    for s in range(N_SITES):
        if k > 0:
            d2e = np.linalg.norm(POS[s] - POS[edge_idx], axis=1)
            j = int(np.argmin(d2e))
            dist_edge = d2e[j]
        if k > 0 and dist_edge <= 150.0:            # served at edge
            rtt[s] = 2 * dist_edge * US_PER_KM / 1e3 + PROC_MS
            avail[s] = path_availability(dist_edge) * A_EDGE
            served_local += DEMAND[s]
            edge_load[edge_idx[j]] += DEMAND[s]
        else:                                       # served in cloud
            rtt[s] = 2 * CLOUD_DIST[s] * US_PER_KM / 1e3 + PROC_MS
            avail[s] = path_availability(CLOUD_DIST[s]) * A_CLOUD
    w = DEMAND / DEMAND.sum()
    mean_rtt = float(np.sum(w * rtt))
    mean_avail = float(np.sum(w * avail))
    local_share = 100.0 * served_local / DEMAND.sum()

    # energy: edge servers (1 per active edge site) + cloud share
    power = 0.0
    n_servers = 0
    for e in np.where(edge_load > 0)[0]:
        # dimension edge sites to their load, with the same rule as the cloud
        srv = int(np.ceil(edge_load[e] / LOAD_PER_SERVER))
        util = min(1.0, edge_load[e] / (srv * LOAD_PER_SERVER))
        power += srv * (P_IDLE + (P_FULL - P_IDLE) * util) * PUE_EDGE
        n_servers += srv
    cloud_demand = DEMAND.sum() - served_local
    cloud_servers = int(np.ceil(cloud_demand / LOAD_PER_SERVER)) if cloud_demand > 0 else 0
    for _ in range(cloud_servers):
        util = min(1.0, cloud_demand / max(cloud_servers, 1) / LOAD_PER_SERVER)
        power += (P_IDLE + (P_FULL - P_IDLE) * util) * PUE_CLOUD
    n_servers += cloud_servers
    embodied = n_servers * EMBODIED_PER_SERVER
    return mean_rtt, power, embodied, mean_avail, local_share


def main():
    names = ["mean RTT [ms]", "power [W]", "embodied [kgCO2e/y]", "availability"]
    directions = ["min", "min", "min", "max"]
    configs, points, local = [], [], []
    for k in range(N_SITES + 1):
        r = evaluate(k)
        configs.append({"edge_sites": k})
        points.append(list(r[:4]))
        local.append(r[4])
    points = np.asarray(points)
    local = np.asarray(local)

    for req in (0.0, 60.0, 100.0):
        feas = local >= req
        if not feas.any():
            print(f"localization >= {req:.0f} %: no feasible config")
            continue
        norm, _ = pareto.normalize(points, directions)
        front = pareto.pareto_front(norm[feas])
        base = 0
        cand_pool = np.where(feas)[0]
        # Declared selection priority: minimum operational power among
        # compliant candidates (ties resolved by candidate order).
        cand = int(cand_pool[np.argmin(points[cand_pool, 1])])
        cls = pareto.classify_change(norm[base], norm[cand], front,
                                     destination_feasible=bool(feas[cand]))
        d0 = pareto.distance_to_front(norm[base], front)
        print(f"localization >= {req:3.0f} %: baseline feas={bool(feas[base])} "
              f"d={d0:.2f}; candidate k={configs[cand]['edge_sites']} "
              f"{np.round(points[cand], 2)} -> {cls}")

    feas = local >= 60.0
    pareto.save_points(OUT / "uc3_points.json", names, directions, points,
                       configs, extra={"local_share": local.tolist(),
                                       "localization_req": 60.0})
    # figure: RTT vs power. Highlight the SELECTED candidate at the 60 % requirement
    # using the same minimum-power rule as the table and sweep above.
    disp = points[:, [1, 0]]
    _norm, _ = pareto.normalize(points, directions)
    _front = pareto.pareto_front(_norm[feas])
    _pool = np.where(feas)[0]
    sel = int(_pool[np.argmin(points[_pool, 1])])
    ann = [
        (0, "cloud-only:\ninfeasible", (0.30, 0.90), {"color": pareto.C_BASE}),
        (sel, f"k={configs[sel]['edge_sites']}: lowest-power\ncompliant",
         (0.46, 0.44), {"color": pareto.C_IMP}),
    ]
    _cls = pareto.classify_change(_norm[0], _norm[sel], _front,
                                  destination_feasible=bool(feas[sel]))
    pareto.plot_front_2d(disp, ["power [W]", "mean RTT [ms]"],
                         baseline_idx=0, highlight_idx=sel,
                         feasible=feas, annotate=ann,
                         legend_loc="upper right",
                         front_mask=pareto.feasible_front_mask(
                             points, directions, feas),
                         highlight_label=pareto.highlight_tag(_cls),
                         fname=FIG / "uc3_front_power_rtt.pdf")
    pareto.plot_parallel(points, ["RTT", "power", "embodied", "avail."],
                         directions, baseline_idx=0, highlight_idx=sel,
                         fname=FIG / "uc3_parallel.pdf")


if __name__ == "__main__":
    main()
