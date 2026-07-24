"""
UC3 — Edge vs. cloud placement of a latency-sensitive service.

Model: 12 access sites on a 400x400 km region, central cloud DC outside the
region (500 km away). Configuration: number k of edge sites (greedy by
demand) serving their nearest sites; remaining demand goes to the cloud.

Literature anchors (cite in paper):
  - propagation: 5 us/km in fiber (~1 ms per 200 km one-way)
  - server power: idle 100 W / full 200 W (SPECpower-class 2S server)
  - PUE: hyperscale cloud 1.2 vs. edge site 1.6 (ISO/IEC 30134-2 context)
  - embodied carbon: ~1300 kgCO2e/server amortized over 5 y (Dell LCA class)
  - element availability: edge site 0.999 (no redundancy), cloud 0.9999
    (redundant), per-100km fiber path segment 0.9995

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
        util = min(1.0, edge_load[e] / LOAD_PER_SERVER)
        power += (P_IDLE + (P_FULL - P_IDLE) * util) * PUE_EDGE
        n_servers += 1
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
        # improved: feasible config minimizing distance to front, prefer low k
        dists = [pareto.distance_to_front(norm[i], front) for i in cand_pool]
        cand = int(cand_pool[int(np.argmin(dists))])
        cls = (pareto.classify_change(norm[base], norm[cand], front)
               if cand != base else "-")
        d0 = pareto.distance_to_front(norm[base], front)
        print(f"localization >= {req:3.0f} %: baseline feas={bool(feas[base])} "
              f"d={d0:.2f}; candidate k={configs[cand]['edge_sites']} "
              f"{np.round(points[cand], 2)} -> {cls}")

    feas = local >= 60.0
    pareto.save_points(OUT / "uc3_points.json", names, directions, points,
                       configs, extra={"local_share": local.tolist(),
                                       "localization_req": 60.0})
    # figure: RTT vs power, annotate k
    disp = points[:, [1, 0]]
    first_feas = int(np.where(feas)[0][0])
    ann = [
        (0, "baseline\n(cloud-only, infeasible)", (0.24, 0.93),
         {"color": pareto.C_BASE}),
        (first_feas,
         f"target (minimal\ncompliant edge, k={configs[first_feas]['edge_sites']})",
         (0.40, 0.60)),
    ]
    if 12 != first_feas:
        ann.append((12, "full edge (k=12)", (0.80, 0.35)))
    pareto.plot_front_2d(disp, ["power [W]", "mean RTT [ms]"],
                         baseline_idx=0, highlight_idx=first_feas,
                         feasible=feas, annotate=ann,
                         legend_loc="upper right",
                         fname=FIG / "uc3_front_power_rtt.pdf")
    pareto.plot_parallel(points, ["RTT", "power", "embodied", "avail."],
                         directions, baseline_idx=0,
                         highlight_idx=int(np.where(feas)[0][0]),
                         fname=FIG / "uc3_parallel.pdf")


if __name__ == "__main__":
    main()
