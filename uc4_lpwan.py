"""
UC4 — LPWAN technology selection for a status-update service:
      LoRaWAN vs. NB-IoT vs. LTE-M.

Literature-based parameters (order-of-magnitude values):
  LoRaWAN (EU868, 14 dBm, 51 B payload, 125 kHz, CR4/5):
    airtime: SF7 0.113 s / SF9 0.411 s / SF12 2.466 s (Semtech calculator)
    TX power draw ~145 mW (44 mA @ 3.3 V) + 20 mJ RX-window overhead
    link budget = 14 dBm - sensitivity: SF7 137 dB / SF9 143 dB / SF12 151 dB
    ETSI duty cycle 1 % -> min period = airtime / 0.01 (governance constraint)
  NB-IoT (3GPP coverage classes, energy per report incl. sync/RACH,
    Lauridsen/Michelinakis-class magnitudes):
    CE0 (<=144 dB MCL): 1.5 J, latency 2 s
    CE1 (<=154 dB MCL): 3.0 J, latency 5 s
    CE2 (<=164 dB MCL): 7.0 J, latency 10 s
  LTE-M (MCL 155.7 dB): 1.0 J per report, latency 0.5 s
  Device sleep floor 16.5 uW (5 uA @ 3.3 V); battery 6.2 Wh (2xAA),
  lifetime capped at 15 y (shelf life).

Metrics (use case U = "battery-powered status updates, deep-indoor metering"):
  - quality/time-based (AoI): mean age of information = T/2 + latency [min]
  - resource/energy:          battery lifetime [years]              [max]
  Constraints (governance/coverage):
  - radio duty cycle (LoRaWAN, ETSI EN 300 220)
  - link budget >= required coverage depth (scenario)

Config space X: technology/mode x update period T in [1 min .. 6 h].
Baseline x0: LTE-M with T = 5 min ("one-size-fits-all" over-reporting).
Parameter study: required coverage depth in {140, 154, 164} dB.
"""
import numpy as np
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"

BATTERY_WH = 6.2
P_SLEEP = 16.5e-6            # W
LIFECAP_Y = 15.0
YEAR_S = 365.25 * 24 * 3600

TECHS = {
    # name: (energy per msg J, latency s, link budget dB, min period s | None)
    "LoRa SF7":  (0.145 * 0.113 + 0.02, 0.113 + 1.0, 137.0, 0.113 / 0.01),
    "LoRa SF9":  (0.145 * 0.411 + 0.02, 0.411 + 1.0, 143.0, 0.411 / 0.01),
    "LoRa SF12": (0.145 * 2.466 + 0.02, 2.466 + 1.0, 151.0, 2.466 / 0.01),
    "NB-IoT CE0": (1.5, 2.0, 144.0, None),
    "NB-IoT CE1": (3.0, 5.0, 154.0, None),
    "NB-IoT CE2": (7.0, 10.0, 164.0, None),
    "LTE-M":     (1.0, 0.5, 155.7, None),
}
PERIODS_S = np.array([60, 120, 300, 600, 900, 1800, 3600, 7200, 21600], float)
AOI_REQ_MIN = 10.0           # application requirement: mean AoI <= 10 min


def evaluate(tech, T):
    e_msg, lat, lb, min_T = TECHS[tech]
    aoi_min = (T / 2 + lat) / 60.0
    p_avg = e_msg / T + P_SLEEP
    life_y = min(LIFECAP_Y, BATTERY_WH * 3600 / p_avg / YEAR_S)
    duty_ok = (min_T is None) or (T >= min_T)
    return aoi_min, life_y, lb, duty_ok


def main():
    names = ["mean AoI [min]", "battery lifetime [y]"]
    directions = ["min", "max"]
    configs, points, lbs, duty = [], [], [], []
    for tech in TECHS:
        for T in PERIODS_S:
            aoi, life, lb, dok = evaluate(tech, T)
            configs.append({"tech": tech, "period_s": T})
            points.append([aoi, life])
            lbs.append(lb)
            duty.append(dok)
    points = np.asarray(points)
    lbs, duty = np.asarray(lbs), np.asarray(duty)
    base = configs.index({"tech": "LTE-M", "period_s": 300.0})

    for cov in (140.0, 154.0, 164.0):
        feas = duty & (lbs >= cov)
        norm, _ = pareto.normalize(points, directions)
        front = pareto.pareto_front(norm[feas])
        # improved: feasible, AoI requirement met, max lifetime
        ok = feas & (points[:, 0] <= AOI_REQ_MIN)
        cand = int(np.where(ok, points[:, 1], -np.inf).argmax()) if ok.any() else base
        cls = pareto.classify_change(norm[base], norm[cand], front)
        d0 = pareto.distance_to_front(norm[base], front)
        c = configs[cand]
        print(f"coverage {cov:.0f} dB: base(LTE-M,5min) feas={bool(feas[base])} "
              f"d={d0:.2f} {np.round(points[base],1)} | best {c['tech']} "
              f"T={c['period_s']/60:.0f}min {np.round(points[cand],1)} -> {cls} "
              f"| {feas.sum()}/{len(points)} feasible")

    # figure for deep-indoor scenario (164 dB): only NB-IoT CE2 feasible
    # and mid scenario 154 dB: LoRa SF12(151) infeasible, NB CE1/LTE-M ok
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    markers = {"LoRa SF7": "o", "LoRa SF9": "s", "LoRa SF12": "D",
               "NB-IoT CE0": "^", "NB-IoT CE1": "v", "NB-IoT CE2": "<",
               "LTE-M": "*"}
    cols = {"LoRa SF7": "#1a73e8", "LoRa SF9": "#4c8df5", "LoRa SF12": "#8ab4f8",
            "NB-IoT CE0": "#188038", "NB-IoT CE1": "#34a853", "NB-IoT CE2": "#81c995",
            "LTE-M": "#d93025"}
    for tech in TECHS:
        idx = [i for i, c in enumerate(configs) if c["tech"] == tech and duty[i]]
        p = points[idx]
        ax.plot(p[:, 0], p[:, 1], "-", c=cols[tech], lw=0.7, alpha=0.7)
        ax.scatter(p[:, 0], p[:, 1], marker=markers[tech], s=14, c=cols[tech],
                   label=tech)
    ax.scatter(*points[base], marker="s", s=60, facecolors="none",
               edgecolors="k", label="baseline $x_0$")
    ax.axvline(AOI_REQ_MIN, color="k", lw=0.6, ls="--")
    # place the label in the open gap between the LoRa SF9 and SF12 fronts
    ax.text(AOI_REQ_MIN * 1.15, 6.0, "AoI req.", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("mean AoI [min]")
    ax.set_ylabel("battery lifetime [y]")
    ax.legend(frameon=False, fontsize=5.5, ncol=1, loc="upper left")
    fig.savefig(FIG / "uc4_aoi_lifetime.pdf")
    fig.savefig(FIG / "uc4_aoi_lifetime.png")

    pareto.save_points(OUT / "uc4_points.json", names, directions, points,
                       configs, extra={"link_budget": lbs.tolist(),
                                       "duty_ok": duty.tolist(),
                                       "baseline": base,
                                       "aoi_req_min": AOI_REQ_MIN})


if __name__ == "__main__":
    main()
