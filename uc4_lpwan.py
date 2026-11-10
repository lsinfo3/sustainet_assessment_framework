"""
UC4 — LPWAN technology selection for a status-update service:
      LoRaWAN vs. NB-IoT vs. LTE-M.

Input provenance (physical formula plus explicitly assumed profiles):
  LoRaWAN (EU868, 14 dBm, 51 B PHY payload, 125 kHz, CR4/5, 8-symbol preamble,
  explicit header, CRC on, low-data-rate optimize at SF11/SF12):
    airtime is COMPUTED by lora_airtime() from the time-on-air formula of the
    Semtech SX1276/77/78/79 datasheet Rev. 7, sec. 4.1.1.7, giving
    SF7 0.1027 s / SF9 0.3287 s / SF12 2.4658 s. Earlier releases carried the
    hard-coded triple 0.113 / 0.411 / 2.466 s, which no single configuration
    reproduces: 2.466 s is the 51 B value for SF12, but 0.113 and 0.411 s
    correspond to payloads about 8 and 17 bytes larger.
    TX draw 145 mW and 20 mJ RX overhead are assumed representative values.
    SF7/SF9/SF12 link budgets 137/143/151 dB are scenario assumptions;
    the implied sensitivities are -123/-129/-137 dBm at 14 dBm TX power.
    ETSI duty cycle 1 % -> min period = airtime / 0.01 (governance constraint;
    the 1 % allowance is the one for the 865-868.6 MHz sub-band)
  NB-IoT: the following energy, latency and coverage tuples are assumed
    representative class profiles, NOT values extracted as tuples from
    Lauridsen et al. That paper anchors state-power and coverage magnitudes.
    Assumed energy includes synchronization/access overhead:
    CE0 (<=144 dB MCL): 1.5 J, latency 2 s
    CE1 (<=154 dB MCL): 3.0 J, latency 5 s
    CE2 (<=164 dB MCL): 7.0 J, latency 10 s
  LTE-M: 155.7 dB budget, 1.0 J/report and 0.5 s latency are scenario
    assumptions with no exact cited source for the tuple.
  Sleep floor 16.5 uW (5 uA @ 3.3 V), battery 6.2 Wh and 15 y shelf-life
  cap are assumptions. Lauridsen measured 13 uW PSM draw, not 16.5 uW.
  AoI assumes deterministic periodic generation and fixed latency; battery
  power is E_message/T + P_sleep, without a separate packet-loss model.

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
FIG.mkdir(exist_ok=True)   # so each script also runs standalone

BATTERY_WH = 6.2
P_SLEEP = 16.5e-6            # W
LIFECAP_Y = 15.0
YEAR_S = 365.25 * 24 * 3600

# --- LoRa time on air, Semtech SX1276 datasheet Rev. 7, sec. 4.1.1.7 ---
LORA_PAYLOAD_B = 51          # PHY payload [B]
LORA_BW = 125e3              # Hz
LORA_CR = 1                  # coding rate 4/5
LORA_PREAMBLE = 8            # symbols
LORA_TX_W = 0.145            # W during transmission (44 mA @ 3.3 V)
LORA_RX_J = 0.02             # J of receive-window overhead per message
LORA_DUTY = 0.01             # ETSI 865-868.6 MHz sub-band allowance


def lora_airtime(sf, payload_b=LORA_PAYLOAD_B, bw=LORA_BW, cr=LORA_CR,
                 preamble=LORA_PREAMBLE, explicit_header=True, crc=True):
    """Time on air [s] of one LoRa frame.

    Low-data-rate optimize is mandatory when the symbol duration exceeds
    16 ms, i.e. at SF11 and SF12 on a 125 kHz channel.
    """
    t_sym = (2 ** sf) / bw
    de = 1 if t_sym > 16e-3 else 0
    ih = 0 if explicit_header else 1
    num = 8 * payload_b - 4 * sf + 28 + 16 * int(crc) - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(int(np.ceil(num / den)) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n_payload * t_sym


_AT = {sf: lora_airtime(sf) for sf in (7, 9, 12)}

TECHS = {
    # name: (energy per msg J, latency s, link budget dB, min period s | None)
    "LoRa SF7":  (LORA_TX_W * _AT[7] + LORA_RX_J, _AT[7] + 1.0, 137.0,
                  _AT[7] / LORA_DUTY),
    "LoRa SF9":  (LORA_TX_W * _AT[9] + LORA_RX_J, _AT[9] + 1.0, 143.0,
                  _AT[9] / LORA_DUTY),
    "LoRa SF12": (LORA_TX_W * _AT[12] + LORA_RX_J, _AT[12] + 1.0, 151.0,
                  _AT[12] / LORA_DUTY),
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
        cls = pareto.classify_change(norm[base], norm[cand], front,
                                     destination_feasible=bool(feas[cand]))
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
    ax.scatter(*points[base], marker="s", s=45, c=pareto.C_BASE,
               edgecolors="white", linewidths=0.5, label="baseline $x_0$",
               zorder=5)
    ax.axvline(AOI_REQ_MIN, color="k", lw=0.6, ls="--")
    # place the label in the open gap between the LoRa SF9 and SF12 fronts
    ax.text(AOI_REQ_MIN * 1.15, 6.0, "AoI req.", fontsize=6)
    # headroom so the legend block clears the 15-year plateau
    ax.set_ylim(-0.7, 15.0 * (1.0 + 0.62 * 2.0 / pareto.FIG_H))
    ax.set_xscale("log")
    ax.set_xlabel("mean AoI [min]")
    ax.set_ylabel("battery lifetime [y]")
    # Two legends, one in each top corner: a single block in the top left
    # reaches across the AoI-requirement line and sits on top of it.
    _h, _l = ax.get_legend_handles_labels()
    _kw = dict(frameon=False, fontsize=5.5, ncol=1, handletextpad=0.4,
               labelspacing=0.25, borderaxespad=0.3)
    ax.add_artist(ax.legend(_h[:3], _l[:3], loc="upper left", **_kw))
    ax.legend(_h[3:], _l[3:], loc="upper right", **_kw)
    fig.savefig(FIG / "uc4_aoi_lifetime.pdf")
    fig.savefig(FIG / "uc4_aoi_lifetime.png")

    pareto.save_points(OUT / "uc4_points.json", names, directions, points,
                       configs, extra={"link_budget": lbs.tolist(),
                                       "duty_ok": duty.tolist(),
                                       "baseline": base,
                                       "aoi_req_min": AOI_REQ_MIN})


if __name__ == "__main__":
    main()
