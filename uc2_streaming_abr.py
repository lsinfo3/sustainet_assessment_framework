"""
UC2 — Adaptive video streaming: ABR policy vs. QoE vs. resource demand.

Model: time-slotted (1 s) throughput trace (log-normal around a diurnal mean),
DASH-like ladder, rate-based ABR with safety margin theta and estimator
window w. Buffer dynamics with rebuffering. Baseline: static top bitrate
(no adaptation).

Literature anchors (cite in paper):
  - ladder & resolution-MOS mapping: BBB ladder as in the SustaiNet demo,
    MOS 1..5 for 240p..1080p (cf. ITU-T P.1203-style quality mapping)
  - network energy intensity: 0.1 kWh/GB transport network (order of
    magnitude from energy-intensity literature, e.g., Coroama/Hilty,
    Schien et al.) -> resource metric reported as data volume AND energy

Metrics (use case U = "mobile video streaming, 1 h session"):
  - quality/QoE:        mean MOS (session, stall-penalized)  [max]
  - continuity:         stall time share [%]                 [min]
  - resource/energy:    network energy per hour [Wh]         [min]
  Governance/QoE floor constraint: stall share <= 1 %

Config space X: theta in [0.3..1.2] x estimator window w in {2, 8} slots,
plus static ladder levels (no ABR). Baseline x0: static 1080p.
Parameter study: mean capacity scenario in {2, 4, 8} Mbps.
"""
import numpy as np
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"

LADDER_MBPS = np.array([0.40, 0.75, 1.50, 2.50, 4.50])
LADDER_MOS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
SEG = 1.0                    # s per slot
SESSION = 3600               # slots (1 h)
WH_PER_GB = 100.0            # 0.1 kWh/GB network energy intensity
STALL_SLA = 1.0              # % stall share (governance/QoE floor)
BUF_MAX = 30.0               # s


def trace(mean_mbps, n, seed):
    rng = np.random.default_rng(seed)
    # log-normal slot capacity, sigma=0.5, slow variation via AR(1)
    x = np.zeros(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = 0.98 * x[i - 1] + 0.2 * rng.standard_normal()
    return mean_mbps * np.exp(x - x.var() / 2)


def run(policy, cap, theta=1.0, w=3, static_level=None):
    """Simulate one session; returns (mean MOS, stall share %, energy Wh/h)."""
    buf, stall, bits = 0.0, 0.0, 0.0
    mos_acc = []
    hist = []
    for t in range(SESSION):
        c = cap[t]
        if static_level is not None:
            lvl = static_level
        else:
            est = len(hist) and (len(hist) / np.sum(1.0 / np.array(hist))) or c
            lvl = int(np.searchsorted(LADDER_MBPS, theta * est, side="right") - 1)
            lvl = max(0, min(lvl, len(LADDER_MBPS) - 1))
        hist.append(c)
        hist = hist[-w:]
        rate = LADDER_MBPS[lvl]
        # download SEG seconds of video at 'rate' using capacity c
        dl_time = SEG * rate / c
        bits += rate * 1e6 * SEG
        if dl_time <= SEG + buf:          # arrives in time (buffer absorbs)
            buf = min(BUF_MAX, buf + SEG - dl_time)
            mos_acc.append(LADDER_MOS[lvl])
        else:                              # rebuffering
            stall += dl_time - (SEG + buf)
            buf = 0.0
            mos_acc.append(1.0)            # stall slot -> worst experience
    gb = bits / 8 / 1e9
    return float(np.mean(mos_acc)), 100.0 * stall / (SESSION + stall), gb * WH_PER_GB


def scenario(mean_mbps, seed=11):
    cap = trace(mean_mbps, SESSION, seed)
    names = ["mean MOS", "stall share [%]", "energy [Wh/h]"]
    directions = ["max", "min", "min"]
    configs, points = [], []
    # baseline: static top bitrate
    configs.append({"policy": "static-1080p"})
    points.append(run("static", cap, static_level=len(LADDER_MBPS) - 1))
    for lvl in range(len(LADDER_MBPS) - 1):
        configs.append({"policy": f"static-{lvl}"})
        points.append(run("static", cap, static_level=lvl))
    for w in (2, 8):
        for theta in np.arange(0.3, 1.21, 0.1):
            configs.append({"policy": "abr", "theta": round(float(theta), 2), "w": w})
            points.append(run("abr", cap, theta=theta, w=w))
    points = np.asarray(points)
    feasible = points[:, 1] <= STALL_SLA
    return names, directions, configs, points, feasible


def main():
    all_out = {}
    for mean_mbps in (2.0, 4.0, 8.0):
        names, dirs, configs, pts, feas = scenario(mean_mbps)
        norm, _ = pareto.normalize(pts, dirs)
        front = pareto.pareto_front(norm[feas]) if feas.any() else norm
        base = 0
        # improved: feasible, max MOS per Wh -> pick min distance to front w/ MOS>=3
        ok = feas & (pts[:, 0] >= 3.0)
        if ok.any():
            cand = int(np.where(ok, pts[:, 2], np.inf).argmin())
        elif feas.any():   # fallback: feasible config with best MOS
            cand = int(np.where(feas, pts[:, 0], -np.inf).argmax())
        else:
            cand = base
        cls = pareto.classify_change(norm[base], norm[cand], front)
        d0 = pareto.distance_to_front(norm[base], front)
        all_out[mean_mbps] = (configs, pts, feas, base, cand, cls, d0)
        print(f"C={mean_mbps} Mbps: base {np.round(pts[base],2)} d={d0:.2f} | "
              f"improved {configs[cand]} {np.round(pts[cand],2)} -> {cls} | "
              f"{feas.sum()}/{len(pts)} feasible")

    # main scenario for the figure: 4 Mbps
    configs, pts, feas, base, cand, cls, d0 = all_out[4.0]
    pareto.save_points(OUT / "uc2_points.json",
                       ["mean MOS", "stall share [%]", "energy [Wh/h]"],
                       ["max", "min", "min"], pts, configs,
                       extra={"baseline": base, "improved": cand,
                              "classification": cls, "scenario_mbps": 4.0})
    # display space: energy (min) vs -MOS -> plot energy vs MOS with max-y
    disp = np.column_stack([pts[:, 2], -pts[:, 0]])
    fig, ax = pareto.plot_front_2d(
        disp, ["energy [Wh/h]", "negative MOS (lower = better QoE)"],
        baseline_idx=base, highlight_idx=cand, feasible=feas,
        fname=FIG / "uc2_front_energy_mos.pdf")
    # main figure (Fig. 5): standard front grammar for the primary 4 Mbps
    # scenario -- dominated / infeasible / Pareto front / baseline / improved,
    # matching UC1/UC3/UC5/UC6 -- with the 2 and 8 Mbps fronts overlaid as
    # light reference curves to convey the capacity-dependence of the result.
    import matplotlib.pyplot as plt
    cfg4, p4, f4, b4, c4, _, _ = all_out[4.0]
    en, mos = p4[:, 2], p4[:, 0]                 # energy [Wh/h], mean MOS
    dmask = np.zeros(len(p4), bool)              # feasible & non-dominated
    dmask[np.where(f4)[0][pareto.pareto_mask(
        np.column_stack([en, -mos])[f4])]] = True

    fig2, ax2 = plt.subplots()
    # reference fronts for the 2 and 8 Mbps capacity scenarios
    for mbps, col in ((2.0, "#e0928c"), (8.0, "#8fbf9f")):
        cfg, p, f, b, c_, _, _ = all_out[mbps]
        m = np.zeros(len(p), bool)
        m[np.where(f)[0][pareto.pareto_mask(
            np.column_stack([p[:, 2], -p[:, 0]])[f])]] = True
        fr = p[m][np.argsort(p[m][:, 2])]
        ax2.plot(fr[:, 2], fr[:, 0], "-", c=col, lw=1.0, alpha=0.9, zorder=2,
                 label=f"{mbps:.0f} Mbps front (ref.)")
    # primary 4 Mbps scenario in the shared plot grammar
    ax2.scatter(en[f4 & ~dmask], mos[f4 & ~dmask], s=12, c=pareto.C_DOM,
                label="dominated", zorder=3)
    if (~f4).any():
        ax2.scatter(en[~f4], mos[~f4], s=14, facecolors="none",
                    edgecolors=pareto.C_DOM, linewidths=0.7,
                    label="infeasible", zorder=3)
    order = np.argsort(en[dmask])
    ax2.plot(en[dmask][order], mos[dmask][order], "-o", c=pareto.C_FRONT, ms=4,
             label="Pareto front $\\hat{\\mathcal{P}}$ (4 Mbps)", zorder=4)
    ax2.scatter(en[b4], mos[b4], marker="s", s=45, c=pareto.C_BASE,
                edgecolors="white", linewidths=0.5, label="baseline $x_0$", zorder=5)
    ax2.scatter(en[c4], mos[c4], marker="*", s=90, c=pareto.C_IMP,
                edgecolors="white", linewidths=0.5, label="improved $x'$", zorder=5)
    ax2.annotate("", xy=(en[c4], mos[c4]), xytext=(en[b4], mos[b4]),
                 arrowprops=dict(arrowstyle="->", color=pareto.C_IMP, lw=1.0,
                                 alpha=0.7), zorder=4)
    ax2.set_xlabel("energy [Wh/h]")
    ax2.set_ylabel("mean MOS")
    ax2.margins(0.16)
    ax2.legend(frameon=False, loc="lower right", labelspacing=0.3,
               handlelength=1.4, fontsize=6.5)
    fig2.savefig(FIG / "uc2_param_study.pdf")
    fig2.savefig(FIG / "uc2_param_study.png")


if __name__ == "__main__":
    main()
