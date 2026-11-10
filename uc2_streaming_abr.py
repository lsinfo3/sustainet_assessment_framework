"""
UC2: physical-clock adaptive streaming of one hour of video content.

Capacity is piecewise constant in one-second physical-time slots. A seeded
AR(1) log-normal 3600-slot trace is rescaled to its nominal period mean and
repeated periodically if playback extends beyond one hour. Downloads integrate
capacity across slot boundaries. Each policy sees the same physical trace,
not the same capacity sample for every segment index.

The first complete segment starts playback; startup delay is reported separately
and excluded from rebuffering. Subsequent stalls occur only after the buffered
complete segments have played out. Waiting at the 30 s buffer limit advances
the physical clock. The estimator is the harmonic mean of the last w completed
segment throughputs; the initial estimate is the observed capacity at time zero.

Representation scores 1..5 and assigning score 1 to any segment whose arrival
causes rebuffering are author-defined QoE surrogates, not a calibrated MOS model.
Energy is attributed volume times an assumed 100 Wh/GB, reported as
Wh/content-hour. Device and content-production energy are outside this scope.
"""
import numpy as np
from pathlib import Path
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)   # so each script also runs standalone

LADDER_MBPS = np.array([0.40, 0.75, 1.50, 2.50, 4.50])
LADDER_MOS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
SEG = 1.0                    # seconds of content per complete segment
SESSION = 3600               # segments: one content-hour
WH_PER_GB = 100.0            # 0.1 kWh/GB network energy intensity
STALL_SLA = 1.0              # % stall share (governance/QoE floor)
BUF_MAX = 30.0               # s


def trace(mean_mbps, n, seed):
    rng = np.random.default_rng(seed)
    # AR(1) log capacity: stationary SD sqrt(.2**2 / (1-.98**2)) ~= 1.005
    x = np.zeros(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = 0.98 * x[i - 1] + 0.2 * rng.standard_normal()
    # Exponentiate, then rescale so the realised trace mean is exactly
    # `mean_mbps`. Subtracting only half the variance (the log-normal
    # correction) left the sample mean of x uncompensated and biased every
    # trace high by exp(mean(x)).
    y = np.exp(x)
    return mean_mbps * y / y.mean()


def download_finish(start, megabits, cap, prefix=None):
    """Finish time integrating a periodic, one-second piecewise-constant trace."""
    cap = np.asarray(cap, dtype=float)
    if cap.ndim != 1 or len(cap) == 0 or np.any(~np.isfinite(cap)) or np.any(cap <= 0):
        raise ValueError("capacity must be a finite positive vector")
    if start < 0 or megabits < 0:
        raise ValueError("time and downloaded volume must be nonnegative")
    if prefix is None:
        prefix = np.r_[0.0, np.cumsum(cap)]
    period, total = len(cap), float(prefix[-1])
    cycles = int(np.floor(start / period))
    phase = start - cycles * period
    slot = min(int(np.floor(phase)), period - 1)
    delivered = cycles * total + prefix[slot] + (phase - slot) * cap[slot]
    target = delivered + megabits
    end_cycles = int(np.floor(target / total))
    residual = target - end_cycles * total
    end_slot = min(int(np.searchsorted(prefix, residual, side="right") - 1), period - 1)
    return float(end_cycles * period + end_slot + (residual - prefix[end_slot]) / cap[end_slot])


def run(policy, cap, theta=1.0, w=3, static_level=None, *, segments=SESSION,
        return_details=False):
    """Return (score, rebuffer %, Wh/content-hour); optionally diagnostic data.

    Rebuffer share = stall / (content duration + stall), excluding startup.
    Segment quality is its representation score unless its arrival causes a
    positive rebuffer interval, when the score is 1. Complete segments only.
    """
    cap = np.asarray(cap, dtype=float)
    if w < 1 or segments < 1:
        raise ValueError("window and segment count must be positive")
    prefix = np.r_[0.0, np.cumsum(cap)]
    clock, buf, stall, bits, startup = 0.0, 0.0, 0.0, 0.0, 0.0
    mos_acc, hist = [], []
    for t in range(segments):
        # Wait before requesting another complete segment if it would exceed
        # the storage limit; playback continues throughout this physical wait.
        idle = max(buf - (BUF_MAX - SEG), 0.0)
        clock += idle
        buf -= idle
        if static_level is not None:
            lvl = static_level
        else:
            est = len(hist) / np.sum(1.0 / np.array(hist)) if hist else cap[0]
            lvl = int(np.searchsorted(LADDER_MBPS, theta * est, side="right") - 1)
            lvl = max(0, min(lvl, len(LADDER_MBPS) - 1))
        rate = LADDER_MBPS[lvl]
        end = download_finish(clock, SEG * rate, cap, prefix)
        dl_time = end - clock
        interrupted = max(dl_time - buf, 0.0)
        if t == 0:
            startup = dl_time
            interrupted = 0.0
        else:
            stall += interrupted
        buf = max(buf - dl_time, 0.0) + SEG
        clock = end
        hist.append(SEG * rate / dl_time)
        hist = hist[-w:]
        bits += rate * 1e6 * SEG
        mos_acc.append(1.0 if interrupted > 1e-9 else LADDER_MOS[lvl])
    content = segments * SEG
    elapsed = clock + buf  # finish playing all downloaded complete segments
    assert np.isclose(elapsed, content + startup + stall, rtol=1e-9, atol=1e-7)
    gb = bits / 8 / 1e9
    metrics = (float(np.mean(mos_acc)), 100.0 * stall / (content + stall),
               gb * WH_PER_GB / (content / 3600.0))
    if return_details:
        return {"metrics": metrics, "startup_s": startup, "rebuffer_s": stall,
                "content_s": content, "elapsed_s": elapsed, "volume_GB": gb,
                "energy_Wh": gb * WH_PER_GB}
    return metrics


def scenario(mean_mbps, seed=11):
    cap = trace(mean_mbps, SESSION, seed)
    names = ["mean quality score", "stall share [%]", "energy [Wh/content-h]"]
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
        # Select minimum attributed energy among feasible candidates with score >=3.
        ok = feas & (pts[:, 0] >= 3.0)
        if ok.any():
            cand = int(np.where(ok, pts[:, 2], np.inf).argmin())
        elif feas.any():   # fallback: feasible config with best MOS
            cand = int(np.where(feas, pts[:, 0], -np.inf).argmax())
        else:
            raise ValueError("No feasible candidate in this capacity scenario; no class assigned")
        cls = pareto.classify_change(norm[base], norm[cand], front,
                                     destination_feasible=bool(feas[cand]))
        d0 = pareto.distance_to_front(norm[base], front)
        all_out[mean_mbps] = (configs, pts, feas, base, cand, cls, d0)
        print(f"C={mean_mbps} Mbps: base {np.round(pts[base],2)} d={d0:.2f} | "
              f"improved {configs[cand]} {np.round(pts[cand],2)} -> {cls} | "
              f"{feas.sum()}/{len(pts)} feasible")

    # main scenario for the figure: 4 Mbps
    configs, pts, feas, base, cand, cls, d0 = all_out[4.0]
    pareto.save_points(OUT / "uc2_points.json",
                       ["mean quality score", "stall share [%]", "energy [Wh/content-h]"],
                       ["max", "min", "min"], pts, configs,
                       extra={"baseline": base, "improved": cand,
                              "timing_model": "physical-clock periodic capacity; complete segments",
                              "startup_excluded_from_rebuffer": True,
                              "classification": cls, "scenario_mbps": 4.0})
    # display space: energy (min) vs -MOS -> plot energy vs MOS with max-y
    disp = np.column_stack([pts[:, 2], -pts[:, 0]])
    fmask = pareto.feasible_front_mask(pts, dirs, feas)
    fig, ax = pareto.plot_front_2d(
        disp, ["energy [Wh/content-h]", "negative score (lower = better QoE)"],
        baseline_idx=base, highlight_idx=cand, feasible=feas,
        front_mask=fmask, highlight_label=pareto.highlight_tag(cls),
        fname=FIG / "uc2_front_energy_mos.pdf")
    # main figure (Fig. 5): standard front grammar for the primary 4 Mbps
    # scenario -- dominated / infeasible / Pareto front / baseline / improved,
    # matching UC1/UC3/UC5/UC6 -- with the 2 and 8 Mbps fronts overlaid as
    # light reference curves to convey the capacity-dependence of the result.
    import matplotlib.pyplot as plt
    cfg4, p4, f4, b4, c4, _, _ = all_out[4.0]
    en, mos = p4[:, 2], p4[:, 0]                 # energy [Wh/content-h], mean quality score
    # non-dominated over all three objectives, not over the two plotted ones
    dmask = pareto.feasible_front_mask(p4, ["max", "min", "min"], f4)

    fig2, ax2 = plt.subplots()
    # reference fronts for the 2 and 8 Mbps capacity scenarios
    for mbps, col in ((2.0, "#e0928c"), (8.0, "#8fbf9f")):
        cfg, p, f, b, c_, _, _ = all_out[mbps]
        m = pareto.feasible_front_mask(p, ["max", "min", "min"], f)
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
                edgecolors="white", linewidths=0.5,
                label=pareto.highlight_tag(all_out[4.0][5]), zorder=5)
    ax2.annotate("", xy=(en[c4], mos[c4]), xytext=(en[b4], mos[b4]),
                 arrowprops=dict(arrowstyle="->", color=pareto.C_IMP, lw=1.0,
                                 alpha=0.7), zorder=4)
    ax2.set_xlabel("energy [Wh/content-h]")
    ax2.set_ylabel("mean quality score")
    ax2.margins(0.16)
    # seven entries do not fit inside a one-column-wide panel without covering
    # data, so this figure keeps its legend in a strip below the axes
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24 * 2.0 / pareto.FIG_H),
               ncol=3, labelspacing=0.2, handletextpad=0.4, handlelength=1.1,
               columnspacing=0.9, borderaxespad=0.0, fontsize=6.5)
    fig2.savefig(FIG / "uc2_param_study.pdf")
    fig2.savefig(FIG / "uc2_param_study.png")


if __name__ == "__main__":
    main()
