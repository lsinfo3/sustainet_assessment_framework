"""
SUSTAINET evaluation harness: normalization, Pareto analysis, plots.

Implements the formal model of Section IV-A:
  - min-max normalization with direction alignment (smaller = better)
  - Pareto dominance, front extraction
  - hypervolume (2D exact, nD Monte Carlo), distance to front
  - change classification: strict improvement / trade-off improvement /
    burden shift / degradation
  - paper-ready plots (2D front, parallel coordinates)

All functions take points as np.ndarray of shape (N, n_metrics), raw units.
`directions` is a list of 'min'/'max' per metric ('max' metrics are negated
before normalization so that smaller is always better internally).
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- style
# One IEEE column wide, so figures are included at their natural size and their
# 8 pt text stays 8 pt. The height is the only free dimension; the per-figure
# layout below scales with FIG_H so legends keep their absolute clearances.
FIG_W, FIG_H = 3.33, 1.70

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.figsize": (FIG_W, FIG_H),
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
})

C_DOM = "#9aa0a6"    # dominated points
C_FRONT = "#1a73e8"  # front
C_BASE = "#d93025"   # baseline
C_IMP = "#188038"    # improvement


# ---------------------------------------------------------- normalization
def align(points: np.ndarray, directions) -> np.ndarray:
    """Negate 'max' metrics so smaller is better for all."""
    pts = np.asarray(points, dtype=float).copy()
    for i, d in enumerate(directions):
        if d == "max":
            pts[:, i] = -pts[:, i]
    return pts


def normalize(points: np.ndarray, directions, bounds=None):
    """Min-max normalize direction-aligned points to [0, 1] (0 = best).

    Returns (normalized_points, (mins, maxs)) so that additional points can
    be normalized consistently with `apply_normalization`.
    """
    pts = align(points, directions)
    if bounds is None:
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
    else:
        mins, maxs = bounds
    span = np.where(maxs - mins > 0, maxs - mins, 1.0)
    return (pts - mins) / span, (mins, maxs)


def apply_normalization(points, directions, bounds):
    pts = align(points, directions)
    mins, maxs = bounds
    span = np.where(maxs - mins > 0, maxs - mins, 1.0)
    return (pts - mins) / span


# ------------------------------------------------------------- dominance
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff a dominates b (minimization, normalized space)."""
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (minimization)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dom = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        if np.any(dom):
            mask[i] = False
    return mask


def pareto_front(points: np.ndarray) -> np.ndarray:
    return np.asarray(points)[pareto_mask(points)]


def feasible_front_mask(points, directions, feasible=None) -> np.ndarray:
    """Boolean mask over *all* rows marking the feasible non-dominated set.

    Dominance is evaluated in the FULL objective space, on direction-aligned
    raw values. Negating a maximized column reverses its numerical direction
    together with its preference, giving a minimization coordinate. Positive
    affine normalization then preserves aligned dominance; a constant column
    contributes no preference information and is normalized to zero.

    Use this instead of running `pareto_mask` on the two columns that happen to
    be plotted: a configuration that is non-dominated over all objectives can
    look dominated once the objective vector is projected onto two axes.
    """
    pts = np.asarray(points, dtype=float)
    feas = (np.ones(len(pts), bool) if feasible is None
            else np.asarray(feasible, bool))
    mask = np.zeros(len(pts), bool)
    if feas.any():
        aligned = align(pts, directions)
        mask[np.where(feas)[0][pareto_mask(aligned[feas])]] = True
    return mask


# ------------------------------------------------------------- indicators
def hypervolume(points: np.ndarray, ref=None) -> float:
    """Hypervolume dominated by the front of `points` w.r.t. `ref`
    (normalized minimization space; default ref = (1.1, ..., 1.1)).
    Exact for 2D, Monte Carlo for higher dimensions."""
    front = pareto_front(points)
    n = front.shape[1]
    if ref is None:
        ref = np.full(n, 1.1)
    ref = np.asarray(ref, dtype=float)
    front = front[np.all(front <= ref, axis=1)]
    if len(front) == 0:
        return 0.0
    if n == 2:
        f = front[np.argsort(front[:, 0])]
        hv, prev_y = 0.0, ref[1]
        for x, y in f:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return float(hv)
    rng = np.random.default_rng(42)
    samples = rng.uniform(np.zeros(n), ref, size=(200_000, n))
    dominated = np.zeros(len(samples), dtype=bool)
    for p in front:
        dominated |= np.all(samples >= p, axis=1)
    return float(dominated.mean() * np.prod(ref))


def distance_to_front(point: np.ndarray, front: np.ndarray) -> float:
    """Euclidean distance of a (normalized) point to the closest front point."""
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        raise ValueError("Distance is undefined for an empty feasible front")
    return float(np.min(np.linalg.norm(front - np.asarray(point), axis=1)))


# Compact legend tags for the four outcome classes. The legend sits inside a
# one-column figure, so the full class names do not fit beside the callouts;
# the caption spells the class out.
CLASS_TAG = {
    "strict improvement": "strict",
    "trade-off improvement": "trade-off",
    "burden shifting": "burden shift.",
    "degradation": "degradation",
}


def highlight_tag(cls: str | None) -> str:
    """Legend entry for the highlighted configuration, naming its outcome
    class instead of asserting a generic 'improvement'."""
    return "$x'$: " + ("unclassified" if cls is None else CLASS_TAG.get(cls, cls))


def classify_change(m0, m1, front, *, destination_feasible: bool) -> str | None:
    """Classify an admissible change in normalized minimization coordinates.

    The caller must evaluate all use-case constraints and explicitly pass the
    destination's feasibility. An infeasible destination is outside this rule's
    domain and raises ValueError. Equal metric vectors are indistinguishable
    under the selected objectives and return None (no outcome class).
    The baseline may be infeasible; restoring feasibility can be assessed.
    """
    if not destination_feasible:
        raise ValueError("An infeasible destination has no outcome class")
    m0, m1 = np.asarray(m0, dtype=float), np.asarray(m1, dtype=float)
    if np.array_equal(m0, m1):
        return None
    if dominates(m1, m0):
        return "strict improvement"
    if dominates(m0, m1):
        return "degradation"
    d0, d1 = distance_to_front(m0, front), distance_to_front(m1, front)
    if d1 < d0 - 1e-12:
        return "trade-off improvement"
    return "burden shifting"


# ------------------------------------------------------------------ plots
C_LEADER = "#9aa0a6"   # thin leader lines for callout labels
C_TEXT = "#202124"     # callout text


def anchor_label(ax, xy_data, text, anchor, color=C_TEXT, fontsize=6.5,
                 weight=None):
    """Place a callout `text` at `anchor` (axes-fraction coords) and connect it
    to the data point `xy_data` with a thin leader line.

    Anchoring in axes fraction guarantees the label stays inside the plotting
    box (no spilling into the page margin) and lets each use case place labels
    in empty regions so they never overlap. Horizontal/vertical alignment is
    derived from the anchor so the text block always grows toward the interior.
    """
    ax_x, ax_y = anchor
    ha = "left" if ax_x < 0.45 else ("right" if ax_x > 0.55 else "center")
    va = "bottom" if ax_y < 0.45 else ("top" if ax_y > 0.55 else "center")
    ax.annotate(
        text, xy=xy_data, xycoords="data",
        xytext=(ax_x, ax_y), textcoords="axes fraction",
        ha=ha, va=va, fontsize=fontsize, color=color, weight=weight,
        linespacing=1.0, zorder=6,
        arrowprops=dict(arrowstyle="-", color=C_LEADER, lw=0.5,
                        shrinkA=1, shrinkB=3),
    )


def plot_front_2d(points, labels, baseline_idx=None, highlight_idx=None,
                  feasible=None, fname=None, title=None, annotate=None,
                  legend_loc="best", legend_bbox=None, legend_ncol=1,
                  front_mask=None, highlight_label="selected $x'$"):
    """Scatter all operating points, draw the Pareto front (of feasible
    points), mark the baseline red and an optional selected configuration green.

    points: raw-unit (N,2) array, both metrics 'min'-oriented for display.
    feasible: optional boolean mask; infeasible points drawn as open circles.
    annotate: optional list of callouts, each ``(idx, text, anchor)`` where
        ``anchor`` is an ``(x_frac, y_frac)`` position in axes-fraction
        coordinates, optionally followed by a dict of styling overrides
        (e.g. ``{"color": C_BASE, "weight": "bold"}``). A bare ``(idx, text)``
        is still accepted and defaults to an interior anchor.
    legend_loc: matplotlib legend location (default "best").
    front_mask: optional boolean mask over all points marking the
        non-dominated set, as returned by `feasible_front_mask`. Pass the mask
        computed in the FULL objective space; the two display columns are a
        projection, and a point that is non-dominated over all objectives can
        appear dominated in that projection. When omitted, the mask is derived
        from the display columns, which is correct only when those columns are
        the complete objective set.
    highlight_label: legend entry for the highlighted configuration. Defaults
        to the neutral "selected $x'$"; pass the outcome class of Section IV-A
        where the change is not an unqualified improvement.
    """
    pts = np.asarray(points, dtype=float)
    feas = np.ones(len(pts), bool) if feasible is None else np.asarray(feasible)
    fig, ax = plt.subplots()
    if front_mask is None:
        # fallback: display columns are the whole objective space
        mask = np.zeros(len(pts), bool)
        mask[np.where(feas)[0][pareto_mask(pts[feas])]] = True
    else:
        mask = np.asarray(front_mask, bool) & feas

    ax.scatter(*pts[~mask & feas].T, s=12, c=C_DOM, label="dominated", zorder=2)
    if np.any(~feas):
        ax.scatter(*pts[~feas].T, s=14, facecolors="none", edgecolors=C_DOM,
                   linewidths=0.7, label="infeasible", zorder=2)
    fr = pts[mask][np.argsort(pts[mask][:, 0])]
    ax.plot(*fr.T, "-o", c=C_FRONT, ms=4, label="Pareto front $\\hat{\\mathcal{P}}$", zorder=3)
    if baseline_idx is not None:
        ax.scatter(*pts[baseline_idx], marker="s", s=45, c=C_BASE,
                   edgecolors="white", linewidths=0.5,
                   label="baseline $x_0$", zorder=5)
    if highlight_idx is not None:
        ax.scatter(*pts[highlight_idx], marker="*", s=90, c=C_IMP,
                   edgecolors="white", linewidths=0.5,
                   label=highlight_label, zorder=5)
        if baseline_idx is not None:
            ax.annotate("", xy=pts[highlight_idx], xytext=pts[baseline_idx],
                        arrowprops=dict(arrowstyle="->", color=C_IMP, lw=1.0,
                                        alpha=0.7), zorder=3)
    # margins first so axis limits are settled before anchoring callouts
    ax.margins(0.16)
    for item in (annotate or []):
        idx, text = item[0], item[1]
        anchor = item[2] if len(item) > 2 else (0.5, 0.5)
        opts = item[3] if len(item) > 3 else {}
        anchor_label(ax, pts[idx], text, anchor, **opts)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title:
        ax.set_title(title)
    # Legend in a strip below the axes: at one column width the plot area is too
    # small to hold a multi-entry legend without covering data or callouts.
    if legend_bbox is None and legend_loc == "below":
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26),
                  ncol=legend_ncol, handlelength=1.1, labelspacing=0.2,
                  columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0)
    else:
        ax.legend(frameon=False, loc=legend_loc, handlelength=1.1, ncol=legend_ncol,
                  borderaxespad=0.2, labelspacing=0.2, columnspacing=0.8,
                  handletextpad=0.4, bbox_to_anchor=legend_bbox)
    if fname:
        fig.savefig(fname)
        fig.savefig(str(fname).rsplit(".", 1)[0] + ".png")
    return fig, ax


def plot_parallel(points, labels, directions, baseline_idx=None,
                  highlight_idx=None, fname=None):
    """Parallel-coordinates plot in normalized space (0 = best)."""
    norm, _ = normalize(points, directions)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = np.arange(norm.shape[1])
    for i, row in enumerate(norm):
        if i == baseline_idx or i == highlight_idx:
            continue
        ax.plot(xs, row, c=C_DOM, alpha=0.35, lw=0.6)
    if baseline_idx is not None:
        ax.plot(xs, norm[baseline_idx], c=C_BASE, lw=1.6, label="baseline $x_0$")
    if highlight_idx is not None:
        ax.plot(xs, norm[highlight_idx], c=C_IMP, lw=1.6, label="selected $x'$")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("normalized cost (0 = best)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, handlelength=1.1, borderaxespad=0.0)
    if fname:
        fig.savefig(fname)
        fig.savefig(str(fname).rsplit(".", 1)[0] + ".png")
    return fig, ax


# ------------------------------------------------------------------- io
def save_points(fname, names, directions, points, configs, extra=None):
    """Persist operating points + configs as JSON for later comparison."""
    out = {
        "metrics": list(names),
        "directions": list(directions),
        "points": np.asarray(points, dtype=float).tolist(),
        "configs": configs,
    }
    if extra:
        out.update(extra)
    with open(fname, "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------- smoke test
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (60, 2))
    m = pareto_mask(pts)
    assert m.sum() >= 1
    # front points must be mutually non-dominated
    fr = pts[m]
    for i in range(len(fr)):
        for j in range(len(fr)):
            if i != j:
                assert not dominates(fr[i], fr[j])
    hv = hypervolume(pts)
    assert 0 < hv < 1.21
    # a point on the front has distance 0
    assert distance_to_front(fr[0], fr) == 0.0
    # classification sanity
    front = np.array([[0.1, 0.5], [0.3, 0.2], [0.6, 0.1]])
    assert classify_change([0.5, 0.5], [0.4, 0.4], front, destination_feasible=True) == "strict improvement"
    assert classify_change([0.4, 0.4], [0.5, 0.5], front, destination_feasible=True) == "degradation"
    assert classify_change([0.5, 0.5], [0.35, 0.52], front, destination_feasible=True) in (
        "trade-off improvement", "burden shifting")
    print("pareto.py smoke test OK; hv(random)= %.3f" % hv)
