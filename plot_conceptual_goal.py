"""
Conceptual goal figure (Fig. 2): an illustrative point cloud with its Pareto
front, a "current operational point" far from the front, and a "target optimal
point" on it. Purely didactic (synthetic data) -- it visualizes the project's
overarching idea of moving operation closer to the front without burden
shifting, using the shared plotting style of pareto.py.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Make sure we can import pareto
sys.path.append(str(Path(__file__).parent))
import pareto

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

def main():
    # 1. Generate some random but structured points representing possible configurations
    rng = np.random.default_rng(42)
    x = rng.normal(loc=60, scale=20, size=300)
    y = rng.normal(loc=60, scale=20, size=300)
    
    # Filter points to create a nice convex front shape (allowing quality to go down to 5)
    valid = (x > 5) & (y > 5) & ((x-5)*(y-5) > 300)
    points = np.column_stack((x[valid], y[valid]))
    
    # Normalize to [0.1, 1.0] so the points do not touch the axes
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    points = 0.1 + 0.9 * (points - pmin) / (pmax - pmin)
    
    # 2. Pick a baseline point that is far from the front (Current Operational Point)
    baseline_idx = np.argmin((points[:,0] - 0.8)**2 + (points[:,1] - 0.8)**2)
    
    # 3. Find the Pareto front
    front_mask = pareto.pareto_mask(points)
    
    # 4. Pick an improved point on the front
    front_indices = np.where(front_mask)[0]
    highlight_idx = front_indices[np.argmin((points[front_indices,0] - 0.35)**2 + (points[front_indices,1] - 0.35)**2)]

    # 5. Plot using the existing pareto style
    fig, ax = pareto.plot_front_2d(
        points, 
        ["Environmental and economic impact\n(normalized, lower = better)",
         "Network quality\n(normalized, lower = better)"],
        baseline_idx=baseline_idx,
        highlight_idx=highlight_idx
    )
    
    # Update legend to match requested terminology but keep it compact
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for l in labels:
        if "baseline" in l:
            new_labels.append("Current operational point")
        elif "improved" in l:
            new_labels.append("Target optimal point")
        else:
            new_labels.append(l)
    
    # Legend below the axes, as for the use-case figures, so it cannot cover data
    ax.legend(handles, new_labels, frameon=False, loc="upper left", ncol=1,
              handlelength=1.1, labelspacing=0.2, handletextpad=0.4,
              columnspacing=0.8, borderaxespad=0.2)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.0 + 0.6 * 2.0 / pareto.FIG_H)   # headroom for the legend and the callout
    
    # Annotate the arrow to show "Improvement potential" and "Goal of SUSTAINET"
    bx, by = points[baseline_idx]
    hx, hy = points[highlight_idx]
    
    mid_x = (bx + hx) / 2
    mid_y = (by + hy) / 2
    
    # The point cloud is confined to [0.1, 1.0] in both axes while the limits run to
    # 1.4, so the upper-right corner is empty and the callout cannot cover any point.
    ax.annotate("improvement\npotential",
                xy=(mid_x, mid_y), xytext=(1.37, 0.80),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8),
                fontsize=7, ha='right', va='center', color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9aa0a6", alpha=0.9))
                
    
    fname = FIG / "conceptual_goal.pdf"
    fig.savefig(fname)
    fig.savefig(str(fname).replace(".pdf", ".png"))
    print(f"Saved {fname}")

if __name__ == "__main__":
    main()
