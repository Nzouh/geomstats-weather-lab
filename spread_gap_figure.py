"""
Standalone figure: the naive (chordal) mean and the geodesic (Frechet) mean diverge as the
data spreads over the sphere. Geometry pillar of the blog.

The naive mean minimizes CHORD distance (a straight line through the sphere's interior); the
Frechet mean minimizes GEODESIC distance (along the surface). For a small patch the two nearly
coincide (a sphere looks flat locally); as the data spreads, the chord short-cuts through the
interior and the two means pull apart. We plot that gap (km) vs the angular spread, and overlay
the three real Tool 1 benchmark cases (tight cluster, continental, global).

Run:  venv/Scripts/python.exe spread_gap_figure.py   ->  writes spread_gap.png
"""

import numpy as np

from tool1_centroid_tracker import (
    latlong_to_sphere, frechet_karcher, naive_chordal, R_EARTH_KM,
)
from tool3_geodesic_pca import exp_map


def geodesic_km(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)) * R_EARTH_KM)


def spread_deg(points, mu):
    return float(np.degrees(np.arccos(np.clip(points @ mu, -1, 1)).max()))


def gap_and_spread(points):
    w = np.ones(len(points))
    muF, _ = frechet_karcher(points, w)
    muC = naive_chordal(points, w)
    return spread_deg(points, muF), geodesic_km(muF, muC)


# fixed ASYMMETRIC shape at the equator, scaled up to sweep the spread continuously
MU0 = latlong_to_sphere(0.0, 0.0)                 # (1,0,0)
EAST = np.array([0.0, 1.0, 0.0])
NORTH = np.array([0.0, 0.0, 1.0])
SHAPE = [(1.0, 0.0), (-0.5, 0.75), (-0.55, -0.45), (0.25, 0.95)]   # asymmetric


def scaled_config(scale):
    vs = np.array([scale * (e * EAST + n * NORTH) for e, n in SHAPE])
    return exp_map(MU0, vs)


# the three real Tool 1 benchmark city sets (reproduces B4)
CITY_SETS = {
    "tight cluster": {"Dallas": (32.78, -96.80), "Houston": (29.76, -95.37),
                      "OKC": (35.47, -97.52)},
    "continental": {"NewYork": (40.71, -74.01), "London": (51.51, -0.13),
                    "Reykjavik": (64.15, -21.94), "Lisbon": (38.72, -9.14)},
    "global spread": {"Quito": (-0.18, -78.47), "Svalbard": (78.22, 15.65),
                      "McMurdo": (-77.85, 166.67), "Singapore": (1.35, 103.82)},
}


def make(png="spread_gap.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # deterministic trend: one fixed asymmetric shape, scaled from tiny to hemispheric
    scales = np.linspace(0.03, 1.55, 90)
    curve = np.array([gap_and_spread(scaled_config(s)) for s in scales])
    xs, ys = curve[:, 0], curve[:, 1]

    refs = {}
    for name, cities in CITY_SETS.items():
        pts = np.array([latlong_to_sphere(la, lo) for la, lo in cities.values()])
        refs[name] = gap_and_spread(pts)
    global_gap = refs["global spread"][1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(0, 20, color="#eef3f7", zorder=0)
    ax.plot(xs, ys, color="#12355b", lw=2.5, zorder=3,
            label="gap vs spread (a typical configuration)")

    # the near-antipodal worst case, as a reference ceiling (not a point on the curve)
    ax.axhline(global_gap, color="#c81e1e", lw=2.0, ls="--", zorder=2)
    ax.text(2, global_gap + 180, f"maximum gap  ~{global_gap:,.0f} km",
            color="#c81e1e", fontsize=10, fontweight="bold", va="bottom")

    for name, col in [("tight cluster", "#1a7a3a"), ("continental", "#e08a00")]:
        sp, gp = refs[name]
        ax.scatter([sp], [gp], color=col, s=110, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"{name}\n{gp:,.0f} km", (sp, gp), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, color=col, fontweight="bold",
                    ha="left", va="bottom")

    ax.text(9.5, 6600, "locally flat:\nnaive ~ geodesic",
            fontsize=9, color="#5b6770", ha="center")

    ax.set_xlabel("angular spread of the data (degrees)", fontsize=11)
    ax.set_ylabel("gap between naive and geodesic mean (km)", fontsize=11)
    ax.set_title("Euclidean is fine locally, wrong globally:\n"
                 "the naive-vs-geodesic gap grows with how far the data spreads",
                 fontsize=13)
    ax.set_xlim(0, 100)
    ax.set_ylim(-300, global_gap * 1.18)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"saved {png}")
    for name, (sp, gp) in refs.items():
        print(f"  {name:<15} spread {sp:5.1f} deg   gap {gp:8.1f} km")


if __name__ == "__main__":
    make()
