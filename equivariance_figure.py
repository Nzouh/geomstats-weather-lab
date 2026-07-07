"""
Standalone figure: the Frechet mean is SO(3)-equivariant; the naive lat/long mean is not.

Group-theory (algebra) pillar of the blog. We take a cluster of points on S2, compute its
Frechet mean and the naive lat/long mean, then apply a rotation R in SO(3) (spin the globe).
Recomputing on the rotated data:
    Frechet mean  ->  exactly R * (original mean)      (equivariant)
    naive lat/long mean  ->  drifts, because lat/long is a coordinate frame that distorts.

Run:  venv/Scripts/python.exe equivariance_figure.py   ->  writes equivariance.png
"""

import numpy as np

from tool1_centroid_tracker import (
    latlong_to_sphere, sphere_to_latlong, frechet_karcher, naive_latlon,
    load_coastlines, R_EARTH_KM,
)
from tool3_geodesic_pca import planted_cloud


def rodrigues(axis, angle):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def geodesic_km(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)) * R_EARTH_KM)


def means(cluster):
    """Frechet (geodesic) mean and naive lat/long mean of a cluster on S2."""
    w = np.ones(len(cluster))
    mu, _ = frechet_karcher(cluster, w)
    lat, lon = sphere_to_latlong(cluster)
    nlat, nlon = naive_latlon(lat, lon, w)
    return mu, latlong_to_sphere(nlat, nlon)


def draw_globe(ax, elev, azim, cloud, mean_geo, mean_naive):
    ax.computed_zorder = False
    cam = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                    np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                    np.sin(np.radians(elev))])
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)), color="#dce6ee",
                    linewidth=0, antialiased=True, shade=False, zorder=0)
    for line in load_coastlines():
        xyz = latlong_to_sphere(line[:, 1], line[:, 0]) * 1.002
        near = xyz @ cam > 0.02
        seg = []
        for p, vis in zip(xyz, near):
            if vis:
                seg.append(p)
            elif seg:
                s = np.array(seg); ax.plot(s[:, 0], s[:, 1], s[:, 2],
                                           color="#8a97a0", lw=0.6, zorder=1); seg = []
        if seg:
            s = np.array(seg); ax.plot(s[:, 0], s[:, 1], s[:, 2],
                                       color="#8a97a0", lw=0.6, zorder=1)
    C = cloud * 1.006
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], color="#4a6fa5", s=14, alpha=0.6,
               depthshade=False, zorder=3)
    mg = mean_geo * 1.02
    ax.scatter(*mg, color="#1a7a3a", s=460, marker="*", edgecolors="white",
               linewidths=0.8, depthshade=False, zorder=6)
    mn = mean_naive * 1.02
    ax.scatter(*mn, color="#c81e1e", s=90, marker="X", edgecolors="white",
               linewidths=0.8, depthshade=False, zorder=6)
    ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-0.7, 0.7)
    ax.view_init(elev=elev, azim=azim)


def make(png="equivariance.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D      # noqa: F401

    # a compact cluster at mid-latitude (naive works here)
    cluster, _, _ = planted_cloud((22, -100), (22, -55), n=60,
                                  sigma_along=0.11, sigma_perp=0.10, seed=4)
    muL, nL = means(cluster)

    # rotate the globe: tilt the cluster up toward the pole (a rotation in SO(3))
    lam = np.radians(-100.0)
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])   # east axis at lon -100
    R = rodrigues(east, np.radians(-64.0))              # tilt cluster up to ~86 N (near pole)
    clusterR = cluster @ R.T
    muR, nR = means(clusterR)
    ring = R @ muL                                       # predicted equivariant mean

    equiv_err = geodesic_km(muR, ring)
    drift_L = geodesic_km(nL, muL)
    drift_R = geodesic_km(nR, muR)
    latR, lonR = sphere_to_latlong(muR)
    print(f"left  mean (lat,lon):  Frechet {tuple(np.round(sphere_to_latlong(muL),1))}"
          f"   naive drift {drift_L:.0f} km")
    print(f"right mean (lat,lon):  Frechet ({latR:.1f},{lonR:.1f})   naive drift {drift_R:.0f} km")
    print(f"equivariance error  |mu(R.X) - R.mu(X)| = {equiv_err:.5f} km")

    fig = plt.figure(figsize=(13, 6.6))
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    draw_globe(axL, 18, -100, cluster, muL, nL)
    draw_globe(axR, 66, -108, clusterR, muR, nR)

    fig.suptitle("The Frechet mean is SO(3)-equivariant; the naive lat/long mean is not",
                 fontsize=15, y=0.99)
    fig.text(0.27, 0.9, "Near equator: locally flat", ha="center", fontsize=12)
    fig.text(0.73, 0.9, "After rotating the globe by R (a rotation in SO(3))",
             ha="center", fontsize=12)
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a7a3a",
               markersize=16, label="Frechet mean (geodesic)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#c81e1e",
               markersize=11, label="naive lat/long mean"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a6fa5",
               markersize=9, label="weather data point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.05))
    fig.text(0.5, 0.005,
             f"The Frechet mean transforms with the globe: it stays centered in the cluster "
             f"(equivariance error {equiv_err:.4f} km). "
             f"The naive lat/long mean drifts {drift_R:.0f} km off the cluster near the pole.",
             ha="center", fontsize=10, color="#333333")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12, wspace=0.02)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"saved {png}")


if __name__ == "__main__":
    make()
