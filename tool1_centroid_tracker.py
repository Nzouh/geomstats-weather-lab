"""
Tool 1 - Weather-pattern centroid tracker on S2 (Card S6).

Correctness-first implementation. Every object maps to the paper
(Papillon, Sanborn, Mathe et al. 2025, "Beyond Euclid", Mach. Learn.: Sci. Technol. 6 031002):

  - Data model = Card S6: a signal T : S2 -> R. Domain S2 is the manifold (locations),
    codomain R is Euclidean (temperature). "Threshold classifies, Frechet mean locates."
  - Geodesic distance on S2:  d(p, q) = arccos(<p, q>)   (great-circle angle).
  - Centroid = weighted Frechet mean (figure 6):
        mu = argmin_{p in S2}  sum_i w_i * d(p, p_i)^2
    solved by the Karcher iteration with exp/log maps (the paper's math kernel):
        v = sum_i w_i log_mu(p_i) / sum_i w_i ;  mu <- exp_mu(v) ;  repeat until ||v|| -> 0.

Baselines, each with a precise, distinct failure mode:
  - naive_latlon : average the degrees. Coordinate-dependent; breaks at the dateline.
  - naive_chordal: normalize(sum w_i p_i). On-sphere but minimizes CHORD, not geodesic.
  - frechet      : minimizes GEODESIC distance. Correct.

Discretization correctness: the lat/long grid over-samples the poles, so the discrete sum
approximates the S2 integral only if each cell is weighted by the area element cos(lat)
(the Riemannian volume element). This is applied throughout.
"""

import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

SPHERE = Hypersphere(dim=2)
R_EARTH_KM = 6371.0


# ---------------------------------------------------------------------------
# Coordinate maps (intrinsic lat/long  <->  extrinsic 3D unit vector on S2)
# ---------------------------------------------------------------------------
def latlong_to_sphere(lat_deg, lon_deg):
    """(lat, lon) in degrees -> unit 3D vector(s). Vectorized over arrays."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def sphere_to_latlong(v):
    """Unit 3D vector -> (lat, lon) in degrees. Inverse of latlong_to_sphere."""
    v = np.asarray(v)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, lon


# ---------------------------------------------------------------------------
# Grid + synthetic field (von Mises-Fisher blob with a KNOWN center)
# ---------------------------------------------------------------------------
def make_grid(n_lat=180, n_lon=360):
    """Regular lat/long grid of CELL CENTERS. Returns flat lats, lons, points, area weights."""
    dlat, dlon = 180.0 / n_lat, 360.0 / n_lon
    lat_c = -90.0 + (np.arange(n_lat) + 0.5) * dlat
    lon_c = -180.0 + (np.arange(n_lon) + 0.5) * dlon
    lon_g, lat_g = np.meshgrid(lon_c, lat_c)
    lats = lat_g.ravel()
    lons = lon_g.ravel()
    points = latlong_to_sphere(lats, lons)
    area = np.cos(np.radians(lats))          # spherical volume element (Jacobian)
    return lats, lons, points, area


def vmf_field(points, center_latlon, kappa=40.0):
    """
    von Mises-Fisher blob: T(p) = exp(kappa * <c, p>), c = center direction.
    Rotationally symmetric about c, so its population Frechet mean is EXACTLY c
    (any rotation about the c-axis fixes the distribution; only c and -c are fixed
    points, and c is the minimizer). This gives an analytic ground truth.
    """
    c = latlong_to_sphere(*center_latlon)
    return np.exp(kappa * (points @ c))


def hot_weights(field, area, keep_fraction=0.10):
    """
    Threshold the codomain (classify), then form domain weights (locate).
    Keep the hottest `keep_fraction` of AREA; weight kept cells by excess heat * area.
    Because T depends only on <c, p>, the kept set is a geodesic cap around c -> symmetric.
    """
    order = np.argsort(field)[::-1]
    csum = np.cumsum(area[order])
    tau_idx = np.searchsorted(csum, keep_fraction * area.sum())
    tau = field[order[min(tau_idx, len(order) - 1)]]
    excess = np.clip(field - tau, 0.0, None)
    return excess * area                      # zero outside the hot cap


# ---------------------------------------------------------------------------
# Centroid estimators
# ---------------------------------------------------------------------------
def frechet_karcher(points, weights, tol=1e-14, max_iter=500):
    """
    Weighted Frechet mean via the Karcher iteration, written from the definition.
    Returns (mu, residual) where residual = ||sum_i w_i log_mu(p_i)|| is the gradient
    of the Frechet functional -> must be ~0 at a stationary point (benchmark B3).
    """
    w = weights / weights.sum()
    mu = points.T @ w
    mu = mu / np.linalg.norm(mu)              # init: chordal mean
    residual = np.inf
    for _ in range(max_iter):
        cos_t = np.clip(points @ mu, -1.0, 1.0)
        theta = np.arccos(cos_t)              # geodesic distances to mu
        perp = points - cos_t[:, None] * mu   # component orthogonal to mu
        norm_perp = np.linalg.norm(perp, axis=1)
        safe = norm_perp > 1e-15
        logs = np.zeros_like(points)          # log_mu(p_i); 0 where p_i == mu
        logs[safe] = (theta[safe] / norm_perp[safe])[:, None] * perp[safe]
        v = logs.T @ w                        # weighted mean of log maps (tangent)
        residual = np.linalg.norm(v)
        if residual < tol:
            break
        nv = residual
        mu = np.cos(nv) * mu + np.sin(nv) * (v / nv)   # exp_mu(v)
    return mu, residual


def frechet_geomstats(points, weights, epsilon=1e-14, max_iter=1000):
    """
    Weighted Frechet mean via GeomStats (library cross-check for B2).
    GeomStats' default optimizer stops at epsilon=1e-4 / max_iter=32, which leaves it
    ~4e-4 rad (~2.7 km) short of the true minimum. Tighten it so the library and the
    from-definition Karcher agree, confirming both target the same stationary point.
    """
    fm = FrechetMean(SPHERE)
    fm.optimizer.epsilon = epsilon
    fm.optimizer.max_iter = max_iter
    fm.fit(points, weights=weights)
    return fm.estimate_


def naive_chordal(points, weights):
    """normalize(sum w_i p_i): on-sphere, but minimizes chordal (Euclidean) distance."""
    m = points.T @ (weights / weights.sum())
    return m / np.linalg.norm(m)


def naive_latlon(lats, lons, weights):
    """Weighted average of degrees. Coordinate-dependent; breaks at the dateline."""
    w = weights / weights.sum()
    return float(np.sum(w * lats)), float(np.sum(w * lons))


def geodesic_km(a_latlon, b_latlon):
    """Great-circle distance between two (lat, lon) points, in km."""
    a = latlong_to_sphere(*a_latlon)
    b = latlong_to_sphere(*b_latlon)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)) * R_EARTH_KM)


# ---------------------------------------------------------------------------
# Public API surface (Surface 1 from DESIGN.md)
# ---------------------------------------------------------------------------
class CentroidTracker:
    """Track a field's hot-region centroid across days via the Frechet mean on S2."""

    def __init__(self, keep_fraction=0.10):
        self.keep_fraction = keep_fraction

    def centroid(self, field, lats, lons, area):
        w = hot_weights(field, area, self.keep_fraction)
        mu, _ = frechet_karcher(latlong_to_sphere(lats, lons), w)
        return sphere_to_latlong(mu)          # (lat, lon)

    def fit(self, fields, lats, lons, area):
        centroids = [self.centroid(f, lats, lons, area) for f in fields]
        migration = [geodesic_km(centroids[d], centroids[d + 1])
                     for d in range(len(centroids) - 1)]
        return {
            "centroids_latlon": centroids,
            "migration_km": migration,
            "total_path_km": float(np.sum(migration)),
        }


# ---------------------------------------------------------------------------
# Visualization (Surface 2, Option B: the manifold itself, a 3D globe)
# ---------------------------------------------------------------------------
def load_coastlines(cache="ne_110m_coastline.geojson"):
    """
    Return world coastlines as a list of (N,2) lon/lat arrays.
    Downloads a small Natural Earth file once and caches it locally (no cartopy).
    Returns [] if offline so the figure still renders (just without coastlines).
    """
    import json
    import os
    import urllib.request
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_coastline.geojson")
    try:
        if not os.path.exists(cache):
            urllib.request.urlretrieve(url, cache)
        with open(cache, "r", encoding="utf-8") as fh:
            gj = json.load(fh)
    except Exception as exc:                     # offline / URL moved: degrade gracefully
        print(f"  (coastlines unavailable: {exc}; drawing bare globe)")
        return []
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        parts = ([geom["coordinates"]] if geom["type"] == "LineString"
                 else geom["coordinates"])
        for part in parts:
            lines.append(np.array(part))         # columns are [lon, lat]
    return lines


def geodesic_arc(a, b, n=40):
    """Sample the great-circle (geodesic) from sphere point a to b via slerp."""
    theta = np.arccos(np.clip(a @ b, -1.0, 1.0))
    if theta < 1e-9:
        return np.array([a, b])
    t = np.linspace(0.0, 1.0, n)
    return (np.sin((1 - t)[:, None] * theta) * a
            + np.sin(t[:, None] * theta) * b) / np.sin(theta)


def plot_track(png="tool1_track.png"):
    """Clean concept figure: a heat wave's center tracked across the globe."""
    import matplotlib
    matplotlib.use("Agg")                       # no window; just write a file
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D      # noqa: F401  (enables 3d projection)

    lats, lons, pts, area = make_grid(120, 240)
    path_latlon = [(20.0 + 3.0 * d, -120.0 + 6.0 * d) for d in range(10)]
    fields = [vmf_field(pts, c, kappa=40.0) for c in path_latlon]
    out = CentroidTracker(keep_fraction=0.10).fit(fields, lats, lons, area)
    centers = np.array([latlong_to_sphere(la, lo)
                        for la, lo in out["centroids_latlon"]])

    elev, azim = 22.0, -100.0
    cam = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                    np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                    np.sin(np.radians(elev))])  # camera direction (for near-side culling)

    # base globe: light "ocean" at low temperature -> warm at the hot core,
    # so most of the sphere reads as a pale globe with one clear red hot spot.
    ocean = LinearSegmentedColormap.from_list(
        "ocean_heat", ["#dce6ee", "#f4d58d", "#e8894a", "#b30000"])
    m = 260
    LON, LAT = np.meshgrid(np.linspace(-np.pi, np.pi, 2 * m),
                           np.linspace(-np.pi / 2, np.pi / 2, m))
    X = np.cos(LAT) * np.cos(LON)
    Y = np.cos(LAT) * np.sin(LON)
    Z = np.sin(LAT)
    # paint where the heat wave passed over ALL days (max over the daily blobs),
    # so the path runs down the spine of the hot corridor instead of dangling off
    # a single day's blob.
    heat = np.zeros_like(X)
    for la, lo in path_latlon:
        cc = latlong_to_sphere(la, lo)
        heat = np.maximum(heat, np.exp(40.0 * (X * cc[0] + Y * cc[1] + Z * cc[2])))
    heat = ((heat - heat.min()) / (heat.max() - heat.min())) ** 2

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=ocean(heat), rcount=m, ccount=2 * m,
                    linewidth=0, antialiased=True, shade=False, zorder=0)

    # --- coastlines on the near hemisphere only (so far side does not bleed) ---
    for line in load_coastlines():
        xyz = latlong_to_sphere(line[:, 1], line[:, 0]) * 1.003
        near = xyz @ cam > 0.02
        seg = []
        for point, vis in zip(xyz, near):
            if vis:
                seg.append(point)
            elif seg:
                s = np.array(seg); ax.plot(s[:, 0], s[:, 1], s[:, 2],
                                           color="#5b6770", lw=0.6, zorder=3)
                seg = []
        if seg:
            s = np.array(seg); ax.plot(s[:, 0], s[:, 1], s[:, 2],
                                       color="#5b6770", lw=0.6, zorder=3)

    # --- one clean path along the surface ---
    arcs = np.vstack([geodesic_arc(centers[i], centers[i + 1])
                      for i in range(len(centers) - 1)]) * 1.01
    ax.plot(arcs[:, 0], arcs[:, 1], arcs[:, 2], color="#12355b", lw=2.5, zorder=6)
    ax.scatter(centers[:, 0] * 1.01, centers[:, 1] * 1.01, centers[:, 2] * 1.01,
               color="#12355b", s=22, depthshade=False, zorder=7)

    # label the start (text only; a scatter marker here re-sorts 3D depth and hides
    # the coastlines, so we avoid it)
    lab = latlong_to_sphere(path_latlon[0][0], path_latlon[0][1] - 8) * 1.12
    ax.text(lab[0], lab[1], lab[2], "Day 1", color="#12355b",
            fontsize=12, fontweight="bold", ha="center", zorder=9)

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-0.72, 0.72)                        # zoom in, kill the white margin
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("Where is the heat centered, and how does it move?\n"
                 "A heat wave's center tracked across the globe over 10 days",
                 fontsize=13)
    # tiny plain-language caption instead of a jargon legend
    ax.text2D(0.5, 0.02,
              "red = where the heat wave passed over 10 days   |   "
              "navy line = its center, day by day",
              transform=ax.transAxes, ha="center", fontsize=10, color="#333333")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}")


# ---------------------------------------------------------------------------
# Benchmarks (each proves a specific mathematical property; see table in chat)
# ---------------------------------------------------------------------------
def _line(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<28} {detail}")


def bench():
    lats, lons, pts, area = make_grid(180, 360)   # 1-degree grid

    print("B1  Recover a vMF blob's known center (analytic ground truth)")
    truth = (35.0, -100.0)
    field = vmf_field(pts, truth, kappa=40.0)
    w = hot_weights(field, area, 0.10)
    mu, res = frechet_karcher(pts, w)
    got = sphere_to_latlong(mu)
    err = geodesic_km(truth, got)
    _line("frechet recovers center", err < 20,
          f"truth={truth} got=({got[0]:.3f},{got[1]:.3f}) err={err:.2f} km")

    print("\nB2  Hand-rolled Karcher == GeomStats FrechetMean")
    mu_gs = frechet_geomstats(pts, w)
    agree = float(np.arccos(np.clip(mu @ mu_gs, -1, 1)) * R_EARTH_KM)
    _line("karcher vs geomstats", agree < 1e-3, f"geodesic gap = {agree:.2e} km")

    print("\nB3  First-order optimality: ||sum w_i log_mu(p_i)|| ~ 0")
    _line("gradient residual ~ 0", res < 1e-8, f"residual = {res:.2e}")

    print("\nB4  Chordal (naive) vs geodesic (Frechet) gap grows with SPREAD")
    print("      (a symmetric blob has gap 0 by symmetry - local flatness; the gap")
    print("       appears only for asymmetric/spread data, as in figure 6 / hello_sphere)")
    datasets = [
        ("tight cluster", {"Dallas": (32.78, -96.80), "Houston": (29.76, -95.37),
                           "OKC": (35.47, -97.52)}),
        ("continental", {"NewYork": (40.71, -74.01), "London": (51.51, -0.13),
                         "Reykjavik": (64.15, -21.94), "Lisbon": (38.72, -9.14)}),
        ("global spread", {"Quito": (-0.18, -78.47), "Svalbard": (78.22, 15.65),
                           "McMurdo": (-77.85, 166.67), "Singapore": (1.35, 103.82)}),
    ]
    for name, cities in datasets:
        cpts = np.array([latlong_to_sphere(la, lo) for la, lo in cities.values()])
        ww = np.ones(len(cpts))
        muf, _ = frechet_karcher(cpts, ww)
        muc = naive_chordal(cpts, ww)
        spread = float(np.arccos(np.clip(
            (cpts @ muf).min(), -1, 1)) * np.degrees(1))   # ang. radius, deg
        gap = float(np.arccos(np.clip(muf @ muc, -1, 1)) * R_EARTH_KM)
        print(f"      {name:<15} spread_radius={spread:5.1f} deg  "
              f"chordal-vs-geodesic gap = {gap:8.1f} km")

    print("\nB5  Dateline + SO(3) equivariance")
    truth_dl = (10.0, 180.0)                       # blob straddling the +/-180 seam
    f = vmf_field(pts, truth_dl, kappa=40.0)
    ww = hot_weights(f, area, 0.10)
    muf, _ = frechet_karcher(pts, ww)
    err_f = geodesic_km(truth_dl, sphere_to_latlong(muf))
    ll = naive_latlon(lats, lons, ww)
    err_ll = geodesic_km(truth_dl, ll)
    _line("frechet ok at dateline", err_f < 20, f"err = {err_f:.2f} km")
    _line("naive latlon FAILS", err_ll > 1000,
          f"err = {err_ll:.0f} km  (avg lon -> {ll[1]:.1f} deg, wrong side)")
    # equivariance: relabel the longitude origin by +70 deg, recompute, undo the shift.
    lons2 = ((lons + 70 + 180) % 360) - 180
    pts2 = latlong_to_sphere(lats, lons2)
    muf2, _ = frechet_karcher(pts2, ww)
    lat2, lon2 = sphere_to_latlong(muf2)
    back = geodesic_km(sphere_to_latlong(muf), (lat2, ((lon2 - 70 + 180) % 360) - 180))
    _line("SO(3)-equivariant", back < 1e-3, f"invariance error = {back:.2e} km")

    print("\nB6  Recover a MOVING planted trajectory (10 days)")
    tracker = CentroidTracker(keep_fraction=0.10)
    truth_path = [(20.0 + 3.0 * d, -120.0 + 6.0 * d) for d in range(10)]
    fields = [vmf_field(pts, c, kappa=40.0) for c in truth_path]
    out = tracker.fit(fields, lats, lons, area)
    errs = [geodesic_km(t, g) for t, g in zip(truth_path, out["centroids_latlon"])]
    _line("trajectory recovered", max(errs) < 20,
          f"max_err={max(errs):.2f} km  total_path={out['total_path_km']:.0f} km")


if __name__ == "__main__":
    import sys
    bench()
    if "--plot" in sys.argv:
        print()
        plot_track()
