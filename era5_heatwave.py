"""
Tool 1 on REAL data: track a heat wave's centroid with ERA5 2m temperature.

Demo event: the June 2021 Pacific Northwest "heat dome" (24-30 June 2021), one of
the most extreme heat events on record. We pull ERA5 2m temperature over a North
America window, then reuse the Tool 1 engine (threshold -> weighted Frechet mean on S2)
to locate and track the hot region's center day by day.

On real data there is no planted ground truth, so instead of recovered-vs-truth we show
Frechet vs the naive lat/long centroid, and report their daily gap in km.
"""

import os
import numpy as np
import cdsapi
import xarray as xr

from tool1_centroid_tracker import (
    latlong_to_sphere, geodesic_arc, load_coastlines,
    frechet_karcher, hot_weights, naive_latlon, geodesic_km,
)

EVENT = dict(year="2021", month="06",
             days=["24", "25", "26", "27", "28", "29", "30"],
             time="21:00",                      # ~afternoon local, peak heat
             area=[62, -140, 30, -95])          # N, W, S, E  (Pacific NW + W. Canada)
NC = "era5_heatdome_2021.nc"
KEEP_FRACTION = 0.10


def fetch(nc=NC):
    """Download the ERA5 field once; reuse the cached netCDF afterwards."""
    if os.path.exists(nc):
        print(f"using cached {nc}")
        return nc
    print("requesting ERA5 (may queue for a bit)...")
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [EVENT["year"]], "month": [EVENT["month"]],
        "day": EVENT["days"], "time": [EVENT["time"]],
        "data_format": "netcdf", "download_format": "unarchived",
        "area": EVENT["area"],
    }, nc)
    return nc


def load(nc=NC):
    """Load ERA5 netCDF into the arrays the Tool 1 engine expects."""
    ds = xr.open_dataset(nc)
    tname = "valid_time" if "valid_time" in ds["t2m"].dims else "time"
    t2m = ds["t2m"] - 273.15                     # kelvin -> celsius
    lat = ds["latitude"].values
    lon = ((ds["longitude"].values + 180) % 360) - 180   # -> [-180, 180]
    LON, LAT = np.meshgrid(lon, lat)
    lats, lons = LAT.ravel(), LON.ravel()
    points = latlong_to_sphere(lats, lons)
    area = np.cos(np.radians(lats))              # spherical area element
    fields = [t2m.isel({tname: i}).values.ravel()
              for i in range(t2m.sizes[tname])]
    labels = [str(v)[:10] for v in ds[tname].values]
    return dict(lats=lats, lons=lons, points=points, area=area,
                fields=fields, labels=labels, shape=LAT.shape, lat=lat, lon=lon)


def track(data):
    """Per-day Frechet centroid vs naive lat/long centroid, with the daily gap."""
    rows = []
    for f, label in zip(data["fields"], data["labels"]):
        w = hot_weights(f, data["area"], KEEP_FRACTION)
        mu, _ = frechet_karcher(data["points"], w)
        fla = np.degrees(np.arcsin(np.clip(mu[2], -1, 1)))
        flo = np.degrees(np.arctan2(mu[1], mu[0]))
        nla, nlo = naive_latlon(data["lats"], data["lons"], w)
        gap = geodesic_km((fla, flo), (nla, nlo))
        rows.append(dict(day=label, frechet=(fla, flo), naive=(nla, nlo), gap_km=gap))
    return rows


def print_report(rows):
    print(f"\n{'day':<12}{'Frechet (lat,lon)':<24}{'naive (lat,lon)':<24}{'gap km':>8}")
    for r in rows:
        print(f"{r['day']:<12}({r['frechet'][0]:6.2f},{r['frechet'][1]:7.2f})     "
              f"({r['naive'][0]:6.2f},{r['naive'][1]:7.2f})     {r['gap_km']:6.1f}")
    fpath = [r["frechet"] for r in rows]
    total = sum(geodesic_km(fpath[i], fpath[i + 1]) for i in range(len(fpath) - 1))
    print(f"\ntotal Frechet centroid migration: {total:.0f} km over {len(rows)} days")


def plot(data, rows, png="era5_heatdome.png"):
    """Clean globe: real ERA5 heat patch + the tracked centroid path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D      # noqa: F401

    # hottest display day (highest 95th-percentile temperature)
    didx = int(np.argmax([np.percentile(f, 95) for f in data["fields"]]))
    T = data["fields"][didx].reshape(data["shape"])
    LON, LAT = np.meshgrid(data["lon"], data["lat"])
    Xr = np.cos(np.radians(LAT)) * np.cos(np.radians(LON))
    Yr = np.cos(np.radians(LAT)) * np.sin(np.radians(LON))
    Zr = np.sin(np.radians(LAT))
    vmin, vmax = np.percentile(T, 5), np.percentile(T, 99)
    norm = np.clip((T - vmin) / (vmax - vmin), 0, 1)

    # aim the camera at the analysis window's center and zoom in, so the region
    # fills the frame instead of sitting as a small patch on a full globe
    elev, azim = 45.0, -116.0
    cam = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                    np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                    np.sin(np.radians(elev))])

    fig = plt.figure(figsize=(8, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False                  # respect manual zorder (labels vs map)

    # pale base globe
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, np.pi, 45)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)),
                    np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)),
                    color="#dce6ee", linewidth=0, antialiased=True,
                    shade=False, zorder=0)
    # real ERA5 temperature patch, sitting just above the surface
    ax.plot_surface(Xr * 1.002, Yr * 1.002, Zr * 1.002,
                    facecolors=cm.YlOrRd(norm), rcount=T.shape[0], ccount=T.shape[1],
                    linewidth=0, antialiased=True, shade=False, zorder=1)
    # coastlines (near side only)
    for line in load_coastlines():
        xyz = latlong_to_sphere(line[:, 1], line[:, 0]) * 1.004
        near = xyz @ cam > 0.02
        seg = []
        for p, vis in zip(xyz, near):
            if vis:
                seg.append(p)
            elif seg:
                s = np.array(seg)
                ax.plot(s[:, 0], s[:, 1], s[:, 2], color="#5b6770", lw=0.6, zorder=3)
                seg = []
        if seg:
            s = np.array(seg)
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color="#5b6770", lw=0.6, zorder=3)
    # tracked centroid path
    centers = np.array([latlong_to_sphere(*r["frechet"]) for r in rows])
    arcs = np.vstack([geodesic_arc(centers[i], centers[i + 1])
                      for i in range(len(centers) - 1)]) * 1.01
    ax.plot(arcs[:, 0], arcs[:, 1], arcs[:, 2], color="#12355b", lw=2.5, zorder=6)
    ax.scatter(centers[:, 0] * 1.01, centers[:, 1] * 1.01, centers[:, 2] * 1.01,
               color="#12355b", s=22, depthshade=False, zorder=7)
    # Day 1 / Day N labels (text only, so they do not re-sort the map)
    n_days = len(rows)
    for idx, txt in [(0, "Day 1"), (n_days - 1, f"Day {n_days}")]:
        lab = centers[idx] * 1.13
        ax.text(lab[0], lab[1], lab[2], txt, color="#12355b", fontsize=11,
                fontweight="bold", ha="center", zorder=9)

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    # zoom: clamp the axes box around the region's center on the sphere
    zoom = 0.40
    ax.set_xlim(cam[0] * 0.62 - zoom, cam[0] * 0.62 + zoom)
    ax.set_ylim(cam[1] * 0.62 - zoom, cam[1] * 0.62 + zoom)
    ax.set_zlim(cam[2] * 0.62 - zoom, cam[2] * 0.62 + zoom)
    ax.view_init(elev=elev, azim=azim)

    fig.suptitle("Tracking a Heat Wave's Center with Geodesics, Not Euclidean Distance",
                 fontsize=14, y=0.975)
    fig.text(0.5, 0.925,
             "A Frechet mean on the sphere follows the hot region day by day, "
             "independent of map coordinates",
             ha="center", fontsize=10, style="italic", color="#333333")
    fig.text(0.5, 0.045,
             f"June 2021 Pacific Northwest heat dome (ERA5 window, field shown on "
             f"{data['labels'][didx]}).\n"
             "Hot cells are selected from the temperature field; their sphere-correct "
             "center is the navy path.\nBecause the center is SO(3)-equivariant, rotating "
             "the globe rotates the answer instead of changing it.",
             ha="center", fontsize=9.5, color="#333333")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}")


if __name__ == "__main__":
    d = load(fetch())
    r = track(d)
    print_report(r)
    plot(d, r)
