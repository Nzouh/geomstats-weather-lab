"""
Tool 3 on REAL data: interannual variability of North American summer heat, 1991-2020.

For each summer we take the ERA5 JJA-mean 2m temperature ANOMALY (that year minus the
30-year climatology), locate the center of the anomalously-hot region with Tool 1
(threshold -> weighted Frechet mean on S2), giving one point per year. Then Tool 3's PGA
finds the dominant geodesic axis of how that center varies across the 30 years.

Anomaly, not absolute temperature: absolute summer heat always centers on the SW deserts
(a near-static centroid). The anomaly centroid captures where each summer was UNUSUALLY hot,
which is the real interannual signal.
"""

import os
import numpy as np
import cdsapi
import xarray as xr

from tool1_centroid_tracker import (
    latlong_to_sphere, sphere_to_latlong, frechet_karcher, hot_weights,
)
from tool3_geodesic_pca import GeodesicPCA, plot_pga, geodesic_km

YEARS = [str(y) for y in range(1991, 2021)]     # 30 summers
AREA = [60, -130, 25, -70]                       # N, W, S, E  (contiguous US + S. Canada)
NC = "era5_jja_monthly_1991_2020.nc"
KEEP_FRACTION = 0.10


def fetch(nc=NC):
    if os.path.exists(nc):
        print(f"using cached {nc}")
        return nc
    print("requesting 30 years of ERA5 monthly means (may queue)...")
    cdsapi.Client().retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": YEARS, "month": ["06", "07", "08"], "time": ["00:00"],
        "data_format": "netcdf", "download_format": "unarchived", "area": AREA,
    }, nc)
    return nc


def load(nc=NC):
    ds = xr.open_dataset(nc)
    tname = "valid_time" if "valid_time" in ds["t2m"].dims else "time"
    t = (ds["t2m"] - 273.15).values                     # (n_months, nlat, nlon), celsius
    yr = np.array([int(str(v)[:4]) for v in ds[tname].values])
    lat = ds["latitude"].values
    lon = ((ds["longitude"].values + 180) % 360) - 180
    LON, LAT = np.meshgrid(lon, lat)
    lats, lons = LAT.ravel(), LON.ravel()
    points = latlong_to_sphere(lats, lons)
    area = np.cos(np.radians(lats))

    years = sorted(set(yr))
    jja = np.array([t[yr == y].mean(0).ravel() for y in years])   # JJA mean per year
    anomaly = jja - jja.mean(0)                                    # minus 30-yr climatology
    return dict(lats=lats, lons=lons, points=points, area=area,
                years=years, anomaly=anomaly)


def yearly_centroids(d):
    centers = []
    for a in d["anomaly"]:
        w = hot_weights(a, d["area"], KEEP_FRACTION)   # hottest-anomaly region that year
        mu, _ = frechet_karcher(d["points"], w)
        centers.append(mu)
    return np.array(centers)


def report(d, centers, pga):
    print(f"\n{'year':<8}{'heat-anomaly center (lat, lon)':<32}")
    for y, c in zip(d["years"], centers):
        la, lo = sphere_to_latlong(c)
        print(f"{y:<8}({la:6.2f}, {lo:7.2f})")

    mla, mlo = pga.mean_latlon
    r1, r2 = 100 * pga.variance_ratio_
    # +/- 2 sigma endpoints of axis 1, as lat/lon, and the span in km
    arc = pga.principal_geodesic(0, 2.0)
    a_end, b_end = arc[0], arc[-1]
    span_km = geodesic_km(a_end, b_end)
    print("\n--- actionable result ---")
    print(f"mean center of anomalous summer heat: ({mla:.1f}, {mlo:.1f})")
    print(f"axis 1 explains {r1:.0f}% of interannual variance (axis 2: {r2:.0f}%)")
    print(f"axis 1 (+/-2 sigma) runs {sphere_to_latlong(a_end)[0]:.1f},"
          f"{sphere_to_latlong(a_end)[1]:.1f}  <->  "
          f"{sphere_to_latlong(b_end)[0]:.1f},{sphere_to_latlong(b_end)[1]:.1f}"
          f"  (span ~{span_km:.0f} km)")


if __name__ == "__main__":
    data = load(fetch())
    centers = yearly_centroids(data)
    pga = GeodesicPCA().fit(centers)
    report(data, centers, pga)
    plot_pga(pga, centers, png="era5_pga_interannual.png",
             title="Interannual variability of N. American summer heat (1991-2020)")
