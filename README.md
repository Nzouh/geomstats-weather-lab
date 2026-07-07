# Weather Data Lives on a Sphere. Our Statistics Should Too.

Geometric statistics for weather data on Earth, built with
[GeomStats](https://geomstats.github.io/) and [ERA5](https://cds.climate.copernicus.eu/):
Frechet means, geodesic distances, and Principal Geodesic Analysis on the sphere, applied to
real temperature fields, with the Euclidean failure modes measured rather than assumed.

**Read the article:** [article.html](article.html) (or the deployed site's front page).

**Reproduce everything:** the companion notebook runs the whole evidence base end to end,
- [view it in the browser](companion_notebook.html) with all outputs, no setup, or
- run [companion_notebook.ipynb](companion_notebook.ipynb) yourself.

## Highlights

- The June 2021 Pacific Northwest heat dome's center, tracked day by day with a weighted
  Frechet mean on $S^2$: it migrated 2,635 km north in seven days.
- Thirty years of summer heat-anomaly centers (1991-2020) analyzed with tangent-space PCA:
  77% of the interannual variance lies along a single, roughly east-west geodesic axis.
- Where the flat shortcut breaks: the naive mean is off by 0 km for a tight cluster, 41 km at
  continental spread, 7,759 km for near-antipodal data, and 17,805 km across the date line.
- Why the geometric statistics never break: they are equivariant under the rotation group
  SO(3) (measured equivariance error 0.0001 km).

Methods are Karcher (1977) and Fletcher et al. (2004), as surveyed in Papillon, Sanborn,
Mathe et al., *Beyond Euclid* (Mach. Learn.: Sci. Technol. 6 031002, 2025;
[arXiv:2407.09468](https://arxiv.org/abs/2407.09468)).

## Running it

Python **3.12** (geomstats 2.8.0 needs `numpy<2.1`, which has no Python 3.13 wheels):

```
pip install -r requirements.txt
```

```
python tool1_centroid_tracker.py        # Frechet-mean benchmarks (B1-B6)
python tool3_geodesic_pca.py            # PGA benchmarks (B1-B6)
python spread_gap_figure.py             # geometry figure
python equivariance_figure.py           # group-theory figure
python era5_heatwave.py                 # heat-dome track + figure
python era5_pga_interannual.py          # 30-year PGA + figure
```

The two ERA5 scripts and the notebook run in **quick mode** against the cached netCDF files
in this repo (`era5_heatdome_2021.nc`, `era5_jja_monthly_1991_2020.nc`), so no API key is
needed. Delete the caches to re-download from the Copernicus Climate Data Store, which
requires a free CDS account, a Personal Access Token in `~/.cdsapirc`, and accepting the ERA5
licence on the dataset page.

## Data attribution

Contains modified Copernicus Climate Change Service information (ERA5). Neither the European
Commission nor ECMWF is responsible for any use of this information. Coastlines from
[Natural Earth](https://www.naturalearthdata.com/) (public domain).

## Author

Nabil Zouhari, 2026.
