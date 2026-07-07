# GeomStats represents a point on the sphere S2 as a 3D unit vector

#So we have to convert (lat, lon) in degrees -> [x, y, z]

import math
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
import numpy as np

def latlong_to_sphere(lat_deg, lon_deg):
    x = np.cos(np.radians(lat_deg)) * np.cos(np.radians(lon_deg))
    y = np.cos(np.radians(lat_deg)) * np.sin(np.radians(lon_deg))
    z = np.sin(np.radians(lat_deg))

    return [x, y, z]


def compare(name, cities):
    sphere = Hypersphere(dim=2)
    # cities.items() gives (key, value) pairs. The value is a (lat, lon) tuple,
    # so "name_of_city, (lat, lon)" unpacks both at once in the loop header.
    points = np.array([
        latlong_to_sphere(lat, lon)
        for city, (lat, lon) in cities.items()
    ])

    fm = FrechetMean(sphere)
    fm.fit(points)
    frechet = fm.estimate_
    # Naive mean — average the raw vectors, then renormalize back onto the sphere.
    raw_avg = np.mean(points, axis=0)
    naive = raw_avg / np.linalg.norm(raw_avg)

    # Geodesic gap between the two answers, converted to km.
    gap_rad = sphere.metric.dist(frechet, naive)
    gap_km = gap_rad * 6371

    print(f"{name}: gap = {gap_km:.1f} km  (raw avg length was {np.linalg.norm(raw_avg):.4f})")
    return gap_km
    

    


texas  = {"Dallas": (32.78, -96.80), "Houston": (29.76, -95.37), "OKC": (35.47, -97.52)}
spread = {"Quito": (-0.18, -78.47), "Svalbard": (78.22, 15.65), "McMurdo": (-77.85, 166.67), "Singapore": (1.35, 103.82)}

compare("Texas", texas)
compare("Spread", spread)
