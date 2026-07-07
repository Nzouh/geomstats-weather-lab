"""
Tool 3 - Principal Geodesic Analysis (PGA) of locations on S2.

Tool 1 finds WHERE a pattern is; Tool 3 finds the dominant DIRECTIONS and AMOUNT of how that
position varies - correctly on the sphere, not on distortion-prone lat/long.

Method = tangent-space PCA at the Frechet mean (paper figure 8, row 2; Fletcher et al. 2004,
Principal Geodesic Analysis). Paper = Papillon, Sanborn, Mathe et al. 2025, "Beyond Euclid".

Math contract (see TOOL3_DESIGN.md):
  1. mu   = Frechet mean of the points            (paper figure 6; reused from Tool 1)
  2. u_i  = log_mu(p_i)  in the tangent plane      (exp/log kernel)
  3. C    = (1/N) sum u_i u_i^T                     (3x3, rank <= 2)
  4. eig  C = sum lambda_k e_k e_k^T               (tangent PCA)
  5. gamma_k(t) = exp_mu(t e_k)                     (principal geodesics)
  6. score s_ik = <u_i, e_k>                        (signed geodesic distance along axis k)
  7. ratio r_k  = lambda_k / sum lambda            (explained variance)

Honest note: this is tangent PCA (a first-order linearization at mu), not exact PGA; the two
coincide as the cloud concentrates. We report the cloud's angular spread and cross-check the
hand-rolled result against GeomStats TangentPCA (B2).
"""

import numpy as np
from geomstats.learning.pca import TangentPCA

from tool1_centroid_tracker import (
    SPHERE, R_EARTH_KM, latlong_to_sphere, sphere_to_latlong,
    frechet_karcher, geodesic_arc,
)


# ---------------------------------------------------------------------------
# exp / log maps (from the definition; the kernel PGA is built on)
# ---------------------------------------------------------------------------
def log_map(mu, points):
    """log_mu(p): tangent vectors at mu pointing toward each p, length = geodesic dist."""
    pts = np.atleast_2d(points)
    cos_t = np.clip(pts @ mu, -1.0, 1.0)
    theta = np.arccos(cos_t)
    perp = pts - cos_t[:, None] * mu
    norm = np.linalg.norm(perp, axis=1)
    out = np.zeros_like(pts)
    safe = norm > 1e-15
    out[safe] = (theta[safe] / norm[safe])[:, None] * perp[safe]
    return out


def exp_map(mu, v):
    """exp_mu(v): push a tangent vector back onto the sphere."""
    v = np.atleast_2d(v)
    nv = np.linalg.norm(v, axis=1)
    out = np.tile(mu, (len(v), 1)).astype(float)
    safe = nv > 1e-15
    out[safe] = (np.cos(nv[safe])[:, None] * mu
                 + np.sin(nv[safe])[:, None] * v[safe] / nv[safe][:, None])
    return out


# ---------------------------------------------------------------------------
# Geodesic PCA (public API surface)
# ---------------------------------------------------------------------------
class GeodesicPCA:
    """Tangent-space PCA of points on S2, at their Frechet mean."""

    def fit(self, points, weights=None):
        points = np.asarray(points, dtype=float)
        w = np.ones(len(points)) if weights is None else np.asarray(weights)
        self.mean_, _ = frechet_karcher(points, w)
        u = log_map(self.mean_, points)                 # (N,3) tangent vectors
        C = (u.T * (w / w.sum())) @ u                   # weighted 3x3 tangent covariance
        vals, vecs = np.linalg.eigh(C)                  # ascending
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        self.eigenvalues_ = vals[:2]                    # 3rd ~ 0 (along mu)
        self.axes_ = vecs[:, :2].T                      # e_1, e_2 as rows
        self.variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        self.scores_ = u @ self.axes_.T                 # (N,2) signed distances
        # angular spread of the cloud (radians) -> tells us the linearization regime
        self.spread_rad_ = float(np.linalg.norm(u, axis=1).max())
        self._u = u
        return self

    @property
    def mean_latlon(self):
        return sphere_to_latlong(self.mean_)

    def principal_geodesic(self, k=0, n_sigma=2.0, n=60):
        """Arc of axis k, spanning +/- n_sigma along that principal geodesic."""
        L = n_sigma * np.sqrt(self.eigenvalues_[k])
        a = exp_map(self.mean_, -L * self.axes_[k])[0]
        b = exp_map(self.mean_, L * self.axes_[k])[0]
        mid = geodesic_arc(a, self.mean_, n // 2)
        return np.vstack([mid, geodesic_arc(self.mean_, b, n // 2)])

    def confidence_ellipse(self, n_sigma=2.0, n=120):
        """Closed geodesic ellipse at n_sigma (the normal envelope)."""
        th = np.linspace(0, 2 * np.pi, n)
        s1, s2 = np.sqrt(self.eigenvalues_)
        w = (n_sigma * (np.cos(th)[:, None] * s1 * self.axes_[0]
                        + np.sin(th)[:, None] * s2 * self.axes_[1]))
        return exp_map(self.mean_, w)

    def reconstruct(self, n_components=1):
        """Rebuild points from the top n_components scores (for the reduction benchmark)."""
        u_hat = self.scores_[:, :n_components] @ self.axes_[:n_components]
        return exp_map(self.mean_, u_hat)


def geodesic_km(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)) * R_EARTH_KM)


# ---------------------------------------------------------------------------
# Figure (Surface 2): principal geodesic axes on a clean globe + scree bar
# ---------------------------------------------------------------------------
def plot_pga(pga, points, png="tool3_pga.png",
             title="How a feature's location varies: principal geodesic axes"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D      # noqa: F401
    from tool1_centroid_tracker import load_coastlines

    elev, azim = 30.0, -100.0
    cam = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                    np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                    np.sin(np.radians(elev))])
    r1, r2 = 100 * pga.variance_ratio_

    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False                  # respect manual zorder (fixes 3D depth bug)

    # pale base globe
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, np.pi, 45)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)), color="#dce6ee",
                    linewidth=0, antialiased=True, shade=False, zorder=0)
    # coastlines (near side)
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
    # the point cloud
    P = points * 1.006
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="#4a6fa5", s=12, alpha=0.55,
               depthshade=False, zorder=3)
    # 2-sigma confidence ellipse (the normal envelope)
    el = pga.confidence_ellipse(2.0) * 1.008
    ax.plot(el[:, 0], el[:, 1], el[:, 2], color="#12355b", lw=1.3, ls="--", zorder=4)
    # principal geodesic axes
    a1 = pga.principal_geodesic(0, 2.0) * 1.01
    a2 = pga.principal_geodesic(1, 2.0) * 1.01
    ax.plot(a1[:, 0], a1[:, 1], a1[:, 2], color="#b30000", lw=3.0, zorder=5)
    ax.plot(a2[:, 0], a2[:, 1], a2[:, 2], color="#1a7a3a", lw=2.0, zorder=5)
    # Frechet mean
    mns = pga.mean_ * 1.012
    ax.scatter(*mns, color="#111111", s=45, depthshade=False, zorder=6)
    # axis labels (text only)
    for arc, txt, col in [(a1, f"axis 1: {r1:.0f}%", "#b30000"),
                          (a2, f"axis 2: {r2:.0f}%", "#1a7a3a")]:
        t = arc[-1] * 1.13
        ax.text(t[0], t[1], t[2], txt, color=col, fontsize=11,
                fontweight="bold", ha="center", zorder=7)

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-0.72, 0.72)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title + "\n"
                 f"mean at ({pga.mean_latlon[0]:.0f}, {pga.mean_latlon[1]:.0f}); "
                 f"axis 1 explains {r1:.0f}% of the spread", fontsize=12)

    # scree bar: variance split at a glance
    iax = fig.add_axes([0.13, 0.13, 0.15, 0.16])
    iax.bar([1, 2], [r1, r2], color=["#b30000", "#1a7a3a"], width=0.6)
    iax.set_xticks([1, 2]); iax.set_xticklabels(["axis 1", "axis 2"], fontsize=8)
    iax.set_ylabel("% variance", fontsize=8)
    iax.set_ylim(0, 100); iax.tick_params(labelsize=8)
    for s in ("top", "right"):
        iax.spines[s].set_visible(False)

    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}")


# ---------------------------------------------------------------------------
# Synthetic clouds with KNOWN structure (ground truth for benchmarks)
# ---------------------------------------------------------------------------
def planted_cloud(mu_latlon, axis_latlon, n=200, sigma_along=0.30,
                  sigma_perp=0.05, seed=0):
    """
    Cloud around mu whose principal axis is the geodesic toward `axis_latlon`.
    sigma_along >> sigma_perp makes axis 1 well-defined; both are tangent-space std (rad).
    Returns (points, mu_true, axis_dir_true).
    """
    rng = np.random.default_rng(seed)
    mu = latlong_to_sphere(*mu_latlon)
    toward = latlong_to_sphere(*axis_latlon)
    e1 = log_map(mu, toward)[0]
    e1 = e1 / np.linalg.norm(e1)                        # principal direction (unit tangent)
    e2 = np.cross(mu, e1)                               # orthogonal tangent direction
    u = (sigma_along * rng.normal(size=n)[:, None] * e1
         + sigma_perp * rng.normal(size=n)[:, None] * e2)
    return exp_map(mu, u), mu, e1


def axis_angle_deg(a, b):
    """Unsigned angle between two axes (mod 180), in degrees."""
    c = abs(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))
    return float(np.degrees(np.arccos(c)))


# ---------------------------------------------------------------------------
# Benchmarks (each proves a property AND prints an actionable number)
# ---------------------------------------------------------------------------
def _line(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<30} {detail}")


def bench():
    print("B1  Recover a planted principal axis (ground truth)")
    pts, mu_true, e1_true = planted_cloud((35, -100), (55, -70), n=300,
                                          sigma_along=0.30, sigma_perp=0.05)
    pga = GeodesicPCA().fit(pts)
    ax_err = axis_angle_deg(pga.axes_[0], e1_true)
    mu_err = geodesic_km(pga.mean_, mu_true)
    _line("axis recovered", ax_err < 3,
          f"axis within {ax_err:.2f} deg; captured {100*pga.variance_ratio_[0]:.1f}% "
          f"variance; mean off {mu_err:.0f} km")

    print("\nB2  Hand-rolled == GeomStats TangentPCA (integrity)")
    gs = TangentPCA(SPHERE)
    gs.fit(pts, base_point=pga.mean_)
    ang = axis_angle_deg(pga.axes_[0], gs.components_[0])
    dv = abs(pga.variance_ratio_[0] - gs.explained_variance_ratio_[0])
    _line("matches library", ang < 1e-3 and dv < 1e-3,
          f"axis angle diff {ang:.2e} deg; var-ratio diff {dv:.2e}")

    print("\nB3  Isotropic -> no spurious axis; anisotropic -> real axis")
    iso, _, _ = planted_cloud((0, 0), (0, 40), n=400, sigma_along=0.20,
                              sigma_perp=0.20, seed=1)
    ani, _, _ = planted_cloud((0, 0), (0, 40), n=400, sigma_along=0.30,
                              sigma_perp=0.03, seed=1)
    r_iso = GeodesicPCA().fit(iso).variance_ratio_
    r_ani = GeodesicPCA().fit(ani).variance_ratio_
    _line("isotropic ratio ~ 1", abs(r_iso[0] / r_iso[1] - 1) < 0.4,
          f"lambda1/lambda2 = {r_iso[0]/r_iso[1]:.2f}")
    _line("anisotropic ratio high", r_ani[0] / r_ani[1] > 20,
          f"lambda1/lambda2 = {r_ani[0]/r_ani[1]:.1f}")

    print("\nB4  PGA vs naive lat/long PCA near a pole (why geometry)")
    pts_hi, mu_hi, e1_hi = planted_cloud((78, 0), (78, 90), n=300,
                                         sigma_along=0.25, sigma_perp=0.03, seed=2)
    pga_hi = GeodesicPCA().fit(pts_hi)
    pga_err = axis_angle_deg(pga_hi.axes_[0], e1_hi)
    # naive: PCA on raw (lat, lon) degrees, then map the axis into the tangent plane
    lat, lon = sphere_to_latlong(pts_hi)
    ll = np.column_stack([lat, lon]).astype(float)
    ll -= ll.mean(0)
    _, _, vt = np.linalg.svd(ll, full_matrices=False)
    dlat, dlon = vt[0]
    east = np.cross(np.array([0, 0, 1.0]), mu_hi); east /= np.linalg.norm(east)
    north = np.cross(mu_hi, east)
    naive_axis = dlon * east + dlat * north
    naive_err = axis_angle_deg(naive_axis, e1_hi)
    _line("PGA axis correct at pole", pga_err < 5, f"PGA off {pga_err:.1f} deg")
    _line("naive lat/long axis wrong", naive_err > 15,
          f"naive off {naive_err:.1f} deg  (a {naive_err/max(pga_err,1e-9):.0f}x worse axis)")

    print("\nB5  1-axis reconstruction error (justifies dimensionality reduction)")
    rec = pga.reconstruct(n_components=1)
    rms = np.sqrt(np.mean([geodesic_km(p, r) ** 2 for p, r in zip(pts, rec)]))
    _line("1 axis reconstructs cloud", rms < 400,
          f"RMS geodesic error {rms:.0f} km  (cloud spread {np.degrees(pga.spread_rad_):.0f} deg)")

    print("\nB6  SO(3) equivariance (coordinate-free)")
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    pga_rot = GeodesicPCA().fit(pts @ Q.T)
    rot_err = axis_angle_deg(pga_rot.axes_[0], Q @ pga.axes_[0])
    dv = abs(pga_rot.variance_ratio_[0] - pga.variance_ratio_[0])
    _line("axes rotate with the data", rot_err < 1e-3 and dv < 1e-6,
          f"axis angle diff {rot_err:.2e} deg; var-ratio diff {dv:.2e}")


if __name__ == "__main__":
    import sys
    bench()
    if "--plot" in sys.argv:
        print()
        # a demo cloud over North America with a clear SW-NE axis
        demo, _, _ = planted_cloud((40, -100), (55, -70), n=140,
                                   sigma_along=0.32, sigma_perp=0.07, seed=7)
        plot_pga(GeodesicPCA().fit(demo), demo)
