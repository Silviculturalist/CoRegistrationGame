import numpy as np
import sys
from pathlib import Path
from scipy.spatial import cKDTree

# Ensure project root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ficp import FractionalICP


def _make_cloud(n=200, seed=9):
    rng = np.random.default_rng(seed)
    xy = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.3], [0.0, 0.6]]).T
    z = np.linspace(0.0, 20.0, n).reshape(-1, 1) + rng.normal(scale=0.02, size=(n, 1))
    return np.hstack([xy, z])


def _apply_rigid(src, angle_deg, t_xy):
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    xy = src[:, :2] @ R.T + np.asarray(t_xy)
    return np.hstack([xy, src[:, 2][:, None]])


def _nn_rmsd(A, B):
    tree = cKDTree(B)
    d, _ = tree.query(A, k=1)
    return np.sqrt(np.mean(d**2))


def _inlier_fraction_xy(aligned, target, tol=0.10):
    tree = cKDTree(target[:, :2])
    d, _ = tree.query(aligned[:, :2], k=1)
    return np.mean(d < tol)


def test_basic_rigid_exact():
    """Exact rigid transform recovery with equal cardinality."""
    src = _make_cloud(n=150, seed=1)
    angle, t = 27.0, np.array([1.6, -2.2])
    tgt = _apply_rigid(src, angle, t)

    ficp = FractionalICP(src.copy(), tgt)
    aligned = ficp.run()

    # Decompose A via least-squares on XY (expect A ≈ R, no scale)
    X, Y = src[:, :2], aligned[:, :2]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    A = np.linalg.lstsq(Xc, Yc, rcond=None)[0].T
    ang_est = np.rad2deg(np.arctan2(A[1, 0], A[0, 0]))
    detA = np.linalg.det(A)

    assert np.allclose(aligned[:, 2], src[:, 2])
    assert abs(detA - 1.0) < 1e-2
    assert abs(((ang_est - angle + 180) % 360) - 180) < 0.2
    assert _nn_rmsd(aligned, tgt) < 2e-3


def test_missing_points_frmsd():
    """Target has 50% missing points; FRMSD should still find the correct pose."""
    src = _make_cloud(n=200, seed=2)
    angle, t = 31.0, np.array([2.5, -1.8])
    full = _apply_rigid(src, angle, t)

    rng = np.random.default_rng(123)
    keep = rng.choice(full.shape[0], size=full.shape[0] // 2, replace=False)
    tgt = full[keep]

    ficp = FractionalICP(src.copy(), tgt)
    aligned = ficp.run()

    assert np.allclose(aligned[:, 2], src[:, 2])
    assert _nn_rmsd(aligned, tgt) < _nn_rmsd(src, tgt) * 0.4
    assert _inlier_fraction_xy(aligned, tgt, tol=0.12) > 0.55


def test_missing_plus_outliers_frmsd():
    """Target missing 50% + 30% outliers; FRMSD should ignore outliers."""
    src = _make_cloud(n=200, seed=3)
    angle, t = -22.0, np.array([-1.2, 2.0])
    clean = _apply_rigid(src, angle, t)

    rng = np.random.default_rng(7)
    keep = rng.choice(clean.shape[0], size=clean.shape[0] // 2, replace=False)
    tgt = clean[keep]

    # Add 30% distractor outliers
    m = tgt.shape[0]
    num_out = int(0.3 * m)
    out_xy = rng.uniform(low=-20, high=20, size=(num_out, 2))
    out_z = rng.uniform(low=-5, high=25, size=(num_out, 1))
    tgt_noisy = np.vstack([tgt, np.hstack([out_xy, out_z])])

    ficp = FractionalICP(src.copy(), tgt_noisy)
    aligned = ficp.run()

    assert np.allclose(aligned[:, 2], src[:, 2])
    assert _nn_rmsd(aligned, tgt_noisy) < _nn_rmsd(src, tgt_noisy) * 0.5
    assert _inlier_fraction_xy(aligned, clean, tol=0.12) > 0.90
