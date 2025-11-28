import numpy as np
from scipy.spatial import cKDTree


class FractionalICP:
    def __init__(
        self,
        source,
        target,
        lambda_val=3.0,
        threshold=1e-6,
        max_iterations=1000,
        allow_reflection=False,
    ):
        """
        Fractional ICP (rigid 2D only):
          • Finds correspondences / FRMSD in 3D (so Z marks are respected).
          • Solves a rigid planar transform (rotation + XY translation, no scaling).
          • Applies it only to XY; Z and extra columns are left unchanged.

        Parameters
        ----------
        source, target : array-like, shape (N, D)
            Point sets. If D >= 3, FRMSD/correspondences use XYZ; XY moves only.
        lambda_val : float
            FRMSD lambda (reset internally between stages).
        threshold : float
            Convergence threshold on FRMSD improvement.
        max_iterations : int
            Maximum iterations per stage.
        allow_reflection : bool
            If False, enforce det(R) = +1 (no flips).
        """
        self.source = np.array(source, dtype=float)
        self.target = np.array(target, dtype=float)

        if self.source.ndim != 2 or self.target.ndim != 2:
            raise ValueError("source and target must be 2D arrays (N, D).")
        if self.source.shape[0] == 0 or self.target.shape[0] == 0:
            raise ValueError("source and target must be non-empty.")

        self.match_dims = 3 if (self.source.shape[1] >= 3 and self.target.shape[1] >= 3) else 2
        self.lambda_val = lambda_val
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.allow_reflection = allow_reflection

    # ----------------- helpers -----------------
    def _xy(self, pts):
        return np.ascontiguousarray(pts[:, :2])

    def _xyz_or_xy(self, pts):
        return np.ascontiguousarray(pts[:, : self.match_dims])

    # ----------------- FRMSD & matching -----------------
    def frmsd(self, fraction, num_elements, subset_source, corresponding_targets):
        """Fractional RMSD computed in XYZ (or XY if no Z)."""
        if num_elements == 0:
            return float("inf")
        diff = self._xyz_or_xy(subset_source) - self._xyz_or_xy(corresponding_targets)
        rmse = np.sqrt(np.sum(diff**2) / num_elements)
        return (1.0 / (fraction**self.lambda_val)) * rmse

    def get_n_first_elements(self, num_elements, distances):
        return np.argsort(distances)[:num_elements]

    def find_correspondences(self, source, target):
        if len(target) == 0 or len(source) == 0:
            return np.empty_like(source), np.array([])
        tree = cKDTree(self._xyz_or_xy(target))
        dists, idx = tree.query(self._xyz_or_xy(source), k=1)
        return target[idx], dists

    def find_optimal_fraction(self, corresponding_targets, distances):
        """Pick the subset size that minimizes FRMSD."""
        N = len(self.source)
        if N == 0:
            return 0.0, 0
        order = np.argsort(distances)
        best_val, best_frac, best_N = float("inf"), 0.0, 0
        for k in range(1, N + 1):
            frac = k / N
            sel = order[:k]
            val = self.frmsd(frac, k, self.source[sel], corresponding_targets[sel])
            if val < best_val:
                best_val, best_frac, best_N = val, frac, k
        return best_frac, best_N

    # ----------------- rigid 2D transform -----------------
    def compute_optimal_transform_2d(self, source_subset, target_subset):
        """Compute 2D rigid transform (rotation + translation, no scaling)."""
        X = self._xy(source_subset)
        Y = self._xy(target_subset)
        cs = X.mean(axis=0)
        ct = Y.mean(axis=0)
        Xc = X - cs
        Yc = Y - ct

        H = Xc.T @ Yc
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if not self.allow_reflection and np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = ct - (cs @ R.T)

        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t
        return T

    def apply_transform_2d_xy_only(self, points, T):
        """Apply 2D rigid transform to XY only; preserve Z and other attributes."""
        out = points.copy()
        xy = self._xy(points)
        ones = np.ones((xy.shape[0], 1))
        xy_t = (np.hstack([xy, ones]) @ T.T)[:, :2]
        out[:, :2] = xy_t
        return out

    # ----------------- ICP loop -----------------
    def _iterate(self):
        corr, d = self.find_correspondences(self.source, self.target)
        frac, k = self.find_optimal_fraction(corr, d)
        if k == 0:
            return self.source

        sel = self.get_n_first_elements(k, d)
        current_frmsd = self.frmsd(frac, k, self.source[sel], corr[sel])

        it = 0
        while it < self.max_iterations:
            sel = self.get_n_first_elements(k, d)
            T = self.compute_optimal_transform_2d(self.source[sel], corr[sel])
            self.source = self.apply_transform_2d_xy_only(self.source, T)

            corr, d = self.find_correspondences(self.source, self.target)
            frac, k = self.find_optimal_fraction(corr, d)
            sel = self.get_n_first_elements(k, d)
            new_frmsd = self.frmsd(frac, k, self.source[sel], corr[sel])

            if current_frmsd - new_frmsd <= self.threshold:
                break
            current_frmsd = new_frmsd
            it += 1

        return self.source

    def run(self):
        """Two-stage Fractional ICP (rigid only)."""
        self._iterate()
        self.lambda_val = 0.95 if self.match_dims == 3 else 1.3
        self._iterate()
        return self.source
