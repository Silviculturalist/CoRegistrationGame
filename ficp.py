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
        scale_mode="uniform",          # "none" | "uniform" | "anisotropic"
        scale_bounds=None,             # e.g., (0.5, 2.0) or None for unbounded
        allow_reflection=False,        # keep False to avoid flips
    ):
        """
        Fractional ICP that:
          • finds correspondences / FRMSD in 3D when available (so Z marks are respected),
          • but solves/applies a 2D transform in the plane (rotation about Z, XY translation),
          • with optional XY-only scaling (uniform or anisotropic), never scaling Z.

        Parameters
        ----------
        source, target : array-like, shape (N, D)
            Point sets. If D >= 3 we match in XYZ; only XY is moved.
        lambda_val : float
            FRMSD lambda (reset between stages below).
        threshold : float
            Convergence threshold on FRMSD improvement.
        max_iterations : int
            Max iterations per stage.
        scale_mode : {"none","uniform","anisotropic"}
            "none"       -> rigid (rotate+translate only, no scaling)
            "uniform"    -> one scalar s for both X and Y
            "anisotropic"-> two scalars (s_x, s_y); still no shear, no Z scaling.
        scale_bounds : tuple or None
            If given, clamp scale(s) to [min_s, max_s].
        allow_reflection : bool
            If False, enforce proper rotation (det(R)=+1) and positive scales.
        """
        self.source = np.array(source, dtype=float)
        self.target = np.array(target, dtype=float)

        if self.source.ndim != 2 or self.target.ndim != 2:
            raise ValueError("source and target must be (N, D) arrays.")
        if self.source.shape[0] == 0 or self.target.shape[0] == 0:
            raise ValueError("source and target must be non-empty.")

        self.match_dims = 3 if (self.source.shape[1] >= 3 and self.target.shape[1] >= 3) else 2
        self.lambda_val = lambda_val
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.scale_mode = scale_mode
        self.scale_bounds = scale_bounds
        self.allow_reflection = allow_reflection

    # --------------------------- helpers ---------------------------
    def _xy(self, pts):
        return np.ascontiguousarray(pts[:, :2])

    def _xyz_or_xy(self, pts):
        return np.ascontiguousarray(pts[:, :self.match_dims])

    @staticmethod
    def _clamp_scale(s, bounds, allow_reflection):
        if bounds is None:
            return s if allow_reflection else np.abs(s)
        lo, hi = bounds
        if np.ndim(s) == 0:
            s = np.clip(s, lo, hi)
            return s if allow_reflection else abs(s)
        s = np.clip(s, lo, hi)
        return s if allow_reflection else np.abs(s)

    # ---------------------- FRMSD & matching -----------------------
    def frmsd(self, fraction, num_elements, subset_source, corresponding_targets):
        """
        Fractional RMSD computed in matching space (XYZ when available, else XY).
        """
        if num_elements == 0:
            return float('inf')
        dif = self._xyz_or_xy(subset_source) - self._xyz_or_xy(corresponding_targets)
        rmse = np.sqrt(np.sum(dif**2) / num_elements)
        return (1.0 / (fraction ** self.lambda_val)) * rmse

    def get_n_first_elements(self, num_elements, distances):
        return np.argsort(distances)[:num_elements]

    def find_correspondences(self, source, target):
        """
        Nearest-neighbor search in 3D (or 2D if Z not present).
        """
        if len(target) == 0 or len(source) == 0:
            return np.empty_like(source), np.array([])
        tree = cKDTree(self._xyz_or_xy(target))
        dists, idx = tree.query(self._xyz_or_xy(source), k=1)
        return target[idx], dists

    def find_optimal_fraction(self, corresponding_targets, distances):
        total_points = len(self.source)
        if total_points == 0:
            return 0.0, 0
        order = np.argsort(distances)
        best_frmsd = float('inf')
        best_frac, best_N = 0.0, 0
        for N in range(1, total_points + 1):
            frac = N / total_points
            sel = order[:N]
            val = self.frmsd(frac, N, self.source[sel], corresponding_targets[sel])
            if val < best_frmsd:
                best_frmsd, best_frac, best_N = val, frac, N
        return best_frac, best_N

    # ---------------- 2D transform (XY-only motion) ----------------
    def compute_optimal_transform_2d(self, source_subset, target_subset):
        """
        Solve for A (2x2) and t (2,) such that in ROW form:  y = x A^T + t
        where A = R @ S, with:
          • R: 2D rotation (det=+1 unless allow_reflection=True)
          • S: I             if scale_mode="none"
               s*I           if scale_mode="uniform"
               diag(sx, sy)  if scale_mode="anisotropic"
        Z is never used/scaled; only XY moves.

        Returns a 3x3 homogeneous 2D transform T (for row-wise application via xy_h @ T.T).
        """
        src = self._xy(source_subset)
        tgt = self._xy(target_subset)

        cs = np.mean(src, axis=0)
        ct = np.mean(tgt, axis=0)

        X = src - cs
        Y = tgt - ct

        # --- rotation (Kabsch) ---
        H = X.T @ Y
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            if not self.allow_reflection:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            # else: keep reflection

        # --- scale in XY only (never Z) ---
        eps = 1e-12
        if self.scale_mode == "uniform":
            # From Yc @ R ≈ Xc * s  -> s = sum(X * (Y @ R)) / sum(X*X)
            Z = Y @ R
            s_num = np.sum(X * Z)
            s_den = np.sum(X * X) + eps
            s = s_num / s_den
            s = self._clamp_scale(s, self.scale_bounds, self.allow_reflection)
            S = np.eye(2) * s

        elif self.scale_mode == "anisotropic":
            # From Yc @ R ≈ Xc @ diag(sx, sy) -> per-axis LS
            Z = Y @ R                                   # (N,2)
            den = np.sum(X * X, axis=0) + eps           # (2,)
            num = np.sum(X * Z, axis=0)                 # (2,)
            s_vec = num / den                           # (2,)
            s_vec = self._clamp_scale(s_vec, self.scale_bounds, self.allow_reflection)
            S = np.diag(s_vec)

        elif self.scale_mode == "none":
            S = np.eye(2)
        else:
            raise ValueError("scale_mode must be one of {'none','uniform','anisotropic'}")

        A = R @ S
        # Row-form translation: t = ct - cs @ A^T   (works for any S, including anisotropic)
        t = ct - (cs @ A.T)

        # Build homogeneous 2D transform for row-wise application
        T = np.eye(3)
        T[:2, :2] = A
        T[:2, 2] = t
        return T

    def apply_transform_2d_xy_only(self, points, T_2D):
        """
        Apply the 2D transform to XY only; pass through Z and any extra columns unchanged.
        """
        out = points.copy()
        xy = self._xy(points)
        ones = np.ones((xy.shape[0], 1))
        xy_h = np.hstack([xy, ones])          # (N, 3)
        xy_t = (xy_h @ T_2D.T)[:, :2]         # row-wise application
        out[:, :2] = xy_t
        return out

    # -------------------------- main loop --------------------------
    def _iterate(self):
        corresponding_targets, distances = self.find_correspondences(self.source, self.target)
        optimal_fraction, optimal_num_elements = self.find_optimal_fraction(corresponding_targets, distances)
        if optimal_num_elements == 0:
            return self.source

        sel = self.get_n_first_elements(optimal_num_elements, distances)
        current_frmsd = self.frmsd(optimal_fraction, optimal_num_elements,
                                   self.source[sel], corresponding_targets[sel])

        improvement = float('inf')
        iteration = 0
        while improvement > self.threshold and iteration < self.max_iterations:
            sel = self.get_n_first_elements(optimal_num_elements, distances)
            src_subset = self.source[sel]
            tgt_subset = corresponding_targets[sel]

            T_2D = self.compute_optimal_transform_2d(src_subset, tgt_subset)
            self.source = self.apply_transform_2d_xy_only(self.source, T_2D)

            # Recompute matches & FRMSD in 3D space (Z marks still drive correspondences)
            corresponding_targets, distances = self.find_correspondences(self.source, self.target)
            optimal_fraction, optimal_num_elements = self.find_optimal_fraction(corresponding_targets, distances)
            sel = self.get_n_first_elements(optimal_num_elements, distances)
            new_frmsd = self.frmsd(optimal_fraction, optimal_num_elements,
                                   self.source[sel], corresponding_targets[sel])

            improvement = current_frmsd - new_frmsd
            current_frmsd = new_frmsd
            iteration += 1

        return self.source

    def run(self):
        """
        Two-stage FICP:
          • Stage 1 with initial lambda
          • Stage 2 with lambda tuned to matching dimensionality (3D vs 2D)
        """
        self._iterate()
        self.lambda_val = 0.95 if self.match_dims == 3 else 1.3
        self._iterate()
        return self.source
