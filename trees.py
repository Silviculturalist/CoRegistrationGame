import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Optional, Tuple, Dict


def _resolve_mapping_value(mapping, key, default, *, allow_none: bool = False):
    """Return a cleaned mapping value, optionally allowing None for blank entries."""

    if not mapping:
        return default

    value = mapping.get(key, default)
    if value is None:
        return None if allow_none else default

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None if allow_none else default

    return value


class Tree:
    # Default Näslund params (a, b, c) – match UI defaults
    NASLUND_DEFAULT = (1.74105089, 0.35979281, 3.56879791)

    def __init__(
        self,
        tree_id,
        x: float,
        y: float,
        species: Optional[str] = None,
        stemdiam_cm: Optional[float] = None,
        height_dm: Optional[float] = None,
        original_plot=None,
        naslund_params: Optional[Tuple[float, float, float]] = None,
    ):
        """A single tree point with optional DBH/height and Näslund parameters.

        Args:
            tree_id: Identifier from CSV (string/number).
            x, y: World coordinates (meters).
            species: Optional species code.
            stemdiam_cm: DBH in centimeters (if provided).
            height_dm: Height in decimeters (if provided). Internally stored as meters.
            original_plot: Back‑reference, used by plot split tool.
            naslund_params: Optional (a, b, c) for Näslund (1936) H–D model.

        Notes:
            Internally, self.stemdiam and self.height are stored in meters.
        """
        self.tree_id = tree_id
        self.x = x
        self.y = y
        self.currentx = x
        self.currenty = y
        self.original_plot = original_plot
        self.species = species
        self.naslund_params = tuple(naslund_params) if naslund_params is not None else None

        # Store provided measurements in meters without automatic imputation.

        self.stemdiam = stemdiam_cm / 100 if stemdiam_cm is not None else None
        self.height = height_dm / 10 if height_dm is not None else None

    @staticmethod
    def naslund_1936(diameter: float, *params: float) -> float:
        """
        Näslund (1936) height model.
        Inputs:
            diameter (m): DBH in meters (as stored internally).
            params: (a, b, c) calibrated for DBH in centimeters.
        Returns:
            Height in meters.
        """
        d_cm = diameter * 100.0
        a, b, c = params
        return 1.3 + (d_cm / (a + b * d_cm)) ** c

    def get_height(self, diameter: float) -> float:
        """Height (m) from diameter (m) via Näslund (1936) [params in cm]."""
        params = self.naslund_params or self.NASLUND_DEFAULT
        return self.naslund_1936(diameter, *params)

    def get_diameter(self, height: float) -> float:
        """Diameter (m) from height (m) by inverting Näslund (params in cm) via 1D minimize."""
        params = self.naslund_params or self.NASLUND_DEFAULT


        def find_diameter(height: float, *params: float) -> float:
            def objective(x):
                return (height - self.naslund_1936(x, *params)) ** 2

            result = minimize_scalar(objective, bounds=(0, 100), method="bounded")
            return result.x

        return min(find_diameter(height, *params), 1.5)

    def impute_height(self, naslund_params: Optional[Tuple[float, float, float]] = None) -> None:
        """Impute missing height using existing diameter and Näslund parameters."""
        if self.height is not None or self.stemdiam is None:
            return
        if naslund_params is not None:
            self.naslund_params = tuple(naslund_params)
        self.height = self.get_height(self.stemdiam)

    def impute_dbh(self, naslund_params: Optional[Tuple[float, float, float]] = None) -> None:
        """Impute missing diameter using existing height and Näslund parameters."""
        if self.stemdiam is not None or self.height is None:
            return
        if naslund_params is not None:
            self.naslund_params = tuple(naslund_params)
        self.stemdiam = self.get_diameter(self.height)




class PlotIterator:
    def __init__(self, plot):
        self._plot = plot
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._plot.trees):
            result = self._plot.trees[self._index]
            self._index += 1
            return result
        raise StopIteration


class Plot:
    def __init__(self, plotid, center=None):
        self.plotid = plotid
        self.trees = []
        self.center = center if center is not None else (0, 0)
        self.current_center = self.center
        self.operations = []
        self.current_action = None
        self.current_translation = (0.0, 0.0)
        self.current_rotation = 0
        self.flipped = False

    def _update_centroid(self):
        if self.trees:
            self.current_center = np.mean(np.array([(tree.currentx, tree.currenty) for tree in self.trees]), axis=0)
        else:
            self.current_center = self.center

    def _apply_translation(self, value):
        """Translate all tree coordinates rigidly in 2D."""
        self.current_center = (
            self.current_center[0] + value[0],
            self.current_center[1] + value[1],
        )
        for tree in self.trees:
            tree.currentx += value[0]
            tree.currenty += value[1]

    def _apply_matrix_about_center(self, matrix: np.ndarray):
        """Apply a 2×2 linear transform about the current center (XY only)."""
        center = np.asarray(self.current_center, dtype=float)
        for tree in self.trees:
            vec = np.array([tree.currentx, tree.currenty], dtype=float) - center
            x_new, y_new = matrix @ vec
            tree.currentx = float(x_new + center[0])
            tree.currenty = float(y_new + center[1])

    def _apply_rotation(self, value):
        angle_radians = np.radians(value)
        rot = np.array(
            [[np.cos(angle_radians), -np.sin(angle_radians)],
             [np.sin(angle_radians),  np.cos(angle_radians)]]
        )
        self._apply_matrix_about_center(rot)

    def _apply_flip(self):
        flip = np.array([[1.0, 0.0], [0.0, -1.0]])
        self._apply_matrix_about_center(flip)

    def translate_plot(self, value):
        """Translate the plot by a given vector.

        Args:
            value (tuple[float, float]): ``(dx, dy)`` translation in world units.

        Returns:
            None
        """
        self._apply_translation(value)
        self.current_translation = (
            self.current_translation[0] + value[0],
            self.current_translation[1] + value[1],
        )

    def rotate_plot(self, value):
        """Rotate the plot around its current center.

        Args:
            value (float): Rotation angle in degrees (counterclockwise).

        Returns:
            None
        """
        self._apply_rotation(value)
        self.current_rotation += value

    def coordinate_flip(self):
        """Flip the plot vertically about its current center."""
        angle_radians = np.radians(self.current_rotation)
        rot = np.array(
            [[np.cos(angle_radians), -np.sin(angle_radians)],
             [np.sin(angle_radians),  np.cos(angle_radians)]]
        )
        reflection = rot @ np.array([[1.0, 0.0], [0.0, -1.0]]) @ rot.T
        self._apply_matrix_about_center(reflection)
        self.flipped = not self.flipped

    def reset_transformations(self):
        """Reset tree positions and transformation state to original values."""
        for tree in self.trees:
            tree.currentx = tree.x
            tree.currenty = tree.y
        self.current_center = self.center
        self.flipped = False
        self.current_translation = (0.0, 0.0)
        self.current_rotation = 0

    def get_tree_source_array(self):
        """Return an array of [tree_id, x, y, height] for trees."""
        if not self.trees:
            return np.empty((0, 4))
        return np.array([[tree.tree_id, tree.x, tree.y, tree.height] for tree in self.trees])

    def get_tree_current_array(self):
        """Return an array of [tree_id, currentx, currenty, height] for trees."""
        if not self.trees:
            return np.empty((0, 4))
        return np.array([[tree.tree_id, tree.currentx, tree.currenty, tree.height] for tree in self.trees])



    def get_transform(self):
        """Compute the rigid 2D transform from original to current tree positions.

        Uses a Procrustes-style solution (SVD) to estimate rotation (R) and
        translation (t) between the source coordinates (X, Y) at load time and
        the current coordinates after interactive edits.

        Returns:
            tuple[np.ndarray, np.ndarray, bool]:
                - R: (2, 2) rotation matrix (counterclockwise positive).
                - t: (2,) translation vector such that current ≈ R @ source + t.
                - flipped: Whether a vertical flip has been applied.

        Raises:
            ValueError: If there are no trees in the plot.
        """
        source_array = self.get_tree_source_array()[:, 1:3].astype(float)
        target_array = self.get_tree_current_array()[:, 1:3].astype(float)
        if source_array.size == 0 or target_array.size == 0:
            raise ValueError("No trees available to compute transform.")
        mu_s = np.mean(source_array, axis=0)
        mu_t = np.mean(target_array, axis=0)
        source_centered = source_array - mu_s
        target_centered = target_array - mu_t
        H = np.dot(source_centered.T, target_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        # Correct Procrustes translation: t = mu_t - R.dot(mu_s)
        t = mu_t - R.dot(mu_s)
        return R, t, self.flipped

    def append_tree(self, tree):
        """Add a tree to the plot and update its centroid.

        Args:
            tree (Tree): Tree instance to append.

        Returns:
            None
        """
        tree.currentx = tree.x
        tree.currenty = tree.y
        self.trees.append(tree)
        self._update_centroid()

    def update_tree_positions(self, update_array):
        """Update current positions of trees from an array of coordinates.

        Args:
            update_array (np.ndarray): Array of shape ``(n, 2)`` with new ``x`` and
                ``y`` coordinates.

        Raises:
            ValueError: If the array length does not match the number of trees.

        Returns:
            None
        """
        if len(self.trees) != update_array.shape[0]:
            raise ValueError('Update array length does not match number of trees in the plot')
        for tree, (x, y) in zip(self.trees, update_array):
            tree.currentx = x
            tree.currenty = y
        self._update_centroid()


class StandIterator:
    def __init__(self, stand):
        self._stand = stand
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._stand.plots):
            result = self._stand.plots[self._index]
            self._index += 1
            return result
        raise StopIteration


class Stand:
    def __init__(
        self,
        ID,
        file_path,
        mapping: Optional[Dict[str, str]] = None,
        sep: str = '\t',
        impute_dbh: bool = True,
        impute_h: bool = True,
        naslund_params: Optional[Tuple[float, float, float]] = None,
    ):
        self.standid = ID
        self.plots = []
        self.center = None
        self.impute_dbh = impute_dbh
        self.impute_h = impute_h
        self.naslund_params = (
            tuple(naslund_params) if naslund_params is not None else None
        )

        # Read CSV using the provided separator.
        reader = pd.read_csv(file_path, sep=sep).to_dict(orient='records')

        # Determine the column names using the mapping, if provided.
        # For StandID, if the user did not select a column (empty string), assume all rows belong to the provided stand.
        if mapping:
            stand_col = _resolve_mapping_value(mapping, 'StandID', '', allow_none=True)
            plot_col = _resolve_mapping_value(mapping, 'PlotID', 'PLOT')
            tree_col = _resolve_mapping_value(mapping, 'TreeID', 'TreeID')
            x_col = _resolve_mapping_value(mapping, 'X', 'X_GROUND')
            y_col = _resolve_mapping_value(mapping, 'Y', 'Y_GROUND')
            dbh_col = _resolve_mapping_value(mapping, 'DBH', 'STEMDIAM')
            h_col = _resolve_mapping_value(mapping, 'H', 'H', allow_none=True)
            species_col = _resolve_mapping_value(mapping, 'Species', 'Species')
            xc_col = _resolve_mapping_value(mapping, 'XC', x_col)
            yc_col = _resolve_mapping_value(mapping, 'YC', y_col)
        else:
            # Use default column names.
            stand_col = 'Stand'
            plot_col = 'PLOT'
            tree_col = 'TreeID'
            x_col = 'X_GROUND'
            y_col = 'Y_GROUND'
            dbh_col = 'STEMDIAM'
            h_col = 'H'
            species_col = 'Species'
            xc_col = 'XC'
            yc_col = 'YC'

        # If a stand column is specified, filter rows; otherwise, assume all rows belong to this stand.
        if stand_col:
            reader = [row for row in reader if row.get(stand_col) is not None and int(row[stand_col]) == int(ID)]
        else:
            # No stand column mapping provided—assume every row belongs to the provided stand.
            for row in reader:
                row['Stand'] = ID

        if not reader:
            raise ValueError(f"No data found for Stand ID: {ID}")

        # Process each row to create trees and plots.
        for row in reader:
            plot_id = row.get(plot_col)
            tree_id = row.get(tree_col)
            raw_dbh = row.get(dbh_col) if dbh_col in row else None
            try:
                stemdiam_cm = float(raw_dbh) if raw_dbh not in (None, "") else None
            except (ValueError, TypeError):
                stemdiam_cm = None
            # Optional height (meters) -> decimeters for Tree
            height_dm = None
            if h_col and h_col in row and row.get(h_col) not in (None, ''):
                try:
                    height_dm = float(row[h_col]) * 10.0
                except (ValueError, TypeError):
                    height_dm = None

            tree = Tree(
                tree_id,
                x=row.get(x_col),
                y=row.get(y_col),
                species=row.get(species_col),
                stemdiam_cm=stemdiam_cm,
                height_dm=height_dm,

                naslund_params=self.naslund_params
                if (self.impute_dbh or self.impute_h)
                else None,
            )
            if self.impute_h:
                tree.impute_height(self.naslund_params)
            if self.impute_dbh:
                tree.impute_dbh(self.naslund_params)

            # Check if the plot already exists; if not, create it.
            plot = next((p for p in self.plots if p.plotid == plot_id), None)
            if not plot:
                center = (row.get(xc_col, row.get(x_col)), row.get(yc_col, row.get(y_col)))
                plot = Plot(plotid=plot_id, center=center)
                self.add_plot(plot)
            plot.append_tree(tree)
            self._update_center()

    def _update_center(self):
        centers = [p.current_center for p in self.plots if p.center is not None]
        if not centers:
            self.center = None
            return
        sum_x = sum(c[0] for c in centers)
        sum_y = sum(c[1] for c in centers)
        self.center = (sum_x / len(centers), sum_y / len(centers))

    def add_plot(self, plot):
        self.plots.append(plot)
        self._update_center()

    def write_out(self):
        def _safe_cm(stemdiam_m):
            try:
                if stemdiam_m is None or (isinstance(stemdiam_m, float) and np.isnan(stemdiam_m)):
                    return np.nan
                return float(stemdiam_m) * 100.0
            except Exception:
                return np.nan

        def _safe_height(height_m):
            try:
                if height_m is None or (isinstance(height_m, float) and np.isnan(height_m)):
                    return np.nan
                return float(height_m)
            except Exception:
                return np.nan

        data = [(plot.plotid, tree.tree_id, tree.currentx, tree.currenty, _safe_cm(tree.stemdiam), _safe_height(tree.height))
                for plot in self.plots for tree in plot.trees]
        return pd.DataFrame(data, columns=['PlotID', 'TreeID', 'CurrentX', 'CurrentY', 'Diameter_cm', 'Height_m'])

    def __iter__(self):
        return StandIterator(self)


class SavedStand(Stand):
    def __init__(self, ID, file_path, naslund_params: Optional[Tuple[float, float, float]] = None):
        self.standid = ID
        self.plots = []
        self.center = None
        self.fp = file_path
        self.naslund_params = tuple(naslund_params) if naslund_params is not None else None
        reader = pd.read_csv(file_path).to_dict(orient='records')
        for row in reader:
            plot_id = row['PlotID']
            tree_id = row['TreeID']
            height_dm = None
            if 'Height_m' in row and row['Height_m'] not in (None, ''):
                try:
                    height_dm = float(row['Height_m']) * 10.0
                except (ValueError, TypeError):
                    height_dm = None
            # Parse DBH (cm) robustly (may be '', None, or numeric)
            raw_dbh = row.get('Diameter_cm')
            try:
                stemdiam_cm = float(raw_dbh) if raw_dbh not in (None, "") else None
            except (ValueError, TypeError):
                stemdiam_cm = None
            tree = Tree(
                tree_id,
                x=row['CurrentX'],
                y=row['CurrentY'],
                stemdiam_cm=stemdiam_cm,
                height_dm=height_dm,
                naslund_params=self.naslund_params,
            )
            plot = next((p for p in self.plots if p.plotid == plot_id), None)
            if not plot:
                plot = Plot(plotid=plot_id)
                self.add_plot(plot)
            plot.append_tree(tree)
            plot._update_centroid()
            self._update_center()
        for plot in self.plots:
            plot.center = plot.current_center

    def write_out(self):
        def _safe_cm(stemdiam_m):
            try:
                if stemdiam_m is None or (isinstance(stemdiam_m, float) and np.isnan(stemdiam_m)):
                    return np.nan
                return float(stemdiam_m) * 100.0
            except Exception:
                return np.nan

        def _safe_height(height_m):
            try:
                if height_m is None or (isinstance(height_m, float) and np.isnan(height_m)):
                    return np.nan
                return float(height_m)
            except Exception:
                return np.nan

        data = [(plot.plotid, tree.tree_id, tree.currentx, tree.currenty, _safe_cm(tree.stemdiam), _safe_height(tree.height))
                for plot in self.plots for tree in plot.trees]
        return pd.DataFrame(data, columns=['PlotID', 'TreeID', 'CurrentX', 'CurrentY', 'Diameter_cm', 'Height_m'])
