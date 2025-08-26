import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from copy import deepcopy

class Tree:
    def __init__(self, tree_id, x, y, species=None, stemdiam_cm=None, height_dm=None, original_plot=None):
        self.tree_id = tree_id
        self.x = x
        self.y = y
        self.currentx = x
        self.currenty = y
        self.original_plot = original_plot
        self.species = species
        if stemdiam_cm is not None and height_dm is not None:
            self.stemdiam = stemdiam_cm / 100
            self.height = height_dm / 10
        elif stemdiam_cm is None and height_dm is not None:
            self.stemdiam = self.get_diameter(height_dm / 10)
            self.height = height_dm / 10
        elif stemdiam_cm is not None and height_dm is None:
            self.stemdiam = stemdiam_cm / 100
            self.height = self.get_height(stemdiam_cm / 100)

    @staticmethod
    def naslund_1936(diameter, *params):
        return 1.3 + (diameter / (params[0] + params[1] * diameter)) ** params[2]

    def get_height(self, diameter):
        """Estimate tree height from diameter using the Näslund model.

        Args:
            diameter (float): Stem diameter in meters.

        Returns:
            float: Estimated height in meters.
        """
        params = [0.01850804, 0.12908718, 1.86770878]
        return self.naslund_1936(diameter, *params)

    def get_diameter(self, height):
        """Estimate diameter from tree height using the Näslund model.

        Args:
            height (float): Tree height in meters.

        Returns:
            float: Estimated stem diameter in meters capped at ``1.5`` m.
        """
        params = [0.01850804, 0.12908718, 1.86770878]

        def find_diameter(height, *params):
            def objective(x):
                return (height - self.naslund_1936(x, *params)) ** 2

            result = minimize_scalar(objective, bounds=(0, 100), method='bounded')
            return result.x

        return min(find_diameter(height, *params), 1.5)


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
        self.current_translation = [0, 0]
        self.current_rotation = 0
        self.flipped = False

    def _update_centroid(self):
        if self.trees:
            self.current_center = np.mean(np.array([(tree.currentx, tree.currenty) for tree in self.trees]), axis=0)
        else:
            self.current_center = self.center

    def _apply_translation(self, value):
        self.current_center = (self.current_center[0] + value[0], self.current_center[1] + value[1])
        for tree in self.trees:
            tree.currentx += value[0]
            tree.currenty += value[1]

    def _apply_rotation(self, value):
        angle_radians = np.radians(value)
        cos_angle = np.cos(angle_radians)
        sin_angle = np.sin(angle_radians)
        for tree in self.trees:
            # Translate tree coordinates to origin
            x_translated = tree.currentx - self.current_center[0]
            y_translated = tree.currenty - self.current_center[1]
            # Rotate coordinates
            x_rotated = x_translated * cos_angle - y_translated * sin_angle
            y_rotated = x_translated * sin_angle + y_translated * cos_angle
            # Translate back from origin
            tree.currentx = x_rotated + self.current_center[0]
            tree.currenty = y_rotated + self.current_center[1]

    def _apply_flip(self):
        for tree in self.trees:
            y_translated = tree.currenty - self.current_center[1]
            y_translated = -y_translated
            tree.currenty = y_translated + self.current_center[1]

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
        self._apply_rotation(-self.current_rotation)
        self._apply_flip()
        self._apply_rotation(self.current_rotation)
        self.flipped = not self.flipped

    def reset_transformations(self):
        """Reset tree positions and transformation state to original values."""
        for tree in self.trees:
            tree.currentx = tree.x
            tree.currenty = tree.y
        self.current_center = self.center
        self.flipped = False
        self.current_translation = [0, 0]
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
        source_array = self.get_tree_source_array()[:, 1:3]
        target_array = self.get_tree_current_array()[:, 1:3]
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
    def __init__(self, ID, file_path, mapping=None, sep='\t'):
        self.standid = ID
        self.plots = []
        self.center = None

        # Read CSV using the provided separator.
        reader = pd.read_csv(file_path, sep=sep).to_dict(orient='records')

        # Determine the column names using the mapping, if provided.
        # For StandID, if the user did not select a column (empty string), assume all rows belong to the provided stand.
        if mapping:
            stand_col = mapping.get('StandID', '').strip()  # May be empty.
            plot_col = mapping.get('PlotID', 'PLOT')
            tree_col = mapping.get('TreeID', 'TreeID')
            x_col = mapping.get('X', 'X_GROUND')
            y_col = mapping.get('Y', 'Y_GROUND')
            dbh_col = mapping.get('DBH', 'STEMDIAM')
            h_col   = mapping.get('H', 'H')
            species_col = mapping.get('Species', 'Species')
            xc_col = mapping.get('XC', x_col)
            yc_col = mapping.get('YC', y_col)
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
            try:
                stemdiam_cm = float(row.get(dbh_col, 0))
            except (ValueError, TypeError):
                stemdiam_cm = None
            # Optional height (assumed meters) -> Tree expects decimeters
            try:
                height_m = float(row.get(h_col)) if h_col in row else None
            except (ValueError, TypeError):
                height_m = None
            height_dm = height_m * 10 if height_m is not None else None

            tree = Tree(
                tree_id,
                x=row.get(x_col),
                y=row.get(y_col),
                species=row.get(species_col),
                stemdiam_cm=stemdiam_cm,
                height_dm=height_dm,
            )
            # Check if the plot already exists; if not, create it.
            plot = next((p for p in self.plots if p.plotid == plot_id), None)
            if not plot:
                center = (row.get(xc_col, row.get(x_col)), row.get(yc_col, row.get(y_col)))
                plot = Plot(plotid=plot_id, center=center)
                self.add_plot(plot)
            plot.append_tree(tree)
            self._update_center()

    def _update_center(self):
        if not self.plots:
            self.center = None
        else:
            sum_x = sum(plot.current_center[0] for plot in self.plots if plot.center is not None)
            sum_y = sum(plot.current_center[1] for plot in self.plots if plot.center is not None)
            total_plots = len(self.plots)
            self.center = (sum_x / total_plots, sum_y / total_plots)

    def add_plot(self, plot):
        self.plots.append(plot)
        self._update_center()

    def write_out(self):
        data = [(plot.plotid, tree.tree_id, tree.currentx, tree.currenty, tree.stemdiam * 100, tree.height)
                for plot in self.plots for tree in plot.trees]
        return pd.DataFrame(data, columns=['PlotID', 'TreeID', 'CurrentX', 'CurrentY', 'Diameter_cm', 'Height_m'])

    def __iter__(self):
        return StandIterator(self)


class SavedStand(Stand):
    def __init__(self, ID, file_path):
        self.standid = ID
        self.plots = []
        self.center = None
        self.fp = file_path
        reader = pd.read_csv(file_path).to_dict(orient='records')
        for row in reader:
            plot_id = row['PlotID']
            tree_id = row['TreeID']
            tree = Tree(tree_id, x=row['CurrentX'], y=row['CurrentY'], stemdiam_cm=row['Diameter_cm'])
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
        data = [(plot.plotid, tree.tree_id, tree.currentx, tree.currenty, tree.stemdiam * 100, tree.height)
                for plot in self.plots for tree in plot.trees]
        return pd.DataFrame(data, columns=['PlotID', 'TreeID', 'CurrentX', 'CurrentY', 'Diameter_cm', 'Height_m'])
