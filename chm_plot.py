import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from copy import deepcopy
from trees import Tree, Plot
import matplotlib.pyplot as plt
import logging 

def Naslund1936kwargs(diameter, *params):
    """Calculate tree height based on the Näslund 1936 model."""
    return 1.3 + (diameter / (params[0] + params[1] * diameter)) ** params[2]

def plot_height_curve(params):
    """
    Generate a plot of tree height vs. diameter using the Näslund 1936 model.

    Parameters:
        params (list): A list of three numerical parameters for the model.
    
    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
    """
    # Create a range of diameters in centimeters and convert to meters
    diameters_cm = [i * 0.1 for i in range(1, 601)]  # 0.1 cm to 60 cm
    diameters_m = [d / 100.0 for d in diameters_cm]
    heights = [Naslund1936kwargs(d, *params) for d in diameters_m]

    fig, ax = plt.subplots()
    ax.plot(diameters_cm, heights, label="Height Curve")
    ax.set_xlabel("Diameter at breast height (cm)")
    ax.set_ylabel("Height (m)")
    ax.grid(True)
    ax.legend()
    return fig


def to_screen_coordinates(geo_coord, stand_center, scale_factor, screen_size):
    """Convert world coordinates to screen coordinates.

    Args:
        geo_coord (tuple[float, float]): ``(x, y)`` position in world units.
        stand_center (tuple[float, float]): World-space center used as origin.
        scale_factor (float): Pixels per world unit.
        screen_size (tuple[int, int]): Screen width and height in pixels.

    Returns:
        tuple[int, int]: Screen-space ``(x, y)`` coordinates.
    """
    screen_x = (geo_coord[0] - stand_center[0]) * scale_factor + screen_size[0] / 2
    screen_y = (geo_coord[1] - stand_center[1]) * scale_factor + screen_size[1] / 2
    return int(screen_x), int(screen_y)

def get_viewport_scale(stand, screen_size):
    """Compute a pixel-to-world scaling factor for the viewport.

    Args:
        stand (Stand): Stand containing plots and trees.
        screen_size (tuple[int, int]): Screen width and height in pixels.

    Returns:
        float: Scale factor expressed as pixels per world unit.
    """
    all_trees = [tree for plot in stand.plots for tree in plot.trees]
    if not all_trees:
        return 1
    coords = np.array([(tree.currentx, tree.currenty) for tree in all_trees])
    furthest_distance = max(
        np.sqrt((x - stand.center[0]) ** 2 + (y - stand.center[1]) ** 2) for x, y in coords
    )
    max_screen_distance = min(screen_size) / 2 - 20  # 20 pixels padding
    scale_factor = max_screen_distance / (furthest_distance + 2)  # +2 meters buffer
    return scale_factor

class CHMPlot(Plot):
    def __init__(self, file_path, x=None, y=None, dist=40, height_unit='m', mapping=None, sep='\t',
                 impute_dbh: bool = False, impute_h: bool = False, naslund_params=None):
        """Load CHM detections as a single 'plot' and optionally crop by distance.

        Args:
            file_path (str): Path to CSV with CHM detections.
            x (float | None): Center X (world units) for radial crop; if None, no crop.
            y (float | None): Center Y (world units) for radial crop; if None, no crop.
            dist (float): Crop radius in world units when x,y are provided.
            height_unit (str): Unit of the height column: 'm', 'dm', or 'cm'.
            mapping (dict | None): Optional column mapping keys: {'X','Y','H','TreeID','DBH'}.
            sep (str): CSV separator character.
            impute_dbh (bool): Whether to impute DBH when missing.
            impute_h (bool): Whether to impute height when missing.
            naslund_params (tuple | None): Optional Näslund parameters.

        Notes:
            Internally, height is stored in meters and stem diameter in meters.
            The input height is converted to decimeters (height_dm) to match Tree's init.
        Raises:
            FileNotFoundError: If the file cannot be read.
            KeyError: If required columns are missing after mapping.
        """
        self.trees = []
        self.naslund_params = tuple(naslund_params) if naslund_params is not None else None
        self.impute_dbh = impute_dbh
        self.impute_h = impute_h
        self.plotid = 1
        # Read CSV into a DataFrame.
        df = pd.read_csv(file_path, sep=sep)

        # Set column names using mapping if provided.
        if mapping:
            x_col = mapping.get('X', 'X').strip() or 'X'
            y_col = mapping.get('Y', 'Y').strip() or 'Y'
            height_col = mapping.get('H', 'H').strip() or 'H'
            idals_col = mapping.get('TreeID', 'IDALS').strip() or 'IDALS'
            dbh_col = mapping.get('DBH', 'DBH').strip() or 'DBH'
        else:
            x_col, y_col, height_col, idals_col, dbh_col = 'X', 'Y', 'H', 'IDALS', 'DBH'

        # Check if the height column exists in the data.
        missing_height = height_col not in df.columns

        # Filter rows based on distance if x, y, and dist are provided.
        if x is not None and y is not None and dist is not None and dist > 0:
            coordinates = df[[x_col, y_col]].values
            point = np.array([[x, y]])
            distances = cdist(coordinates, point, metric='euclidean')
            df = df[distances[:, 0] <= dist]

        # Convert DataFrame to a list of records.
        records = df.to_dict(orient='records')

        for row in records:
            # If the height column is present, use it.
            if not missing_height:
                try:
                    if height_unit == 'm':
                        height = row[height_col] * 10
                    elif height_unit == 'dm':
                        height = row[height_col]
                    elif height_unit == 'cm':
                        height = row[height_col] / 10
                except Exception as e:
                    logging.error(f"Error processing height for row: {row} - {e}")
                    continue
                # DBH value is not needed if height is provided.
                stemdiam_value = None
            else:
                # Height column is missing. Try to get DBH for imputation.
                try:
                    stemdiam_value = float(row[dbh_col]) if (dbh_col in row and row[dbh_col] not in [None, ""]) else None
                except Exception as e:
                    stemdiam_value = None
                height = None  # Let the Tree class impute height.

            # Optionally skip trees with unrealistic heights.
            if height is not None and height > 450:
                continue

            # Skip rows that provide neither metric
            if (
                (height is None or (isinstance(height, float) and np.isnan(height)))
                and (
                    stemdiam_value is None
                    or (isinstance(stemdiam_value, float) and np.isnan(stemdiam_value))
                )
            ):
                continue

            tree = Tree(
                tree_id=row[idals_col],
                x=row[x_col],
                y=row[y_col],
                stemdiam_cm=stemdiam_value,
                height_dm=height,
                naslund_params=self.naslund_params if (self.impute_dbh or self.impute_h) else None,
            )
            if self.impute_h:
                tree.impute_height(self.naslund_params)
            if self.impute_dbh:
                tree.impute_dbh(self.naslund_params)
            self.append_tree(tree)

        pts = np.array([[tree.x, tree.y] for tree in self.trees])
        if len(pts):
            self.center = np.mean(pts, axis=0)
        else:
            # Avoid NaNs when CHM is empty post-filtering
            self.center = np.array([0.0, 0.0])
        self.removed_stems = []
        self.alltrees = deepcopy(self.trees)



    def remove_matches(self, plot, min_dist_percent=15):
        """Remove CHM trees matched closely to plot trees.

        For each tree in ``plot`` the closest neighbor in the CHM layer is
        removed if it lies within ``min_dist_percent`` of that tree's height.

        Args:
            plot (Plot): Plot whose trees are used to search for matches.
            min_dist_percent (float): Maximum allowed distance expressed as a
                percentage of the tree height.

        Returns:
            None
        """
        current_removal = []
        for tree in plot.trees:
            # Find closest neighbor using scalar distance (convert result to a scalar)
            closest_tree = min(
                self.trees,
                key=lambda x: cdist(
                    np.array([[tree.currentx, tree.currenty, tree.height]]),
                    np.array([[x.currentx, x.currenty, x.height]]),
                ).item(),
            )
            distance = cdist(
                np.array([[tree.currentx, tree.currenty, tree.height]]),
                np.array([[closest_tree.currentx, closest_tree.currenty, closest_tree.height]]),
            ).item()
            if distance < ((min_dist_percent / 100) * tree.height):
                current_removal.append(closest_tree)
                if closest_tree in self.trees:
                    self.trees.remove(closest_tree)
        self.removed_stems.append(current_removal)

    def restore_matches(self):
        """Restore the most recently removed CHM trees.

        Returns:
            None
        """
        if not self.removed_stems:
            return
        last_removal = self.removed_stems.pop()
        for tree in last_removal:
            self.append_tree(tree)


class SavedPlot(CHMPlot):
    def __init__(self, file_path, x=None, y=None, dist=40, naslund_params=None):
        self.trees = []
        self.plotid = 1
        self.naslund_params = tuple(naslund_params) if naslund_params is not None else None
        reader = pd.read_csv(file_path)
        if x is not None and y is not None and dist is not None and dist > 0:
            coordinates = reader[['CurrentX', 'CurrentY']].values
            point = np.array([[x, y]])
            distances = cdist(coordinates, point, metric='euclidean')
            df_filtered = reader[distances[:, 0] <= dist]
            reader = df_filtered.to_dict(orient='records')
        else:
            reader = reader.to_dict(orient='records')
        for row in reader:
            height_dm = None
            if 'Height_m' in row and row['Height_m'] not in (None, ''):
                try:
                    height_dm = float(row['Height_m']) * 10.0
                except (ValueError, TypeError):
                    height_dm = None
            tree = Tree(
                tree_id=row['TreeID'],
                x=row['CurrentX'],
                y=row['CurrentY'],
                stemdiam_cm=row.get('Diameter_cm'),
                height_dm=height_dm,
                naslund_params=self.naslund_params,
            )
            self.append_tree(tree)
        self.center = np.mean(np.array([[tree.x, tree.y] for tree in self.trees]), axis=0) if self.trees else np.array([0.0, 0.0])
        self.removed_stems = []
        self.alltrees = deepcopy(self.trees)

