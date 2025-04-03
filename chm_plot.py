import pandas as pd
import numpy as np
import pygame 
from scipy.spatial.distance import cdist
from copy import deepcopy
from trees import Tree, Plot
import matplotlib.pyplot as plt

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
    # Create a range of diameters (in cm)
    diameters = [i * 0.1 for i in range(1, 601)]  # 0.1 cm to 60 cm
    # Compute heights (in m) using the Näslund model
    heights = [Naslund1936kwargs(d, *params) for d in diameters]

    fig, ax = plt.subplots()
    ax.plot(diameters, heights, label="Height Curve")
    ax.set_xlabel("Diameter (cm)")
    ax.set_ylabel("Height (m)")
    ax.grid(True)
    ax.legend()
    return fig


def to_screen_coordinates(geo_coord, stand_center, scale_factor, screen_size):
    screen_x = (geo_coord[0] - stand_center[0]) * scale_factor + screen_size[0] / 2
    screen_y = (geo_coord[1] - stand_center[1]) * scale_factor + screen_size[1] / 2
    return int(screen_x), int(screen_y)

def get_viewport_scale(stand, screen_size):
    all_trees = [tree for plot in stand.plots for tree in plot.trees]
    if not all_trees:
        return 1
    coords = np.array([(tree.currentx, tree.currenty) for tree in all_trees])
    furthest_distance = max(np.sqrt((x - stand.center[0]) ** 2 + (y - stand.center[1]) ** 2) for x, y in coords)
    max_screen_distance = min(screen_size) / 2 - 20  # 20 pixels padding
    scale_factor = max_screen_distance / (furthest_distance + 2)  # +2 meters buffer
    return scale_factor

class CHMPlot(Plot):
    def __init__(self, file_path, x=None, y=None, dist=40, height_unit='m', mapping=None, sep='\t'):
        self.trees = []
        self.plotid = 1
        # Read CSV with the provided separator.
        reader = pd.read_csv(file_path, sep=sep)
        
        # Use mapping for CHM file columns if provided.
        if mapping:
            x_col = mapping.get('X', 'X').strip() or 'X'
            y_col = mapping.get('Y', 'Y').strip() or 'Y'
            height_col = mapping.get('H', 'H').strip() or 'H'
            idals_col = mapping.get('TreeID', 'IDALS').strip() or 'IDALS'
        else:
            x_col, y_col, height_col, idals_col = 'X', 'Y', 'H', 'IDALS'
        
        # If a point is provided and a distance is specified, filter rows by distance.
        if x is not None and y is not None and dist is not None and dist > 0:
            try:
                coordinates = reader[[x_col, y_col]].values
            except KeyError:
                raise KeyError(f"Columns [{x_col}, {y_col}] not found in the CHM file.")
            point = np.array([[x, y]])
            distances = cdist(coordinates, point, metric='euclidean')
            df_filtered = reader[distances[:, 0] <= dist]
            reader = df_filtered
        # Convert reader to a dictionary list for further processing.
        reader = reader.to_dict(orient='records')
        
        for row in reader:
            # Determine the height based on the provided height_unit.
            if height_unit == 'm':
                height = row[height_col] * 10
            elif height_unit == 'dm':
                height = row[height_col]
            elif height_unit == 'cm':
                height = row[height_col] / 10
            if height > 450:
                continue
            tree = Tree(tree_id=row[idals_col], x=row[x_col], y=row[y_col], height_dm=height)
            self.append_tree(tree)
        self.center = np.mean(np.array([[tree.x, tree.y] for tree in self.trees]), axis=0)
        self.removed_stems = []
        self.alltrees = deepcopy(self.trees)


    def remove_matches(self, plot, min_dist_percent=15):
        """
        Remove closest neighbor for each tree from CHM if within min_dist_percent of tree height.
        """
        current_removal = []
        for tree in plot.trees:
            # Find closest neighbor using scalar distance (convert result to a scalar)
            closest_tree = min(self.trees, key=lambda x: cdist(
                np.array([[tree.currentx, tree.currenty, tree.height]]),
                np.array([[x.currentx, x.currenty, x.height]])
            ).item())
            distance = cdist(
                np.array([[tree.currentx, tree.currenty, tree.height]]),
                np.array([[closest_tree.currentx, closest_tree.currenty, closest_tree.height]])
            ).item()
            if distance < ((min_dist_percent / 100) * tree.height):
                current_removal.append(closest_tree)
                if closest_tree in self.trees:
                    self.trees.remove(closest_tree)
        self.removed_stems.append(current_removal)

    def restore_matches(self):
        if not self.removed_stems:
            return
        last_removal = self.removed_stems.pop()
        for tree in last_removal:
            self.append_tree(tree)

class SavedPlot(CHMPlot):
    def __init__(self, file_path, x=None, y=None, dist=40):
        self.trees = []
        self.plotid = 1
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
            tree = Tree(tree_id=row['TreeID'], x=row['CurrentX'], y=row['CurrentY'], stemdiam_cm=row['Diameter_cm'])
            self.append_tree(tree)
        self.center = np.mean(np.array([[tree.x, tree.y] for tree in self.trees]), axis=0)
        self.removed_stems = []
        self.alltrees = deepcopy(self.trees)


class PlotCenters:
    def __init__(self, stand):
        """
        Compute plot centers from the Stand object's plots.
        Only centers within 70 units of the overall stand center are kept.
        """
        # Assume each plot in the stand has a 'current_center' attribute.
        self.centers = np.array([plot.current_center for plot in stand.plots if plot.current_center is not None])
        # Filter centers: only keep those within a distance of 70 from the stand's overall center.
        if self.centers.size > 0:
            self.centers = self.centers[cdist(self.centers, np.array([stand.center])).squeeze() < 70]
        else:
            self.centers = np.array([])
            
    def draw_single_center(self, screen, center, color, alpha, stand_center, scale_factor, screen_size):
        adjusted_position = to_screen_coordinates(center, stand_center, scale_factor, screen_size)
        adjusted_radius = 2  # constant radius
        alpha_surface = pygame.Surface((adjusted_radius * 2, adjusted_radius * 2), pygame.SRCALPHA)
        alpha_surface.fill((0, 0, 0, 0))
        pygame.draw.circle(alpha_surface, color + (int(255 * alpha),), (adjusted_radius, adjusted_radius), adjusted_radius)
        screen.blit(alpha_surface, (adjusted_position[0] - adjusted_radius, adjusted_position[1] - adjusted_radius))

    def draw_centers(self, screen, color, alpha, stand_center, scale_factor, screen_size):
        for center in self.centers:
            self.draw_single_center(screen, center, color, alpha, stand_center, scale_factor, screen_size)

def draw_tree(screen, tree, tree_scale, color, alpha, stand_center, scale_factor, screen_size, tree_component=False):
    adjusted_position = to_screen_coordinates((tree.currentx, tree.currenty), stand_center, scale_factor, screen_size)
    if tree_component:
        adjusted_radius = max(int(tree.stemdiam * 10 * scale_factor / 2), 1) * tree_scale
    else:
        adjusted_radius = max(int(tree.height / 10 * scale_factor / 2), 1) * tree_scale
    alpha_surface = pygame.Surface((adjusted_radius * 2, adjusted_radius * 2), pygame.SRCALPHA)
    alpha_surface.fill((0, 0, 0, 0))
    pygame.draw.circle(alpha_surface, color + (int(255 * alpha),), (adjusted_radius, adjusted_radius), adjusted_radius)
    screen.blit(alpha_surface, (adjusted_position[0] - adjusted_radius, adjusted_position[1] - adjusted_radius))

def draw_plot(screen, tree_scale, plot, alpha, stand_center, scale_factor, screen_size, tree_component=False, fill_color=(0,0,255)):
    for tree in plot.trees:
        draw_tree(screen, tree, tree_scale, fill_color, alpha, stand_center, scale_factor, screen_size, tree_component)

def draw_chm(stems, screen, tree_scale, alpha, stand_center, scale_factor, screen_size, tree_component=False):
    for tree in stems:
        draw_tree(screen, tree, tree_scale, (107, 107, 107), alpha, stand_center, scale_factor, screen_size, tree_component)

def draw_arrow(screen, arrow_position, target_position, color=(0, 0, 0)):
    dx = target_position[0] - arrow_position[0]
    dy = target_position[1] - arrow_position[1]
    angle = np.arctan2(dy, dx)
    length = 50
    endx = arrow_position[0] + length * np.cos(angle)
    endy = arrow_position[1] + length * np.sin(angle)
    pygame.draw.line(screen, color, arrow_position, (endx, endy), 3)
    arrowhead_length = 10
    arrow_angle = np.pi / 6
    leftx = endx + arrowhead_length * np.cos(angle + np.pi - arrow_angle)
    lefty = endy + arrowhead_length * np.sin(angle + np.pi - arrow_angle)
    rightx = endx + arrowhead_length * np.cos(angle + np.pi + arrow_angle)
    righty = endy + arrowhead_length * np.sin(angle + np.pi + arrow_angle)
    arrowhead_points = [(endx, endy), (leftx, lefty), (rightx, righty)]
    pygame.draw.polygon(screen, color, arrowhead_points)

def draw_polygon(screen, points, color=(0, 255, 0)):
    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 3)
    for point in points:
        pygame.draw.circle(screen, color, point, 5)

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
