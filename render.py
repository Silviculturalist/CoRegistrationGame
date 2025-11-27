import numpy as np
import pygame
from scipy.spatial.distance import cdist
from typing import Tuple


def to_screen_coordinates(geo_coord, stand_center, scale_factor, screen_size) -> Tuple[int, int]:
    """World (x,y) -> screen (px,px)."""
    screen_x = (geo_coord[0] - stand_center[0]) * scale_factor + screen_size[0] / 2
    screen_y = (geo_coord[1] - stand_center[1]) * scale_factor + screen_size[1] / 2
    return int(screen_x), int(screen_y)


def get_viewport_scale(stand, screen_size) -> float:
    """Compute a scale so all trees fit within the window with a margin."""
    all_trees = [tree for plot in stand.plots for tree in plot.trees]
    if not all_trees:
        return 1.0
    coords = np.array([(tree.currentx, tree.currenty) for tree in all_trees])
    furthest_distance = max(np.sqrt((x - stand.center[0]) ** 2 + (y - stand.center[1]) ** 2) for x, y in coords)
    max_screen_distance = min(screen_size) / 2 - 20  # padding
    scale_factor = max_screen_distance / (furthest_distance + 2)
    return scale_factor


class PlotCenters:
    """Utility for drawing/holding plot centers near the stand center."""

    def __init__(self, stand):
        centers = np.array([plot.current_center for plot in stand.plots if plot.current_center is not None])
        self.centers = centers[cdist(centers, np.array([stand.center])).squeeze() < 70] if centers.size else np.array([])

    def draw_single_center(self, screen, center, color, alpha, stand_center, scale_factor, screen_size):
        pos = to_screen_coordinates(center, stand_center, scale_factor, screen_size)
        r = 2
        alpha_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        alpha_surface.fill((0, 0, 0, 0))
        pygame.draw.circle(alpha_surface, color + (int(255 * alpha),), (r, r), r)
        screen.blit(alpha_surface, (pos[0] - r, pos[1] - r))

    def draw_centers(self, screen, color, alpha, stand_center, scale_factor, screen_size):
        for center in self.centers:
            self.draw_single_center(screen, center, color, alpha, stand_center, scale_factor, screen_size)


def _safe_numeric(val, default=0.0):
    try:
        if val is None:
            return default
        v = float(val)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def draw_tree(screen, tree, tree_scale, color, alpha, stand_center, scale_factor, screen_size, tree_component=False):
    pos = to_screen_coordinates((tree.currentx, tree.currenty), stand_center, scale_factor, screen_size)
    if tree_component:  # draw proportional to DBH (m) -> cm
        dbh_m = _safe_numeric(tree.stemdiam, 0.0)
        raw_radius = (dbh_m * 10.0 * scale_factor / 2.0)
    else:               # draw proportional to Height (m) -> “/10” scale as before
        h_m = _safe_numeric(tree.height, 0.0)
        raw_radius = (h_m / 10.0 * scale_factor / 2.0)
    radius = max(int(round(raw_radius * tree_scale)), 1)

    alpha_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    alpha_surface.fill((0, 0, 0, 0))
    pygame.draw.circle(alpha_surface, color + (int(255 * alpha),), (radius, radius), radius)
    screen.blit(alpha_surface, (pos[0] - radius, pos[1] - radius))


def draw_plot(screen, tree_scale, plot, alpha, stand_center, scale_factor, screen_size, tree_component=False, fill_color=(0, 0, 255)):
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
    pygame.draw.polygon(screen, color, [(endx, endy), (leftx, lefty), (rightx, righty)])


def draw_polygon(screen, points, color=(0, 255, 0)):
    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 3)
    for point in points:
        pygame.draw.circle(screen, color, point, 5)


def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
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
