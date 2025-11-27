import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from ficp import FractionalICP  # noqa: E402
from trees import Plot, Tree  # noqa: E402


def _pairwise_distances(xy):
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def test_plot_transforms_are_rigid_in_2d():
    plot = Plot(plotid=1)
    plot.append_tree(Tree(1, 0.0, 0.0))
    plot.append_tree(Tree(2, 2.0, 0.0))
    plot.append_tree(Tree(3, 0.0, 3.0))

    original_xy = plot.get_tree_current_array()[:, 1:3].astype(float)

    plot.translate_plot((5.0, -2.0))
    plot.rotate_plot(35)
    plot.coordinate_flip()

    transformed_xy = plot.get_tree_current_array()[:, 1:3].astype(float)

    np.testing.assert_allclose(
        _pairwise_distances(original_xy), _pairwise_distances(transformed_xy), atol=1e-6
    )


def test_fractional_icp_transform_is_planar_and_preserves_z():
    src = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    angle = 15.0
    t = np.array([0.3, -0.4])
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    tgt_xy = src[:, :2] @ rot.T + t
    tgt = np.hstack([tgt_xy, src[:, 2:]])

    ficp = FractionalICP(src.copy(), tgt)
    T = ficp.compute_optimal_transform_2d(src, tgt)
    transformed = ficp.apply_transform_2d_xy_only(src, T)

    R = T[:2, :2]
    np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-7)
    np.testing.assert_array_equal(transformed[:, 2], src[:, 2])
