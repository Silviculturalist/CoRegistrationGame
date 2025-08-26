import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

class FractionalICP:
    def __init__(self, source, target, lambda_val=3.0, threshold=1e-6, max_iterations=1000):
        """
        Fractional ICP algorithm to align source to target.
        :param source: array-like, source points.
        :param target: array-like, target points.
        :param lambda_val: lambda value parameter.
        :param threshold: convergence threshold.
        :param max_iterations: maximum number of iterations.
        """
        self.source = np.array(source)
        self.target = np.array(target)
        self.lambda_val = lambda_val
        self.threshold = threshold
        self.max_iterations = max_iterations

    def frmsd(self, fraction, num_elements, subset_source, corresponding_targets):
        """
        Fractional Root Mean Squared Distance.
        :param fraction: fraction of points considered.
        :param num_elements: number of elements used in subset.
        :param subset_source: subset of source points.
        :param corresponding_targets: corresponding target points.
        :return: FRMSD value.
        """
        if len(subset_source) == 0:
            return float('inf')
        return (1 / (fraction ** self.lambda_val)) * np.sqrt(np.sum((subset_source - corresponding_targets) ** 2) / num_elements)

    def get_n_first_elements(self, num_elements, distances):
        """
        Return indices of the smallest num_elements in distances.
        """
        sorted_indices = np.argsort(distances)
        return sorted_indices[:num_elements]

    def find_correspondences(self, source, target):
        """
        Find the closest target for each source point (Euclidean).
        Uses a KD-tree for speed on larger inputs.
        Returns:
            (targets[np.ndarray], dists[np.ndarray])
        """
        if len(target) == 0 or len(source) == 0:
            return np.empty_like(source), np.array([])
        tree = cKDTree(target)
        dists, idx = tree.query(source, k=1)
        return target[idx], dists

    def find_optimal_fraction(self, corresponding_targets, distances):
        """
        Find the optimal fraction (subset size) of points that minimizes the FRMSD.
        """
        current_frmsd = float('inf')
        optimal_fraction = 0
        optimal_num_elements = 0
        total_points = len(self.source)
        for num_elements in range(1, total_points + 1):
            fraction = num_elements / total_points
            selected_indices = np.argsort(distances)[:num_elements]
            subset_source = self.source[selected_indices]
            subset_corresponding_targets = corresponding_targets[selected_indices]
            new_frmsd = self.frmsd(fraction, num_elements, subset_source, subset_corresponding_targets)
            if new_frmsd < current_frmsd:
                current_frmsd = new_frmsd
                optimal_fraction = fraction
                optimal_num_elements = num_elements
        return optimal_fraction, optimal_num_elements

    def compute_optimal_transform(self, source_subset, target_subset):
        """
        Compute 2D rotation and translation to align source_subset to target_subset.
        """
        # Compute centroids
        centroid_source = np.mean(source_subset[:, :2], axis=0)
        centroid_target = np.mean(target_subset[:, :2], axis=0)
        # Center the points
        centered_source = source_subset[:, :2] - centroid_source
        centered_target = target_subset[:, :2] - centroid_target
        # Compute covariance matrix
        H = np.dot(centered_source.T, centered_target)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        # Build a 4x4 transformation matrix (2D transform embedded in 3D homogeneous coordinates)
        R_3D = np.eye(4)
        R_3D[:2, :2] = R
        translation_3D = np.zeros(3)
        translation_3D[:2] = centroid_target - np.dot(R, centroid_source)
        R_3D[:3, 3] = translation_3D
        return R_3D

    def apply_transform(self, source, R_3D):
        """
        Apply 2D transformation (rotation and translation) to source points.
        """
        points_homogeneous = np.hstack((source, np.ones((source.shape[0], 1))))
        transformed_points_homogeneous = np.dot(points_homogeneous, R_3D.T)
        transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, [3]]
        return transformed_points

    def _iterate(self):
        """
        Perform iterative transformation updates until convergence.
        """
        corresponding_targets, distances = self.find_correspondences(self.source, self.target)
        optimal_fraction, optimal_num_elements = self.find_optimal_fraction(corresponding_targets, distances)
        selected_indices = self.get_n_first_elements(optimal_num_elements, distances)
        current_frmsd = self.frmsd(
            optimal_fraction,
            optimal_num_elements,
            self.source[selected_indices],
            corresponding_targets[selected_indices],
        )
        improvement = float('inf')
        iteration = 0
        while improvement > self.threshold and iteration < self.max_iterations:
            selected_indices = self.get_n_first_elements(optimal_num_elements, distances)
            source_subset = self.source[selected_indices]
            corresponding_subset = corresponding_targets[selected_indices]
            R_3D = self.compute_optimal_transform(source_subset, corresponding_subset)
            self.source = self.apply_transform(self.source, R_3D)
            corresponding_targets, distances = self.find_correspondences(self.source, self.target)
            optimal_fraction, optimal_num_elements = self.find_optimal_fraction(corresponding_targets, distances)
            selected_indices = self.get_n_first_elements(optimal_num_elements, distances)
            new_frmsd = self.frmsd(
                optimal_fraction,
                optimal_num_elements,
                self.source[selected_indices],
                corresponding_targets[selected_indices],
            )
            improvement = current_frmsd - new_frmsd
            current_frmsd = new_frmsd
            iteration += 1
        return self.source

    def run(self):
        """
        Execute the Fractional ICP algorithm with two-stage iteration.
        """
        # First iteration
        self._iterate()
        # Adjust lambda based on dimensionality
        if self.source.shape[1] == 3:
            self.lambda_val = 0.95
        elif self.source.shape[1] == 2:
            self.lambda_val = 1.3
        else:
            return self.source
        # Second iteration with updated lambda
        self._iterate()
        return self.source
