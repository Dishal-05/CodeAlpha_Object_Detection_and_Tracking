"""
Nearest Neighbor Distance Metric for DeepSORT.
Computes cosine / Euclidean distance between detection features and track appearance galleries.
"""

import numpy as np


def _pdist(a, b):
    """
    Compute pair-wise squared Euclidean distance between points in a and b.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2 = np.square(a).sum(axis=1)[:, np.newaxis]
    b2 = np.square(b).sum(axis=1)[np.newaxis, :]
    r2 = -2.0 * np.dot(a, b.T) + a2 + b2
    return np.clip(r2, 0.0, float(np.inf))


def _cosine_distance(a, b, data_is_normalized=False):
    """
    Compute pair-wise cosine distance between points in a and b.
    Cosine distance = 1 - cosine_similarity.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    if not data_is_normalized:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


class NearestNeighborDistanceMetric:
    """
    A nearest neighbor distance metric that manages a feature gallery for each track.

    Parameters
    ----------
    metric : str
        Distance metric: 'cosine' or 'euclidean'.
    matching_threshold : float
        Threshold above which associations are considered invalid.
    budget : int, optional
        Maximum number of feature vectors saved per track ID.
    """

    def __init__(self, metric="cosine", matching_threshold=0.2, budget=None):
        if metric == "cosine":
            self._metric = _cosine_distance
        elif metric == "euclidean":
            self._metric = _pdist
        else:
            raise ValueError(f"Invalid metric; must be 'cosine' or 'euclidean'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def as_default(self):
        """Reset metric storage."""
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """
        Update the distance metric with new samples.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of target IDs corresponding to features.
        active_targets : list
            List of target IDs currently active in tracker.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets if k in self.samples}

    def distance(self, features, targets):
        """
        Compute minimum distance between detection features and target galleries.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : list
            A list of target IDs.

        Returns
        -------
        ndarray
            An Nxlen(targets) cost matrix.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = np.min(
                self._metric(self.samples[target], features), axis=0
            )
        return cost_matrix

