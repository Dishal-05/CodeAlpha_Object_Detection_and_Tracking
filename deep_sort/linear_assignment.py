"""
Data Association and Matching Cascade for DeepSORT.
Uses Hungarian Algorithm (linear_sum_assignment) and IoU / Mahalanobis cost gating.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

INFTY_COST = 1e5


def iou(bbox, candidates):
    """
    Compute Intersection over Union (IoU) between a bounding box and an array of candidates.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in [top-left-x, top-left-y, width, height] format.
    candidates : ndarray
        An Nx4 array of bounding boxes in the same format.

    Returns
    -------
    ndarray
        Array of IoU overlaps of length N.
    """
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]
    cand_tl = candidates[:, :2]
    cand_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.maximum(bbox_tl, cand_tl)
    br = np.minimum(bbox_br, cand_br)
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh[:, 0] * wh[:, 1]
    area_bbox = bbox[2] * bbox[3]
    area_candidates = candidates[:, 2] * candidates[:, 3]

    union = area_bbox + area_candidates - area_intersection
    return np.where(union > 0, area_intersection / union, 0.0)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    Compute cost matrix based on Intersection over Union (1 - IoU).
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment_gate()
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix


def linear_assignment_gate():
    return INFTY_COST


def min_cost_matching(
    distance_metric,
    max_distance,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """
    Solve linear assignment problem using the Hungarian algorithm (linear_sum_assignment).
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], list(track_indices), list(detection_indices)

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric,
    max_distance,
    cascade_depth,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """
    Run matching cascade to match tracks in order of age / time since update.
    Tracks seen more recently are matched first.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break

        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric,
            max_distance,
            tracks,
            detections,
            track_indices_l,
            unmatched_detections,
        )
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k[0] for k in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    kf,
    cost_matrix,
    tracks,
    detections,
    track_indices,
    detection_indices,
    gated_cost=INFTY_COST,
    only_position=False,
):
    """
    Invalidate impossible associations using the Kalman filter Mahalanobis gating distance.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kf.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

