"""
DeepSORT Multi-Object Tracker.
Combines Kalman Filtering motion prediction, appearance feature matching cascade,
and IoU Hungarian association to maintain unique persistent IDs across video frames.
"""

import numpy as np
from .kalman_filter import KalmanFilter
from .linear_assignment import (
    matching_cascade,
    min_cost_matching,
    gate_cost_matrix,
    iou_cost,
)
from .track import Track
from .detection import Detection
from .nn_matching import NearestNeighborDistanceMetric


class DeepSortTracker:
    """
    DeepSORT multi-object tracker.

    Parameters
    ----------
    max_cosine_distance : float
        Maximum cosine distance for valid appearance match (default 0.25).
    nn_budget : int, optional
        Maximum number of appearance features retained per track gallery.
    max_iou_distance : float
        Maximum 1-IoU distance for geometric matching (default 0.7).
    max_age : int
        Maximum number of consecutive missed frames before track deletion (default 30).
    n_init : int
        Number of consecutive detections required to confirm a track (default 3).
    """

    def __init__(
        self,
        max_cosine_distance=0.25,
        nn_budget=100,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
    ):
        self.metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate state distributions of all active tracks."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """
        Perform data association and state update.

        Parameters
        ----------
        detections : list[Detection]
            A list of Detection objects detected in the current frame.
        """
        # Run matching cascade and IoU association
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # 1. Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # 2. Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 3. Create new tentative tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 4. Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 5. Update appearance distance metric gallery with confirmed tracks
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id] * len(track.features)
            track.features = []  # clear buffer once committed to metric gallery
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
        """
        Two-stage association:
        Stage 1: Appearance matching cascade on confirmed tracks.
        Stage 2: IoU association for unconfirmed tracks and unmatched recent tracks.
        """

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices
            )
            return cost_matrix

        # Split tracks into confirmed and unconfirmed
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()
        ]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Stage 1: Matching cascade on confirmed tracks with appearance features
        has_features = len(detections) > 0 and detections[0].feature is not None
        if has_features and len(self.metric.samples) > 0:
            matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks,
            )
        else:
            matches_a = []
            unmatched_tracks_a = confirmed_tracks
            unmatched_detections = list(range(len(detections)))

        # Stage 2: IoU matching for remaining candidates
        iou_track_candidates = unconfirmed_tracks + [
            k
            for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k
            for k in unmatched_tracks_a
            if self.tracks[k].time_since_update > 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """Create new track from unmatched detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                class_name=detection.class_name,
                confidence=detection.confidence,
                feature=detection.feature,
            )
        )
        self._next_id += 1

    def update_tracks(self, raw_detections, frame=None):
        """
        Convenience wrapper compatible with standard YOLO bounding boxes.

        Parameters
        ----------
        raw_detections : list of tuples/lists
            Format: [([x1, y1, w, h], confidence, class_name), ...]
        frame : ndarray, optional
            Current frame image (BGR) for feature extraction.

        Returns
        -------
        list[Track]
            List of confirmed active tracks.
        """
        detections = []
        for det in raw_detections:
            tlwh, conf, cls_name = det[0], det[1], det[2]
            feature = self._extract_feature(tlwh, frame)
            detections.append(Detection(tlwh, conf, cls_name, feature))

        self.predict()
        self.update(detections)

        # Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed()]

    def _extract_feature(self, tlwh, frame):
        """
        Extract normalized spatial-color appearance feature vector.
        Acts as a reliable visual fingerprint for Re-ID when deep embedder weights are offline.
        """
        if frame is None or frame.size == 0:
            return np.ones(128, dtype=np.float32) / np.sqrt(128)

        h_img, w_img = frame.shape[:2]
        x1 = max(0, int(tlwh[0]))
        y1 = max(0, int(tlwh[1]))
        x2 = min(w_img, int(tlwh[0] + tlwh[2]))
        y2 = min(h_img, int(tlwh[1] + tlwh[3]))

        if x2 <= x1 or y2 <= y1:
            return np.ones(128, dtype=np.float32) / np.sqrt(128)

        patch = frame[y1:y2, x1:x2]
        # Multi-bin color histogram in HSV space
        import cv2

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
        )
        hist = hist.flatten()
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        else:
            hist = np.ones(128, dtype=np.float32) / np.sqrt(128)
        return hist.astype(np.float32)

    def reset(self):
        """Reset all tracks and ID counter."""
        self.tracks = []
        self._next_id = 1
        self.metric.as_default()

