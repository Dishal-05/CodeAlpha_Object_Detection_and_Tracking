"""
Track class representation for DeepSORT.
Maintains state machine (Tentative, Confirmed, Deleted) and tracking trajectory.
"""

from collections import deque


class TrackState:
    """Enumeration of possible track states."""

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state, Kalman mean/covariance, hit streak,
    and visual appearance features.

    Parameters
    ----------
    mean : ndarray
        Initial 8D state vector.
    covariance : ndarray
        Initial 8x8 state covariance matrix.
    track_id : int
        Unique integer ID assigned to this track.
    n_init : int
        Number of consecutive detections before the track is confirmed.
    max_age : int
        Number of frames after which an unconfirmed/lost track is deleted.
    class_name : str
        Object class label (e.g., 'person', 'car').
    confidence : float
        Initial detection confidence.
    feature : ndarray, optional
        Initial appearance feature vector.
    """

    def __init__(
        self,
        mean,
        covariance,
        track_id,
        n_init,
        max_age,
        class_name="object",
        confidence=1.0,
        feature=None,
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.class_name = class_name
        self.confidence = float(confidence)

        # Class voting dictionary to maintain stable class identity over time
        self.class_votes = {class_name: float(confidence)}

        # Centroid history for drawing smooth trajectory trails (up to 30 past points)
        self.history = deque(maxlen=30)
        self._record_centroid()

    def _record_centroid(self):
        """Record current center coordinate into history."""
        tlbr = self.to_tlbr()
        cx = int((tlbr[0] + tlbr[2]) / 2.0)
        cy = int((tlbr[1] + tlbr[3]) / 2.0)
        self.history.append((cx, cy))

    def to_tlwh(self):
        """
        Get current position in bounding box format: [top-left-x, top-left-y, width, height].
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    def to_tlbr(self):
        """
        Get current position in format: [min_x, min_y, max_x, max_y].
        """
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def predict(self, kf):
        """
        Propagate the state distribution to the current time step using Kalman Filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self._record_centroid()

    def update(self, kf, detection):
        """
        Perform Kalman filter update with associated detection.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = float(detection.confidence)

        # Smooth class label using weighted voting
        self.class_votes[detection.class_name] = (
            self.class_votes.get(detection.class_name, 0.0) + float(detection.confidence)
        )
        self.class_name = max(self.class_votes.items(), key=lambda x: x[1])[0]

        self._record_centroid()

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark track as missed in current frame."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
