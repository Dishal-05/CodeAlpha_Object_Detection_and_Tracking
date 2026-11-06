"""
Detection class representation for DeepSORT.
Stores bounding box, confidence score, class name, and appearance feature vector.
"""

import numpy as np


class Detection:
    """
    Represents an object bounding box detection in a single frame.

    Parameters
    ----------
    tlwh : list or np.ndarray
        Bounding box format: [top-left-x, top-left-y, width, height].
    confidence : float
        Detector confidence score.
    class_name : str
        Class label of detected object (e.g., 'person', 'car').
    feature : np.ndarray, optional
        Appearance feature vector representing the visual embedding.
    """

    def __init__(self, tlwh, confidence, class_name, feature=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.class_name = str(class_name)
        self.feature = feature

    def to_tlbr(self):
        """
        Convert bounding box to [top-left-x, top-left-y, bottom-right-x, bottom-right-y].
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to [center-x, center-y, aspect-ratio, height],
        where aspect ratio is width / height.
        This is the standard state vector format for DeepSORT Kalman Filter.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret

