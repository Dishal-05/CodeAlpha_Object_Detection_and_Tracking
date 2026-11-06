"""
Kalman Filter for Object Tracking in Image Space.

The 8-dimensional state vector:
    x = [u, v, gamma, h, vx, vy, vgamma, vh]
where:
    (u, v): Center coordinates of bounding box
    gamma : Aspect ratio (width / height)
    h     : Bounding box height
    vx, vy, vgamma, vh : Respective velocities

Measurements are 4-dimensional:
    z = [u, v, gamma, h]
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.
    """

    # 0.95 quantile of the chi-square distribution with N degrees of freedom
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
    }

    def __init__(self):
        ndim, dt = 4, 1.0

        # State transition matrix F (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix H (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and measurement uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (u, v, gamma, h) with aspect ratio gamma.

        Returns
        -------
        (ndarray, ndarray)
            The 8D initial mean vector and 8x8 covariance matrix.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8D mean vector at previous time step.
        covariance : ndarray
            The 8x8 covariance matrix at previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Predicted mean and covariance.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            State mean vector (8D).
        covariance : ndarray
            State covariance matrix (8x8).

        Returns
        -------
        (ndarray, ndarray)
            Projected mean (4D) and covariance (4x4).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
            + innovation_cov
        )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter update step with new measurement.

        Parameters
        ----------
        mean : ndarray
            Predicted state mean vector (8D).
        covariance : ndarray
            Predicted state covariance matrix (8x8).
        measurement : ndarray
            4D measurement vector (u, v, gamma, h).

        Returns
        -------
        (ndarray, ndarray)
            Updated mean and covariance.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute gating distance (squared Mahalanobis distance) between
        state distribution and measurements.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

