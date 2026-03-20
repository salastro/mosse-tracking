import numpy as np


class KalmanCA2D:
    """
    2D constant-acceleration Kalman filter.
    State: [x, y, vx, vy, ax, ay]^T
    Measurement: [x, y]^T
    """

    def __init__(self, dt, process_var=1e-2, measurement_var=25.0):
        self.dt = float(dt)

        dt2 = self.dt * self.dt
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0, 0.5 * dt2, 0.0],
            [0.0, 1.0, 0.0, self.dt, 0.0, 0.5 * dt2],
            [0.0, 0.0, 1.0, 0.0, self.dt, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)

        q_1d = self._ca_process_noise_1d(self.dt, process_var)
        self.Q = np.zeros((6, 6), dtype=np.float64)
        idx_x = np.ix_([0, 2, 4], [0, 2, 4])
        idx_y = np.ix_([1, 3, 5], [1, 3, 5])
        self.Q[idx_x] = q_1d
        self.Q[idx_y] = q_1d

        self.R = np.eye(2, dtype=np.float64) * float(measurement_var)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.x = np.zeros((6, 1), dtype=np.float64)

    @staticmethod
    def _ca_process_noise_1d(dt, var):
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        q = float(var)
        return q * np.array([
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ], dtype=np.float64)

    def initialize(self, x, y):
        self.x[:] = 0.0
        self.x[0, 0] = float(x)
        self.x[1, 0] = float(y)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, z):
        z = np.asarray(z, dtype=np.float64).reshape(2, 1)
        innovation = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.x[0, 0], self.x[1, 0]
