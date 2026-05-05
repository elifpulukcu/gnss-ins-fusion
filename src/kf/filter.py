import numpy as np

from .dynamics import build_F, discretize, STATE_DIM
from .noise import build_Q
from coord_frames import skew  

class ErrorStateKF:
    def __init__(self, P0, R_pos=None, R_vel=None):
        self.P = P0

        # Initial tuning values; adjust later
        if R_pos is None:
            self.R_pos = np.eye(3) * 5.0**2   # GNSS position noise: 5 m
        else:
            self.R_pos = R_pos

        if R_vel is None:
            self.R_vel = np.eye(3) * 0.5**2   # GNSS velocity noise: 0.5 m/s
        else:
            self.R_vel = R_vel
    
    def predict(self, state, f_b_meas, dt):
        # This function take in the state matrix, f_b_meas (raw accelerometer 
        # reading that can be passed to build_F), and the time step between ins readings
        # Then it should build_F, discretize, and propogate self.P
        # Probably should use build_F and discretize from dynamics.py,
        # and build_Q from noise.py
         # 1. Bias-corrected specific force
        f_b = f_b_meas - state.bias_a

        # 2. Build continuous-time dynamics matrix F
        F = build_F(state.C_b_e, f_b)

        # 3. Discretize to get Phi
        Phi = discretize(F, dt)

        # 4. Build process noise
        Q = build_Q(dt)

        # 5. Covariance propagation
        self.P = Phi @ self.P @ Phi.T + Q

        # 6. Keep P symmetric (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)
    
    def update(self, state, gnss_pos, gnss_vel = None):
        # compute innovation, Kalman gain, correct state, update self.P
        if gnss_vel is None:
            # Position-only update
            z = gnss_pos - state.pos_ecef

            H = np.zeros((3, STATE_DIM))
            H[:, 0:3] = np.eye(3)

            R = self.R_pos

        else:
            # Position + velocity update
            z = np.hstack([
                gnss_pos - state.pos_ecef,
                gnss_vel - state.vel_ecef,
            ])

            H = np.zeros((6, STATE_DIM))
            H[0:3, 0:3] = np.eye(3)   # position error
            H[3:6, 3:6] = np.eye(3)   # velocity error

            R = np.zeros((6, 6))
            R[0:3, 0:3] = self.R_pos
            R[3:6, 3:6] = self.R_vel

        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Estimated error state
        dx = K @ z

        # Joseph covariance update, numerically safer
        I = np.eye(STATE_DIM)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Apply feedback correction
        self.apply_feedback(state, dx)

        return dx

    def apply_feedback(self, state, dx):
        dp = dx[0:3]
        dv = dx[3:6]
        dpsi = dx[6:9]
        dba = dx[9:12]
        dbg = dx[12:15]

        # Correct position and velocity
        state.pos_ecef += dp
        state.vel_ecef += dv

        # Correct attitude.
        # dpsi is expressed in ECEF, so use left multiplication.
        state.C_b_e = (np.eye(3) + skew(dpsi)) @ state.C_b_e

        # Re-orthonormalize DCM to avoid numerical drift
        U, _, Vt = np.linalg.svd(state.C_b_e)
        state.C_b_e = U @ Vt

        # Correct bias estimates
        state.bias_a += dba
        state.bias_g += dbg