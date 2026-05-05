import numpy as np

from .dynamics import build_F, discretize
from .noise import build_Q

class ErrorStateKF:
    def __init__(self, P0):
        self.P = P0
    
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
    
    def update(self, state, gnss_pos, gnss_vel):
        # compute innovation, Kalman gain, correct state, update self.P
        pass