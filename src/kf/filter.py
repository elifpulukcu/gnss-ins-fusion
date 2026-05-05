class ErrorStateKF:
    def __init__(self, P0):
        self.P = P0
    
    def predict(self, state, f_b_meas, dt):
        # This function take in the state matrix, f_b_meas (raw accelerometer 
        # reading that can be passed to build_F), and the time step between ins readings
        # Then it should build_F, discretize, and propogate self.P
        # Probably should use build_F and discretize from dynamics.py,
        # and build_Q from noise.py
    
    def update(self, state, gnss_pos, gnss_vel):
        # compute innovation, Kalman gain, correct state, update self.P
    