import sys
import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['CONSTANT_SPEED'] = False

class KalmanFilter:
    def __init__(self):
        self.v = 0
        self.prev_time = 0
        self.error = np.matrix([[0.],
                                [0.]])

        self.tolerance = np.matrix([[0.5],
                                    [0.5]])
        # Initial State
        self.x = np.matrix([[0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[1000.0, 0.],
                            [0., 1000.0]])

        # Next State Function
        self.F = np.matrix([[1., 1.],
                            [0., 1.]])

        # Measurement Function
        self.H = np.matrix([[1., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[0.01]])

        # Identity Matrix
        self.I = np.matrix([[1., 0.],
                            [0., 1.]])

        self.predict_x = self.x

    def predict(self,t):
        # Calculate dt.
        dt = t - self.prev_time
        oldx = self.x
        # Put dt into the state transition matrix.
        self.F[0,1] = dt
        self.x = self.F * self.x
        self.P = self.F * self.P * np.transpose(self.F)
        self.predict_x = self.x
        return self.x[0,0]
    def measure_and_update(self,measurements,t):
        dt = t - self.prev_time
        self.F[0,1] = dt
        Z = np.matrix(measurements)
        y = np.transpose(Z) - self.H * self.x
        S = self.H * self.P * np.transpose(self.H) + self.R
        K = self.P * np.transpose(self.H) * np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (self.I - K * self.H) * self.P

        self.error = np.matrix.clip( np.abs(self.x - self.predict_x) - self.tolerance, 0.0, 10.0 )
        self.P += self.error * 400.0
        print ("self.error = ", self.error)


        self.v = self.x[1,0]
        self.prev_time = t
        return


sim_run(options,KalmanFilter)
