import numpy as np
import math
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [10,10]

options['DRIVE_IN_CIRCLE'] = False
# If False, measurements will be x,y.
# If True, measurements will be x,y, and current angle of the car.
# Required if you want to pass the driving in circle.
options['MEASURE_ANGLE'] = True
options['RECIEVE_INPUTS'] = False

class KalmanFilter:
    def __init__(self):
        # Expected measurement noise
        self.threshold = np.matrix([[0.1],
                                    [0.1]])
        ## Initial State (x, y, angle, delta_x, delta_y, delta_angle)
        # Initial State (x, y, delta_x, delta_y)
        # size: #states x 1
        self.x = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.],
                            [0.]])

        # Uncertainty Matrix
        # size: #states x #states
        self.P = np.matrix([[1000.0, 0., 0., 0., 0.],
                            [0., 1000.0, 0., 0., 0.],
                            [0., 0., 1000.0, 0., 0.],
                            [0., 0., 0., 1000.0, 0.],
                            [0., 0., 0., 0., 1000.0]])

        # Next State Function
        # size: #states x #states
        self.F = np.matrix([[1., 0., 1., 0., 0.],
                            [0., 1., 1., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 1.],
                            [0., 0., 0., 0., 1.]])

        # Measurement Function
        # size: #measurements x #states
        self.H = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 1., 0.]])

        # Measurement Uncertainty
        # size: #measurements x #measurements
        self.R = np.matrix([[ 50.0,  0.0,  0.0],
                            [  0.0, 50.0,  0.0],
                            [  0.0,  0.0, 20.0]])

        # Identity Matrix
        # size: #states x #states
        self.I = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 1.]])

    def predict(self, dt):
        # Put dt into the state transition matrix.
        self.F[0,2] = dt * math.cos( self.x[3] )
        self.F[1,2] = dt * math.sin( self.x[3] )
        self.F[3,4] = dt
        self.x = self.F * self.x
        self.P = self.F * self.P * np.transpose(self.F)
        return [ [self.x[0]], [self.x[1]], [self.x[2]] ]

    def measure_and_update(self,measurements, dt):
        self.F[0,2] = dt * math.cos( self.x[3] )
        self.F[1,2] = dt * math.sin( self.x[3] )
        self.F[3,4] = dt
        Z = np.matrix( measurements )
        # sizeof y: 2x1
        y = np.transpose(Z) - self.H * self.x

        # goal: if derivation of predicted value and the measured value is really high, we increase the Uncertainty matrix
        # abs(y) ==> derivation of measured value to predicted value
        # Subtract the threshold, i.e. all derivation below threshold is considered as measurement noise.
        # Clip and scale it the result to map it to Uncertainty.
        #self.P += np.diag( np.matrix.clip( np.abs(y) - self.threshold, 0.0, 10.0 ) * 400.0 + np.matrix( [[10.1], [10.1]] ) )
        self.P[0,0] +=  0.5
        self.P[1,1] +=  0.5
        self.P[2,2] +=  1.0
        self.P[3,3] +=  0.5
        self.P[4,4] +=  0.5

        # sizeof S: 2x2
        S = self.H * self.P * np.transpose(self.H) + self.R
        # sizeof K: 4x2
        K = self.P * np.transpose(self.H) * np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (self.I - K * self.H) * self.P
        
        return [ [self.x[0]], [self.x[1]], [self.x[2]] ]


    def recieve_inputs(self, u_steer, u_pedal):
        return

sim_run(options,KalmanFilter)
