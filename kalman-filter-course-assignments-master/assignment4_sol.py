import math
import numpy as np
from sim.sim2d_prediction import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
#options['ALLOW_SPEEDING'] = False
options['ALLOW_SPEEDING'] = True

class KalmanFilter:
    def __init__(self):
        # Initial State (x, y, angle, delta_x, delta_y, delta_angle)
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

    def predictHelper(self, dt):
        F_new = np.copy(self.F)
        # Put dt into the state transition matrix.
        F_new[0,2] = dt * math.cos( self.x[3] )
        F_new[1,2] = dt * math.sin( self.x[3] )
        F_new[3,4] = dt
        x_new = F_new * self.x
        P_new = F_new * self.P * np.transpose(F_new)
        return [ F_new, P_new, x_new ]

    def predict(self, dt):
        [ self.F, self.P, self.x ] = self.predictHelper(dt)
        return [ [self.x[0]], [self.x[1]] ]

    def measure_and_update(self,measurements, dt):
        self.F[0,2] = dt * math.cos( self.x[3] )
        self.F[1,2] = dt * math.sin( self.x[3] )
        self.F[3,4] = dt
        Z = np.matrix( [measurements[0], measurements[1], 0] )
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
        
        return [ [self.x[0]], [self.x[1]] ]

    def predict_red_light(self,light_location):
        light_duration = 3
        pr = self.predictHelper( light_duration )
        x_new = pr[2]
        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]

    def predict_red_light_speed(self, light_location):
        light_duration = 3
        check = self.predict_red_light( light_location )
        if( check[0] ):
          return check
        #pr = self.predictHelper( light_duration )
        #x_new = pr[2]
        F_new = np.copy(self.F)

        dt = 1
        F_new[0,2] = dt * math.cos( self.x[3] )
        F_new[1,2] = dt * math.sin( self.x[3] )
        F_new[3,4] = dt
        x_new = F_new * self.x
        x_new[2] += 1.5

        dt = light_duration - dt
        # Put dt into the state transition matrix.
        F_new[0,2] = dt * math.cos( x_new[3] )
        F_new[1,2] = dt * math.sin( x_new[3] )
        F_new[3,4] = dt
        x_new = F_new * x_new

        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]


for i in range(0,5):
    sim_run(options,KalmanFilter,i)
