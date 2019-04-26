import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['CONSTANT_SPEED'] = False

class KalmanFilterToy:
    def __init__(self):
        self.v = 0
        self.prev_x = 0
        self.prev_t = 0
    def predict(self,t):
        prediction = self.prev_x + self.v * ( t - self.prev_t )
        return prediction
    def measure_and_update(self,x,t):
        # assignment1: set weight to constant.
        # Result: 0.1 (small values) ==> high latency, low noise
        #         0.9 (higher values) ==> fast adaption, high noise
        #weight = 0.5
        # assignment2: for high deviation of measured value and estimated value, we assume a change of velocity
        # Result: fast adaption, low noise when estimated speed and actual speed get close together
        weight = min( 1.0, abs( x - self.predict(t) ) * 0.8 )

        measured_v = ( x - self.prev_x ) / ( t - self.prev_t )
        self.v += ( measured_v - self.v ) * weight
        self.prev_x = x
        self.prev_t = t
        return


sim_run(options,KalmanFilterToy)
