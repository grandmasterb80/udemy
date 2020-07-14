import numpy as np
import math
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['FULL_RECALCULATE'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference = [50, 0, 0]

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        v_t = prev_state[3] # m/s
        a_t = pedal   # air resistance & influence of accelerator pedal
        x_t_1 = x_t + v_t * dt
        v_t_1 = v_t + a_t * dt - v_t * dt * 0.2
        return [x_t_1, 0, 0, v_t_1]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0
        v_max = 10.0                # speed limit
        pedal_prev = 0.0
        for k in range( self.horizon ):
          state = self.plant_model( state, self.dt, u[k], 0.0 );
          if state[3] * 3.6 > v_max:
            cost += pow( state[3] * 3.6 - v_max, 2.0) * 1000.0
          cost += pow( state[0] - self.reference[0], 2.0 )
          cost += pow( u[k] - pedal_prev, 2.0 ) * 3.0
          pedal_prev = u[k]
        return cost

sim_run(options, ModelPredictiveControl)
