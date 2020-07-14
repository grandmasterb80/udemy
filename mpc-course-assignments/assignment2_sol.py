import numpy as np
import math
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['FULL_RECALCULATE'] = False
options['OBSTACLES'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 4
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 10, 0]
        self.reference2 = [10, 2, 3.14/2]

        # description of the car
        self.wheel_base = 2.5

    def my_clamp(self, value, min_value, max_value):
      return max(min_value, min( max_value, value ))

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        self.beta += self.my_clamp( steering - self.beta, -math.pi * 2 * dt, math.pi * 2 * dt )
        #self.beta = steering
        a_t = pedal   # air resistance & influence of accelerator pedal
        x_t = x_t + math.cos( psi_t ) * v_t * dt
        y_t = y_t + math.sin( psi_t ) * v_t * dt
        v_t = v_t + a_t * dt - v_t * dt * 0.2
        psi_t = psi_t + dt * v_t * math.tan( self.beta ) / self.wheel_base

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0
        self.beta = 0.0

        v_max = 10.0                # speed limit
        for k in range( self.horizon ):
          #gear = np.sign( state[3] )
          state = self.plant_model( state, self.dt, u[ k*2 ], u[ k*2 + 1 ] );
          dist2 = pow( state[0] - ref[0], 2.0 ) + pow( state[1] - ref[1], 2.0 )
          delta_angle = state[2] - ref[2]
          # normalize delta_angle to a value between -pi and pi
          while delta_angle > math.pi:
            delta_angle -= 2*math.pi
          while delta_angle < -math.pi:
            delta_angle += 2*math.pi
          #if gear * np.sign( state[3] ) < 0:   # extra costs for changing gear
            #cost += 1.0
          if abs( state[3] * 3.6 ) > v_max:   # limit speed
            cost += pow( state[3] * 3.6 - v_max, 2.0) * 1000.0
          cost += dist2 * 2.0
          cost += pow( delta_angle, 2.0 )
        return cost

sim_run(options, ModelPredictiveControl)
