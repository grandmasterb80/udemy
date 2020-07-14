import numpy as np
import math
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 10
        #self.horizon = 2
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0]
        #self.reference2 = [10, 2, 3.14/2]
        self.reference2 = None
        #self.reference = self.reference1

        # description of the car
        self.wheel_base = 2.5

        self.x_obs = 5
        self.y_obs = 0.1

    def clamp(self, value, min_value, max_value):
      return max(min_value, min( max_value, value ))

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        # model inertia of steering ==> it is better to consider this in the cost function first!
        #self.beta += self.my_clamp( steering - self.beta, -math.pi * 2 * dt, math.pi * 2 * dt )
        self.beta = steering
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

        dim_x = 1.0
        dim_y_front = 3.0
        dim_y_rear = 1.0

        v_max = 10.0                # speed limit

        for k in range( self.horizon ):
          v_start = state[3]
          state = self.plant_model( state, self.dt, u[ k*2 ], u[ k*2 + 1 ] );
          delta_angle = state[2] - ref[2]
          # normalize delta_angle to a value between -pi and pi
          while delta_angle > math.pi:
            delta_angle -= 2.0 * math.pi
          while delta_angle < -math.pi:
            delta_angle += 2.0 * math.pi

          # costs for obstacle
          dist_to_obs = pow( state[0] - self.x_obs, 2.0 ) + pow( state[1] - self.y_obs, 2.0 )
          if dist_to_obs < 10.0:
            cost += 100.0 / ( 1.0 + dist_to_obs * dist_to_obs * dist_to_obs * dist_to_obs )

          # distance
          dist2 = pow( state[0] - ref[0], 2.0 ) + pow( state[1] - ref[1], 2.0 )
          dist = math.sqrt( dist2 )
          cost_dist = dist * 4.0

          # alignment costs (they get higher the closer we get to the target)
          cost_align = pow( delta_angle, 2.0 ) * ( 2.0 / max( 1.0 / 6.0, dist2 ) )

          # steering costs
          if k >= 1:
            cost_steering = ( u[ k*2 + 1 ] - u[ (k-1)*2 + 1 ] ) ** 2
          else:
            cost_steering = 0.0
          #cost_steering = 0.0

          # acceleration costs
          cost_acc = (state[3] - v_start) ** 2.0 * 10 * self.clamp( state[3] / ( 2.0 / max( 1.0 / 2.0, dist2 ) ), 0.1, 1.0 )
          #cost_acc = 0.0

          # fuel consumption: (if pedal position has same sign as velocity, consider it as throttle)
          if state[3] * u[ k*2 ] > 0:
            cost_fuel = math.fabs( u[ k*2 ] ) * ( 1.0 / max( 1.0 / 2.0, dist2 ) )
          else:
            cost_fuel = 0.0
          #cost_fuel = 0.0

          # speed limit
          if abs( state[3] * 3.6 ) > v_max:
            cost_v_max = pow( state[3] * 3.6 - v_max, 2.0) * 1000.0
          else:
            cost_v_max = 0.0
          #cost_v_max = 0.0

          cost += cost_fuel + cost_v_max + cost_dist + cost_acc + cost_steering + cost_align


        return cost

sim_run(options, ModelPredictiveControl)

