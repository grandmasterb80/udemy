import math
import numpy as np
from sim.sim_play import sim_run
# further imnports by Daniel to add drawings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = False

class Run:
    def __init__(self):
        self.dt = 0.2
        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0] #3.14/2]
        self.reference2 = None #[10, 2, 3.14/2]
        self.firstIter = True

    def bezier( self, p_x, p_y, t ):
        p = [ 0.0, 0.0 ]        # x coord, y coord, curvature at (x, y)
        v = [ 0.0, 0.0 ]        # normalized first derivative of curvature / tangent of curvature
        c = [ 0.0, 0.0 ]        # second derivative of curvature
        if len( p_x ) != len( p_y ):
            print "bezier: check length of inputs"
            return (p, v, c, 0.0)
        N = len( p_x )
        n = N - 1
        for i in range(0, N):
            n_choose_i = 1.0 * math.factorial( n ) / ( math.factorial( n - i ) * math.factorial( i ) )
            # helper variable for b(t,p[])
            s = n_choose_i * math.pow( 1.0 - t, n - i ) * math.pow( t, i )
            # helper variable for b'(t,p[]) (first derivative of b
            if i == 0:
              s_der = - n_choose_i * n * math.pow( 1.0 - t, n - 1 )
            elif i == n:
              s_der = n_choose_i * n * math.pow( t, n - 1 )
            else:
              s_der = n_choose_i * ( 1.0 * i * math.pow( 1.0 - t, n - i ) * math.pow( t, i - 1 ) - ( n - i ) * math.pow( 1.0 - t, n - 1 - i ) * math.pow( t, i ) )
            # helper variable for b"(t,p[]) (second derivative of b
            if i == 0:
              s_der2 = n_choose_i * n * ( n -1 ) * math.pow( 1.0 - t, n - 2 )
            elif i == 1:
              s_der2 = n_choose_i * ( ( n - 1 ) * math.pow( 1 - t, n - 1 ) + ( n - 1 ) * ( n - 2 ) * t * math.pow( 1 - t, n - 3 ) - ( n - 1 ) * math.pow( 1 - t, n - 2 ) )
            elif i == n-1:
              s_der2 = n_choose_i * ( ( n - 1 ) * ( n - 2 ) * math.pow( t, n - 3 ) * ( 1 - t ) + ( n - 1 ) * math.pow( t, n - 2 ) - math.pow( t, n - 2 ) )
            elif i == n:
              s_der2 = n_choose_i * n * ( n - 1 ) * math.pow( t, n - 2 )
            else:
              s_der2 = n_choose_i * ( i * ( i - 1 ) * math.pow( t, i - 2 ) * math.pow( 1 - t, n - i ) - i * ( n - i ) * math.pow( t, i - 1 ) * math.pow( 1 - t, n - i - 1 ) + ( n - i )  * ( n - i - 1 ) * math.pow( t, i ) * math.pow( 1 - t, n - i - 2 ) - ( n - i ) * i * math.pow( t, i - 1 ) * math.pow( 1 - t, n - i - 1 ) )
            #print "x[",i,"] = ", p_x[ i ], "   y[",i,"] = ", p_y[ i ], "   s = ", s
            p[ 0 ] = p[ 0 ] + s * p_x[ i ]
            p[ 1 ] = p[ 1 ] + s * p_y[ i ]
            v[ 0 ] = v[ 0 ] + s_der * p_x[ i ]
            v[ 1 ] = v[ 1 ] + s_der * p_y[ i ]
            c[ 0 ] = c[ 0 ] + s_der2 * p_x[ i ]
            c[ 1 ] = c[ 1 ] + s_der2 * p_y[ i ]
        l_v = math.sqrt( v[ 0 ] * v[ 0 ] + v[ 1 ] * v[ 1 ] )
        v[ 0 ] = v[ 0 ] / l_v
        v[ 1 ] = v[ 1 ] / l_v
        c[ 0 ] = c[ 0 ] / l_v
        c[ 1 ] = c[ 1 ] / l_v
        c_a = math.sqrt( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] )
        return (p, v, c, c_a)

    def gen_track( self, plt, x0, y0, yaw0, x1, y1, yaw1 ):
        d0 = 7
        support_x0 = x0 + math.cos( yaw0 ) * d0
        support_y0 = y0 + math.sin( yaw0 ) * d0
        d1 = 7
        support_x1 = x1 - math.cos( yaw1 ) * d1
        support_y1 = y1 - math.sin( yaw1 ) * d1
        bp_x = [ x0, support_x0, support_x1, x1 ]
        bp_y = [ y0, support_y0, support_y1, y1 ]
        
        # plot points and supports
        plt.plot( bp_x, bp_y, 'g' )

        pl = []
        bg_x = []
        bg_y = []
        num_points = 100
        for ti in range( 0, num_points ):
            t = 1.0 * ti / num_points
            (np, tang, curv, cs) = self.bezier( bp_x, bp_y, t )
            print "bezier (", np[ 0 ], ", ", np[ 1 ], ", ", tang[ 0 ], ", ", tang[ 1 ], ", ", curv[ 0 ], ", ", curv[ 1 ], ", ", cs, ")"
            pl.append( np )
            bg_x.append( np[ 0 ] )
            bg_y.append( np[ 1 ] )
            if ti / 10 * 10 == ti:
              tang_x = [ np[0], np[0] + tang[0]  ]
              tang_y = [ np[1], np[1] + tang[1]  ]
              plt.plot( tang_x, tang_y, 'b' )

        # plot graph
        plt.plot( bg_x, bg_y, 'rx' )

        return pl

    def run( self, current_state):
        x_t = current_state[0] # X Location [m]
        y_t = current_state[1] # Y Location [m]
        psi_t = current_state[2] # Angle [rad]
        v_t = current_state[3] # Speed [m/s]
        pedal = 1 # Max: 5, Min: -5
        steering = 0.0 # Max; 0.8, Min: -0.8
        
        if self.firstIter:
            self.firstIter = False
            x_vals = [x_t, self.reference1[0] ]
            y_vals = [y_t, self.reference1[1] ]
            plt.plot( x_vals, y_vals, 'b' )
            points = self.gen_track( plt, x_t, y_t, psi_t, self.reference1[0], self.reference1[1], self.reference1[2] )
            self.lastX = x_t
            self.lastY = y_t
        else:
            x_vals = [self.lastX, x_t]
            y_vals = [self.lastY, y_t]
            plt.plot( x_vals, y_vals, 'rx' )
            self.lastX = x_t
            self.lastY = y_t

        acceleration_target = 1 # intended maximal deceleration to come to a full stop
        deceleration_target = -1 # intended maximal deceleration to come to a full stop
        stop_dist_tolerance = 0.05 # in meter
        distance_x = self.reference1[0] - x_t;
        distance_to_stop = -0.5 * v_t * v_t / deceleration_target
        if abs( distance_x ) > 0.0:
          desired_deceleration = np.max( [ deceleration_target, -0.5 * v_t * v_t / abs( distance_x ) ] )
        else:
          desired_deceleration = deceleration_target

        if distance_x > distance_to_stop + stop_dist_tolerance and v_t < 5:
          pedal = acceleration_target * 3.6
          print "**** ACCELERATION"
        elif ( abs( distance_x ) <= distance_to_stop + stop_dist_tolerance ) or ( distance_x < 0 ):
          pedal = desired_deceleration * 50
          print "**** BRAKING"
        else:
          pedal = 0
          print "**** CRUISING"

        print "distance_x = ", distance_x, "   v = ", v_t, "      distance to stop = ", distance_to_stop, "    pedal = ", pedal

        return [pedal, steering]

sim_run(options, Run)

