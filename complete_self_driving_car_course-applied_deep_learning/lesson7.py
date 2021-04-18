import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def draw( x1, x2 ):
    ln = plt.plot( x1, x2 )

def sigmoid( score ):
    #print ( "sigmoid:", score )
    return 1 / (1 + np.exp( -score ) )

def compute_error( line_parameters, points, y ):
    p = sigmoid( points * line_parameters )
    return -( np.log( p ).T * y + np.log( 1 - p ).T * ( 1 - y ) ) / points.shape[ 0 ]

def gradient_descent( line_parameters, points, y, alpha ):
    m = points.shape[ 0 ]
    for i in range( 500 ):
        p = sigmoid( points * line_parameters )
        gradient = points.T * ( p - y ) * ( alpha / m )
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item( 0 )
        w2 = line_parameters.item( 1 )
        bias = line_parameters.item( 2 )
        x1 = np.array( [ points[ :, 0 ].min(), points[ :, 0 ].max() ] )
        if w2 != 0:
            x2 = -bias / w2 + x1 * ( -w1 / w2 )
        else:
            x2 = 0
    draw ( x1, x2 )

# prepare upper sample points (representing people having diabetis)
n_pts = 100
np.random.seed( 0 )
bias = np.ones( n_pts )

random_x1_values = np.random.normal( 10, 2, n_pts )
random_x2_values = np.random.normal( 12, 2, n_pts )
upper_region = np.array( [ random_x1_values, random_x2_values, bias ] ).T

# prepare lower sample points (representing people not having diabetis)
#np.random.seed(0)
random_x1_values = np.random.normal(5, 2, n_pts)
random_x2_values = np.random.normal(6, 2, n_pts)
lower_region = np.array( [ random_x1_values, random_x2_values, bias ] ).T

# concatenate lists of points
all_points = np.vstack( ( upper_region, lower_region ) )

line_parameters = np.matrix( [ np.zeros( 3 ) ] ).T
y = np.array( [ np.zeros( n_pts ), np.ones( n_pts ) ] ).reshape( n_pts * 2, 1 )

# render points into graph
_, ax = plt.subplots( figsize = ( 4, 4 ) )
ax.scatter( lower_region [ :, 0 ], lower_region [ :, 1 ], color='b' )
ax.scatter( upper_region [ :, 0 ], upper_region [ :, 1 ], color='r' )

gradient_descent( line_parameters, all_points, y, 0.06 )
plt.show()
