import cv2
import numpy as np
#import matplotlib.pyplot as plt

def make_coordinates( image, line ):
    slope, intercept = line
    y1 = image.shape[ 0 ]
    y2 = int( y1 * (3/5) )
    x1 = int( ( y1 - intercept ) / slope )
    x2 = int( ( y2 - intercept ) / slope )
    return np.array( [ x1, y1, x2, y2 ] )

def average_slope_intercept( iamge, lines ):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit( (x1, x2), (y1, y2), 1 )
        slope = parameters[ 0 ]
        intercept = parameters[ 1 ]
        if slope > 0:
            right_fit.append( ( slope, intercept ) )
        else:
            left_fit.append( ( slope, intercept ) )

    lines_to_return = []
    if len( left_fit ):
        left_fit_average = np.average( left_fit, axis = 0 )
        if len( left_fit_average ):
            lines_to_return.append( make_coordinates( iamge, left_fit_average ) )

    if len( right_fit ):
        right_fit_average = np.average( right_fit, axis = 0 )
        if len( right_fit_average ):
            lines_to_return.append( make_coordinates( iamge, right_fit_average ) )

    return np.array( lines_to_return )


def canny( image ):
    gray = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
    blur = cv2.GaussianBlur( gray, (5, 5), 0 )
    canny = cv2.Canny( blur, 50, 150 )
    return canny


def region_of_interest( image ):
    height = image.shape[ 0 ]
    polygons = np.array( [
        [ ( 200,height ), ( 1100, height ), ( 550, 250 ) ]
        ] )
    mask = np.zeros_like( image )
    cv2.fillPoly( mask, polygons, 255 )
    masked_image = cv2.bitwise_and( image, mask )
    return masked_image

def display_lines( image, lines ):
    line_image = np.zeros_like( image )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape( 4 )
            cv2.line( line_image, (x1, y1), (x2, y2), (255, 0, 0), 10 )
    return line_image

print ( "" )
print ( "" )
print ( "Press 'q' to stop video!" )
print ( "" )
print ( "" )
run = True
filename = "US_50_Roadtrip.mp4"
#filename = "test2.mp4"
cap = cv2.VideoCapture( filename )
while( run and cap.isOpened() ):
    ret, frame = cap.read()
    while run and not ret:
        cap.release()
        cap = cv2.VideoCapture("test2.mp4")
        ret, frame = cap.read()
        if cv2.waitKey( 1 ) == ord('q'):
            run = False
    lane_image = np.copy( frame )
    canny_img = canny( lane_image )
    canny_roi = region_of_interest( canny_img )
    lines = cv2.HoughLinesP( canny_roi, 2, 1*np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )
    averaged_lines = average_slope_intercept( lane_image, lines )
    image_with_lanes = display_lines( lane_image, averaged_lines )
    combo_image = cv2.addWeighted( lane_image, 0.8, image_with_lanes, 1.0, 1.0 )
    # cv2.imshow( "Source", frame );
    #cv2.imshow( "Result", canny_img );
    #cv2.imshow( "Region of Interest", canny_roi );
    cv2.imshow( "Lanes", combo_image );
    if cv2.waitKey( 1 ) & 0xFF == ord('q'):
        run = False

cap.release()
cv2.destroyAllWindows()
