import cv2
import numpy as np
#import matplotlib.pyplot as plt
import argparse

###########################################################################

def make_coordinates( image, line ):
    slope, intercept = line
    y1 = image.shape[ 0 ]
    y2 = int( y1 * (3/5) )
    x1 = int( ( y1 - intercept ) / slope )
    x2 = int( ( y2 - intercept ) / slope )
    # print ("slope, intercept = ", slope, intercept, "   (x1,y1) - (x2,y2) = (", x1, y1, ") - (", x2, y2, ")")
    return np.array( [ x1, y1, x2, y2 ] )

###########################################################################

def average_slope_intercept( iamge, lines ):
    ###########################################################################
# // partition via our partitioning function
#     std::vector<int> labels;
#     int equilavenceClassesCount = cv::partition(linesWithoutSmall, labels, [](const Vec4i l1, const Vec4i l2){
#         return extendedBoundingRectangleLineEquivalence(
#             l1, l2,
#             // line extension length - as fraction of original line width
#             0.2,
#             // maximum allowed angle difference for lines to be considered in same equivalence class
#             2.0,
#             // thickness of bounding rectangle around each line
#             10);
#     });
#
#     std::cout << "Equivalence classes: " << equilavenceClassesCount << std::endl;
#
#     // grab a random colour for each equivalence class
#     RNG rng(215526);
#     std::vector<Scalar> colors(equilavenceClassesCount);
#     for (int i = 0; i < equilavenceClassesCount; i++){
#         colors[i] = Scalar(rng.uniform(30,255), rng.uniform(30, 255), rng.uniform(30, 255));;
#     }
#
#     // draw original detected lines
#     for (int i = 0; i < linesWithoutSmall.size(); i++){
#         Vec4i& detectedLine = linesWithoutSmall[i];
#         line(detectedLinesImg,
#              cv::Point(detectedLine[0], detectedLine[1]),
#              cv::Point(detectedLine[2], detectedLine[3]), colors[labels[i]], 2);
#     }
#
#     // build point clouds out of each equivalence classes
#     std::vector<std::vector<Point2i>> pointClouds(equilavenceClassesCount);
#     for (int i = 0; i < linesWithoutSmall.size(); i++){
#         Vec4i& detectedLine = linesWithoutSmall[i];
#         pointClouds[labels[i]].push_back(Point2i(detectedLine[0], detectedLine[1]));
#         pointClouds[labels[i]].push_back(Point2i(detectedLine[2], detectedLine[3]));
#     }
#
#     // fit line to each equivalence class point cloud
#     std::vector<Vec4i> reducedLines = std::accumulate(pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{}, [](std::vector<Vec4i> target, const std::vector<Point2i>& _pointCloud){
#         std::vector<Point2i> pointCloud = _pointCloud;
#
#         //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
#         // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
#         Vec4f lineParams; fitLine(pointCloud, lineParams, CV_DIST_L2, 0, 0.01, 0.01);
#
#         // derive the bounding xs of point cloud
#         decltype(pointCloud)::iterator minXP, maxXP;
#         std::tie(minXP, maxXP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const Point2i& p1, const Point2i& p2){ return p1.x < p2.x; });
#
#         // derive y coords of fitted line
#         float m = lineParams[1] / lineParams[0];
#         int y1 = ((minXP->x - lineParams[2]) * m) + lineParams[3];
#         int y2 = ((maxXP->x - lineParams[2]) * m) + lineParams[3];
#
#         target.push_back(Vec4i(minXP->x, y1, maxXP->x, y2));
#         return target;
#     });
#
#     for(Vec4i reduced: reducedLines){
#         line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), Scalar(255, 255, 255), 2);
#     }
#
#     imshow("Detected Lines", detectedLinesImg);
#     imshow("Reduced Lines", reducedLinesImg);
#     waitKey();
#
#     return 0;
    ###########################################################################
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        # print ( "average input:", x1, y1,x2, y2 )
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
            # print ( "left: ", left_fit_average )
            lines_to_return.append( make_coordinates( iamge, left_fit_average ) )

    if len( right_fit ):
        right_fit_average = np.average( right_fit, axis = 0 )
        if len( right_fit_average ):
            # print ( "right: ", right_fit_average )
            lines_to_return.append( make_coordinates( iamge, right_fit_average ) )

    return np.array( lines_to_return )

###########################################################################

def canny( image_gray ):
    blur = cv2.GaussianBlur( image_gray, (5, 5), 0 )
    canny = cv2.Canny( blur, 50, 100 )
    return canny

###########################################################################

def region_of_interest( image ):
    height, width = image.shape
    polygons = np.array( [
        [
            ( int( 0.0 * width ), height ),
            ( int( 1.0 * width ), height ),
            ( int( 1.0 * width ), int( 0.7 * height ) ),
            ( int( 0.8 * width ), int( 0.4 * height ) ),
            ( int( 0.2 * width ), int( 0.4 * height ) ),
            ( int( 0.0 * width ), int( 0.7 * height ) )
        ] ] )
    mask = np.zeros_like( image )
    cv2.fillPoly( mask, polygons, 255 )
    return mask
    #masked_image = cv2.bitwise_and( image, mask )
    #return masked_image

###########################################################################

def display_lines( image, lines ):
    line_image = np.zeros_like( image )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape( 4 )
            # print (x1, y1, x2, y2)
            cv2.line( line_image, (x1, y1), (x2, y2), (255, 0, 0), 10 )
    return line_image

###########################################################################

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--file", help = "Video to use for lane detection" )
parser.add_argument( "-d", "--denseopticalflow", action = "store_true", help = "Enable dense optical flow analysis" )
parser.add_argument( "-o", "--opticalflow", action = "store_true", help = "Enable optical flow analysis" )
args = parser.parse_args()

print ( "" )
print ( "" )
parser.print_help()
print ( "" )
print ( "" )
print ( "Press 'q' to stop video!" )
print ( "" )
print ( "" )
run = True
filename = "US_50_Roadtrip.mp4"
oticalflow = False
dense_opt_flow = False
if args.file:
    filename = args.file
if args.denseopticalflow:
    dense_opt_flow = True
    print ( "Dense Optical Flow Analysis enabled" )
if args.opticalflow:
    oticalflow = True
    print ( "Optical Flow Analysis enabled" )

#filename = "test2.mp4"

###### prepare optical flow analysis
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.1,
                       minDistance = 10,
                       blockSize = 32 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 8,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

# Create some random colors
color = np.random.randint(0,255,(1000,3))
first_frame = True
tracked_images = 0
######
# flag for first frame in dense optical flow analysis
first_frame_dof = True
######

cap = cv2.VideoCapture( filename )

while( run and cap.isOpened() ):
    ret, frame = cap.read()
    while run and not ret:
        cap.release()
        cap = cv2.VideoCapture( filename )
        ret, frame = cap.read()
        if cv2.waitKey( 1 ) == ord('q'):
            run = False
    lane_image = np.copy( frame )
    gray = cv2.cvtColor( lane_image, cv2.COLOR_RGB2GRAY )
    roi_mask = region_of_interest( gray )
    if dense_opt_flow:
        if first_frame_dof:
            dof_prvs = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            hsv = np.zeros_like( frame )
            hsv[ ..., 1 ] = 255
            first_frame_dof = False
        else:
            next = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            flow = cv2.calcOpticalFlowFarneback( dof_prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0 )

            mag, ang = cv2.cartToPolar( flow[...,0], flow[...,1] )
            hsv[ ..., 0 ] = ang * 180 / np.pi / 2
            hsv[ ..., 2 ] = cv2.normalize( mag, None, 0, 255, cv2.NORM_MINMAX )
            rgb = cv2.cvtColor( hsv, cv2.COLOR_HSV2BGR )

            cv2.imshow('Dense Optical Flow',rgb )
            dof_prvs = next

    if oticalflow:
        if first_frame:
            print ( gray.shape )
            # Take first frame and find corners in it
            p0 = cv2.goodFeaturesToTrack( gray, mask = None, **feature_params )
            #p0 = cv2.goodFeaturesToTrack( gray, mask = roi_mask, **feature_params )
            # Create a mask image for drawing purposes
            mask = np.zeros_like( frame )
            first_frame = ( p0 is None )
            tracked_images = 0
        else:
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK( old_gray, gray, p0, None, **lk_params )
            if len( p0 ) > 0 and len( p1 ) > 0:
                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]

                mask = np.zeros_like( frame )
                # draw the tracks
                for i,( new, old ) in enumerate( zip( good_new, good_old ) ):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line( mask, (a,b),(c,d), color[i].tolist(), 2 )
                    frame = cv2.circle( frame, (a,b), 5, color[i].tolist(), -1 )
                flow_img = cv2.add( frame, mask )
                cv2.imshow( "Flow", flow_img );
                if len( good_new ) < 100 or tracked_images > 25:
                    p0 = cv2.goodFeaturesToTrack( gray, mask = None, **feature_params )
                    #p0 = cv2.goodFeaturesToTrack( gray, mask = roi_mask, **feature_params )
                    tracked_images = 0
                else:
                    p0 = good_new.reshape( -1, 1, 2 )
                    tracked_images += 1
            else:
                first_frame = True # needs a reinitialization

            # Now update the previous frame and previous points
        old_gray = gray.copy()

    canny_img = canny( gray )
    canny_roi = cv2.bitwise_and( canny_img, roi_mask )
    lines = cv2.HoughLinesP( canny_roi, 2, 1*np.pi/180, 100, np.array( [ ] ), minLineLength=30, maxLineGap=5 )
    #averaged_lines = average_slope_intercept( lane_image, lines )
    #image_with_lanes = display_lines( lane_image, averaged_lines )
    image_with_lanes = display_lines( lane_image, lines )
    combo_image = cv2.addWeighted( lane_image, 0.8, image_with_lanes, 1.0, 1.0 )
    # cv2.imshow( "Source", frame );
    #cv2.imshow( "Result", canny_img );
    cv2.imshow( "Region of Interest", canny_roi );
    cv2.imshow( "Lanes", combo_image );
    if cv2.waitKey( 1 ) & 0xFF == ord('q'):
        run = False

cap.release()
cv2.destroyAllWindows()
