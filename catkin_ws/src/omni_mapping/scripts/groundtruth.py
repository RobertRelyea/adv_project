#!/usr/bin/env python

import rospy
from visualization_msgs.msg import MarkerArray
from utils import pubMarkerArrayZ, pubMarkerArray
import numpy as np

##### Light Corner Measurements and Accuracy

def I2M(np_arr):
    ''' Convert all dimensions in the given numpy array to meters from inches
        @param np_arr The numpy array with dimensions to be converted
    '''
    return np_arr * 0.0254

def correctPos(np_arr):
    ''' Transforms the given points to match the origin of the robot map
        @param np_arr The xyz points to be transformed
    '''
    # Reflect Y axis (my bad...)
    np_arr *= np.array([1,-1,1])
    # Offset from robot map origin
    np_arr += np.array([-2.2352, 0.2032, -1.1938])
    return np_arr

def findClosestPoint(point, lights):
    ''' Finds the closest point and returns the squared difference for use in 
        the MSE calculation

        @param point The point to search for
        @param lights The GT light corner positions
    '''
    min_dist = 9999.0
    for light in lights:
        for corner in light:
            dist = np.sum(np.square(np.subtract(point, corner)))
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

def getMSE(points, lights):
    ''' Calculates the Mean-Squared Error between a set of measured map points 
        the ground truth light corner points.

    @param points The measured map points
    @param lights The GT light corner positions
    '''
    squared_error = 0.0
    for point in points:
        squared_error += findClosestPoint(point, lights)

    mean_squared_error = squared_error / len(points)
    return mean_squared_error

# Hardcoded ceiling light positions obtained through tiresome manual measurement.
light_0 = np.array([[0, 0, 94], [0, 24, 94], [48, 0, 94], [48, 24, 94]])
light_1 = np.array([[93, 0, 132], [93, 24, 132], [141, 0, 132], [141, 24, 132]])
light_2 = np.array([[184, 0, 132], [184, 24, 132], [232, 0, 132], [232, 24, 132]])
light_3 = np.array([[280, 0, 132], [280, 24, 132], [328, 0, 132], [328, 24, 132]])
light_4 = np.array([[0, 74, 94], [0, 98, 94], [48, 74, 94], [48, 98, 94]])
light_5 = np.array([[93, 74, 132], [93, 98, 132], [141, 74, 132], [141, 98, 132]])
light_6 = np.array([[186, 69, 132], [186, 93, 132], [234, 69, 132], [234, 93, 132]])
light_7 = np.array([[282, 74, 132], [282, 98, 132], [330, 74, 132], [330, 98, 132]])
light_8 = np.array([[93, -74, 94], [93, -50, 94], [141, -74, 94], [141, -50, 94]])
light_9 = np.array([[189, -74, 94], [189, -50, 94], [237, -74, 94], [237, -50, 94]])
light_10 = np.array([[287, -74, 94], [287, -50, 94], [335, -74, 94], [335, -50, 94]])

# Gather em up
lights = [light_0, light_1, light_2, light_3, light_4, light_5, light_6, light_7, 
          light_8, light_9, light_10]

# Convert from inch measurements to meters and transform to robot land
for light_idx in range(len(lights)):
    lights[light_idx] = correctPos(I2M(lights[light_idx]))

my_mapped_points = [[-1.02, 0.222, 1.164],
[-1.015, -0.397, 1.2612],
[0.187, 0.138, 2.141],
[0.053, -0.34, 2.43],
[1.342, 0.136, 2.08],
[1.3307, -0.335, 2.27],
[2.597, 0.109, 2.338],
[2.614, -0.349, 2.496],
[3.798, 0.116, 2.16],
[3.778, -0.427, 2.26],
[5.1, 0.153, 2.436],
[5.186, -0.4, 2.213],
[6.19, 0.04, 2.09],
[6.16, -0.545, 2.08],
[1.316, -1.636, 2.31],
[1.33, -2.18, 2.337],
[2.457, -1.533, 2.117],
[3.752, -1.624, 2.156],			
[6.176, -1.676, 2.302],
[6.1674, -2.271, 2.012],
[0.1448, 1.514, 1.202],
[1.367, 1.482, 1.33],
[2.57, 2.243, 1.428],
[2.791, 1.377, 1.292],
[3.85, 1.386, 1.159],
[5.259, 1.195, 1.215],
[6.562, 1.108, 0.996]]

svo_mapped_points = [[-0.9828, 0.1411, 1.192],
[-0.968, -0.40515, 1.2005],
[-2.14, 0.122, 1.238], 
[0.0752, -0.3496, 2.185],
[0.225, 0.1487, 2.159],
[1.3777, 0.18996, 2.178],
[1.31 , -0.3898, 2.1386],
[2.4226, -0.361, 2.2244],
[2.407, 0.1944, 2.1934],
[3.6288, 0.1754, 2.0744], 
[4.913, 0.1986, 2.1299],
[4.8937, -0.3252, 2.116],
[6.1 , -0.3525, 2.14434], 
[-2.179, -2.2957, 1.3534],
[-2.216, -1.743, 1.345],
[-0.99, -1.71159, 1.324],
[-0.9644, -2.22, 1.283],
[0.195, -1.679, 2.138],
[0.209, -2.198, 2.203],
[1.3486, -1.6766, 2.2259], 
[2.5 , -2.11, 2.126],
[2.489, -1.59311, 2.159777],
[3.694, -2.1165, 2.17313],
[3.6395, -1.4906, 2.208],
[4.8736, -2.619, 2.2377],
[4.8395, -1.659, 2.15], 
[0.12307, 1.4399, 1.16733849],
[0.07654, 2.0318, 1.192],
[1.34345, 1.479, 1.19786],
[1.28277, 2.04338, 1.3],
[2.589, 1.44669, 1.1544],
[3.7116, 1.4356, 1.2589],
[5.14157, 1.604, 1.4413],
[6.1094, 1.51, 1.374]]


print("My Mapping MSE: {}".format(getMSE(my_mapped_points, lights)))
print("SVO Mapping MSE: {}".format(getMSE(svo_mapped_points, lights)))

##### Odometry Measurements and Accuracy

        # 1st G,            2nd G,            End
gt_odom = [[[1.8288, 0.9144]],[[3.6576, -0.9144]], [[5.842, 0.0]]]
raw_odom= [[1.836, 0.8269], [3.373, -1.4786],  [6.0715, -0.9406]]
my_odom = [[1.8849, 1.045], [3.673, -1.0449], [6.16066, -0.1455]]
svo_odom= [[1.8106, 0.9309], [3.6509, -0.855], [5.758, 0.05188]]

print("Raw Odom MSE: {}".format(getMSE(raw_odom, gt_odom)))
print("My Odom MSE: {}".format(getMSE(my_odom, gt_odom)))
print("SVO Odom MSE: {}".format(getMSE(svo_odom, gt_odom)))

svo_gt = [[[1.8106, 0.9309]], [[3.6509, -0.855]], [[5.758, 0.05188]]]
print("Raw Odom MSE (Compared to SVO): {}".format(getMSE(raw_odom, svo_gt)))
print("My Odom MSE (Compared to SVO): {}".format(getMSE(my_odom, svo_gt)))


# Yellow odometry marker coords
ycoords = [[0.0, 0.0], [5.842, 0.0],   [0.0, 5.842]]
# Green odometry marker coords
gcoords = [[1.8288, 0.9144],[3.6576, -0.9144],    [-0.9144, 1.8288],[0.9144, 3.6576]]
# Purple odometry marker coords
pcoords = [[1.8288, -0.9144],[3.6576, 0.9144],    [0.9144, 1.8288],[-0.9144, 3.6576]]

def pub_odom_landmarks():
    rospy.init_node('light_landmarks_node', anonymous=True)
    marker_pub = rospy.Publisher("light_landmarks_node", MarkerArray, queue_size=10)
    r = rospy.Rate(10)

    while(not rospy.is_shutdown()):
        marker_id = 0
        for light in lights:
            pubMarkerArrayZ(light, "imhs/odom", marker_pub, (1.0,0.0,0.0), marker_id=marker_id)
            pubMarkerArrayZ(light, "world_gt", marker_pub, (1.0,0.0,0.0), marker_id=marker_id)
            marker_id += len(light)

        pubMarkerArray(ycoords, "imhs/odom", marker_pub, (0.0, 1.0, 1.0), True, marker_id=marker_id)
        pubMarkerArray(ycoords, "world_gt", marker_pub, (0.0, 1.0, 1.0), True, marker_id=marker_id)
        marker_id += len(ycoords)
        pubMarkerArray(gcoords, "imhs/odom", marker_pub, (0.0, 1.0, 0.0), True, marker_id=marker_id)
        pubMarkerArray(gcoords, "world_gt", marker_pub, (0.0, 1.0, 0.0), True, marker_id=marker_id)
        marker_id += len(gcoords)
        pubMarkerArray(pcoords, "imhs/odom", marker_pub, (1.0, 0.0, 0.7), True, marker_id=marker_id)
        pubMarkerArray(pcoords, "world_gt", marker_pub, (1.0, 0.0, 0.7), True, marker_id=marker_id)

        r.sleep()



if __name__ == '__main__':
    try:
        # point = np.array([0,0,0], dtype='float32')
        # points = np.array([[[0,0,9]], [[0,0,4]]], dtype='float32')
        # print(findClosestPoint(point, points))
        pub_odom_landmarks()
    except rospy.ROSInterruptException:
        pass
