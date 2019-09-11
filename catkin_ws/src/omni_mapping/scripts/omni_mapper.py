#!/usr/bin/env python

from concurrent.futures import ThreadPoolExecutor

# For ROS communication
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# Data processing
import numpy as np
import cv2
import math

# For debugging
import pdb

# from map_builder import OrbMatcher
from orb_matcher import OrbTracker

last_frame = np.array([])
last_frame_time = 0
bridge = CvBridge()

# Odometry data
current_odom_t = np.array([0.0, 0.0, 0.0])
current_odom_r = np.array([0.0, 0.0, 0.0])
last_odom_time = 0
last_odom_t = np.array([0.0, 0.0, 0.0])
last_odom_r = np.array([0.0, 0.0, 0.0])
new_frame = False

matcher = {}
# cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)

def handleNewFrame():
    global last_frame, last_frame_time, last_odom_t, matcher
    matcher.newFrame(last_frame, last_odom_t, last_odom_r)
    print("Submitted frame")

# Handles new omnivision image data.
def omnivisionCB(data):
    global last_frame, last_frame_time, current_odom_t, last_odom_t, current_odom_r, last_odom_r, new_frame, executor

    if type(data) == CompressedImage:
        np_arr = np.fromstring(data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        frame = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (1440, 1440))
    try:
        # if (np.sqrt(np.sum(np.square(current_odom_t - last_odom_t))) > 0.05) or \
        #     (abs(current_odom_r[2] - last_odom_r[2]) > 0.01):
        last_frame = frame
        # print(current_odom_t)
        last_odom_t = current_odom_t
        last_odom_r = current_odom_r
        # Reset the current local odometry vector
        current_odom_r = np.asarray([0.0, 0.0, 0.0])
        current_odom_t = np.asarray([0.0, 0.0, 0.0])
        last_frame_time = rospy.get_time()
        # Let the executor handle the new frame
        # executor.submit(handleNewFrame)
        matcher.newFrame(last_frame, last_odom_t, last_odom_r)

    except CvBridgeError, e:
        rospy.logerr(e)

overall_odom = np.array([0.,0.,0.])
overall_th = 0

# Odometry Callback
def odomCallback(data):
    ''' Update a running position and orientation displacement vector
    '''
    current_odom_time = rospy.get_time()
    global current_odom_t, current_odom_r, last_odom_time
    # First reading will be used as the reference start time
    if last_odom_time == 0:
        last_odom_time = current_odom_time
    # Subsequent readings will now have a time delta
    else:
        time_delta = current_odom_time - last_odom_time
        # Save latest odometry information
        linear = data.twist.twist.linear
        angular = data.twist.twist.angular
        # Integrate angular velocity into current orientation vector
        th_change = angular.z * time_delta
        current_odom_r += np.array([0.0, 0.0, th_change])

        # Integrate linear velocity into current position vector using current
        x_change = linear.x * np.cos(current_odom_r[2]) * time_delta
        y_change = linear.x * np.sin(current_odom_r[2]) * time_delta

        current_odom_t += np.array([x_change, y_change, 0.0])
        # overall_odom += np.array([x_change, y_change, 0.0])
        # print(overall_th)
        last_odom_time = current_odom_time

# Runs the mapping node
def omni_mapper():
    global matcher
    matcher = OrbTracker()
    rospy.init_node('omni_mapper', anonymous=True)
    # last_frame_time = rospy.get_time()
    rospy.Subscriber('/odom', Odometry, odomCallback)
    rospy.Subscriber('/omni_cam/image_raw', Image, omnivisionCB)
    # rospy.Subscriber('/resize/image', Image, omnivisionCB)
    # rospy.Subscriber('/omni_cam/image_raw/compressed', CompressedImage, omnivisionCB)
    print("GO!")

    while(last_frame_time == 0):
        pass

    rate = rospy.Rate(10)

    last_kf_num = 0

    while(rospy.get_time() - last_frame_time < 1):
        # Publish the triangulated points whenever we have a new keyframe
        if len(matcher.KeyFrames) > last_kf_num:
            matcher.publishMap()
            last_kf_num = len(matcher.KeyFrames)
        matcher.publishKFMarkers()
        matcher.publishCameraMarker()
        rate.sleep()

    print("Total Keyframes: {}".format(len(matcher.KeyFrames)))
    path = '/home/imhs/Robert/advanced_robotics_2019/term_project/maps/test.p'
    print("Saving to {}".format(path))
    matcher.save(path)
    print("Saved!")

    print("Loading map from {}".format(path))
    matcher = OrbTracker()
    matcher.load(path)
    print("Loaded!")
    print("Publishing map...")
    matcher.publishMap()
    rospy.spin()



if __name__ == '__main__':
    try:
        omni_mapper()
    except rospy.ROSInterruptException:
        pass
