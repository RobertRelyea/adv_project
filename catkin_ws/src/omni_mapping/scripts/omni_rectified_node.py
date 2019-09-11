#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from omnicam import omnicam

rectified_pub = {}
bridge = CvBridge()

# Create omnicam object
omni = omnicam()

# Handles new omnivision image data.
def omnivisionCB(data):
    global rectified_pub
    if type(data) == CompressedImage:
        np_arr = np.fromstring(data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        frame = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (1440, 1440))
    try:
        image_msg = bridge.cv2_to_imgmsg(omni.undistortP(frame))
        rectified_pub.publish(image_msg)

    except CvBridgeError, e:
        rospy.logerr(e)

# Collects and saves data from various sensors
def omni_rectified_node():
    global rectified_pub
    rospy.init_node('omni_rectified_node', anonymous=True)
    rectified_pub = rospy.Publisher('/omni_cam/image_rect', Image, queue_size=10)
    rospy.Subscriber('/omni_cam/image_raw', Image, omnivisionCB)
    rospy.spin()


if __name__ == '__main__':
    try:
        omni_rectified_node()
    except rospy.ROSInterruptException:
        pass
