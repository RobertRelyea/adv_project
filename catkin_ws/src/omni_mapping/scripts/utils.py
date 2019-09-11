import tf
import numpy as np
import cv2
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField

def pubMarkerArray(coords, frame_id, pub_handle, color, cubes=False, marker_id=0):
    marr = MarkerArray()

    for coord in coords:
        if cubes:
            marker = makeCubeMarker(coord, frame_id, color, marker_id)
        else:
            marker = makeMarker(coord, frame_id, color, marker_id)
        marr.markers.append(marker)
        marker_id += 1

    pub_handle.publish(marr)

def pubMarkerArrayZ(coords, frame_id, pub_handle, color, marker_id=0):
    marr = MarkerArray()

    for coord in coords:
        marker = makeCubeMarkerZ(coord, frame_id, color, marker_id)
        marr.markers.append(marker)
        marker_id += 1

    pub_handle.publish(marr)

def makeMarker(coord, frame_id, color, marker_id=0):
    pi = 3.14159
    th = pi
    if len(coord) > 2:
        th = coord[2]
    quat = tf.transformations.quaternion_from_euler(0,0,th)

    marker = Marker()
    marker.header.frame_id=frame_id
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.scale.x = 0.3
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker.pose.position.x = coord[0]
    marker.pose.position.y = coord[1]
    marker.pose.position.z = 0.0
    marker.id = marker_id

    return marker

def makeCubeMarker(coord, frame_id, color, marker_id=0):
    pi = 3.14159
    th = pi
    if len(coord) > 2:
        th = coord[2]
    quat = tf.transformations.quaternion_from_euler(0,0,th)

    marker = Marker()
    marker.header.frame_id=frame_id
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker.pose.position.x = coord[0]
    marker.pose.position.y = coord[1]
    marker.pose.position.z = 0.0
    marker.id = marker_id

    return marker

def makeCubeMarkerZ(coord, frame_id, color, marker_id=0):
    pi = 3.14159
    th = pi
    quat = tf.transformations.quaternion_from_euler(0,0,th)

    marker = Marker()
    marker.header.frame_id=frame_id
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker.pose.position.x = coord[0]
    marker.pose.position.y = coord[1]
    marker.pose.position.z = coord[2]
    marker.id = marker_id

    return marker

def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12*points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()

    return msg 

def triangulateFeatures(R, t, K, p1, p2):
    # Create relative transformation matricies
    T1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    T2 = np.hstack((R, t))

    # Compute relative projection matricies for both frames
    # (Camera matrix with transforms applied)
    proj1 = np.dot(K,  T1)
    proj2 = np.dot(K,  T2)

    # Perform triangulation using matched points and frame projection matricies
    points_hom = cv2.triangulatePoints(proj1, proj2, np.expand_dims(p1, axis=1), 
                                        np.expand_dims(p2, axis=1))
    points = points_hom / np.tile(points_hom[-1, :], (4, 1))
    points3d = points[:3, :].T
    # Clamp dimensions of 3d points (done by feel...)
    points3d = points3d[np.where(np.abs(points3d[:,0]) < 30)]
    points3d = points3d[np.where(np.abs(points3d[:,1]) < 30)]
    points3d = points3d[np.where(points3d[:,2] > 0)]
    # points3d = points3d[np.where(points3d[:,2] < 8)]

    return points3d


def coord2kp(coord_arr):
    kp_arr = []
    for coord in coord_arr:
        kp = cv2.KeyPoint(x=coord[0], y=coord[1],
                          _size=5,
                          _angle=0,
                          _response=0,
                          _octave=0,
                          _class_id=0)
        kp_arr.append(kp)
    return kp_arr

def kp2coord(kp_arr):
    ''' Convert an array of OpenCV KeyPoint objects into an array of pixel coordinates
        @param kp_arr Array of OpenCV KeyPoint objects
        @return coord_arr An array of pixel coordinates
    '''
    coord_arr = np.array([kp.pt for kp in kp_arr], dtype='float32')
    return coord_arr

def createMatches(indices):
    ''' Generate an array of DMatch objects from a list of indices.
        @param indices List of indices
    '''

    matches = []
    for index in indices:
        match = cv2.DMatch(index, index, index, 0)
        matches.append(match)
    return matches

def matches2coords(kp1, kp2, matches):
    '''
        Convert an array of OpenCV Matches objects with the associated KeyPoint arrays
        into two coordinate arrays for use with cv2.findEssentialMat().
        @param kp1 The first array of KeyPoint objects
        @param kp1 The second array of KeyPoint objects
        @param matches The array of Matches objects
        @return coord1_arr The coordinates of matches from kp1
        @return coord2_arr The coordinates of matches from kp2
    '''
    coord1_arr = []
    coord2_arr = []
    for match in matches:
        coord1 = kp1[match.queryIdx].pt
        coord2 = kp2[match.trainIdx].pt
        coord1_arr.append(coord1)
        coord2_arr.append(coord2)
    return np.asarray(coord1_arr), np.asarray(coord2_arr)