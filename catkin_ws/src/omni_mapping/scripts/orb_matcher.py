#!/usr/bin/env python
import numpy as np
import cv2
from omnicam import omnicam
import tf
import rospy
import time
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from utils import *

import math
import pickle
import pdb

# cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)

PROFILING = True
ORB_FEATURES = True
ROS_PC_PUBLISH = False


# Criteria for third+ keyframe generation
MIN_TRANSLATION = 0.3
MIN_ROTATION = 0.1
MIN_FRAMES = 5
if ORB_FEATURES:
    MIN_FEATURE_COUNT = 420
    STEREO_INIT_THRESH = 200
    KEYFRAME_THRESH = 200
else:
    MIN_FEATURE_COUNT = 35
    KEYFRAME_THRESH = 30

# Post multiply x rotation by pi
WORLD_ROT = np.array([[ 1, 0, 0, 0],
                      [ 0,-1, 0, 0],
                      [ 0, 0,-1, 0],
                      [ 0, 0, 0, 1]])


Y_ROT = np.array([[-1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                    [ 0, 0,-1, 0],
                    [ 0, 0, 0, 1]])
Z_ROT = np.array([[-1, 0, 0, 0],
                    [ 0,-1, 0, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1]])
X_ROT = np.array([[ 1, 0, 0, 0],
                    [ 0,-1, 0, 0],
                    [ 0, 0,-1, 0],
                    [ 0, 0, 0, 1]])

# WORLD_ROT = np.eye(4)

# Create feature extractor
orb = cv2.ORB_create(nfeatures=1000)

# Create omnicam object
omni = omnicam()

def detectFeatures(frame):
    ''' Detect features in given frame
        @param frame The image to detect features in
        @return kp KeyPoints of features detected
        @return des Descriptors of features detected
    '''
    kp = []
    kp, des = orb.detectAndCompute(frame, None)
    return kp, des

def trackFeatures(frame1, frame2, p1):
    '''
    Track previously detected features in frame1 in frame2 using 
    Lucas-Kanade pyramid optical flow

    @param frame1 The first frame containing previously detected features
    @param frame2 The second frame to track features in
    @param p1 An Nx2 numpy array containing points in pixel coordinate form
    @param kp1 The same as p1 but in OpenCV KeyPoint object form
    '''
    # Perform optical flow on features from the first frame to the second
    lk_params = dict( winSize  = (21, 21),
                    maxLevel = 5,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))

    # Compute sparse optical flow using given points from frame1 to frame2
    flow_p2, status, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p1.astype('float32'), None, **lk_params)
    err = err.reshape(err.shape[0])
    status = status.reshape(status.shape[0])
    # Keep only the points that were successfully tracked
    # flow_p1 = p1[status==1]
    # flow_p2 = np.array(flow_p2)[status==1]
    # flow_err = np.interp(err[status==1], (err.min(), err.max()), (255, 0)).astype('uint8')
    return flow_p2, status, err

def matchFeatures(kp1, des1, kp2, des2, frame1=np.array([]), frame2=np.array([])):
    # Find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    # Refine matches
    # Apply ratio test
    good = matches

    # if frame1.shape[0] > 0:
    #     # img = np.array([])
    #     # img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:100],None, flags=2)
    #     # cv2.imshow("Matches", img)
    #     # cv2.waitKey(0)
    return good

class KeyFrame():
    '''Object for holding KeyFrame information
    '''
    def __init__(self, frame, kp, des, pose_t, pose_r, H=np.eye(4), 
                 prev_idx=0, points3D=np.array([])):
        ''' Constructor for KeyFrame object
            @param frame The camera frame for this KeyFrame
            @param kp KeyPoints for this KeyFrame's features
            @param des Descriptors for this KeyFrame's features
            @param H The homogeneous transformation matrix between this 
                     keyframe and the previous
        '''
        self.frame = frame
        self.kp = np.asarray(kp)
        self.des = np.asarray(des)

        self.pose_t = pose_t
        self.pose_r = pose_r

        self.H = H.copy()
        self.prev_idx = prev_idx

        self.points3D = points3D
        self.updated = 1


class OrbTracker():
    def __init__(self):
        # KeyFrame information
        self.KeyFrames = []
        self.currentKF = 0
        # Motion estimation information
        self.previousFrame = np.array([])
        self.previousPoints = np.array([])
        self.trackedIdx = np.array([])
        self.frameCount = 0
        self.localOdom_t = np.zeros(3)
        self.localOdom_r = np.zeros(3)
        self.world_t = np.zeros(3)
        self.world_r = np.zeros(3)
        # Camera parameters
        self.K = omni.p_K
        self.pp = (self.K[0,2], self.K[1,2])
        self.fc = self.K[0,0]
        # Accumulated map
        self.map = np.array([])
        # Odom test
        self.localOdom_H = np.eye(4)
        self.marker_frame_id = 'imhs/odom'
        # Visualization and data publication
        # if ROS_PC_PUBLISH:
        self.marker_pub = rospy.Publisher("marker_pub", MarkerArray, queue_size=10)
        self.camera_marker_pub = rospy.Publisher("camera_marker_pub", MarkerArray, queue_size=10)
        self.point_marker_pub = rospy.Publisher("point_marker_pub", MarkerArray, queue_size=10)
        self.map_pc_pub = rospy.Publisher("map_pc_pub", PointCloud2, queue_size=10)
        self.new_pc_pub = rospy.Publisher("new_pc_pub", PointCloud2, queue_size=10)
    
    def save(self, path):
        save_dict = {}
        save_dict['KeyFrames'] = self.KeyFrames
        save_dict['Map'] = self.map
        pickle.dump(save_dict, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        load_dict = pickle.load(open(path, 'rb'))
        self.KeyFrames = load_dict['KeyFrames']
        self.map = load_dict['Map']

    def publishMap(self):
        ''' Publish the current map pointcloud as a PointCloud2 message.
        '''
        # Store all map points
        map_points = np.array([])

        map_coords = []

        # Accumulate point cloud
        # Iterate through every keyframe
        # for kf_idx in range(len(self.KeyFrames)):
        for kf_idx in range(1, len(self.KeyFrames)):
            # Extract 3d points from keyframe
            points3D = self.KeyFrames[kf_idx].points3D
            if points3D.shape[0] == 0:
                continue
            ones = np.reshape(np.ones(points3D.shape[0]), (points3D.shape[0], 1))
            points3D = np.hstack((points3D, ones))
            # Get latest transform for current keyframe
            # H = np.matmul(self.getWorldH(kf_idx - 1), WORLD_ROT)
            # H = self.getWorldH(kf_idx, world_rot=False)
            # H = self.getWorldH(kf_idx - 1, world_rot=False)
            # H = self.getWorldH(kf_idx)
            H = self.getWorldH(kf_idx - 1)
            H = np.matmul(H, X_ROT)
            H = np.matmul(H, Z_ROT)
            # Transform points into world coordinates
            H_inv= np.linalg.inv(H)
            # points3D = np.matmul(points3D, X_ROT)
            # points3D = np.matmul(points3D, H)
            # points3D = np.matmul(points3D, H_inv)
            # points3D = np.matmul(H, points3D.T).T
            points3D = np.matmul(H, points3D.T).T

            # world_inv = np.linalg.inv(WORLD_ROT)
            # points3D = np.matmul(points3D, world_inv)

            # points3D = np.matmul(points3D, X_ROT)
            # H = np.matmul(H, X_ROT)

            map_coords.append(self.transform2Coord(H))
            pubMarkerArray(map_coords, self.marker_frame_id, self.point_marker_pub, (1.0,1.0,0.))


            # points3D = np.matmul(points3D, Y_ROT)

            # Add to map points
            if map_points.shape[0] == 0:
                map_points = points3D[:,:3]
            else:
                map_points = np.vstack((map_points, points3D[:,:3]))
        
        # Publish as a PointCloud2 message
        pc2 = xyz_array_to_pointcloud2(map_points, frame_id=self.marker_frame_id)
        self.map_pc_pub.publish(pc2)

    def publishKFMarkers(self):
        ''' Publishes visualization markers for all keyframe positions
        '''
        coords = []
        for kf_idx in range(len(self.KeyFrames)):
            H = self.getWorldH(kf_idx)
            coords.append(self.transform2Coord(H))

        pubMarkerArray(coords, self.marker_frame_id, self.marker_pub, (0.,1.0,0.))

    def transform2Coord(self, H):
        ''' Convert homogeneous transform matrix into x, y and theta coordinate
            @param H The transform matrix
        '''
        pose_t = H[:,3,][:2]
        _, _, angles, _, _ = tf.transformations.decompose_matrix(H)
        th = angles[2]
        coord = [pose_t[0], pose_t[1], th]
        return coord

    def publishCameraMarker(self):
        ''' Publishes a marker showing the current camera position in the world.
        '''
        # Get current world pose of camera
        # current_pose_t, current_pose_r = self.getWorldPose()
        # # Create a coordinate for marker publisher
        # coord = (-1 * current_pose_t[:2]).tolist()
        # coord.append(current_pose_r[2] + math.pi)

        H = self.getWorldH()
        H = np.matmul(H, self.localOdom_H)
        pose_t = H[:,3,][:2]
        _, _, angles, _, _ = tf.transformations.decompose_matrix(H)
        th = angles[2]
        coord = [pose_t[0], pose_t[1], th]

        # Publish marker
        pubMarkerArray([coord], self.marker_frame_id, self.camera_marker_pub, (0.,0.,1.0))

    def updateLocalOdom(self, odom_t, odom_r):
        ''' Update odometry offset between keyframe and current frame.
            The absolute position of the camera will be the absolute position of the
            current KeyFrame plus the estimated camera motion scaled by the local odom.
        '''
        self.localOdom_r += odom_r.copy()
        self.localOdom_t += odom_t.copy()

        x = self.localOdom_t[0]
        y = self.localOdom_t[1]
        th = self.localOdom_r[2]

        # Create H matrix for local odom frame
        H = np.array([[ np.cos(th), np.sin(th), 0, x],
                      [-np.sin(th), np.cos(th), 0,-y],
                      [          0,          0, 1, 0],
                      [          0,          0, 0, 1]])

        # Update H matrix
        self.localOdom_H = H

    def resetLocalOdom(self):
        ''' Reset local odometry. Used after a new keyframe is set.
        '''
        self.localOdom_r = np.array([0., 0., 0.])
        self.localOdom_t = np.array([0., 0., 0.])

        self.localOdom_H = np.eye(4)

    def setWorldOdom(self, kf_idx):
        ''' Updates the current camera world pose reference to a given keyframe index.
            @param kf_idx the keyframe to base the world pose off of
        '''
        kf_t = self.KeyFrames[kf_idx].pose_t.copy()
        kf_r = self.KeyFrames[kf_idx].pose_r.copy()
        
        self.world_t = kf_t + self.localOdom_t.copy()
        self.world_r = kf_r + self.localOdom_r.copy()

    def getWorldPose(self):
        ''' Return the current world pose of the camera frame.
            @return current_pose_t The current world position
            @return current_pose_r The current world orientation
        '''
        #  Compute the current world pose
        current_pose_t = self.world_t.copy() + self.localOdom_t.copy()
        current_pose_r = self.world_r.copy() + self.localOdom_r.copy()
        return current_pose_t, current_pose_r

    def distToNthNearestKeyFrame(self, N):
        ''' Returns the distance to the Nth nearest keyframe based on the relative position
            with respect to the current keyframe.
            @param N The index for the desired keyframe distance
        '''
        if len(self.KeyFrames) == 0:
            return 0

        # Compute the current world pose
        current_pose_t, current_pose_r = self.getWorldPose()
        print("World Pose: {}".format(current_pose_t))

        # Compute distances to all keyframes
        dists = self.distToAllKeyframes(current_pose_t)
        # Return keypoint index and shortest distance
        if N >= len(dists):
            return dists[-1] # Return last entry
        return dists[N]

    def distToAllKeyframes(self, pose_t):
        ''' Returns a sorted list of keyframe indexes and distances to a given world
            pose.
            @param pose_t The world pose to compare keyframe positions to
            @return A sorted list of [KeyFrame Index, Distance] elements
        '''
        distances = []
        
        if len(self.KeyFrames) == 0:
            return distances

        for kf_idx in range(len(self.KeyFrames)):
            dist = np.sqrt(np.sum(np.square(pose_t - self.KeyFrames[kf_idx].pose_t)))
            distances.append([kf_idx, dist])

        distances.sort(key=lambda x: x[1])
        return distances

    def matchToCurrentKF(self, frame):
        ''' Match features from the current keyframe to features detected in a new frame
        '''
        kf = self.KeyFrames[self.currentKF]

        # Extract features from given frame
        new_kp, new_des = detectFeatures(frame)

        # Match features to current KeyFrame features
        matches = matchFeatures(kf.kp, kf.des, new_kp, new_des, kf.frame, frame)
        # Create new tracked features array based on the current keyframe's features
        index_size = len(matches)
        self.trackedIdx = np.linspace(0, index_size - 1, num=index_size, dtype="int32")
        # Convert keypoint matches into numpy coordinate arrays
        # Keypoints from the new frame will be reordered to share the same
        # indices as their matches from the current keyframe.
        kf_coords, new_coords = matches2coords(kf.kp, new_kp, matches)

        # Update tracked features
        self.previousPoints = new_coords
        self.previousFrame = frame

    def drawKeypointMatches(self, window):
        ''' Draw matches between latest frame and current keyframe
            @param window The cv2 normal window name to display to (str.)
        '''
        # Only display if we have a keyframe and a previous frame
        if self.previousFrame.shape[0] > 0 and len(self.KeyFrames) > 0:
            # Get keypoints from latest frame
            p1 = self.previousPoints[self.trackedIdx]
            kp1 = coord2kp(p1)
            frame1 = self.previousFrame
            # Get keypoints from current keyframe
            kp2 = self.KeyFrames[self.currentKF].kp[self.trackedIdx]
            frame2 = self.KeyFrames[self.currentKF].frame
            # Create matches array
            matches = createMatches(self.trackedIdx)
            # Display
            img = np.array([])
            img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches, None, flags=2)
            cv2.imshow(window, img)
            cv2.waitKey(1)

    def getCurrentH(self, R, t):
        ''' Composes the current homogeneous transformation matrix from a given
            translation vector and rotation matrix in addition to the current local
            odometry readings for scale.
            @param R Rotation matrix
            @param t translation vector
            @return The composed H matrix
            @return trans_mag The maginitude of the localOdom translation
        '''
        odom_scale = np.sqrt(np.sum(np.square(self.localOdom_t)))
        trans = t * odom_scale

        trans[2,0] = 0.0
        trans_mag = np.sqrt(np.sum(np.square(trans)))
        # Extract translation and rotation information
        H = np.hstack((R, trans))
        H = np.vstack((H, [0,0,0,1]))
        return H, trans, trans_mag

    def getWorldH(self, kf_idx=-1, world_rot=True):
        ''' Calculates the homogeneous transformation matrix from the map origin to 
            the latest keyframe or the specified keyframe.
            @return The calculated world transform matrix
        '''
        H = np.eye(4)
        # If no kf_idx specified, calculate transform to the latest keyframe
        if kf_idx == -1:
            kf_idx = len(self.KeyFrames) - 1
        if world_rot:
            H = np.matmul(WORLD_ROT, H)
        # Iterate through keyframes in order (post multiply)
        for idx in range(kf_idx + 1):
            H = np.matmul(H, self.KeyFrames[idx].H)

        # else:
        return H

    def newFrame(self, frame, odom_t, odom_r):
        ''' Receives a new camera frame and odometry translation pair. If first frame,
            initializes trackable points. Subsequent frames are used for tracking
            features and estimating movement throughout the scene. Triangulates tracked
            features and produces world coordinates.

            @param frame The latest frame to process
            @param odom_t 3x1 np array containing the overall odometry translation'''

        frame = omni.undistortP(frame)
        self.updateLocalOdom(odom_t, odom_r)
        # print("Odom_t: {}".format(odom_t))
        # print("LocalOdom_t: {}".format(self.localOdom_t))

        # First Frame
        if len(self.KeyFrames) == 0:
            # Detect features in first frame
            kp, des = detectFeatures(frame)
            # Sets the world position of the first keyframe as (0,0)
            self.resetLocalOdom()
            # Create a new KeyFrame object using the first frame we receive
            kf = KeyFrame(frame, kp, des, self.localOdom_t.copy(), self.localOdom_r.copy())
            # Add keyframe to current keyframes
            self.KeyFrames.append(kf)
            # Update current keyframe index
            self.currentKF = 0
        elif len(self.KeyFrames) == 1:
            ### STEREO INITIALIZATION
            # Initialize map and second keyframe as soon as we can successfully recover
            # pose from matched features in a new camera frame.
            # TODO: Change this to track features similarly to the map expansion phase
            
            ### FEATURE EXTRACTION
            start_feat = time.time()
            # Compute new features from new camera frame
            new_kp, new_des = detectFeatures(frame)
            # Obtain current keyframe
            kf = self.KeyFrames[self.currentKF]
            # Match new camera frame features to current keyframe features
            matches = matchFeatures(kf.kp, kf.des, new_kp, new_des)
            end_feat = time.time()
            # print("Number of matches: {}".format(len(matches)))

            ### MOTION ESTIMATION
            # Determine relative motion of the camera with respect to the first keyframe.
            start_emat = time.time()
            # Convert matches and keypoint lists into coordinates for findEssentialMat
            kf_coords, new_coords = matches2coords(kf.kp, new_kp, matches)
            # Find essential matrix between original features and flow features
            E, e_mask = cv2.findEssentialMat(kf_coords, new_coords, self.fc, self.pp)
            e_mask = e_mask.reshape(e_mask.shape[0])
            # Maintain list of features that fit essential matrix model
            good_p1 = kf_coords[e_mask == 1]
            good_p2 = new_coords[e_mask == 1]
            end_emat = time.time()

            # Recover camera pose from calculated essential matrix
            start_rec = time.time()
            _, R, t, rp_mask = cv2.recoverPose(E, good_p1, good_p2, self.K)
            H, trans, trans_mag = self.getCurrentH(R, t)
            # Maintain list of features that fit RANSAC model
            rp_mask = rp_mask.reshape(rp_mask.shape[0])
            good_p1 = good_p1[rp_mask == 255]
            good_p2 = good_p2[rp_mask == 255]
            end_rec = time.time()
            # print("Number of good coords: {}".format(good_p1.shape[0]))

            ### KEYFRAME GENERATION
            # Determine if we meet criteria for stereo initialization
            # TODO: Fix translation for point triangulation
            if good_p1.shape[0] > STEREO_INIT_THRESH:
                # Add second keyframe to our keyframes
                points_3d = triangulateFeatures(R, t, self.K, good_p1, good_p2)
                kf = KeyFrame(frame, new_kp, new_des, self.localOdom_t, self.localOdom_r, 
                              H=H, prev_idx=-1, points3D=points_3d)
                self.KeyFrames.append(kf)
                self.setWorldOdom(self.currentKF)
                self.resetLocalOdom()
                self.currentKF = self.distToNthNearestKeyFrame(0)[0]
                # Initialize tracking information
                self.matchToCurrentKF(frame)

                print("Number of good projections: {}".format(points_3d.shape[0]))
                print("\t\t### STEREO INITIALIZATION DONE ###")
                if ROS_PC_PUBLISH:
                    pc2 = xyz_array_to_pointcloud2(points_3d, frame_id=self.marker_frame_id)
                    self.new_pc_pub.publish(pc2)
                    self.publishKFMarkers()


            if PROFILING:
                print("Feature tracking exec time: {}".format(end_feat - start_feat))
                print("Essential mat exec time: {}".format(end_emat - start_emat))
                print("Recover pose exec time: {}".format(end_rec - start_rec))

        else:
            ### MAP EXPANSION
            # After the first two keyframes have been initialized, we will attempt to 
            # acquire additional good keyframe candidates in order to expand the map.
            self.frameCount += 1

            ### FEATURE TRACKING
            # When expanding the map, features from the previous keyframe will be tracked
            # using sparse optical flow to ensure the survival of useful features.
            start_feat = time.time()
            # Perform optical flow using latest tracked features
            # TODO: Do we need to pass in all keyframe coords even if we've lost tracking?
            #       This might be causing slowdown
            new_coords, status, _ = trackFeatures(self.previousFrame, frame, 
                                                    self.previousPoints[self.trackedIdx])
            # Use only valid coords for motion estimation
            new_coords = new_coords[status > 0]
            # Keep a running list of points that are being tracked
            self.trackedIdx = self.trackedIdx[status > 0]
            self.previousPoints[self.trackedIdx] = new_coords


            end_feat = time.time()
            # print("Number of tracked features: {}".format(new_coords.shape[0]))

            ### MOTION ESTIMATION
            # Motion estimation updates at this point will be performed using the most
            # recent keyframe.
            start_emat = time.time()

            # Find essential matrix between original features and flow features
            kp_coords = kp2coord(self.KeyFrames[self.currentKF].kp)[self.trackedIdx]
            E, e_mask = cv2.findEssentialMat(kp_coords, new_coords, self.fc, self.pp)
            e_mask = e_mask.reshape(e_mask.shape[0])
            # Maintain list of features that fit essential matrix model
            kp_coords = kp_coords[e_mask == 1]
            new_coords = new_coords[e_mask == 1]
            end_emat = time.time()
            # print("Number of e_mask {}".format(new_coords.shape[0]))
            # if new_coords.shape[0] < MIN_FEATURE_COUNT:
            #     self.matchToCurrentKF(frame)

            # Recover camera pose from calculated essential matrix
            start_rec = time.time()
            _, R, t, rp_mask = cv2.recoverPose(E, kp_coords, new_coords, self.K)
            H, trans, trans_mag = self.getCurrentH(R, t)

            # Maintain list of features that fit RANSAC model
            rp_mask = rp_mask.reshape(rp_mask.shape[0])
            kp_coords = kp_coords[rp_mask > 0]
            new_coords = new_coords[rp_mask > 0]
            end_rec = time.time()
            # print("Number of good coords: {}".format(new_coords.shape[0]))

            ### KEYFRAME GENERATION
            # Determine if we meet criteria for keyframe creation
            if new_coords.shape[0] > KEYFRAME_THRESH:
                # Calculate odometry scaling for relative translation between the current
                # KeyPoint and the current frame
                # Determine current position in world coordinates
                ref_pose_t = self.world_t.copy()
                ref_pose_r = self.world_r.copy()
                _, _, angles, _, _ = tf.transformations.decompose_matrix(H)
                angles[:2] = [0]*2 # Locking rotations to z-axis only
                new_angle = ref_pose_r + angles
                new_pose = ref_pose_t
                new_pose += np.squeeze(trans, trans.shape[1])

                # More criteria for keyframe creation
                if self.frameCount > MIN_FRAMES and trans_mag > MIN_TRANSLATION:
                    # Obtain original KeyFrame features that were successfully tracked

                    # Add second keyframe to our keyframes
                    new_kp, new_des = detectFeatures(frame)
                    # Triangulate tracked features with current KeyFrame features
                    points_3d = triangulateFeatures(R, trans, self.K, kp_coords, new_coords)
                    kf = KeyFrame(frame, new_kp, new_des, new_pose, new_angle, 
                                  H=H, prev_idx=self.currentKF, points3D=points_3d)
                    self.KeyFrames.append(kf)
                    # Update current keyframe
                    self.setWorldOdom(self.currentKF)
                    self.currentKF = self.distToNthNearestKeyFrame(0)[0]
                    self.matchToCurrentKF(frame)
                    self.resetLocalOdom()
                    self.frameCount = 0

                    # Pointcloud visualization

                    # Transform points and accumulate in self.map
                    points_3d = transformPoints(points_3d, trans=new_pose)
                    print("### NEW KEYFRAME ###\nNumber of good projections: {}".format(points_3d.shape[0]))
                    if ROS_PC_PUBLISH:
                        pc2 = xyz_array_to_pointcloud2(points_3d, frame_id=self.marker_frame_id)
                        self.new_pc_pub.publish(pc2)
                        self.showMap()
                        self.publishKFMarkers()


                    # Add new points to map
                    if self.map.shape[0] == 0:
                        self.map = points_3d
                    else:
                        self.map = np.vstack((self.map, points_3d))

            self.previousFrame = frame

    def findBestKeyFrame(self, new_kp, new_des, pose):
        ''' Determine best keyframe for tracking position given the current
            pose estimate and the latest frame's features
        '''
        # Sort keyframes by distance to current pose estimate
        # TODO: May need to take pose?
        kf_ordered_idx = self.distToAllKeyFrames()

        good_point_counts = []
        # Iterate through all keyframes in order of distance
        for kf_info in kf_ordered_idx:
            # Perform feature matching using keyframe features and current features
            kf = self.KeyFrames[kf_info[0]]
            good = matchFeatures(kf.kp, kf.des, new_kp, new_des)
            kp_coords, new_coords = matches2coords(kf.kp, new_kp, good)
            # Perform pose reconstruction using matched features
            # Find essential matrix between keyframe features and new features
            E, e_mask = cv2.findEssentialMat(kp_coords, new_coords, self.fc, self.pp)
            e_mask = e_mask.reshape(e_mask.shape[0])
            # Maintain list of features that fit essential matrix model
            kp_coords = kp_coords[e_mask == 1]
            new_coords = new_coords[e_mask == 1]

            # Recover camera pose from calculated essential matrix
            _, R, t, rp_mask = cv2.recoverPose(E, kp_coords, new_coords, self.K)
            # Maintain list of features that fit RANSAC model
            rp_mask = rp_mask.reshape(rp_mask.shape[0])
            kp_coords = kp_coords[rp_mask > 0]
            new_coords = new_coords[rp_mask > 0]
            # Store the number of valid points for reconstruction
            good_point_counts.append(new_coords.shape[0])
            # Continue until some threshold for valid points is exceeded
            if new_coords.shape[0] > STEREO_INIT_THRESH:
                break

        # Set current keyframe index to the best match
        best_idx = kf_ordered_idx[np.argmax(good_point_counts)][0]
        self.currentKF = best_idx

        # Set tracked features to indices of matched features
        kf = self.KeyFrames[self.currentKF]
        good = matchFeatures(kf.kp, kf.des, new_kp, new_des)
        kp_coords, new_coords = matches2coords(kf.kp, new_kp, good)

        # Set current pose to new triangulated pose
        self.lastodom = pose   

    def relocalize(self, new_kp, new_des):
        '''
            Relocalizes current camera view by matching extracted features to
            features in every KeyFrame. The KeyFrame with the greatest number
            of matches will be declared the current KeyFrame. The current position 
            estimate will be triangulated using the current features and the 
            KeyFrame features.

            @param new_kp KeyPoints of features from the newest frame
            @param new_des Descriptors of features from the newest frame
        '''
        kf = self.KeyFrames[self.currentKF]
        matches = []
        # Convert matches and keypoint lists into coordinates for findEssentialMat
        kf_coords, new_coords = matches2coords(kf.kp, new_kp, matches)

def transformPoints(points, rot=np.array([]), trans=np.array([])):
    ''' Perform a rotation and/or a translation transformation to a set of points.

        @param points The points to transform
        @param rot The rotation matrix
        @param trans The translation vector
        @return The transformed points
    '''

    if rot.shape[0] == 0:
        rot = np.eye(3)

    # Create homogenous transformation matrix
    M = np.hstack((rot, np.reshape(trans, (trans.shape[0],1))))
    M = np.vstack((M, np.asarray([0., 0., 0., 1.])))
    M_inv= np.linalg.inv(M)
    rot = M_inv[:3,:3]
    trans = M_inv[:3,3]

    if rot.shape[0] != 0:
        # Perform rotation
        points = np.dot(points, rot)
    if trans.shape[0] != 0:
        # Perform translation
        points += np.reshape(trans, trans.shape[0])
    return points