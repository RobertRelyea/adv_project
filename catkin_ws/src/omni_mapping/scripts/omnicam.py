import cv2
import numpy as np

class omnicam():
    def __init__(self, cameraPath=""):
        self.fs_path = "/home/imhs/Robert/advanced_robotics_2019/term_project/catkin_ws/src/omni_calib/images/out_camera_params.xml"
        self.loadParams()
        if len(cameraPath) != 0:
            self.cap = cv2.VideoCapture(cameraPath)
            self.cap.set(3,1440)
            self.cap.set(4,1440)
        else:
            self.cap = None
        
    def loadParams(self):
        '''
        Loads camera calibration parameters from self.fs_path.
        '''
        # Load from calibration file fs_path
        cv_file = cv2.FileStorage(self.fs_path, cv2.FILE_STORAGE_READ)

        # Omnidirectional camera model params
        self.K = cv_file.getNode("camera_matrix").mat()
        self.D = cv_file.getNode("distortion_coefficients").mat()
        self.X = np.array(cv_file.getNode("xi").real())

        # New rectified perspective camera matrix
        # https://docs.opencv.org/3.4.1/dd/d12/tutorial_omnidir_calib_main.html
        self.p_K = np.array([[1440.0/4, 0, 1440.0/2],
                             [0, 1440.0/4, 1440.0/2],
                             [0,0,1]])

        # Calculate distortion maps for rectified perspective image
        self.p_maps = cv2.omnidir.initUndistortRectifyMap(self.K, self.D, self.X, None, self.p_K, 
                    (1440, 1440), cv2.CV_16SC2, flags=cv2.omnidir.RECTIFY_PERSPECTIVE)

        # New rectified cylindrical camera matrix
        # https://docs.opencv.org/3.4.1/dd/d12/tutorial_omnidir_calib_main.html
        self.c_K = np.array([[1440.0*2/3.1415, 0, 1440*2],
                             [0, 1440.0/3.1415, 1440],
                             [0,0,1]])

        # Calculate distortion maps for rectified cylindrical image
        self.c_maps = cv2.omnidir.initUndistortRectifyMap(self.K, self.D, self.X, None, self.c_K, 
            (1440*4, int(1440/1.3)), cv2.CV_16SC2, flags=cv2.omnidir.RECTIFY_CYLINDRICAL)

    def read(self):
        '''
        Read a new frame from the video capture.

        @return (ret, frame) - Return value of video capture and the read frame
        '''
        assert (self.cap != None), "No video capture for this omnicam object!"
        return self.cap.read()

    def readP(self):
        '''
        Read a new frame from the video capture and perform perspective rectification.

        @return The perspective rectified captured frame
        '''
        ret, frame = self.read()
        if ret:
            return self.undistort(frame, self.p_maps)
        else:
            return np.array([])

    def readC(self):
        '''
        Read a new frame from the video capture and perform cylindrical rectification.

        @return The cylindrical rectified captured frame
        '''
        ret, frame = self.read()
        if ret:
            return self.undistort(frame, self.c_maps)
        else:
            return np.array([])

    def undistort(self, frame, maps):
        '''
        Remove distortion from the provided image using the provided mappings.

        @param frame: Image to remove distortion from
        @param maps:  Tuple containing two mappings for distortion removal
        @return:      The undistorted image
        '''
        undistorted_frame = cv2.remap(frame, maps[0], maps[1], interpolation=cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_CONSTANT)
        return undistorted_frame

    def undistortP(self, frame):
        '''
        Remove distortion from the provided image and return a perspective
        rectified image.

        @param frame: Image to remove distortion from
        @return:      The undistorted and perspective rectified image
        '''
        return self.undistort(frame, self.p_maps)

    def undistortC(self, frame):
        '''
        Remove distortion from the provided image and return a cylindrical
        rectified image.

        @param frame: Image to remove distortion from
        @return:      The undistorted and cylindrical rectified image
        '''
        return self.undistort(frame, self.c_maps)