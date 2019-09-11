import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import pdb

CHECKERBOARD = (6,8)
SHOW_CHECKERBOARD = False
CALIBRATE = True

ROI_factor = 8.0

# ROI = [int(1440.0*(1.0/ROI_factor)), int(1440.0*( 1.0 - (1.0)/ROI_factor))]
ROI = [0, 1440]

images = glob.glob('./calibration_images/*.jpg')

cv2.namedWindow("undistort", cv2.WINDOW_NORMAL)
cv2.namedWindow("original", cv2.WINDOW_NORMAL)

def getROI(image):
    return image[ROI[0]:ROI[1], ROI[0]:ROI[1]]


if CALIBRATE:
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    
    cv2.namedWindow("checkerboard", cv2.WINDOW_NORMAL)

    for fname in images:
        img = getROI(cv2.imread(fname))
        # pdb.set_trace()
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            if SHOW_CHECKERBOARD:
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
                cv2.imshow("checkerboard", img)
                cv2.waitKey(0)
            imgpoints.append(corners)

    N_OK = len(objpoints)
    print(N_OK)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    xi = np.array([])
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    calib_flag = cv2.omnidir.CALIB_USE_GUESS
    print(type(calib_flag))
    # rms, _, _, _, _, _, _ = \
    cv2.omnidir.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            xi,
            D,
            # rvecs,
            # tvecs,
            calib_flag,
            # calibration_flags
            (cv2.TERM_CRITERIA_EPS, 30, 1e-6)
        )

    print(rms)
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    DIM=_img_shape[::-1]
else:
    DIM=(1440, 1440)
    K=np.array([[366.8624462136205, 0.0, 719.8795191660734], [0.0, 367.0339507631779, 721.0954243064517], [0.0, 0.0, 1.0]])
    D=np.array([[0.19949284971025222], [-0.18044759013193445], [0.08203161937421583], [-0.014534396126957652]])

w,h = DIM

def undistort(img, map1, map2):
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

# K_NEW = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), fov_scale=0.5)
# map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_NEW, DIM, cv2.CV_16SC2)

# img = getROI(cv2.imread(images[0]))
# dst = undistort(img, map1, map2)

# cv2.imshow('original', img)
# cv2.imshow('undistort',dst)
# cv2.waitKey(0)


# K_NEW = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), fov_scale=0.3)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

for fname in images:
    img = getROI(cv2.imread(fname, 0))
    dst = undistort(img, map1, map2)

    cv2.imshow('original', img)
    cv2.imshow('undistort',dst)
    cv2.waitKey(0)