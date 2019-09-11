import numpy as np
import cv2
from omnicam import omnicam

orig = cv2.imread("pixpro_frame.jpg")

omni = omnicam()
rect = omni.undistortP(orig)

cv2.imwrite("pixpro_rect.jpg", rect)