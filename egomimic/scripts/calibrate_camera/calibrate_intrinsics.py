import os

folder_path = os.path.join(os.path.dirname(__file__))

import numpy as np
import cv2
import argparse
import json
import h5py
from tqdm import tqdm

from egomimic.utils.egomimicUtils import (
    # WIDE_LENS_ROBOT_LEFT_K,
    # WIDE_LENS_ROBOT_LEFT_D,
    ARIA_INTRINSICS,
    LOGITECH_INTRINSICS
)

from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from deoxys_vision.utils.markers.apriltag_detector import AprilTagDetector


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h5py-path",
        type=str,
    )

    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()



def main():
    args = parse_args()

    calib = h5py.File(args.h5py_path, "r+")

    CHECKERBOARD = (7, 10)


    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []


    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
        -1, 2
    )
    # objectp3d *= .0264
    objectp3d *= 1
    prev_img_shape = None

    calib = calib["data"]
    for key in calib.keys():
        demo = calib[key]
        T, H, W, _ = demo["obs/front_img_1"].shape
        t = 0
        while t < T:

            image = demo["obs/front_img_1"][t]

            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(
                grayColor,
                CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                threedpoints.append(objectp3d)

                # Refining pixel coordinates
                # for given 2d points.
                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)

                twodpoints.append(corners2)

                # Draw and display the corners
                # image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
                # cv2.imshow('img', image)

                t += 10
            else:
                t += 1
            

    print(f"==========================")
    print(f"Number of valid images: {len(threedpoints)}")


    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None
    )


    # Displaying required output
    print(" Camera matrix:")
    print(repr(matrix))

    print("\n Distortion coefficient:")
    print(repr(distortion))


if __name__ == "__main__":
    main()
