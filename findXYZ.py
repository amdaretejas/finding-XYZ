import cv2
import numpy as np
import pickle

def load_calibration(calib_file="cam_calib.pkl"):
    with open(calib_file, "rb") as f:
        data = pickle.load(f)
    return data["mtx"], data["dist"]

def undistort_point(pt, mtx, dist):
    pts = np.array([[pt]], dtype=np.float32)
    und = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    return und[0][0]

def image_to_world(pt_pixel, depth_Z, mtx, dist):
    x_u, y_u = undistort_point(pt_pixel, mtx, dist)
    fx, fy = mtx[0,0], mtx[1,1]
    cx, cy = mtx[0,2], mtx[1,2]

    X = (x_u - cx) * depth_Z / fx
    Y = (y_u - cy) * depth_Z / fy
    Z = depth_Z
    return np.array([X, Y, Z])

# Example Usage
if __name__ == "__main__":
    mtx, dist = load_calibration()
    # Suppose RealSense gives this:
    img_pt = (650, 480)
    depth_z = 1.35  # in meters

    X, Y, Z = image_to_world(img_pt, depth_z, mtx, dist)
    print(f"World coordinates: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
