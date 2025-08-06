import cv2
import numpy as np
import glob
import pickle

def calibrate_and_save(images_glob, checkerboard=(9,6), square_size=1.0, save_file="cam_calib.pkl"):
    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1,2)
    objp *= square_size

    objpoints, imgpoints = [], []
    images = glob.glob(images_glob)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
            cv2.drawChessboardCorners(gray, checkerboard, corners2, ret)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration RMS error:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coeffs:", dist.ravel())

    with open(save_file, "wb") as f:
        pickle.dump({"mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}, f)
    print(f"Saved calibration to {save_file}")
    return mtx, dist

if __name__ == "__main__":
    calibrate_and_save("data/checkerboard/*.png", checkerboard=(15,10), square_size=25)
