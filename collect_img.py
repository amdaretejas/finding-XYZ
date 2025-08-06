import cv2
import numpy as np

# Try different indices to find your cameras
cap = cv2.VideoCapture(0) # May be 0, 1, 2, etc.

width = 640
height = 480

CHECKERBOARD = (15, 10)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

id_image = 0
path = "data"
file_name = "checkerboard"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frames.")
        break  

    cv2.imshow('Camera', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if (ret == True):
        corners2L= cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2L, ret)
        cv2.imshow("checkerboard", gray)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            print(f"Images {id_image} saved for left cameras")
            cv2.imwrite(f"{path}/{file_name}/{id_image}.png", frame)
            id_image=id_image+1
        else:
            print('Images not saved')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()