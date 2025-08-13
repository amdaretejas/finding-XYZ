import math
from ultralytics import YOLO
import cv2
import numpy as np

def euclidean_distance_np(point1, point2):
    """
    Calculates the Euclidean distance between two points using NumPy.

    Args:
        point1 (array-like): The coordinates of the first point.
        point2 (array-like): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model
# model = YOLO("best.pt")  # load a custom model
model = YOLO("result/train/weights/best.pt")  # load a custom model


image = cv2.imread("images/test6.jpg")

# Predict with the model
results = model(image)  # predict on an image


def compute_heading(head, tail):
    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = (math.degrees(angle_rad) + 360) % 360
    return angle_deg


text_y_len = 30

# Access the results
for result in results:
    obb_data = result.obb
    xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
    confs = result.obb.conf  # confidence score of each box
    classes = obb_data.cls
    for i, box in enumerate(xyxyxyxy):
        conf = confs[i].item()
        cls = int(classes[i].item())
        cordinates = box.tolist()
        cordinates2 = xywhr[i].tolist()
        cordinates = [[math.floor(cordinates[0][0]), math.floor(cordinates[0][1])], [math.floor(cordinates[1][0]), math.floor(cordinates[1][1])], [math.floor(cordinates[2][0]), math.floor(cordinates[2][1])], [math.floor(cordinates[3][0]), math.floor(cordinates[3][1])]]
        cv2.rectangle(image, (cordinates[0][0], cordinates[0][1]), (cordinates[2][0], cordinates[2][1]), (255, 0, 0), 2)
        
        adjacent_side_y = euclidean_distance_np(cordinates[1], [cordinates[1][0], cordinates[2][1]])
        adjacent_side_x = euclidean_distance_np(cordinates[1], [cordinates[0][0], cordinates[1][1]])
        hypotenuse1 = euclidean_distance_np(cordinates[0], cordinates[1])
        hypotenuse2 = euclidean_distance_np(cordinates[1], cordinates[2])
        
        if hypotenuse2 > hypotenuse1:
            angle_radians = math.acos(adjacent_side_y/hypotenuse2)
            angle_degrees = round(math.degrees(angle_radians), 2)
        else:
            angle_radians = math.acos(adjacent_side_x/hypotenuse1)
            angle_degrees = round(math.degrees(angle_radians) + 90, 2)

        print(cordinates2)
        print(f"angle: {angle_degrees}")
        print(f"A: {euclidean_distance_np(cordinates[0], cordinates[1])} | B: {euclidean_distance_np(cordinates[1], cordinates[2])} | {euclidean_distance_np(cordinates[2], cordinates[3])} | {euclidean_distance_np(cordinates[3], cordinates[0])}" )
        cv2.putText(image, f"{i}). {angle_degrees}", (20, text_y_len), 1, 2.0, (0, 0, 0), 2, 1)
        cv2.putText(image, f"{i}", (int(cordinates2[0]), int(cordinates2[1])), 1, 2.0, (0, 0, 0), 2, 1)
        cv2.putText(image, f"{round(conf, 2)}", (int(cordinates2[0]), int(cordinates2[1]) - 30), 1, 2.0, (0, 0, 0), 2, 1)

        text_y_len+=30
resize_image = cv2.resize(image, (math.floor(image.shape[1]/2), math.floor(image.shape[0]/2)))
cv2.imshow("result", resize_image)
cv2.waitKey(0)