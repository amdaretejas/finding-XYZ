import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math



frame_size = [640, 480]
frame_center = [int(frame_size[0]/2), int(frame_size[1]/2)]
fps = 30

def euclidean_distance_np(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)

def get_bounding_box_angle(points):

    p1 = points[0]
    p2 = points[1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle_rad = math.atan2(dy, dx)

    angle_deg = math.degrees(angle_rad)
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg

model = YOLO("best.pt")
# model = YOLO("yolo11n-obb.pt")
# model = YOLO("yolo11n.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, fps)
config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, fps)

pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # OBJECT DETECTION
        results = model(color_image)

        ## FOR OBB MODEL
        for result in results:
            obb_data = result.obb
            xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
            xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
            names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
            confs = result.obb.conf  # confidence score of each box
            classes = obb_data.cls
            count = 0
            for i, box in enumerate(xyxyxyxy):
                conf = confs[i].item()
                cls = int(classes[i].item())
                cordinates = box.tolist()
                cordinates2 = xywhr[i].tolist()
                angle = get_bounding_box_angle(cordinates)
                cordinates = [[math.floor(cordinates[0][0]), math.floor(cordinates[0][1])], [math.floor(cordinates[1][0]), math.floor(cordinates[1][1])], [math.floor(cordinates[2][0]), math.floor(cordinates[2][1])], [math.floor(cordinates[3][0]), math.floor(cordinates[3][1])]]
                print("cordinates", cordinates)
                cv2.line(color_image, (cordinates[0]), (cordinates[1]), (0, 255, 255), 2, cv2.LINE_4)
                cv2.line(color_image, (cordinates[1]), (cordinates[2]), (255, 0, 255), 2, cv2.LINE_4) # 
                cv2.line(color_image, (cordinates[2]), (cordinates[3]), (255, 255, 0), 2, cv2.LINE_4)
                cv2.line(color_image, (cordinates[3]), (cordinates[0]), (255, 255, 255), 2, cv2.LINE_4)
                cv2.line(color_image, (cordinates[1]), (cordinates[1][0], cordinates[2][1]), (0, 0, 0), 2, cv2.LINE_4) #
                
                cv2.circle(color_image, (cordinates[0]), 10,(0, 255, 255), 10, cv2.LINE_4)
                cv2.circle(color_image, (cordinates[1]), 10,(255, 0, 255), 10, cv2.LINE_4)
                cv2.circle(color_image, (cordinates[2]), 10,(255, 255, 0), 10, cv2.LINE_4)
                cv2.circle(color_image, (cordinates[3]), 10,(255, 255, 255), 10, cv2.LINE_4)
                cv2.circle(color_image, ([int(cordinates2[0]), int(cordinates2[1])]), 5,(0, 0, 255), 5, cv2.LINE_4)
                xc = (cordinates[0][0] + cordinates[1][0] + cordinates[2][0] + cordinates[3][0])/4
                yc = (cordinates[0][1] + cordinates[1][1] + cordinates[2][1] + cordinates[3][1])/4
                cv2.circle(color_image, ([int(xc), int(yc)]), 5,(255, 0, 0), 5, cv2.LINE_4)
                depth_value = depth_frame.get_distance(int(cordinates2[0]), int(cordinates2[1]))
                depth_mm = int(depth_value * 1000)

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

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(cordinates2[0]), int(cordinates2[1])], depth_value)
                X_mm, Y_mm, Z_mm = X * 1000, Y * 1000, Z * 1000
                cv2.putText(color_image, f"X: {round(X_mm, 2)} mm", (int(10 + count*150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(color_image, f"Y: {round(Y_mm, 2)} mm", (int(10 + count*150), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(color_image, f"Z: {round(Z_mm, 2)} mm", (int(10 + count*150), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(color_image, f"A: {angle_degrees} O", (int(10 + count*150), int(80)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                print(f"angle: {angle} ")
                count += 1

        ## FOR NORMAL MODEL
        # for result in results:
        #     result_obj = result.boxes
        #     cls = result_obj.cls
        #     conf = result_obj.conf
        #     xyxy = result_obj.xyxy
        #     xywh = result_obj.xywh
        #     for i, box in enumerate(xyxy):
        #         if int(cls[i]) == 0: 
        #             cx, cy = xywh[i][0], xywh[i][1]
        #             depth_value = depth_frame.get_distance(int(cx), int(cy))
        #             depth_mm = int(depth_value * 1000)
        #             x1, y1, x2, y2 = map(int, box)
        #             pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32).reshape((-1, 1, 2))
        #             cv2.putText(color_image, f"z: {round(depth_mm, 2)} mm", (int(cx), int(cy+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        #             cv2.polylines(color_image, [pts], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
            
            # print("result :", result.boxes)
            # color_image = result.plot()

        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        # print("depth_image", depth_color.shape)
        # print("color_image", color_image.shape)
        cv2.line(color_image, ([frame_center[0], 0]), ([frame_center[0], frame_size[1]]), (0, 255, 255), 2, cv2.LINE_4)
        cv2.line(color_image, ([0, frame_center[1]]), ([frame_size[0], frame_center[1]]), (0, 255, 255), 2, cv2.LINE_4)
        depth_image = cv2.resize(depth_color, (frame_size[0], frame_size[1]))
        combined_image = np.hstack((color_image, depth_image))
        cv2.imshow('RGB + Depth', combined_image)
        if cv2.waitKey(1) in [27, ord('q')]:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
