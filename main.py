import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusServerContext, ModbusSlaveContext
import threading
from ultralytics import YOLO
import time

def euclidean_distance_np(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)

def proper_angle(angle):
    if angle <= 10:
        angle = 0
    elif angle >= 170:
        angle = 0
    return angle

def best_box_picker(boxes):
    # Z -> X -> Y
    # Y -> X -> Z
    max_y = 1500 
    max_x = 850
    max_z = 1050
    z_dif = 80
    y_dif = 200
    new_list = []
    last_box_h = 0
    last_box_l = 5000
    boxes = sorted(boxes, key=lambda box:(box[2]))
    print("Real: ", boxes)
    for box in boxes:
        if (box[0] <= max_x) and (box[1] <= max_y) and (box[2] <= max_z):
            if last_box_h <= box[2]:
                last_box_h = box[2]
                if len(new_list) > 0:
                    if (new_list[0][2] + z_dif) > box[2]:
                        new_list.append(box)
                else:
                    new_list.append(box)
    print("For Z: ", new_list)
    last_box_h = 0
    last_box_l = 5000
    boxes = sorted(new_list, key=lambda box:(box[1]))
    new_list = []
    for box in boxes:
        if last_box_h <= box[1]:
            last_box_h = box[1]
            if len(new_list) > 0:
                if (new_list[0][1] + y_dif) > box[1]:
                    new_list.append(box)
            else:
                new_list.append(box)
    print("For Y: ", new_list)
    boxes = sorted(new_list, key=lambda box:(-box[0]))
    print("For X: ", boxes)
    if boxes != []:
        best_box = boxes[0]
    else:
        best_box = []        
    return best_box

model = YOLO('result/train2/weights/best.pt') # best by me
# model = YOLO('runs/obb/tune/weights/best.pt') # best by yolo

port = 502
host = "0.0.0.0"
frame_size = [640, 480]
frame_center = [int(frame_size[0]/2), int(frame_size[1]/2)]
fps = 30

register1 = 10 # PLC WILL SEND FOR ACTIVATING THE PREDICTION PROCESS
register2 = 11 # PLC WILL RECIEVE FOR PREDICTION COMPLITION
register3 = 12 # PLC WILL RECIEVE FOR X AXIS
register4 = 14 # PLC WILL RECIEVE FOR Y AXIS
register5 = 16 # PLC WILL RECIEVE FOR Z AXIS
register6 = 18 # PLC WILL RECIEVE FOR R AXIS

listning_value = 0
sending_value = 0

x_gantry = 0
y_gantry = 0
z_gantry = 0
r_gantry = 0

x_offset = 430.72 #388.9 - 56.0
y_offset = 1290 #1329.59 #1311.91
z_offset = 926 #926.47

store = ModbusSlaveContext(
    di=ModbusSequentialDataBlock(0, [0]*100), # Discrete Input
    co=ModbusSequentialDataBlock(0, [0]*100), # Coils
    hr=ModbusSequentialDataBlock(0, [0]*100), # Holding Registers
    ir=ModbusSequentialDataBlock(0, [0]*100), # Input Registers
)

context = ModbusServerContext(slaves=store, single=True)

def _start_server():
    StartTcpServer(context, address=(host, port))

server_thread = threading.Thread(target=_start_server)
server_thread.start()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, fps)
config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, fps)

pipeline.start(config)
align = rs.align(rs.stream.color)
prediction = False
last_listning_value = 0

try:
    while True:
        if sending_value == 1:
            sending_value = 0
            store.setValues(3, register2, [sending_value])
            print(f"sending... | register: {register2} | value: {sending_value}")
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        listning_value = store.getValues(3, register1, 1)[0]
        # print(f"listning... | register: {register1} | value: {listning_value}")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # cv2.imshow("image", color_image)
        
        if last_listning_value == 0 and listning_value == 1:
            prediction = True

        if last_listning_value == 1 and listning_value == 0:
            last_listning_value = 0

        if (listning_value == 1) and prediction:
            prediction = False
            # listning_value = 0
            sending_value = 0
            # store.setValues(3, register1, [listning_value])
            store.setValues(3, register2, [sending_value])
            
            time.sleep(1)
            print(f"sending... | register: {register2} | value: {sending_value}")

            # OBJECT DETECTION
            results = model(color_image)
            boxes = []

            ## FOR OBB MODEL
            for result in results:
                obb_data = result.obb
                xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
                xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
                names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
                confs = result.obb.conf  # confidence score of each box
                classes = obb_data.cls
                if list(xywhr) != []:
                    for i, box in enumerate(xyxyxyxy):
                        print("prediction successful!")
                        conf = confs[i].item()
                        cls = int(classes[i].item())
                        cordinates = box.tolist()
                        cordinates2 = xywhr[i].tolist()

                        cordinates = [[math.floor(cordinates[0][0]), math.floor(cordinates[0][1])], [math.floor(cordinates[1][0]), math.floor(cordinates[1][1])], [math.floor(cordinates[2][0]), math.floor(cordinates[2][1])], [math.floor(cordinates[3][0]), math.floor(cordinates[3][1])]]
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
                        x_mm = X_mm + x_offset
                        y_mm = y_offset - Y_mm
                        z_mm = Z_mm - z_offset
                        a_deg = proper_angle(angle_degrees)
                        boxes.append([abs(x_mm), abs(y_mm), abs(z_mm), abs(a_deg), cordinates, [X_mm, Y_mm, Z_mm, angle_degrees]])
                    
                    best_box = best_box_picker(boxes) 
                    if best_box != []:
                        x_gantry, y_gantry, z_gantry, r_gantry, final_cordinates, original_values = best_box[0], best_box[1], best_box[2], best_box[3], best_box[4], best_box[5]
                        if r_gantry > 10:
                            x_gantry = x_gantry - 35
                            y_gantry = y_gantry - 35
                            # if r_gantry > 45: 
                            #     r_gantry = r_gantry - 5
                            # else:
                            #     r_gantry = r_gantry + 5

                        cv2.putText(color_image, f"X: {original_values[0]} | {round(x_gantry, 2)} mm", (int(10), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(color_image, f"Y: {original_values[1]} | {round(y_gantry, 2)} mm", (int(10), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(color_image, f"Z: {original_values[2]} | {round(z_gantry, 2)} mm", (int(10), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(color_image, f"A: {original_values[3]} | {round(r_gantry, 2)} O", (int(10), int(80)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(color_image, f"N: {len(boxes)} ", (int(10), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.line(color_image, (final_cordinates[0]), (final_cordinates[1]), (0, 0, 255), 2, cv2.LINE_4)
                        cv2.line(color_image, (final_cordinates[1]), (final_cordinates[2]), (0, 0, 255), 2, cv2.LINE_4) # 
                        cv2.line(color_image, (final_cordinates[2]), (final_cordinates[3]), (0, 0, 255), 2, cv2.LINE_4)
                        cv2.line(color_image, (final_cordinates[3]), (final_cordinates[0]), (0, 0, 255), 2, cv2.LINE_4)
                        cv2.line(color_image, (final_cordinates[1]), (final_cordinates[1][0], final_cordinates[2][1]), (0, 0, 0), 2, cv2.LINE_4) #
                        
                        cv2.circle(color_image, (final_cordinates[0]), 10, (0, 0, 255), 10, cv2.LINE_4)
                        cv2.circle(color_image, (final_cordinates[1]), 10, (0, 0, 255), 10, cv2.LINE_4)
                        cv2.circle(color_image, (final_cordinates[2]), 10, (0, 0, 255), 10, cv2.LINE_4)
                        cv2.circle(color_image, (final_cordinates[3]), 10, (0, 0, 255), 10, cv2.LINE_4)
                        
                        x_conversion = abs(int(round(float(x_gantry), 2)))
                        y_conversion = abs(int(round(float(y_gantry), 2)))
                        z_conversion = abs(int(round(float(z_gantry), 2)))
                        r_conversion = abs(int(round(float(r_gantry), 2)))
                        
                        store.setValues(3, register3, [abs(x_conversion)])
                        store.setValues(3, register4, [abs(y_conversion)])
                        store.setValues(3, register5, [abs(z_conversion)])
                        store.setValues(3, register6, [abs(r_conversion)])

                        sending_value = 1
                        store.setValues(3, register2, [sending_value])
                        print(f"sending... | register: {register2} | value: {sending_value}")
                        cv2.imshow('Prediction: ', color_image)
                        last_listning_value = listning_value
                        time.sleep(2)
                    else:
                        print("No best box detected!")
                        store.setValues(3, register3, [-1])
                        store.setValues(3, register4, [-1])
                        store.setValues(3, register5, [-1])
                        store.setValues(3, register6, [-1])
                        sending_value = 1
                        store.setValues(3, register2, [sending_value])
                        print(f"sending... | register: {register2} | value: {sending_value}")
                        cv2.imshow('Prediction: ', color_image)
                        last_listning_value = listning_value
                        time.sleep(2)

                else:
                    print("prediction failed!")
                    store.setValues(3, register3, [-1])
                    store.setValues(3, register4, [-1])
                    store.setValues(3, register5, [-1])
                    store.setValues(3, register6, [-1])
                    sending_value = 1
                    store.setValues(3, register2, [sending_value])
                    print(f"sending... | register: {register2} | value: {sending_value}")
                    cv2.imshow('Prediction: ', color_image)
                    last_listning_value = listning_value
                    time.sleep(2)

                    # listning_value = 0
                    # store.setValues(3, register1, [listning_value])    
        
        cv2.line(color_image, ([frame_center[0], 0]), ([frame_center[0], frame_size[1]]), (0, 255, 255), 2, cv2.LINE_4)
        cv2.line(color_image, ([0, frame_center[1]]), ([frame_size[0], frame_center[1]]), (0, 255, 255), 2, cv2.LINE_4)
        
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        depth_image = cv2.resize(depth_color, (frame_size[0], frame_size[1]))
        combined_image = np.hstack((color_image, depth_image))
        cv2.imshow('RGB + Depth', combined_image)
        if cv2.waitKey(1) in [27, ord('q')]:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
