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
    if angle >= 190:
        angle = 190
    # elif angle >= 100:
    #     angle = 99
    return angle

def best_box_picker(boxes):
    # Z -> X -> Y
    # Y -> X -> Z
    max_y = 1620 
    max_x = 1000
    max_z = 1050
    min_y = 800 
    min_x = -40
    min_z = 150
    # z_dif = 80
    # y_dif = 200
    z_dif = 50 # 50 %
    y_dif = 50 # 50 %
    new_list = []
    last_box_h = 0
    last_box_l = 5000
    boxes = sorted(boxes, key=lambda box:(box[2]))
    print("Real: ", boxes)
    for box in boxes:
        if (box[0] <= max_x) and (box[1] <= max_y) and (box[2] <= max_z) and (box[0] >= min_x) and (box[1] >= min_y) and (box[2] >= min_z):
            if last_box_h <= box[2]:
                last_box_h = box[2]
                if len(new_list) > 0:
                    if (new_list[0][2] + 150) > box[2]: #new_list[0][2]*(z_dif/100)
                        new_list.append(box)
                else:
                    new_list.append(box)
    print("For Z: ", new_list)
    boxes = sorted(new_list, key=lambda box:(box[1]))
    new_list = []
    remaining_list = []
    new_combine_list = []
    while True:
        last_box_h = 0
        last_box_l = 5000
        for box in boxes:
            if last_box_h <= box[1]:
                last_box_h = box[1]
                if len(new_list) > 0:
                    if (new_list[0][1] + max(box[6][0], box[6][1])*(y_dif/100)) > box[1]:
                        new_list.append(box)
                    else:
                        remaining_list.append(box)
                else:
                    new_list.append(box)
        new_combine_list.append(new_list)
        boxes = remaining_list
        if remaining_list == []:
            break
        new_list = []
        remaining_list = []
    print("For Y: ", new_combine_list)
    final_list = []
    for index, box_list in enumerate(new_combine_list):
        boxes = sorted(box_list, key=lambda box:(-box[0]))
        print(f"For X{index}: ", boxes)
        if boxes != []:
            final_list = final_list + boxes
        else:
            best_box = []        
    return final_list

# model = YOLO('result/train2/weights/best.pt') # best by me
model = YOLO('result2/train/weights/best.pt')
# model = YOLO('runs/obb/tune/weights/best.pt') # best by yolo

port = 8000
host = "0.0.0.0"
frame_size = [640, 480]
frame_center = [int(frame_size[0]/2), int(frame_size[1]/2)]
fps = 30

# register1 = 10 # PLC WILL SEND FOR ACTIVATING THE PREDICTION PROCESS
# register2 = 11 # PLC WILL RECIEVE FOR PREDICTION COMPLITION
# register3 = 12 # PLC WILL RECIEVE FOR X AXIS
# register4 = 14 # PLC WILL RECIEVE FOR Y AXIS
# register5 = 16 # PLC WILL RECIEVE FOR Z AXIS
# register6 = 18 # PLC WILL RECIEVE FOR R AXIS

register1 = 8 # PLC WILL SEND FOR ACTIVATING THE PREDICTION PROCESS
register2 = 9 # PLC WILL RECIEVE FOR PREDICTION COMPLITION
register3 = 10 # BOX 1 X
register4 = 11 # BOX 1 Y
register5 = 12 # BOX 1 Z
register6 = 13 # BOX 1 A

register7 = 14 # BOX 2 X
register8 = 15 # BOX 2 Y
register9 = 16 # BOX 2 Z
register10 = 17 # BOX 2 A

register11 = 18 # BOX 3 X
register12 = 19 # BOX 3 Y
register13 = 20 # BOX 3 Z
register14 = 21 # BOX 3 A

register15 = 22 # BOX 4 X
register16 = 23 # BOX 4 Y
register17 = 24 # BOX 4 Z
register18 = 25 # BOX 4 A

register19 = 26 # BOX 5 X
register20 = 27 # BOX 5 Y
register21 = 28 # BOX 5 Z
register22 = 29 # BOX 5 A

register23 = 30 # BOX 6 X
register24 = 31 # BOX 6 Y
register25 = 32 # BOX 6 Z
register26 = 33 # BOX 6 A

register27 = 34 # BOX 7 X
register28 = 35 # BOX 7 Y
register29 = 36 # BOX 7 Z
register30 = 37 # BOX 7 A

register31 = 38 # BOX 8 X
register32 = 39 # BOX 8 Y
register33 = 40 # BOX 8 Z
register34 = 41 # BOX 8 A

register35 = 42 # No. of BOXES
register36 = 43 # Live Status

listning_value = 0
sending_value = 0

x_gantry = 0
y_gantry = 0
z_gantry = 0
r_gantry = 0

x_conversion = [-1, -1, -1, -1, -1, -1, -1, -1]
y_conversion = [-1, -1, -1, -1, -1, -1, -1, -1]
z_conversion = [-1, -1, -1, -1, -1, -1, -1, -1]
r_conversion = [-1, -1, -1, -1, -1, -1, -1, -1]

x_offset = 430.72 #388.9 - 56.0
y_offset = 1290 #1329.59 #1311.91
z_offset = 926 #926.47

x_acc_offset = 100
y_acc_offset = 50

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
        store.setValues(3, register36, [1])
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
        print(f"listning... | register: {register1} | value: {listning_value}")

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
                        
                        # Get the width and height from the xywhr data (cordinates2)
                        width_pixels = cordinates2[2]
                        height_pixels = cordinates2[3]

                        right_edge_pixel_x = int(cordinates2[0] + width_pixels / 2)
                        right_edge_pixel_y = int(cordinates2[1])

                        bottom_edge_pixel_x = int(cordinates2[0])
                        bottom_edge_pixel_y = int(cordinates2[1] + height_pixels / 2)

                        # Deproject the right edge pixel to find its 3D coordinates
                        X_right_edge, _, _ = rs.rs2_deproject_pixel_to_point(depth_intrin, [right_edge_pixel_x, right_edge_pixel_y], depth_value)

                        # Deproject the bottom edge pixel to find its 3D coordinates
                        _, Y_bottom_edge, _ = rs.rs2_deproject_pixel_to_point(depth_intrin, [bottom_edge_pixel_x, bottom_edge_pixel_y], depth_value)

                        width_mm = abs(X_right_edge - X) * 2000
                        height_mm = abs(Y_bottom_edge - Y) * 2000

                        # print(f"Result: X - {X} | Y - {Y} | Z - {Z} | D - {depth_value}")
                        X_mm, Y_mm, Z_mm = X * 1000, Y * 1000, Z * 1000
                        x_mm = X_mm + x_offset
                        # x_mm = x_mm if x_mm > 0 else 0
                        y_mm = y_offset - Y_mm
                        z_mm = Z_mm - z_offset
                        a_deg = proper_angle(angle_degrees + 3)
                        # x_mm = abs(x_mm)
                        # y_mm = abs(y_mm)
                        # z_mm = abs(z_mm)
                        # a_deg = abs(a_de)
                        boxes.append([x_mm, y_mm, z_mm, a_deg, cordinates, [X_mm, Y_mm, Z_mm, angle_degrees], [width_mm, height_mm]])
                    
                    best_boxes = best_box_picker(boxes) 
                    if best_boxes != []:
                        print("boxes prediction successful!")
                        cv2.putText(color_image, f"N: {len(best_boxes)} ", (int(10), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        for index, best_box in enumerate(best_boxes):
                            x_gantry, y_gantry, z_gantry, r_gantry, final_cordinates, original_values = best_box[0], best_box[1], best_box[2], best_box[3], best_box[4], best_box[5]
                            # if r_gantry > 90:
                            #     x_gantry = round(x_gantry - x_acc_offset*math.sin(r_gantry), 2)
                            #     y_gantry = round(y_gantry + y_acc_offset*math.cos(r_gantry), 2)
                            # else:
                            #     x_gantry = round(x_gantry + x_acc_offset*math.cos(r_gantry), 2)
                            #     y_gantry = round(y_gantry - y_acc_offset*math.sin(r_gantry), 2)

                            print(f"x_gantry - {x_gantry} | offset {math.sin(r_gantry)} || y_gantry - {y_gantry} | offset {math.cos(r_gantry)}")
                            # if r_gantry > 10:
                            if True:
                                x_gantry = x_gantry - 40
                                y_gantry = y_gantry + 20
                                # if r_gantry > 90:
                                #     x_gantry = x_gantry + 65
                                #     y_gantry = y_gantry - 35
                                # else:
                                #     x_gantry = x_gantry - 100
                                #     y_gantry = y_gantry - 45

                                # if r_gantry > 45: 
                                #     r_gantry = r_gantry - 5
                                # else:
                                #     r_gantry = r_gantry + 5

                            cv2.putText(color_image, f"X: px {round(original_values[0], 2)} mm {round(x_gantry, 2)} | Y: px {round(original_values[1], 2)} mm {round(y_gantry, 2)} | Z: px {round(original_values[2], 2)} mm {round(z_gantry, 2)} | A: od {round(original_values[3], 2)} md {round(r_gantry, 2)} ", (int(10), int(40) + 20*index), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            # cv2.putText(color_image, f"X: {original_values[0]} | {round(x_gantry, 2)} mm", (int(10), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            # cv2.putText(color_image, f"Y: {original_values[1]} | {round(y_gantry, 2)} mm", (int(10), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            # cv2.putText(color_image, f"Z: {original_values[2]} | {round(z_gantry, 2)} mm", (int(10), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            # cv2.putText(color_image, f"A: {original_values[3]} | {round(r_gantry, 2)} O", (int(10), int(80)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            # cv2.putText(color_image, f"N: {len(boxes)} ", (int(10), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            cv2.line(color_image, (final_cordinates[0]), (final_cordinates[1]), (0, 0, 255), 2, cv2.LINE_4)
                            cv2.line(color_image, (final_cordinates[1]), (final_cordinates[2]), (0, 0, 255), 2, cv2.LINE_4) # 
                            cv2.line(color_image, (final_cordinates[2]), (final_cordinates[3]), (0, 0, 255), 2, cv2.LINE_4)
                            cv2.line(color_image, (final_cordinates[3]), (final_cordinates[0]), (0, 0, 255), 2, cv2.LINE_4)
                            cv2.line(color_image, (final_cordinates[1]), (final_cordinates[1][0], final_cordinates[2][1]), (0, 0, 0), 2, cv2.LINE_4) #
                            
                            cv2.circle(color_image, (final_cordinates[0]), 10, (0, 0, 255), 10, cv2.LINE_4)
                            cv2.circle(color_image, (final_cordinates[1]), 10, (0, 0, 255), 10, cv2.LINE_4)
                            cv2.circle(color_image, (final_cordinates[2]), 10, (0, 0, 255), 10, cv2.LINE_4)
                            cv2.circle(color_image, (final_cordinates[3]), 10, (0, 0, 255), 10, cv2.LINE_4)
                        
                            x_conversion[index] = abs(int(round(float(x_gantry), 2)))
                            y_conversion[index] = abs(int(round(float(y_gantry), 2)))
                            z_conversion[index] = abs(int(round(float(z_gantry), 2)))
                            r_conversion[index] = abs(int(round(float(r_gantry), 2)))
                        
                        # store.setValues(3, register3, [abs(x_conversion)])
                        # store.setValues(3, register4, [abs(y_conversion)])
                        # store.setValues(3, register5, [abs(z_conversion)])
                        # store.setValues(3, register6, [abs(r_conversion)])

                        store.setValues(3, register3, [abs(x_conversion[0])])
                        store.setValues(3, register4, [abs(y_conversion[0])])
                        store.setValues(3, register5, [abs(z_conversion[0])])
                        store.setValues(3, register6, [abs(r_conversion[0])])

                        store.setValues(3, register7, [abs(x_conversion[1])])
                        store.setValues(3, register8, [abs(y_conversion[1])])
                        store.setValues(3, register9, [abs(z_conversion[1])])
                        store.setValues(3, register10, [abs(r_conversion[1])])

                        store.setValues(3, register11, [abs(x_conversion[2])])
                        store.setValues(3, register12, [abs(y_conversion[2])])
                        store.setValues(3, register13, [abs(z_conversion[2])])
                        store.setValues(3, register14, [abs(r_conversion[2])])

                        store.setValues(3, register15, [abs(x_conversion[3])])
                        store.setValues(3, register16, [abs(y_conversion[3])])
                        store.setValues(3, register17, [abs(z_conversion[3])])
                        store.setValues(3, register18, [abs(r_conversion[3])])

                        store.setValues(3, register19, [abs(x_conversion[4])])
                        store.setValues(3, register20, [abs(y_conversion[4])])
                        store.setValues(3, register21, [abs(z_conversion[4])])
                        store.setValues(3, register22, [abs(r_conversion[4])])

                        store.setValues(3, register23, [abs(x_conversion[5])])
                        store.setValues(3, register24, [abs(y_conversion[5])])
                        store.setValues(3, register25, [abs(z_conversion[5])])
                        store.setValues(3, register26, [abs(r_conversion[5])])

                        store.setValues(3, register27, [abs(x_conversion[6])])
                        store.setValues(3, register28, [abs(y_conversion[6])])
                        store.setValues(3, register29, [abs(z_conversion[6])])
                        store.setValues(3, register30, [abs(r_conversion[6])])

                        store.setValues(3, register31, [abs(x_conversion[7])])
                        store.setValues(3, register32, [abs(y_conversion[7])])
                        store.setValues(3, register33, [abs(z_conversion[7])])
                        store.setValues(3, register34, [abs(r_conversion[7])])

                        store.setValues(3, register35, [abs(len(best_boxes))])

                        sending_value = 1
                        store.setValues(3, register2, [sending_value])
                        print(f"sending... | register: {register2} | value: {sending_value}")
                        cv2.imshow('Prediction: ', color_image)
                        last_listning_value = listning_value
                        time.sleep(2)
                    else:
                        print("boxes prediction failed!")
                        # store.setValues(3, register3, [-1])
                        # store.setValues(3, register4, [-1])
                        # store.setValues(3, register5, [-1])
                        # store.setValues(3, register6, [-1])

                        store.setValues(3, register3, [abs(x_conversion[0])])
                        store.setValues(3, register4, [abs(y_conversion[0])])
                        store.setValues(3, register5, [abs(z_conversion[0])])
                        store.setValues(3, register6, [abs(r_conversion[0])])

                        store.setValues(3, register7, [abs(x_conversion[1])])
                        store.setValues(3, register8, [abs(y_conversion[1])])
                        store.setValues(3, register9, [abs(z_conversion[1])])
                        store.setValues(3, register10, [abs(r_conversion[1])])

                        store.setValues(3, register11, [abs(x_conversion[2])])
                        store.setValues(3, register12, [abs(y_conversion[2])])
                        store.setValues(3, register13, [abs(z_conversion[2])])
                        store.setValues(3, register14, [abs(r_conversion[2])])

                        store.setValues(3, register15, [abs(x_conversion[3])])
                        store.setValues(3, register16, [abs(y_conversion[3])])
                        store.setValues(3, register17, [abs(z_conversion[3])])
                        store.setValues(3, register18, [abs(r_conversion[3])])

                        store.setValues(3, register19, [abs(x_conversion[4])])
                        store.setValues(3, register20, [abs(y_conversion[4])])
                        store.setValues(3, register21, [abs(z_conversion[4])])
                        store.setValues(3, register22, [abs(r_conversion[4])])

                        store.setValues(3, register23, [abs(x_conversion[5])])
                        store.setValues(3, register24, [abs(y_conversion[5])])
                        store.setValues(3, register25, [abs(z_conversion[5])])
                        store.setValues(3, register26, [abs(r_conversion[5])])

                        store.setValues(3, register27, [abs(x_conversion[6])])
                        store.setValues(3, register28, [abs(y_conversion[6])])
                        store.setValues(3, register29, [abs(z_conversion[6])])
                        store.setValues(3, register30, [abs(r_conversion[6])])

                        store.setValues(3, register31, [abs(x_conversion[7])])
                        store.setValues(3, register32, [abs(y_conversion[7])])
                        store.setValues(3, register33, [abs(z_conversion[7])])
                        store.setValues(3, register34, [abs(r_conversion[7])])

                        store.setValues(3, register35, [abs(len(best_boxes))])

                        sending_value = 1
                        store.setValues(3, register2, [sending_value])
                        print(f"sending... | register: {register2} | value: {sending_value}")
                        cv2.imshow('Prediction: ', color_image)
                        last_listning_value = listning_value
                        time.sleep(2)

                else:
                    print("prediction failed!")
                    # store.setValues(3, register3, [-1])
                    # store.setValues(3, register4, [-1])
                    # store.setValues(3, register5, [-1])
                    # store.setValues(3, register6, [-1])

                    store.setValues(3, register3, [abs(x_conversion[0])])
                    store.setValues(3, register4, [abs(y_conversion[0])])
                    store.setValues(3, register5, [abs(z_conversion[0])])
                    store.setValues(3, register6, [abs(r_conversion[0])])

                    store.setValues(3, register7, [abs(x_conversion[1])])
                    store.setValues(3, register8, [abs(y_conversion[1])])
                    store.setValues(3, register9, [abs(z_conversion[1])])
                    store.setValues(3, register10, [abs(r_conversion[1])])

                    store.setValues(3, register11, [abs(x_conversion[2])])
                    store.setValues(3, register12, [abs(y_conversion[2])])
                    store.setValues(3, register13, [abs(z_conversion[2])])
                    store.setValues(3, register14, [abs(r_conversion[2])])

                    store.setValues(3, register15, [abs(x_conversion[3])])
                    store.setValues(3, register16, [abs(y_conversion[3])])
                    store.setValues(3, register17, [abs(z_conversion[3])])
                    store.setValues(3, register18, [abs(r_conversion[3])])

                    store.setValues(3, register19, [abs(x_conversion[4])])
                    store.setValues(3, register20, [abs(y_conversion[4])])
                    store.setValues(3, register21, [abs(z_conversion[4])])
                    store.setValues(3, register22, [abs(r_conversion[4])])

                    store.setValues(3, register23, [abs(x_conversion[5])])
                    store.setValues(3, register24, [abs(y_conversion[5])])
                    store.setValues(3, register25, [abs(z_conversion[5])])
                    store.setValues(3, register26, [abs(r_conversion[5])])

                    store.setValues(3, register27, [abs(x_conversion[6])])
                    store.setValues(3, register28, [abs(y_conversion[6])])
                    store.setValues(3, register29, [abs(z_conversion[6])])
                    store.setValues(3, register30, [abs(r_conversion[6])])

                    store.setValues(3, register31, [abs(x_conversion[7])])
                    store.setValues(3, register32, [abs(y_conversion[7])])
                    store.setValues(3, register33, [abs(z_conversion[7])])
                    store.setValues(3, register34, [abs(r_conversion[7])])

                    store.setValues(3, register35, [0])

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
    store.setValues(3, register36, [0])
    pipeline.stop()
    cv2.destroyAllWindows()


store.setValues(3, register36, [0])