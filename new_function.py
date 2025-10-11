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
import json
from new_function import euclidean_distance, euclidean_distance_np, best_box_picker

model = YOLO('result2/train/weights/best.pt')

camera_calibration_path = "/home/smart/HighAccuracySettings.json"

port = 502
host = "0.0.0.0"
x_px_size = 1280
y_px_size = 720

frame_size = [x_px_size, y_px_size]
frame_center = [int(frame_size[0]/2), int(frame_size[1]/2)]

fps = 30
exposer = 10000
gain = 16
laser_power = 360
fill_mode = 2
neighbour_px = 5

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

x_offset = 440 
y_offset = 1280   
z_offset = 1080 

x_box_min = 1600
x_box_max = 0
y_box_min = 1200
y_box_max = 700
z_box_min = 1400
z_box_max = 150
y_dif = 50 #%
z_dif = 50 #%

box_limit = [x_box_max, x_box_min, y_box_max, y_box_min, z_box_max, z_box_min, y_dif, z_dif]

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

profile = pipeline.start(config)
device = profile.get_device()

advanced_mode = rs.rs400_advanced_mode(device)
if not advanced_mode.is_enabled():
    advanced_mode.toggle_advanced_mode(True)

json_obj = None
with open(camera_calibration_path, "r") as f:
    json_obj = json.load(f)
print(f"Json File Uploaded")

advanced_mode.load_json(json.dumps(json_obj))
depth_sensor = device.first_depth_sensor()

if depth_sensor.get_option(rs.option.enable_auto_exposure):
    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

exposure_range = depth_sensor.get_option_range(rs.option.exposure)
print(f"Exposer range: {exposure_range.min} ~ {exposure_range.max}") 
depth_sensor.set_option(rs.option.exposure, exposer)
print(f"Eexposer Activated at ~ {exposer}")

gain_range = depth_sensor.get_option_range(rs.option.gain)
print(f"Gain range: {gain_range.min} ~ {gain_range.max}") 
depth_sensor.set_option(rs.option.gain, gain)
print(f"Gain Activated at ~ {gain}")

laser_range = depth_sensor.get_option_range(rs.option.laser_power)
print(f"Laser power range: {laser_range.min} ~ {laser_range.max}") 
depth_sensor.set_option(rs.option.laser_power, laser_power)
print(f"Laser power Activated at ~ {laser_power}")

hole_filling_filter = rs.hole_filling_filter()
hole_filling_filter.set_option(rs.option.holes_fill, fill_mode)
print(f"Hole Filling Filter Activated")

prediction = False
last_listning_value = 0

try:
    while True:
        store.setValues(3, register36, [1])
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        listning_value = store.getValues(3, register1, 1)[0]

        depth_value = round(depth_frame.get_distance(int(frame_size[0]/2), int(frame_size[1]/2)), 6)
        depth_filter = hole_filling_filter.process(depth_frame)

        colorizer = rs.colorizer()
        depth_image = np.asanyarray(colorizer.colorize(depth_filter).get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # color_image = cv2.resize(color_image, (640, 480))
        # depth_image = cv2.resize(depth_image, (640, 480))

        if last_listning_value == 0 and listning_value == 1:
            prediction = True

        if last_listning_value == 1 and listning_value == 0:
            last_listning_value = 0

        if (listning_value == 1) and prediction:
            prediction = False
            sending_value = 0
            store.setValues(3, register2, [sending_value])

            time.sleep(1)
            print(f"sending... | register: {register2} | value: {sending_value}")

            # OBJECT DETECTION
            results = model(color_image)
            boxes = []
            depth_value_avg = 0

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
                        print(f"{i}] Prediction successful!")
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

                        depth_value = round(depth_frame.get_distance(int(cordinates2[0]), int(cordinates2[1])), 6)
                        depth_value1 = round(depth_frame.get_distance(int(cordinates2[0]+neighbour_px), int(cordinates2[1])), 6)
                        depth_value2 = round(depth_frame.get_distance(int(cordinates2[0]), int(cordinates2[1])+neighbour_px), 6)
                        depth_value3 = round(depth_frame.get_distance(int(cordinates2[0]-neighbour_px), int(cordinates2[1])), 6)
                        depth_value4 = round(depth_frame.get_distance(int(cordinates2[0]), int(cordinates2[1])-neighbour_px), 6)
                        depth_value5 = round(depth_frame.get_distance(int(cordinates2[0]+neighbour_px), int(cordinates2[1])+neighbour_px), 6)
                        depth_value6 = round(depth_frame.get_distance(int(cordinates2[0])-neighbour_px, int(cordinates2[1])-neighbour_px), 6)
                        depth_value7 = round(depth_frame.get_distance(int(cordinates2[0])+neighbour_px, int(cordinates2[1])-neighbour_px), 6)
                        depth_value8 = round(depth_frame.get_distance(int(cordinates2[0])-neighbour_px, int(cordinates2[1])+neighbour_px), 6)

                        depth_mm = int((depth_value*1000  + depth_value1*1000 + depth_value2*1000 + depth_value3*1000 + depth_value4*1000 + depth_value5*1000 + depth_value6*1000 + depth_value7*1000 + depth_value8*1000)/9)
                        print(f"{i}] Avrage Depth: {depth_mm} mm and Center Depth: {depth_value*1000} mm")

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
                        
                        width_pixels = cordinates2[2]
                        height_pixels = cordinates2[3]
                        
                        right_edge_pixel_x = int(cordinates2[0] + width_pixels / 2)
                        right_edge_pixel_y = int(cordinates2[1])

                        bottom_edge_pixel_x = int(cordinates2[0])
                        bottom_edge_pixel_y = int(cordinates2[1] + height_pixels / 2)

                        X_right_edge, _, _ = rs.rs2_deproject_pixel_to_point(depth_intrin, [right_edge_pixel_x, right_edge_pixel_y], depth_value)
                        _, Y_bottom_edge, _ = rs.rs2_deproject_pixel_to_point(depth_intrin, [bottom_edge_pixel_x, bottom_edge_pixel_y], depth_value)

                        width_mm = abs(X_right_edge - X) * 2000
                        height_mm = abs(Y_bottom_edge - Y) * 2000
                        
                        X_mm = X*1000
                        Y_mm = Y*1000
                        print(f"{i}] Center X: {X_mm} mm | Center Y: {Y_mm} mm | Avrage Z: {Z_mm} | Center A: {A_deg} by realsense")
                        
                        X_mm = X_mm + x_offset
                        X_mm = X_mm if X_mm > 0 else 0
                        Y_mm = y_offset - Y_mm
                        Z_mm = depth_mm - z_offset 
                        A_deg = angle_degrees
                        boxes.append([X_mm, Y_mm, Z_mm, A_deg, cordinates, [cordinates2[0], cordinates2[1], depth_mm], [width_mm, height_mm]])

                    best_boxes = best_box_picker(boxes, box_limit) 
                    if best_boxes != []:
                        print("boxes prediction successful!")
                        cv2.putText(color_image, f"N: {len(best_boxes)} ", (int(10), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        for index, best_box in enumerate(best_boxes):
                            x_gantry, y_gantry, z_gantry, r_gantry, cordinates, cordinates2, original_values = best_box[0], best_box[1], best_box[2], best_box[3], best_box[4], best_box[5]
                            cv2.putText(color_image, f"X: px {round(cordinates2[0], 2)} mm {round(x_gantry, 2)} | Y: px {round(cordinates2[1], 2)} mm {round(y_gantry, 2)} | Z: px {round(cordinates2[2], 2)} mm {round(z_gantry, 2)} | A: {round(r_gantry, 2)} ", (int(10), int(40) + 20*index), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            
                            cv2.line(color_image, (cordinates[0]), (cordinates[1]), (0, 0, 255), 2, cv2.LINE_4)
                            cv2.line(color_image, (cordinates[1]), (cordinates[2]), (0, 0, 255), 2, cv2.LINE_4)
                            cv2.line(color_image, (cordinates[2]), (cordinates[3]), (0, 0, 255), 2, cv2.LINE_4)
                            cv2.line(color_image, (cordinates[3]), (cordinates[0]), (0, 0, 255), 2, cv2.LINE_4)
                            cv2.line(color_image, (cordinates[1]), (cordinates[1][0], cordinates[2][1]), (0, 0, 0), 2, cv2.LINE_4)
                            
                            cv2.circle(color_image, (cordinates[0]), 10, (0, 0, 255), 10, cv2.LINE_4)
                            cv2.circle(color_image, (cordinates[1]), 10, (0, 0, 255), 10, cv2.LINE_4)
                            cv2.circle(color_image, (cordinates[2]), 10, (0, 0, 255), 10, cv2.LINE_4)
                            cv2.circle(color_image, (cordinates[3]), 10, (0, 0, 255), 10, cv2.LINE_4)
                            
                            x_conversion[index] = abs(int(round(float(x_gantry), 2)))
                            y_conversion[index] = abs(int(round(float(y_gantry), 2)))
                            z_conversion[index] = abs(int(round(float(z_gantry), 2)))
                            r_conversion[index] = abs(int(round(float(r_gantry), 2)))
                        
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
        cv2.line(color_image, ([frame_center[0], 0]), ([frame_center[0], frame_size[1]]), (0, 255, 255), 2, cv2.LINE_4)
        cv2.line(color_image, ([0, frame_center[1]]), ([frame_size[0], frame_center[1]]), (0, 255, 255), 2, cv2.LINE_4)
        color_image = cv2.resize(color_image, (640, 480))
        depth_image = cv2.resize(depth_image, (640, 480))
        combined_image = np.hstack((color_image, depth_image))
        cv2.imshow('RGB + Depth', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    store.setValues(3, register36, [0])
    pipeline.stop()
    cv2.destroyAllWindows()
    
store.setValues(3, register36, [0])