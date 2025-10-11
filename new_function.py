import math
import numpy as np

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def euclidean_distance_np(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)

def combine_boxes(boxes):
    removelist = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
                if abs(boxes[i][3] - boxes[j][3]) < 10:
                    if i in removelist or j in removelist:
                        continue
                    if euclidean_distance([boxes[i][0], boxes[i][1]], [boxes[j][0], boxes[j][1]]) < min(boxes[0][-1][0], boxes[0][-1][1]) + 30:
                        boxes[i][0] = (boxes[i][0] + boxes[j][0]) / 2
                        boxes[i][1] = (boxes[i][1] + boxes[j][1]) / 2
                        boxes[i][5][0] = (boxes[i][5][0] + boxes[j][5][0]) / 2
                        boxes[i][5][1] = (boxes[i][5][1] + boxes[j][5][1]) / 2
                        boxes[i][4] = [[int((boxes[i][4][0][0] + boxes[j][4][0][0]) / 2), int((boxes[i][4][0][1] + boxes[j][4][0][1]) / 2)], [int((boxes[i][4][1][0] + boxes[j][4][1][0]) / 2), int((boxes[i][4][1][1] + boxes[j][4][1][1]) / 2)], [int((boxes[i][4][2][0] + boxes[j][4][2][0]) / 2), int((boxes[i][4][2][1] + boxes[j][4][2][1]) / 2)], [int((boxes[i][4][3][0] + boxes[j][4][3][0]) / 2), int((boxes[i][4][3][1] + boxes[j][4][3][1]) / 2)]]
                        boxes[j][0] = -100000
                        boxes[j][1] = -100000
                        boxes[i][5][0] = -100000
                        boxes[i][5][1] = -100000
                        removelist.append(j)
    boxes = [sublist for sublist in boxes if -100000 not in sublist]
    return boxes

def best_box_picker(boxes, box_limit):
    # Z -> X -> Y
    # Y -> X -> Z
    new_list = []
    last_box_h = 0
    last_box_l = 5000
    boxes = sorted(boxes, key=lambda box:(box[2]))
    for box in boxes:
        if (box[0] <= box_limit[0]) and (box[1] <= box_limit[2]) and (box[2] <= box_limit[4]) and (box[0] >= box_limit[1]) and (box[1] >= box_limit[3]) and (box[2] >= box_limit[5]):
            if last_box_h <= box[2]:
                last_box_h = box[2]
                if len(new_list) > 0:
                    if (new_list[0][2] + 150) > box[2]: #new_list[0][2]*(z_dif/100)
                        new_list.append(box)
                else:
                    new_list.append(box)
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
                    if (new_list[0][1] + max(box[6][0], box[6][1])*(box_limit[6]/100)) > box[1]:
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
    final_list = []
    for index, box_list in enumerate(new_combine_list):
        boxes = sorted(box_list, key=lambda box:(-box[0]))
        if boxes != []:
            final_list = final_list + boxes
        else:
            best_box = []  
    final_list = combine_boxes(final_list)      
    return final_list