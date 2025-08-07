list = [[459.2420164805651, 1009.5066202545165, 479.530018119812, 1.53, [[254, 496], [427, 492], [421, 268], [248, 272]]], [29.827447204589816, 1037.3227668666839, 479.530018119812, 178.84, [[169, 469], [173, 272], [85, 270], [80, 467]]], [831.8046912860871, 1031.419731464386, 485.5300600814819, 177.31, [[578, 493], [589, 259], [417, 252], [407, 486]]], [68.30069276809695, 1019.7801960372924, 485.5300600814819, 179.75, [[255, 493], [256, 263], [84, 262], [83, 492]]]]


def best_pick(boxes):
    new_list = []
    last_box_h = 0
    last_box_l = 5000
    boxes = sorted(boxes, key=lambda box:(box[2]))
    for box in boxes:
        if last_box_h <= box[2]:
            last_box_h = box[2]
            if len(new_list) > 0:
                if (new_list[0][2] + 50) > box[2]:
                    new_list.append(box)
            else:
                new_list.append(box)

    last_box_h = 0
    last_box_l = 5000
    boxes = sorted(new_list, key=lambda box:(box[1]))
    for box in boxes:
        if last_box_h <= box[1]:
            last_box_h = box[1]
            if len(new_list) > 0:
                if (new_list[0][1] + 20) > box[1]:
                    new_list.append(box)
            else:
                new_list.append(box)
    boxes = sorted(new_list, key=lambda box:(-box[0]))
    best_box = boxes[0]
    return best_box

print(best_pick(list))
