import math
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
old_list = [[100, 200], [150, 150], [200, 100], [120, 180], [180, 150]]
print("Old list before:", old_list)
plt.scatter(*zip(*old_list))
removelist = []
for i in range(len(old_list)):
    for j in range(i + 1, len(old_list)):
        dist = euclidean_distance(old_list[i], old_list[j])
        if  i in removelist or j in removelist:
            continue
        if dist < 31:
            old_list[i] = [int((old_list[i][0] + old_list[j][0]) / 2), int((old_list[i][1] + old_list[j][1]) / 2)]
            old_list[j] = [-1, -1]
            removelist.append(j)
        print(f"Distance between {old_list[i]} and {old_list[j]}: {dist}")
for _ in range(len(old_list)):
    if [-1, -1] in old_list:
        old_list.remove([-1, -1])
    else:
        break
plt.scatter(*zip(*old_list), color='red')
plt.show()
print("Old list after:", old_list)