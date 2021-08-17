import numpy as np
import json

keypoints_cam1 = np.array([216,397, 214,405, 211,414, 208,424,
                            194,397, 192,405, 188,414, 184,424,
                            149,398, 145,406, 140,415, 135,424])

keypoints_cam2 = np.array([276,406, 303,406, 331,406, 357,406,
                            277,419, 305,418, 333,418, 361,418,
                            277,448, 308,448, 339,447, 369,447])

x_len = 0.05 #0.0438
y_len = 0.05 #0.044
n_row = 3
n_col = 4
x = 0.0
y = 0.0
z = 0.0
ref_points = np.zeros((n_row, n_col, 3))
for i in range(n_row):
    y = 0.0
    for j in range(n_col):
        ref_points[i,j] = [x, y, z]
        y -= y_len    
    x -= x_len
    if i == 1: # 비대칭구조
        x -= x_len

# print(ref_points)
save_data_left = {'ref_points': ref_points.reshape(-1,3).tolist(),
                'keypoints': keypoints_cam1.reshape(-1,2).tolist(),}
save_data_right = {'ref_points': ref_points.reshape(-1,3).tolist(),
                'keypoints': keypoints_cam2.reshape(-1,2).tolist(),}
# print(save_data)

with open('experiment_data/samples/references/references_cam1.json', 'w') as json_file:
    json.dump(save_data_left, json_file)
with open('experiment_data/samples/references/references_cam2.json', 'w') as json_file:
    json.dump(save_data_right, json_file)

