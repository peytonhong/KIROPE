import numpy as np
import json

keypoints_cam1 = np.array([157,388, 157,396, 157,405, 158,414, 158,424,
                            137,389, 136,398, 136,407, 135,417, 135,426,
                            116,391, 155,400, 114,409, 113,419, 112,429])

keypoints_cam2 = np.array([206,422, 230,421, 253,420, 276,419, 299,418,
                            203,435, 227,434, 251,433, 275,432, 298,431,
                            199,448, 224,448, 249,447, 274,446, 298,444])

x_len = 0.0438
y_len = 0.044
n_row = 3
n_col = 5
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

