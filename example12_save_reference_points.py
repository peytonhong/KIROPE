import numpy as np
import json

keypoints_left = np.array([514,615, 546,621, 578,627, 610,634, 644,640, 
                            501,631, 534,638, 568,645, 602,652, 636,658, 
                            488,649, 522,656, 557,663, 592,670, 627,678])

keypoints_right = np.array([486,569, 495,583, 504,598, 514,613, 524,630,
                            456,573, 463,587, 472,603, 480,619, 491,636,
                            424,577, 431,592, 439,607, 447,624, 455,641])

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
                'keypoints': keypoints_left.reshape(-1,2).tolist(),}
save_data_right = {'ref_points': ref_points.reshape(-1,3).tolist(),
                'keypoints': keypoints_right.reshape(-1,2).tolist(),}
# print(save_data)

with open('experiment_data/samples/references/references_left.json', 'w') as json_file:
    json.dump(save_data_left, json_file)
with open('experiment_data/samples/references/references_right.json', 'w') as json_file:
    json.dump(save_data_right, json_file)

