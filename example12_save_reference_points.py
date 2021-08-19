import numpy as np
import json

keypoints_cam1 = np.array([45,315, 286,269, 468,353, 180,463])

keypoints_cam2 = np.array([117,430, 206,306, 469,310, 572,435])

# x_len = 0.05 #0.0438
# y_len = 0.05 #0.044
# n_row = 3
# n_col = 4
# x = 0.0
# y = 0.0
# z = 0.0
# ref_points = np.zeros((n_row, n_col, 3))
# for i in range(n_row):
#     y = 0.0
#     for j in range(n_col):
#         ref_points[i,j] = [x, y, z]
#         y -= y_len    
#     x -= x_len
#     if i == 1: # unsymmetric structure
#         x -= x_len
ref_points = np.array([0,0,0, 0.825,0,0, 0.825,-0.825,0, 0,-0.825,0], dtype=np.float32)

# print(ref_points)
save_data_cam1 = {'ref_points': ref_points.reshape(-1,3).tolist(),
                'keypoints': keypoints_cam1.reshape(-1,2).tolist(),}
save_data_cam2 = {'ref_points': ref_points.reshape(-1,3).tolist(),
                'keypoints': keypoints_cam2.reshape(-1,2).tolist(),}
# print(save_data)

with open('experiment_data/samples/references/references_cam1.json', 'w') as json_file:
    json.dump(save_data_cam1, json_file)
with open('experiment_data/samples/references/references_cam2.json', 'w') as json_file:
    json.dump(save_data_cam2, json_file)

