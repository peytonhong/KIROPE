import numpy as np
import json
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import cv2

data_path = 'annotation/real/test/20210819_025345_human/'
human_pose_path = os.path.join(data_path, 'human_pose')
human_pose_json_path = sorted(glob(os.path.join(human_pose_path, '*.json')))
joint_interest = [0,1,2,5,6,7,8,9,10,11,12,13] 
# [nose, L-eye, R-eye, 
# L-shoulder, R-shoulder, 
# L-elbow, R-elbow, 
# L-wrist, R-wrist, 
# L-pelvis, R-pelvis,
# neck]
joint_connections = [[0,1], [0,2], [0,11], [11,3], [11,4], [3,5], [4,6], [5,7], [6,8], [3,9], [4,10], [9,10]]

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# fig.clear()

for frame_i, human_pose_json_path in enumerate(human_pose_json_path):
    with open(human_pose_json_path, 'r') as json_file:
        human_pose_json = json.load(json_file)
    joint_keypoints_1 = human_pose_json['joint_keypoints_1']
    joint_keypoints_2 = human_pose_json['joint_keypoints_2']
    joint_3d_positions = human_pose_json['joint_3d_positions']

    # add neck position (center of shoulders)
    neck_x = (joint_3d_positions[3][0] + joint_3d_positions[4][0])/2
    neck_y = (joint_3d_positions[3][1] + joint_3d_positions[4][1])/2
    neck_z = (joint_3d_positions[3][2] + joint_3d_positions[4][2])/2
    joint_3d_positions.append([neck_x, neck_y, neck_z])

    p3ds = joint_3d_positions
    
    for _c in joint_connections:
        ax.plot(xs = [p3ds[_c[0]][0], p3ds[_c[1]][0]], 
                ys = [p3ds[_c[0]][1], p3ds[_c[1]][1]], 
                zs = [p3ds[_c[0]][2], p3ds[_c[1]][2]], 
                c = 'red')
        ax.view_init(elev=20, azim=-150)
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(0, 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('frame: {}'.format(frame_i))
    # plt.show()
    plt.pause(0.1)
    plt.cla()


    # ret = cv2.waitKey()