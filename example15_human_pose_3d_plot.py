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
cam1_path = data_path + 'cam1'
cam2_path = data_path + 'cam2'
cam1_image_path = sorted(glob(os.path.join(cam1_path, '*.jpg')))
cam2_image_path = sorted(glob(os.path.join(cam2_path, '*.jpg')))
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

    # draw keypoint on images
    image_1 = cv2.imread(cam1_image_path[frame_i])
    image_2 = cv2.imread(cam2_image_path[frame_i])
    for kps1, kps2 in zip(joint_keypoints_1, joint_keypoints_2):
        kps1 = np.array(kps1)
        kps2 = np.array(kps2)
        cv2.circle(image_1, (int(kps1[0]), int(kps1[1])), radius=2, color=(0,255,0), thickness=2)
        cv2.circle(image_2, (int(kps2[0]), int(kps2[1])), radius=2, color=(0,255,0), thickness=2)

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
    fig.canvas.draw()
    image_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_3d = image_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_total = np.hstack((image_2, image_3d, image_1))
    cv2.imwrite(f'visualization_result/human_pose_result/image_human_3d_{frame_i:04d}.png', image_total)
    plt.pause(0.1)
    plt.cla()
    
    # ret = cv2.waitKey()