import numpy as np
import json

with open('annotation/real/test/20210819_025345_human/cam2/0000.json') as json_file:
    label = json.load(json_file)

cam_K = np.array(label['camera']['camera_intrinsic'])
cam_RT = np.array(label['camera']['camera_extrinsic'])
print(cam_RT)
cam_R = cam_RT[:,:3]
cam_t = cam_RT[:,-1]
cam_center = -np.transpose(cam_R) @ cam_t
print(cam_center)
pos = np.array([0,-0.825,0,1]).reshape(-1,1)

kp = cam_K @ cam_RT @ pos
# kp /= kp[-1]
print(kp)

pos2 = np.linalg.pinv(cam_K@cam_RT)@kp
pos2 /= pos2[-1]
print(pos2)