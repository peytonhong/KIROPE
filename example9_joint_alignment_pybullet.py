import os 
import random
import colorsys
import subprocess 
import math
import pybullet as p 
# import pybullet_data
import numpy as np
import simplejson as json
from tqdm import tqdm
from glob import glob
import cv2
import time
from scipy.spatial.transform import Rotation as R

opt = lambda : None
opt.width = 500
opt.height = 500 
opt.noise = False
opt.frame_freq = 8
opt.nb_frames = 10000
opt.inputf1 = 'annotation/train/cam1'
opt.inputf2 = 'annotation/train/cam2'
opt.outf = 'joint_alignment'
opt.idx = 9

make_joint_sphere = False

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
    existing_files = glob(f'{opt.outf}/*')
    for f in existing_files:
        os.remove(f)
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
    
# # # # # # # # # # # # # # # # # # # # # # # # #

def save_keypoint_visualize_image(iter_image_path, target_image_path, iter_keypoints, target_keypoints, iter):
    rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
    bgr_colors = rgb_colors[:, ::-1]
    iter_image = cv2.imread(iter_image_path)
    target_image = cv2.imread(target_image_path)    
    image = np.array(iter_image/2, dtype=np.uint8) + np.array(target_image/2, dtype=np.uint8)
    
    cv2.putText(image, f'Iterations: {iter}', (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,0), 2)
    for i, (iter_keypoint, target_keypoint) in enumerate(zip(iter_keypoints, target_keypoints)):
        cv2.drawMarker(image, (int(iter_keypoint[0]), int(iter_keypoint[1])), color=bgr_colors[i].tolist(), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=1)
        cv2.circle(image, (int(target_keypoint[0]), int(target_keypoint[1])), radius=5, color=bgr_colors[i].tolist(), thickness=2)                
    cv2.imwrite(iter_image_path, image.copy())



def get_my_keypoints(cam_K, cam_R, robotId, joint_world_position, opt):
    # get 2d keypoints from 3d positions using camera K, R matrix (2021.06.30, Hyosung Hong)    
    numJoints = p.getNumJoints(robotId)
    jointPositions = np.zeros((numJoints, 3))
    jointKeypoints = np.zeros((numJoints, 2))
    for l in range(numJoints):
        jointPosition = np.array(list(joint_world_position[l])+[1.]).reshape(4,1)
        jointKeypoint = cam_K@cam_R@jointPosition
        jointKeypoint /= jointKeypoint[-1]
        jointKeypoint[0] = opt.width - jointKeypoint[0]   # OpenGL convention for left-right mirroring
        jointPositions[l] = jointPosition.reshape(-1)[:3]
        jointKeypoints[l] = jointKeypoint.reshape(-1)[:2]
    # print('jointPositions: ', jointPositions)
    # print('jointKeypoints: ', jointKeypoints)
    return jointKeypoints # [numJoints, 2]

def get_joint_keypoints_from_angles(jointAngles, opt, cam_K, cam_R, robotId):
    
    for j in range(len(jointAngles)):
        p.resetJointState(bodyUniqueId=robotId,
                        jointIndex=j,
                        targetValue=(jointAngles[j]),
                        )
    p.stepSimulation()

    # get joint states
    joint_world_position = []
    for link_num in range(len(jointAngles)):    
        link_state = p.getLinkState(bodyUniqueId=robotId, linkIndex=link_num)
        pos_world = list(link_state[4])
        rot_world = link_state[5] # world orientation of the URDF link frame
        if link_num == 4:
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0,-0.04,0.08535])
            pos_world = rot_mat.dot(offset) + pos_world
        if link_num == 5:
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0.0,0.0619,0])
            pos_world = rot_mat.dot(offset) + pos_world
        joint_world_position.append(pos_world)        
    keypoints = get_my_keypoints(cam_K, cam_R, robotId=robotId, joint_world_position=joint_world_position, opt=opt)
    return keypoints # [numJoints, 2]

def clamping(min, val, max):
  # returns clamped value between min and max
  return sorted([min, val, max])[1]

def rot_shift(rot):
    # shift quaternion array from [x,y,z,w] (PyBullet) to [w,x,y,z] (NVISII)
    return [rot[-1], rot[0], rot[1], rot[2]]

def add_tuple(a, b):
    assert len(a)==len(b), "Size of two tuples are not matched!"
    return tuple([sum(x) for x in zip(a, b)])

# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0

physicsClient = p.connect(p.DIRECT) # non-graphical version

# lets create a robot
robotId = p.loadURDF("urdfs/ur3/ur3.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0.0], [0, 0, 0, 1])

numJoints = p.getNumJoints(robotId)

p.setGravity(0, 0, -9.81)

# jointInfo = p.getJointInfo(robotId, 0)
# lower_limit = [p.getJointInfo(robotId, i)[8] for i in range(numJoints)]
# upper_limit = [p.getJointInfo(robotId, i)[9] for i in range(numJoints)]

# init_time = time.time()
# Lets run the simulation for joint alignment. 
label_paths1 = sorted(glob(os.path.join(opt.inputf1, '*.json')))
label_paths2 = sorted(glob(os.path.join(opt.inputf2, '*.json')))
with open(label_paths1[opt.idx]) as json_file:
    label1 = json.load(json_file)
with open(label_paths2[opt.idx]) as json_file:
    label2 = json.load(json_file)

# for OpenGL based image capture (p.getCameraImage)
camera_struct_look_at = {
    'at': label1['camera_data']['camera_look_at']['at'],
    'up': label1['camera_data']['camera_look_at']['up'],
    'eye':label1['camera_data']['camera_look_at']['eye']
}
fov = label1['camera_data']['fov']
opt.width = label1['camera_data']['width']
opt.height = label1['camera_data']['height']
cam_intrinsic = p.computeProjectionMatrixFOV(fov=fov, # [view angle in degree]
                                            aspect=opt.width/opt.height,
                                            nearVal=0.1,
                                            farVal=100,
                                            )
cam_extrinsic_1 = p.computeViewMatrix(cameraEyePosition=camera_struct_look_at['eye'],
                                cameraTargetPosition=camera_struct_look_at['at'],
                                cameraUpVector=camera_struct_look_at['up'],
                                )

# for pinhole camera model based keypoint generation
cam_K = np.array(label1['camera_data']['camera_intrinsics'])
cam_R1 = np.array(label1['camera_data']['camera_extrinsics'])
cam_R2 = np.array(label2['camera_data']['camera_extrinsics'])

targetJointAngles = label1['objects']['joint_angles'] # goal for joint angle
target_keypoints1 = label1['objects']['projected_keypoints'] # goal for 2d joint keypoints [6, 2]
target_keypoints2 = label2['objects']['projected_keypoints'] # goal for 2d joint keypoints [6, 2]
target_keypoints = target_keypoints1 + target_keypoints2 # augmented keypoints from two cameras [12, 2]
target_keypoints = np.array([target_keypoints[i][j] for i in range(len(target_keypoints)) for j in range(2)])  # [24]

# print('target_keypoints: ', target_keypoints)
jointAngles = np.zeros(numJoints)
eps = 1e-6 # epsilon for Jacobian approximation
# eps = np.linspace(1e-6, 1e-6, numJoints)
iterations = 100 # This value can be adjusted.
for iter in range(iterations):    
    print(f'iter: {iter}, jointAngles: {jointAngles}')
    # get joint 2d keypoint from 3d points and camera model
    keypoints1 = get_joint_keypoints_from_angles(jointAngles, opt, cam_K, cam_R1, robotId)
    keypoints2 = get_joint_keypoints_from_angles(jointAngles, opt, cam_K, cam_R2, robotId)
    keypoints = np.vstack((keypoints1, keypoints2)).reshape(-1) # [24]
    
    # Jacobian approximation: keypoint rate (변화량)
    Jacobian = np.zeros((numJoints*2*2, numJoints)) # [24, 6]
    for col in range(numJoints):
        eps_array = np.zeros(numJoints)
        eps_array[col] = eps
        keypoints_eps1 = get_joint_keypoints_from_angles(jointAngles+eps_array, opt, cam_K, cam_R1, robotId)
        keypoints_eps2 = get_joint_keypoints_from_angles(jointAngles+eps_array, opt, cam_K, cam_R2, robotId)
        keypoints_eps = np.vstack((keypoints_eps1, keypoints_eps2)).reshape(-1) # [24]
        Jacobian[:,col] = (keypoints_eps - keypoints)/eps
        
    dy = np.array(target_keypoints - keypoints)
    dx = np.linalg.pinv(Jacobian)@dy
    jointAngles += dx # all joint angle update

    # # LM Algorithm
    # if iter == 0:
    #     lam = np.mean(np.diag(np.transpose(Jacobian)@Jacobian))*1e-3         # LM Algorithm
    # dy = np.array(keypoints - target_keypoints).reshape(-1,1)
    # dx = - np.linalg.inv(np.transpose(Jacobian)@Jacobian + lam*np.eye(numJoints)) @ np.transpose(Jacobian) @ dy # LM Algorithm
    # dx = dx.reshape(-1)
    # jointAngles_new = jointAngles + dx # all joint angle update
    # keypoints_new = get_joint_keypoints_from_angles(jointAngles_new, opt, cam_K, cam_R, robotId)
    # if np.linalg.norm(target_keypoints - keypoints_new) < np.linalg.norm(target_keypoints - keypoints): # accepted
    #     jointAngles += dx
    #     lam /= 10
    # else:
    #     lam *= 10
    #     # continue

    for j in range(numJoints):
        p.resetJointState(bodyUniqueId=robotId,
                        jointIndex=j,
                        targetValue=(jointAngles[j]),
                        )    
    p.stepSimulation()
    
    keypoints1 = get_joint_keypoints_from_angles(jointAngles, opt, cam_K, cam_R1, robotId)
    # print(f'iteration: {str(iter).zfill(5)}/{str(iterations).zfill(5)}')



    # get_my_keypoints(camera_entity=camera, robotId=robotId, joint_world_position=joint_world_position, opt=opt)
    image_arr = p.getCameraImage(opt.width,
                            opt.height,
                            viewMatrix=cam_extrinsic_1,
                            projectionMatrix=cam_intrinsic,
                            shadow=1,
                            lightDirection=[1, 1, 1],
                            renderer=p.ER_BULLET_HARDWARE_OPENGL, #p.ER_TINY_RENDERER, #p.ER_BULLET_HARDWARE_OPENGL
                            )


    image = np.array(image_arr[2]) # [height, width, 4]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    iter_image_path = f"{opt.outf}/{str(iter).zfill(5)}.png"
    cv2.imwrite(iter_image_path, image)
    target_image_path = f"{opt.inputf1}/{str(opt.idx).zfill(5)}.png"
    save_keypoint_visualize_image(iter_image_path, target_image_path, keypoints1, target_keypoints1, iter)
    
    
    criteria = np.linalg.norm(dy)
    print('iter: {}, criteria: {}'.format(iter, criteria))
    # if criteria < 1e-1:
    #     eps *= 0.9
    if criteria < 1e-2:
        print('targetJointAngles: ', np.array(targetJointAngles))
        print('jointAngles: ', np.array(jointAngles))
        break


p.disconnect()
framerate = str(iter/10)
subprocess.call(['ffmpeg', '-y', '-framerate', '2', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 'output.mp4'], cwd=os.path.realpath(opt.outf))
