import numpy as np
import json
import matplotlib.pyplot as plt
from glob import glob
import pybullet as p

data_path = 'annotation/real/test/20211005_220303_human_validation/'
index_begin = 202
index_end = 287
robot_joint_index = 5 # End-effector
human_joint_index = 7 # L-wrist

physicsClient = p.connect(p.DIRECT) # non-graphical version
robotId = p.loadURDF("urdfs/ur3/ur3_gazebo_no_limit.urdf", [0, 0, 0], useFixedBase=True)

basePose = p.getQuaternionFromEuler([0,0,np.pi])
p.resetBasePositionAndOrientation(robotId, [0.4, -0.15, 0.0], basePose) # robot base offset(25cm) from checkerboard
numJoints = p.getNumJoints(robotId)

def get_json_path(data_path, file_number):
    json_path = data_path + str(file_number).zfill(4) + '.json'
    return json_path

robot_pos_3d = []
human_pos_3d = []
for file_number in range(index_begin, index_end):

    with open(get_json_path(data_path, file_number), 'r') as json_file:
        load_data = json.load(json_file)
    jointAngles = load_data['joint_angles']

    with open(get_json_path(data_path+'human_pose/', file_number), 'r') as json_file:
        load_data = json.load(json_file)
    human_world_position = load_data['joint_3d_positions']

    for j in range(numJoints):
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
        if link_num == 0: # sholder
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0,0,0])
            pos_world = rot_mat.dot(offset) + pos_world
        if link_num == 1: # upper_arm
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0,0,0.1198])
            pos_world = rot_mat.dot(offset) + pos_world
        if link_num == 2: # fore_arm
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0,0,0.025])
            pos_world = rot_mat.dot(offset) + pos_world
        if link_num == 3: # wrist 1
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0,0,-0.085])
            pos_world = rot_mat.dot(offset) + pos_world
        if link_num == 4: # wrist 2
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0,-0.045,0])
            pos_world = rot_mat.dot(offset) + pos_world
        if link_num == 5: # wrist 3
            rot_mat = p.getMatrixFromQuaternion(rot_world)
            rot_mat = np.array(rot_mat).reshape(3,3)
            offset = np.array([0,0,0])
            pos_world = rot_mat.dot(offset) + pos_world
        joint_world_position.append(pos_world) 

    joint_world_position = np.array(joint_world_position)
    
    robot_pos_3d.append(joint_world_position[robot_joint_index])
    human_pos_3d.append(human_world_position[human_joint_index])
robot_pos_3d = np.array(robot_pos_3d)
human_pos_3d = np.array(human_pos_3d)
position_error = []
for robot_3d, human_3d in zip(robot_pos_3d, human_pos_3d):
    position_error.append(np.linalg.norm(robot_3d-human_3d))

position_error = np.mean(position_error)

print('position_error: ', position_error)
plt.figure()
plt.subplot(3,1,1)
plt.plot(robot_pos_3d[:,0], label='robot')
plt.plot(human_pos_3d[:,0], label='human')
plt.xlabel('Frames')
plt.ylabel('X [m]')
plt.title('Robot - Human Position Validation')
plt.legend()
plt.subplot(3,1,2)
plt.plot(robot_pos_3d[:,1], label='robot')
plt.plot(human_pos_3d[:,1], label='human')
plt.xlabel('Frames')
plt.ylabel('Y [m]')
plt.legend()
plt.subplot(3,1,3)
plt.plot(robot_pos_3d[:,2], label='robot')
plt.plot(human_pos_3d[:,2], label='human')
plt.xlabel('Frames')
plt.ylabel('Z [m]')
plt.legend()
plt.show()