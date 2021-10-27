import pybullet as p
import pybullet_data
import numpy as np
import cv2
from dataset_load import RobotDataset
from torch.utils.data import DataLoader

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane.urdf", [0, 0, -0.1])
robotId = p.loadURDF("urdfs/ur3/ur3_gazebo_no_limit.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0.4, -0.15, 0.0], p.getQuaternionFromEuler([0,0,np.pi]))
numJoints = p.getNumJoints(robotId)

# to get data from dataset
idx = 0
test_dataset = RobotDataset(data_dir='annotation/real/test')
test_iterator = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

targetJointPoses = [0, -1.57, 0, -1.57, 0, 0]
# targetJointPoses = [-0.014069859181539357, -1.5705331007586878, -0.004110638295308888, -1.5707600752459925, 0.004471239633858204, -0.011967484151021779]
targetJointPoses = test_dataset[idx]['joint_angles']

while(1):
    for j in range(numJoints):
        p.setJointMotorControl2(bodyIndex=robotId,
                                jointIndex=j,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=targetJointPoses[j],
                                targetVelocity=0,
                                force=5000,
                                positionGain=0.1,
                                velocityGain=0.5)

    p.stepSimulation()
    break

def get_joint_keypoints_from_angles(jointAngles, bodyUniqueId, physicsClientId, cam_K, cam_RT, distortion):
    for j in range(len(jointAngles)):
        p.resetJointState(bodyUniqueId=bodyUniqueId,
                        jointIndex=j,
                        targetValue=(jointAngles[j]),
                        physicsClientId=physicsClientId,
                        )
    p.stepSimulation(physicsClientId=physicsClientId)

    # get joint states
    joint_world_position = []
    for link_num in range(len(jointAngles)):    
        link_state = p.getLinkState(bodyUniqueId=bodyUniqueId, linkIndex=link_num, physicsClientId=physicsClientId)
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
    # keypoints = self.get_my_keypoints(cam_K, cam_RT, joint_world_position=joint_world_position)
    
    rvecs = cv2.Rodrigues(cam_RT[:,:-1])[0]
    tvecs = cam_RT[:,-1]
    keypoints, jacobian = cv2.projectPoints(joint_world_position, rvecs, tvecs, cam_K, distortion)
    return keypoints.squeeze() # [numJoints, 2]



cam_K_1 = test_dataset[idx]['cam_K_1']
cam_K_2 = test_dataset[idx]['cam_K_2']
cam_RT_1 = test_dataset[idx]['cam_RT_1']
cam_RT_2 = test_dataset[idx]['cam_RT_2']
distortion_1 = test_dataset[idx]['distortion_1']
distortion_2 = test_dataset[idx]['distortion_2']

keypoint = get_joint_keypoints_from_angles(targetJointPoses, robotId, physicsClient, cam_K_2, cam_RT_2, distortion_2)

print("keypoint: ", keypoint)
print("keypoint GT: ", test_dataset[0]['keypoints_GT_2'])