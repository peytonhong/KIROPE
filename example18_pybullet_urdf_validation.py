import pybullet as p
import pybullet_data
import numpy as np
import cv2
from dataset_load import RobotDataset
from torch.utils.data import DataLoader
import time
import json
import matplotlib.pyplot as plt

"""
Validate pybullet robot model with real robot motion.
Compare with the joint angles.
"""

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane.urdf", [0, 0, -0.1])
robotId = p.loadURDF("urdfs/ur3/ur3_gazebo_no_limit.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0.4, -0.15, 0.0], p.getQuaternionFromEuler([0,0,np.pi]))
numJoints = p.getNumJoints(robotId)

# to get data from dataset
idx = 0
test_dataset = RobotDataset(data_dir='annotation/real/pybullet_validation')
test_iterator = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

targetJointPoses = [0, -1.57, 0, -1.57, 0, 0]
# targetJointPoses = [-0.014069859181539357, -1.5705331007586878, -0.004110638295308888, -1.5707600752459925, 0.004471239633858204, -0.011967484151021779]
targetJointPoses = test_dataset[idx]['joint_angles']

for j in range(numJoints):
    p.resetJointState(bodyUniqueId=robotId,
                                jointIndex=j,
                                targetValue=(targetJointPoses[j]),
                                )


time_begin = time.time()

angle_save = []

# while(1):
for i in range(1000):
    targetJointPoses = test_dataset[i]['joint_angles']
    for _ in range(2):
        for j in range(numJoints):
            p.setJointMotorControl2(bodyIndex=robotId,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=targetJointPoses[j],
                                    targetVelocity=0,
                                    force=5000,
                                    positionGain=0.3,
                                    velocityGain=0.5)

        p.stepSimulation()    
        time.sleep(1/240)
    jointStates = p.getJointStates(robotId, range(numJoints))
    jointAngles_main = np.array([jointStates[i][0] for i in range(len(jointStates))])    
    angle_save.append([targetJointPoses.tolist(),jointAngles_main.tolist()])

with open("pybullet_validation.json", 'w') as json_file:
    json.dump(angle_save, json_file)

angle_save = np.array(angle_save)*180/np.pi # [N, 2, 6]

pad = 0.1
plt.figure()
plt.subplot(3,2,1)
plt.plot(angle_save[:,0,0], label='target')
plt.plot(angle_save[:,1,0], label='estimate', linestyle='dashed')
plt.legend()
plt.title('Joint 1')
# plt.xlabel('steps')
# plt.gca().axes.get_xaxis().set_ticks([])
plt.ylabel('angle [deg]')
plt.grid()
plt.tight_layout(pad=pad)

plt.subplot(3,2,2)
plt.plot(angle_save[:,0,1], label='target')
plt.plot(angle_save[:,1,1], label='estimate', linestyle='dashed')
plt.legend()
plt.title('Joint 2')
# plt.xlabel('steps')
# plt.gca().axes.get_xaxis().set_ticks([])
plt.ylabel('angle [deg]')
plt.grid()
plt.tight_layout(pad=pad)

plt.subplot(3,2,3)
plt.plot(angle_save[:,0,2], label='target')
plt.plot(angle_save[:,1,2], label='estimate', linestyle='dashed')
plt.legend()
plt.title('Joint 3')
# plt.xlabel('steps')
# plt.gca().axes.get_xaxis().set_ticks([])
plt.ylabel('angle [deg]')
plt.grid()
plt.tight_layout(pad=pad)

plt.subplot(3,2,4)
plt.plot(angle_save[:,0,3], label='target')
plt.plot(angle_save[:,1,3], label='estimate', linestyle='dashed')
plt.legend()
plt.title('Joint 4')
# plt.xlabel('steps')
# plt.gca().axes.get_xaxis().set_ticks([])
plt.ylabel('angle [deg]')
plt.grid()
plt.tight_layout(pad=pad)

plt.subplot(3,2,5)
plt.plot(angle_save[:,0,4], label='target')
plt.plot(angle_save[:,1,4], label='estimate', linestyle='dashed')
plt.legend()
plt.title('Joint 5')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.grid()
plt.tight_layout(pad=pad)

plt.subplot(3,2,6)
plt.plot(angle_save[:,0,5], label='target')
plt.plot(angle_save[:,1,5], label='estimate', linestyle='dashed')
plt.legend()
plt.title('Joint 6')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.grid()
plt.tight_layout(pad=pad)

plt.show()