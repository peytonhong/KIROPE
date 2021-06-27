import pybullet as p
import time
import math
from datetime import datetime
from time import sleep

import pybullet_data
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.3])
robotId = p.loadURDF("urdfs/ur3/ur3.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0], [0, 0, 0, 1])
robotEndEffectorIndex = 5
numJoints = p.getNumJoints(robotId)

# #lower limits for null space
# ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# #upper limits for null space
# ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# #joint ranges for null space
# jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# #restposes for null space
# rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# #joint damping coefficents
# jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# #Joint damping coefficents. Using large values for the joints that we don't want to move.
# # jd = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.5]
# # jd=[0.5,0.5,0.5,0.5,0.5,0.5,0.5]

p.setGravity(0, 0, -9.81)
dt = 0.01
t = 0
r = 0.3

hasPrevPose = 0
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
trailDuration = 15

while 1:
  p.stepSimulation()
  for i in range(1):
    # pos = [0, -0.5, 0.26]
    # pos = [r*math.sin(t), -0.8, r*math.cos(t)+0.3]   # singularity 
    # orn = p.getQuaternionFromEuler([3.14/2, 0, 0])   # singularity 
    pos = [0.3, r*math.sin(t), r*math.cos(t)+0.3]
    orn = p.getQuaternionFromEuler([0, 0, 0])
    ls = p.getLinkState(robotId, robotEndEffectorIndex)
    # print("end-effctor orientation: World:{}, Internal:{}".format(ls[1], ls[3]))
    
    t += dt
    # jointPoses = p.calculateInverseKinematics(robotId,
    #                                           robotEndEffectorIndex,
    #                                           pos,
    #                                           orn,
    #                                           jointDamping=jd)
    jointPoses = p.calculateInverseKinematics(robotId, robotEndEffectorIndex, pos, orn)#, ll, ul, jr, rp)
    print("end-effector jointPoses: {}".format(jointPoses[robotEndEffectorIndex]))

  for i in range(numJoints):
    p.setJointMotorControl2(bodyIndex=robotId,
                            jointIndex=i,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=jointPoses[i],
                            targetVelocity=0,
                            force=500,
                            positionGain=0.05,
                            velocityGain=0.5)
  
  # Draw trajectory
  ls = p.getLinkState(robotId, robotEndEffectorIndex)
  if (hasPrevPose):
    p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  prevPose = pos
  prevPose1 = ls[4]
  hasPrevPose = 1
  
  zero_vec = [0.0] * len(jointPoses)
  J = p.calculateJacobian(bodyUniqueId=robotId, linkIndex=robotEndEffectorIndex, localPosition=(0.0, 0.0, 0.02), objPositions=jointPoses, objVelocities=zero_vec, objAccelerations=zero_vec)
  J = np.array(J).reshape((6,numJoints))
  U, S, Vh = np.linalg.svd(J)
  print(S@S.transpose()) # Manipulability Measure: smaller as closer to singularity condition



  sleep(1/240)
