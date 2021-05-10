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
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])

numJoints = p.getNumJoints(kukaId)

p.setGravity(0, 0, -9.81)
dt = 0.01
t = 0

signs = [np.random.choice([-1, 1]) for _ in range(numJoints)]
freqs = np.random.rand(numJoints) + 0.2 # 0.2~1.2 [Hz]

jointInfo = p.getJointInfo(kukaId, 0)
lower_limit = [p.getJointInfo(kukaId, i)[8] for i in range(numJoints)]
upper_limit = [p.getJointInfo(kukaId, i)[9] for i in range(numJoints)]

def clamping(min, val, max):
  # returns clamped value between min and max
  return sorted([min, val, max])[1]

while 1:
  p.stepSimulation()
  for i in range(1):   
    t += dt
    targetJointPoses = [clamping(lower_limit[i], signs[i]*np.sin(freqs[i]*t), upper_limit[i]) for i in range(numJoints)]

    for i in range(numJoints):
      p.setJointMotorControl2(bodyIndex=kukaId,
                              jointIndex=i,
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=targetJointPoses[i],
                              targetVelocity=0,
                              force=5000,
                              positionGain=0.1,
                              velocityGain=0.5)
  sleep(1/240)
