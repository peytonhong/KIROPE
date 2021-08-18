import pybullet as p
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane.urdf", [0, 0, -0.1])
robotId = p.loadURDF("urdfs/ur3/ur3_gazebo.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0.0], p.getQuaternionFromEuler([0,0,0]))
numJoints = p.getNumJoints(robotId)

# targetJointPoses = [0, -1.57, 0, -1.57, 0, 0]
targetJointPoses = [6.634353269507898e-05, -1.5707521351165825, -5.5360102004797795e-05, -1.5708962143338834, 1.7861635648763752e-05, -1.3169758231512674e-05]

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

