import pybullet as p
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane.urdf", [0, 0, -0.1])
robotId = p.loadURDF("urdfs/ur3/ur3_new.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0.0], p.getQuaternionFromEuler([0,0,-180*np.pi/180]))
numJoints = p.getNumJoints(robotId)

# targetJointPoses = [0, -1.57, 0, -1.57, 0, 0]
targetJointPoses = [0.0004315376281738281, -1.572294060383932, -0.0008505026446741226, -1.5651825110064905, -0.0038855711566370132, -0.0020025412188928726]

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

