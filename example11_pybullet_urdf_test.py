import pybullet as p
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane.urdf", [0, 0, -0.1])
robotId = p.loadURDF("urdfs/ur3/ur3_gazebo.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0.0], p.getQuaternionFromEuler([0,0,np.pi]))
numJoints = p.getNumJoints(robotId)

# targetJointPoses = [0, -1.57, 0, -1.57, 0, 0]
targetJointPoses = [-0.014069859181539357, -1.5705331007586878, -0.004110638295308888, -1.5707600752459925, 0.004471239633858204, -0.011967484151021779]

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

