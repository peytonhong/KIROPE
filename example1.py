import pybullet as p
import pybullet_data
import time
import numpy as np

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
# objId = p.loadURDF("./urdfs/panda_urdf/model.urdf",cubeStartPos, cubeStartOrientation, useFixedBase=True)
objId = p.loadURDF("kuka_iiwa/model.urdf",cubeStartPos, cubeStartOrientation, useFixedBase=True)
# boxId = p.loadSDF("./urdfs/panda_sdf/model.sdf")

# load Texture
# texUid = p.loadTexture("tex256.png")
# p.changeVisualShape(objId, 4, textureUniqueId=texUid)
# p.changeVisualShape(objId, -1, rgbaColor=list(np.array([20, 60, 100, 255])/255))
# p.changeVisualShape(objId, 0, rgbaColor=list(np.array([24, 78, 119, 255])/255))
# p.changeVisualShape(objId, 1, rgbaColor=list(np.array([30, 96, 145, 255])/255))
# p.changeVisualShape(objId, 2, rgbaColor=list(np.array([26, 117, 159, 255])/255))
# p.changeVisualShape(objId, 3, rgbaColor=list(np.array([22, 138, 173, 255])/255))
# p.changeVisualShape(objId, 4, rgbaColor=list(np.array([52, 160, 164, 255])/255))
# p.changeVisualShape(objId, 5, rgbaColor=list(np.array([82, 182, 154, 255])/255))
# p.changeVisualShape(objId, 6, rgbaColor=list(np.array([118, 200, 147, 255])/255))
# p.changeVisualShape(objId, 7, rgbaColor=list(np.array([153, 217, 140, 255])/255))
# p.changeVisualShape(objId, 8, rgbaColor=list(np.array([181, 228, 140, 255])/255))
# p.changeVisualShape(objId, 9, rgbaColor=list(np.array([217, 237, 146, 255])/255))
# p.changeTexture(texUid, pixels, width, height)

# motor velocity control
maxForce = 500
mode = p.VELOCITY_CONTROL
p.setJointMotorControl2(bodyUniqueId=objId, jointIndex=6, controlMode=mode, targetVelocity=1, force=maxForce)
# p.setJointMotorControlArray(bodyUniqueId=objId, jointIndices=[3,4], controlMode=mode, targetVelocities=[1,1], forces=[maxForce, maxForce])

for i in range(200):
    p.stepSimulation()
    p.getCameraImage(300, 300, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(objId)
    print(cubePos,cubeOrn)
    time.sleep(0.01)

print("getNumJoints: {}".format(p.getNumJoints(objId)))
print("getJointInfo: {}".format(p.getJointInfo(objId, 6))) # 8: panda_hand_joint
print("getJointState: {}".format(p.getJointState(objId, 6)))
programPause = input("Press the <ENTER> key to continue...")
p.disconnect()