from digital_twin import DigitalTwin
import time
import pybullet as p 
# DT = DigitalTwin(headless=True)
# start_time = time.time()
# DT.forward()
# print("elapsed time: {}".format(time.time()-start_time))

physicsClient1 = p.connect(p.DIRECT)
physicsClient2 = p.connect(p.DIRECT)
kukaId_1 = p.loadURDF("../urdfs/kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=physicsClient1)
kukaId_2 = p.loadURDF("../urdfs/kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=physicsClient2)
p.setGravity(0, 0, -9.81, physicsClientId=physicsClient1)
p.setGravity(0, 0, -9.81, physicsClientId=physicsClient2)