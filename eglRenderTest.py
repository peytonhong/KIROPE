import pybullet as p
import time
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import pybullet_data
import numpy as np
import cv2
import glob

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
print("plugin=", plugin)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf", [0, 0, -1])
# p.loadURDF("r2d2.urdf")
robotId = p.loadURDF("urdfs/ur3/ur3.urdf", [0, 0, 0], useFixedBase=True)

pixelWidth = 320
pixelHeight = 220
camTargetPos = [0, 0, 0.5]
camDistance = 1
pitch = -10.0
roll = 0
upAxisIndex = 2
fov = 43 # [deg]

# noise map generation
for i in range(100):
  cv2.imwrite(f'noisemap/noisemap_{str(i).zfill(5)}.png', np.random.rand(pixelHeight, pixelWidth)*255)

noise_map_paths = glob.glob('noisemap/*.png')
for j in range(7):
  linkTexId = p.loadTexture(np.random.choice(noise_map_paths))
  p.changeVisualShape(robotId, j-1, textureUniqueId=linkTexId)

# while (p.isConnected()):
for _ in range(1):
  for yaw in range(0, 360, 10):
    start = time.time()
    p.stepSimulation()
    stop = time.time()
    print("stepSimulation %f" % (stop - start))

    # viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
                                                     upAxisIndex)
    projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
    ]

    start = time.time()
    img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1],
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
    stop = time.time()
    print("renderImage %f" % (stop - start))
    print(img_arr[0], img_arr[1])
    print(np.array(img_arr[2]).shape)
    print(np.array(img_arr[3]).shape)
    print(np.array(img_arr[4]).shape)
    #time.sleep(.1)
    #print("img_arr=",img_arr)

p.unloadPlugin(plugin)
print('min, max', np.min(img_arr[3]), np.max(img_arr[3]))
print(np.unique(img_arr[4]))
cv2.imwrite('visualization_result/image.png', np.array(img_arr[2]))
cv2.imwrite('visualization_result/depth.png', np.array(img_arr[3]))
cv2.imwrite('visualization_result/seg.png', (np.array(img_arr[4])+1)*100 )