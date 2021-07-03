import os 
import subprocess 
import math
import pybullet as p 
import pkgutil
import pybullet_data
import numpy as np
import simplejson as json
from tqdm import tqdm
import cv2
import glob


opt = lambda : None
opt.width = 500
opt.height = 500 
opt.noise = False
opt.nb_frames = 1000
opt.type = 'temp'
opt.outf = f'annotation/{opt.type}'
opt.random_objects = False
opt.nb_objs = 30

 

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #


def make_annotation_file(
    filename = "tmp.json", #this has to include path as well
    jointPositions = [], # this is a list of joint 3d positions to be exported into 2d keypoints
    jointStates = [], # shape: [numJoints,2] each by (angle, velocity)
    jointKeypoints = [],
    height = 500, 
    width = 500,
    camera_struct = [],
    cam_intrinsic = [],
    cam_extrinsic = [],
    fov = [],
    ):

    dict_out = {
                "camera_data" : {
                    'width' : width,
                    'height' : height,
                    'fov' : fov,
                    'camera_look_at':
                    {
                        'at': [
                            camera_struct['at'][0],
                            camera_struct['at'][1],
                            camera_struct['at'][2],
                        ],
                        'eye': [
                            camera_struct['eye'][0],
                            camera_struct['eye'][1],
                            camera_struct['eye'][2],
                        ],
                        'up': [
                            camera_struct['up'][0],
                            camera_struct['up'][1],
                            camera_struct['up'][2],
                        ]
                    },
                    'camera_intrinsics': cam_intrinsic.tolist(),
                    'camera_extrinsics': cam_extrinsic.tolist(),                    
                }, 
                "objects" : {
                    'projected_keypoints': jointKeypoints.tolist(),
                    'joint_angles': [jointStates[j][0] for j in range(len(jointStates))],
                    'joint_velocities': [jointStates[j][1] for j in range(len(jointStates))],
                }
            }
    
    with open(filename, 'w') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=False)


def get_my_keypoints(cam_K, cam_R, robotId, joint_world_position, opt):
    # get 2d keypoints from 3d positions using camera K, R matrix (2021.06.30, Hyosung Hong)    
    numJoints = p.getNumJoints(robotId)
    jointPositions = np.zeros((numJoints, 3))
    jointKeypoints = np.zeros((numJoints, 2))
    for l in range(numJoints):
        jointPosition = np.array(list(joint_world_position[l])+[1.]).reshape(4,1)
        jointKeypoint = cam_K@cam_R@jointPosition
        jointKeypoint /= jointKeypoint[-1]
        jointKeypoint[0] = opt.width - jointKeypoint[0]   # OpenGL convention for left-right mirroring
        jointPositions[l] = jointPosition.reshape(-1)[:3]
        jointKeypoints[l] = jointKeypoint.reshape(-1)[:2]
    # print('jointPositions: ', jointPositions)
    # print('jointKeypoints: ', jointKeypoints)
    return jointKeypoints # [numJoints, 2]

def uniform(a, b): # for random object placement
        "Get a random number in the range [a, b) or [a, b] depending on rounding."
        return a + (b-a) * np.random.rand()

def clamping(min, val, max):
  # returns clamped value between min and max
  return sorted([min, val, max])[1]

def rot_shift(rot):
    # shift quaternion array from [x,y,z,w] (PyBullet) to [w,x,y,z] (NVISII)
    return [rot[-1], rot[0], rot[1], rot[2]]

def add_tuple(a, b):
    assert len(a)==len(b), "Size of two tuples are not matched!"
    return tuple([sum(x) for x in zip(a, b)])



physicsClient = p.connect(p.DIRECT)

# lets create a robot
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane.urdf", [0, 0, -1])
robotId = p.loadURDF("urdfs/ur3/ur3.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0.0], [0, 0, 0, 1])

# egl = pkgutil.get_loader('eglRenderer')
# if (egl):
#     plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")    
# else:
plugin = p.loadPlugin("eglRendererPlugin")
    
print("plugin=", plugin)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


# camera settings
camera_struct_look_at_1 = {
    'at':[0,0,0.5],
    'up':[0,0,1],
    'eye':[1.5,0,0.5]
}
camera_struct_look_at_2 = {
    'at':[0,0,0.5],
    'up':[0,0,1],
    'eye':[0,1.5,0.5]
}
fov = 43 # Intel Realsense L515
cam_intrinsic = p.computeProjectionMatrixFOV(fov=fov, # [view angle in degree]
                                            aspect=opt.width/opt.height,
                                            nearVal=0.1,
                                            farVal=100,
                                            )
cam_extrinsic_1 = p.computeViewMatrix(cameraEyePosition=camera_struct_look_at_1['eye'],
                                cameraTargetPosition=camera_struct_look_at_1['at'],
                                cameraUpVector=camera_struct_look_at_1['up'],
                                )
cam_extrinsic_2 = p.computeViewMatrix(cameraEyePosition=camera_struct_look_at_2['eye'],
                                cameraTargetPosition=camera_struct_look_at_2['at'],
                                cameraUpVector=camera_struct_look_at_2['up'],
                                )

# cam_K = np.array(cam_intrinsic).reshape(4,4).transpose()
cam_R_1 = np.array(cam_extrinsic_1).reshape(4,4).transpose()
cam_R_2 = np.array(cam_extrinsic_2).reshape(4,4).transpose()
fx = opt.height/(2*np.tan(fov*np.pi/180/2))
fy = opt.height/(2*np.tan(fov*np.pi/180/2))

cam_K = np.array([[fx, 0, opt.height/2],[0, fy, opt.width/2], [0, 0, 1]])
cam_R_1 = cam_R_1[:3]
cam_R_2 = cam_R_2[:3]

# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0

numJoints = p.getNumJoints(robotId)

# texUid = p.loadTexture("../NVISII/examples/content/salle_de_bain_separated/textures/WoodFloor_BaseColor.jpg")
# texUidgradient = p.loadTexture("../NVISII/examples/content/photos_2020_5_11_fst_gray-wall-grunge.jpg")
# p.changeVisualShape(planeUid, -1, textureUniqueId=texUid)

# noise map generation
# for i in range(100):
#     cv2.imwrite(f'noisemap/noisemap_{str(i).zfill(5)}.png', np.random.rand(opt.height, opt.width, 3)*255)
noise_map_paths = glob.glob('noisemap/*.png')


# p.changeVisualShape(robotId, -1, specularColor=10)
# p.changeVisualShape(robotId, 0, specularColor=30)
# p.changeVisualShape(robotId, 1, specularColor=60)
# p.changeVisualShape(robotId, 2, specularColor=90)
# p.changeVisualShape(robotId, 3, specularColor=120)
# p.changeVisualShape(robotId, 4, specularColor=150)
# p.changeVisualShape(robotId, 5, specularColor=180)




p.setGravity(0, 0, -9.81)
t = 0

signs = [np.random.choice([-1, 1]) for _ in range(numJoints)]
freqs = np.random.rand(numJoints) + 0.2 # 0.2~1.2 [Hz]

jointInfo = p.getJointInfo(robotId, 0)
lower_limit = [p.getJointInfo(robotId, i)[8] for i in range(numJoints)]
upper_limit = [p.getJointInfo(robotId, i)[9] for i in range(numJoints)]

rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
bgr_colors = rgb_colors[:, ::-1]

# Lets run the simulation for a few steps. 
for i in tqdm(range(int(opt.nb_frames))):
# for i in range(100):
    steps_per_frame = math.ceil( 1.0 / (seconds_per_step * frames_per_second) )
    for _ in range(steps_per_frame):
        # p.stepSimulation()

        # robot joint pose setting
        t += seconds_per_step
        # if opt.type == 'train' and i%100 == 0: # refresh frequencies at every 100 frames
        #     signs = [np.random.choice([-1, 1]) for _ in range(numJoints)]
        #     freqs = np.random.rand(numJoints) + 0.2 # 0.2~1.2 [Hz]
        targetJointPoses = [clamping(lower_limit[k], 1.0*signs[k]*np.sin(freqs[k]*t), upper_limit[k]) for k in range(numJoints)]
        targetJointPoses[:3] = [0]*3
        for j in range(numJoints):
            p.setJointMotorControl2(bodyIndex=robotId,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=targetJointPoses[j],
                                    targetVelocity=0,
                                    force=5000,
                                    positionGain=0.1,
                                    velocityGain=0.5)
        # for j in range(numJoints+1):
        #     linkTexId = p.loadTexture(np.random.choice(noise_map_paths))
        #     p.changeVisualShape(robotId, j-1, textureUniqueId=linkTexId)
        p.stepSimulation()

    image_arr_1 = p.getCameraImage(opt.width,
                            opt.height,
                            viewMatrix=cam_extrinsic_1,
                            projectionMatrix=cam_intrinsic,
                            shadow=1,
                            lightDirection=[1, 1, 1],
                            renderer=p.ER_BULLET_HARDWARE_OPENGL, #p.ER_TINY_RENDERER, #p.ER_BULLET_HARDWARE_OPENGL
                            )
    image_arr_2 = p.getCameraImage(opt.width,
                            opt.height,
                            viewMatrix=cam_extrinsic_2,
                            projectionMatrix=cam_intrinsic,
                            shadow=1,
                            lightDirection=[1, 1, 1],
                            renderer=p.ER_BULLET_HARDWARE_OPENGL, #p.ER_TINY_RENDERER, #p.ER_BULLET_HARDWARE_OPENGL
                            )


    image_1 = np.array(image_arr_1[2]) # [height, width, 4]
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
    image_2 = np.array(image_arr_2[2]) # [height, width, 4]
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2BGR)
    
    # depth = np.array(image_arr[3])*255
    # print('depth.shape',depth.shape)
    # cv2.imwrite(f'visualization_result/{str(i).zfill(5)}_depth.png', depth)
    
    joint_states = []
    for j in range(numJoints):
        state = p.getJointState(robotId, j)
        joint_states.append([state[0], state[1]])

    # get joint states
    joint_world_position = []
    for link_num in range(numJoints):    
        link_state = p.getLinkState(bodyUniqueId=robotId, linkIndex=link_num)
        pos_world = list(link_state[4])
        # rot_world = link_state[5] # world orientation of the URDF link frame        
        joint_world_position.append(pos_world)
        
    jointKeypoints_1 = get_my_keypoints(cam_K, cam_R_1, robotId=robotId, joint_world_position=joint_world_position, opt=opt)
    jointKeypoints_2 = get_my_keypoints(cam_K, cam_R_2, robotId=robotId, joint_world_position=joint_world_position, opt=opt)
    
    cv2.imwrite(f'{opt.outf}/cam1/{str(i).zfill(5)}.png', image_1)
    cv2.imwrite(f'{opt.outf}/cam2/{str(i).zfill(5)}.png', image_2)

    for j, (keypoint_1, keypoint_2) in enumerate(zip(jointKeypoints_1, jointKeypoints_2)):
        cv2.circle(image_1, (int(keypoint_1[0]), int(keypoint_1[1])), radius=5, color=bgr_colors[j].tolist(), thickness=2)
        cv2.circle(image_2, (int(keypoint_2[0]), int(keypoint_2[1])), radius=5, color=bgr_colors[j].tolist(), thickness=2)
    
    image = np.hstack((image_1, image_2))
    cv2.imwrite(f'{opt.outf}/twocams/{str(i).zfill(5)}.png', image)

    make_annotation_file(
    filename = f"{opt.outf}/cam1/{str(i).zfill(5)}.json",
    jointPositions = joint_world_position,
    jointStates = joint_states,
    jointKeypoints = jointKeypoints_1,
    width=opt.width, 
    height=opt.height, 
    camera_struct = camera_struct_look_at_1,
    cam_intrinsic = cam_K,
    cam_extrinsic = cam_R_1,
    fov = fov,
    )
    make_annotation_file(
    filename = f"{opt.outf}/cam2/{str(i).zfill(5)}.json",
    jointPositions = joint_world_position,
    jointStates = joint_states,
    jointKeypoints = jointKeypoints_2,
    width=opt.width, 
    height=opt.height, 
    camera_struct = camera_struct_look_at_2,
    cam_intrinsic = cam_K,
    cam_extrinsic = cam_R_2,
    fov = fov,
    )
    
p.unloadPlugin(plugin)
p.disconnect()

# subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))
