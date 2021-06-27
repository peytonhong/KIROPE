import os 
import subprocess 
import math
import pybullet as p 
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
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
opt.nb_frames = 10000
opt.type = 'train'
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


def export_to_ndds_file(
    filename = "tmp.json", #this has to include path as well
    jointPositions = [], # this is a list of joint 3d positions to be exported into 2d keypoints
    jointStates = [], # shape: [numJoints,2] each by (angle, velocity)
    height = 500, 
    width = 500,
    camera_name = 'camera',
    camera_struct = None,
    ):
    """
    Method that exports the meta data like NDDS. This includes all the scene information in one 
    scene. 

    :filename: string for the json file you want to export, you have to include the extension
    :obj_names: [string] each entry is a nvisii entity that has the cuboids attached to, these
                are the objects that are going to be exported. 
    :height: int height of the image size 
    :width: int width of the image size 
    :camera_name: string for the camera name nvisii entity
    :camera_struct: dictionary of the camera look at information. Expecting the following 
                    entries: 'at','eye','up'. All three has to be floating arrays of three entries.
                    This is an optional export. 
    :visibility_percentage: bool if you want to export the visibility percentage of the object. 
                            Careful this can be costly on a scene with a lot of objects. 
    :jointPositions: [numJoints, 3] 3d joint position of a robot

    :return nothing: 
    """

    # assume we only use the view camera
    cam_matrix = nvisii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    
    cam_matrix_export = []
    for row in cam_matrix:
        cam_matrix_export.append([row[0],row[1],row[2],row[3]])
    
    cam_world_location = nvisii.entity.get(camera_name).get_transform().get_position()
    cam_world_quaternion = nvisii.entity.get(camera_name).get_transform().get_rotation()
    # cam_world_quaternion = nvisii.quat_cast(cam_matrix)

    cam_intrinsics = nvisii.entity.get(camera_name).get_camera().get_intrinsic_matrix(width, height)

    if camera_struct is None:
        camera_struct = {
            'at': [0,0,0,],
            'eye': [0,0,0,],
            'up': [0,0,0,]
        }

    dict_out = {
                "camera_data" : {
                    "width" : width,
                    'height' : height,
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
                    'camera_view_matrix':cam_matrix_export,
                    'location_world':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                    'quaternion_world_xyzw':[
                        cam_world_quaternion[0],
                        cam_world_quaternion[1],
                        cam_world_quaternion[2],
                        cam_world_quaternion[3],
                    ],
                    'intrinsics':{
                        'fx':cam_intrinsics[0][0],
                        'fy':cam_intrinsics[1][1],
                        'cx':cam_intrinsics[2][0],
                        'cy':cam_intrinsics[2][1]
                    }
                }, 
                "objects" : []
            }

    projected_keypoints, _ = get_joint_keypoints(jointPositions, opt.width, opt.height, camera_name=camera_name)

    # put them in the image space. 
    # for i_p, p in enumerate(projected_keypoints):
    #     projected_keypoints[i_p] = [p[0]*width, p[1]*height]
    # print('projected_keypoints: ', projected_keypoints)
   
    # Final export
    dict_out['objects'].append({
        'projected_keypoints': projected_keypoints,
        'joint_angles': [jointStates[j][0] for j in range(len(jointStates))],
        'joint_velocities': [jointStates[j][1] for j in range(len(jointStates))],
        },
        )
        
    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)


def get_my_keypoints(camera_entity, robotId, joint_world_position, opt):
    # get 2d keypoints from 3d positions using camera K, R matrix (2021.05.03, Hyosung Hong)
    
    numJoints = p.getNumJoints(robotId)
    # Camera intrinsic and extrinsic matrix
    cam_intrinsic = camera_entity.get_camera().get_intrinsic_matrix(opt.width, opt.height)
    cam_extrinsic = camera_entity.get_transform().get_world_to_local_matrix()
    cam_intrinsic_export = []
    cam_extrinsic_export = []
    for row in cam_intrinsic:
        cam_intrinsic_export.append([row[0],row[1],row[2]])
    for row in cam_extrinsic:
        cam_extrinsic_export.append([row[0],row[1],row[2],row[3]])

    cam_K = np.array(cam_intrinsic_export).reshape(-1,3).transpose() # camera intrinsic [3x3]
    cam_R = np.array(cam_extrinsic_export).reshape(-1,4).transpose()[:3] # camera extrinsic [3x4]

    jointPositions = []
    jointKeypoints = []
    for l in range(numJoints):
        jointPosition = np.array(list(joint_world_position[l])+[1.]).reshape(4,1)        
        jointKeypoint = cam_K@cam_R@jointPosition
        jointKeypoint /= jointKeypoint[-1]
        jointPositions.append(jointPosition)
        jointKeypoints.append(jointKeypoint)
    print('jointPositions: ', jointPositions)
    print('jointKeypoints: ', jointKeypoints)

    return jointKeypoints

def uniform(a, b):
        "Get a random number in the range [a, b) or [a, b] depending on rounding."
        return a + (b-a) * np.random.rand()

physicsClient = p.connect(p.DIRECT)

# lets create a robot
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane.urdf", [0, 0, -1])
robotId = p.loadURDF("urdfs/ur3/ur3.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0.0], [0, 0, 0, 1])


# plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
plugin = p.loadPlugin("eglRendererPlugin")
print("plugin=", plugin)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


# camera settings
camera_struct_look_at = {
    'at':[0,0,0.5],
    'up':[0,0,1],
    'eye':[1.5,0,0.5]
}

cam_intrinsic = p.computeProjectionMatrixFOV(fov=43, # [view angle in degree]
                                            aspect=opt.width/opt.height,
                                            nearVal=0.1,
                                            farVal=100,
                                            )
cam_extrinsic = p.computeViewMatrix(cameraEyePosition=camera_struct_look_at['eye'],
                                cameraTargetPosition=camera_struct_look_at['at'],
                                cameraUpVector=camera_struct_look_at['up'],
                                )

cam_K = np.array(cam_intrinsic).reshape(4,4)
cam_R = np.array(cam_extrinsic).reshape(4,4)


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

def clamping(min, val, max):
  # returns clamped value between min and max
  return sorted([min, val, max])[1]

def rot_shift(rot):
    # shift quaternion array from [x,y,z,w] (PyBullet) to [w,x,y,z] (NVISII)
    return [rot[-1], rot[0], rot[1], rot[2]]

def add_tuple(a, b):
    assert len(a)==len(b), "Size of two tuples are not matched!"
    return tuple([sum(x) for x in zip(a, b)])


# Lets run the simulation for a few steps. 
for i in tqdm(range(int(opt.nb_frames))):
    steps_per_frame = math.ceil( 1.0 / (seconds_per_step * frames_per_second) )
    for _ in range(steps_per_frame):
        # p.stepSimulation()

        # robot joint pose setting
        t += seconds_per_step
        # if opt.type == 'train' and i%100 == 0: # refresh frequencies at every 100 frames
        #     signs = [np.random.choice([-1, 1]) for _ in range(numJoints)]
        #     freqs = np.random.rand(numJoints) + 0.2 # 0.2~1.2 [Hz]
        targetJointPoses = [clamping(lower_limit[k], 1.0*signs[k]*np.sin(freqs[k]*t), upper_limit[k]) for k in range(numJoints)]

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

    img_arr = p.getCameraImage(opt.width,
                            opt.height,
                            viewMatrix=cam_extrinsic,
                            projectionMatrix=cam_intrinsic,
                            shadow=1,
                            lightDirection=[5, 5, 5],
                            renderer=p.ER_BULLET_HARDWARE_OPENGL, #p.ER_TINY_RENDERER, #p.ER_BULLET_HARDWARE_OPENGL
                            )
    img = np.array(img_arr[2]) # [height, width, 4]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'visualization_result/{str(i).zfill(5)}.png', img)
    print('img_arr: ', img_arr)
    print('unique:',np.unique(img))
    print('unique:',np.unique(img_arr[3]))
    print('unique:',np.unique(img_arr[4]))
    exit()
    # get joint states
    # joint_states = []
    # for j in range(numJoints):
    #     state = p.getJointState(robotId, j)
    #     joint_states.append([state[0], state[1]])

    # get_my_keypoints(camera_entity=camera, robotId=robotId, joint_world_position=joint_world_position, opt=opt)
    

    # if opt.random_objects:
    #     for obj_entity in random_obj_entities:
    #         set_random_objects(obj_entity)


    # export_to_ndds_file(
    # filename = f"{opt.outf}/{str(i).zfill(5)}.json",
    # jointPositions = joint_world_position,
    # jointStates = joint_states,
    # width=opt.width, 
    # height=opt.height, 
    # camera_struct = camera_struct_look_at,
    # )

    
p.unloadPlugin(plugin)
p.disconnect()

subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))
