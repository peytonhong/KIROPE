import os 
import nvisii
import random
import colorsys
import subprocess 
import math
import pybullet as p 
# import pybullet_data
import numpy as np
import simplejson as json
from tqdm import tqdm
from glob import glob
import cv2

opt = lambda : None
# opt.nb_objects = 2
opt.spp = 100
opt.width = 500
opt.height = 500 
opt.noise = False
opt.frame_freq = 8
opt.nb_frames = 10000
opt.inputf = 'annotation/test'
opt.outf = 'joint_alignment'
opt.idx = 999

make_joint_sphere = False

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
    existing_files = glob(f'{opt.outf}/*')
    for f in existing_files:
        os.remove(f)
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
    
# # # # # # # # # # # # # # # # # # # # # # # # #

def save_keypoint_visualize_image(iter_image_path, target_image_path, iter_keypoints, target_keypoints, iter):
    rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
    bgr_colors = rgb_colors[:, ::-1]
    iter_image = cv2.imread(iter_image_path)
    target_image = cv2.imread(target_image_path)    
    image = np.array(iter_image/2, dtype=np.uint8) + np.array(target_image/2, dtype=np.uint8)
    
    cv2.putText(image, f'Iterations: {iter}', (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,0), 2)
    for i, (iter_keypoint, target_keypoint) in enumerate(zip(iter_keypoints, target_keypoints)):
        cv2.drawMarker(image, (int(iter_keypoint[0]), int(iter_keypoint[1])), color=bgr_colors[i].tolist(), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=1)
        cv2.circle(image, (int(target_keypoint[0]), int(target_keypoint[1])), radius=5, color=bgr_colors[i].tolist(), thickness=2)                
    cv2.imwrite(iter_image_path, image.copy())
    

def create_joint_markers(joint_entiies, jointPositions):    
    for i_p, position in enumerate(jointPositions):
        joint_entiies[i_p].get_transform().set_position(position)
        joint_entiies[i_p].get_transform().set_scale((0.1, 0.1, 0.1))
        joint_entiies[i_p].get_material().set_base_color((0.1,0.9,0.08))  
        joint_entiies[i_p].get_material().set_roughness(0.7) 

def get_joint_keypoints(jointPositions, opt, camera_name = 'camera'):
    """
    reproject the 3d points into the image space for a given object. 
    It assumes you already added the cuboid to the object 

    :obj_id: string for the name of the object of interest
    :camera_name: string representing the camera name in nvisii

    :return: cubdoid + centroid projected to the image, values [0..1]
    """

    cam_matrix = nvisii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    cam_proj_matrix = nvisii.entity.get(camera_name).get_camera().get_projection()

    keypoints = []
    points_cam = []
    for j in range(len(jointPositions)):        
        pos_m = nvisii.vec4(
            jointPositions[j][0],
            jointPositions[j][1],
            jointPositions[j][2],
            1)
        
        p_cam = cam_matrix * pos_m 

        p_image = cam_proj_matrix * (cam_matrix * pos_m) 
        p_image = nvisii.vec2(p_image) / p_image.w        
        p_image = p_image * nvisii.vec2(1,-1)        
        p_image = (p_image + nvisii.vec2(1,1)) * 0.5
        keypoints.append([p_image[0]*opt.width, p_image[1]*opt.height])
        points_cam.append([p_cam[0],p_cam[1],p_cam[2]])        
    return keypoints, points_cam


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

def get_joint_world_position(bidyUniqueId):
    # Lets update the pose of the objects in nvisii 
    link_world_state = []
    for link_num in range(p.getNumJoints(bidyUniqueId)+1):
        # get the pose of the objects
        if link_num==0: # base
            link_state = p.getBasePositionAndOrientation(bodyUniqueId=bidyUniqueId)      
            pos_world = link_state[0]
            rot_world = link_state[1]
        else: # link
            link_state = p.getLinkState(bodyUniqueId=bidyUniqueId, linkIndex=link_num-1)
            pos_world = add_tuple(link_state[4], (0, 0, 0.1)) # world position of the URDF link frame
            rot_world = link_state[5] # world orientation of the URDF link frame
        link_world_state.append([pos_world, rot_world])   # (link position is identical to joint position) [8, 2]
    return np.array(link_world_state)

def get_joint_keypoints_from_angles(jointAngles, opt, camera_name = 'camera'):
    cam_matrix = nvisii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    cam_proj_matrix = nvisii.entity.get(camera_name).get_camera().get_projection()
    
    for j in range(numJoints):
        p.resetJointState(bodyUniqueId=kukaId,
                        jointIndex=j,
                        targetValue=(jointAngles[j]),
                        )
    p.stepSimulation()

    link_world_state = get_joint_world_position(kukaId)   # [num_link, position, rotation]
    jointPositions = link_world_state[1:, 0] #[7, 3]

    keypoints = []
    points_cam = []
    for j in range(len(jointPositions)):        
        pos_m = nvisii.vec4(
            jointPositions[j][0],
            jointPositions[j][1],
            jointPositions[j][2],
            1)
        
        p_cam = cam_matrix * pos_m 

        p_image = cam_proj_matrix * (cam_matrix * pos_m) 
        p_image = nvisii.vec2(p_image) / p_image.w        
        p_image = p_image * nvisii.vec2(1,-1)        
        p_image = (p_image + nvisii.vec2(1,1)) * 0.5
        keypoints.append([p_image[0]*opt.width, p_image[1]*opt.height])
        points_cam.append([p_cam[0],p_cam[1],p_cam[2]])            
    keypoints = np.array([keypoints[i][j] for i in range(len(keypoints)) for j in range(2)])  # [14]
    return keypoints

# show an interactive window, and use "lazy" updates for faster object creation time 
nvisii.initialize(headless=False, lazy_updates=True)

if not opt.noise is True: 
    nvisii.enable_denoiser()

# Create a camera
camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create_from_fov(
        name = "camera", 
        field_of_view = 0.85,
        aspect = float(opt.width)/float(opt.height)
    )
)

# lets store the camera look at information so we can export it
camera_struct_look_at = {
    'at':[0,0,1],
    'up':[0,0,1],
    'eye':[3,0,1]
}

# # # # # # # # # # # # # # # # # # # # # # # # #
camera.get_transform().look_at(
    at = camera_struct_look_at['at'],
    up = camera_struct_look_at['up'],
    eye = camera_struct_look_at['eye']
)
nvisii.set_camera_entity(camera)


# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0
physicsClient = p.connect(p.GUI) # non-graphical version


# Lets set the scene

# Change the dome light intensity
nvisii.set_dome_light_intensity(1.0)

# atmospheric thickness makes the sky go orange, almost like a sunset
nvisii.set_dome_light_sky(sun_position=(5,5,5), atmosphere_thickness=1.0, saturation=1.0)

# Lets add a sun light
sun = nvisii.entity.create(
    name = "sun",
    mesh = nvisii.mesh.create_sphere("sphere"),
    transform = nvisii.transform.create("sun"),
    light = nvisii.light.create("sun")
)
sun.get_transform().set_position((10,10,5))
sun.get_light().set_temperature(5780)
sun.get_light().set_intensity(1000)

floor = nvisii.entity.create(
    name="floor",
    mesh = nvisii.mesh.create_plane("floor"),
    transform = nvisii.transform.create("floor"),
    material = nvisii.material.create("floor")
)
floor.get_transform().set_position((0,0,0))
floor.get_transform().set_scale((10, 10, 10))
floor.get_material().set_roughness(0.1)
floor.get_material().set_base_color((0.5,0.5,0.5))

# Set the collision with the floor mesh
# first lets get the vertices 
vertices = floor.get_mesh().get_vertices()

# get the position of the object
pos = floor.get_transform().get_position()
pos = [pos[0],pos[1],pos[2]]
scale = floor.get_transform().get_scale()
scale = [scale[0],scale[1],scale[2]]
rot = floor.get_transform().get_rotation()
rot = [rot[0],rot[1],rot[2],rot[3]]

# create a collision shape that is a convex hull
obj_col_id = p.createCollisionShape(
    p.GEOM_MESH,
    vertices = vertices,
    meshScale = scale,
)

# create a body without mass so it is static
p.createMultiBody(
    baseCollisionShapeIndex = obj_col_id,
    basePosition = pos,
    baseOrientation= rot,
)    

# lets create a robot
kukaId = p.loadURDF("urdfs/kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0.0], [0, 0, 0, 1])

numJoints = p.getNumJoints(kukaId)

if make_joint_sphere:
    # create joint entities to validate joint position (sphere shape)
    joint_entities = []
    for j in range(numJoints):
        sphere = nvisii.entity.create(
        name=f"sphere{j}",
        mesh = nvisii.mesh.create_sphere(f"sphere{j}"),
        transform = nvisii.transform.create(f"sphere{j}"),
        material = nvisii.material.create(f"sphere{j}")
        )
        joint_entities.append(sphere)


p.setGravity(0, 0, -9.81)

jointInfo = p.getJointInfo(kukaId, 0)
lower_limit = [p.getJointInfo(kukaId, i)[8] for i in range(numJoints)]
upper_limit = [p.getJointInfo(kukaId, i)[9] for i in range(numJoints)]

def clamping(min, val, max):
  # returns clamped value between min and max
  return sorted([min, val, max])[1]

def rot_shift(rot):
    # shift quaternion array from [x,y,z,w] (PyBullet) to [w,x,y,z] (NVISII)
    return [rot[-1], rot[0], rot[1], rot[2]]

def add_tuple(a, b):
    assert len(a)==len(b), "Size of two tuples are not matched!"
    return tuple([sum(x) for x in zip(a, b)])

# load robot link meshes from obj file
obj_list = [f'urdfs/kuka_iiwa/meshes/link_{i}.obj' for i in range(numJoints+1)]
link_meshes = [nvisii.mesh.create_from_file(f'mesh_{i}', obj_list[i]) for i in range(len(obj_list))]

#   <material name="Grey">
#     <color rgba="[0.2, 0.2, 0.2] 1.0"/>
#   </material>
#   <material name="Orange">
#     <color rgba="[1.0, 0.42, 0.04] 1.0"/>
#   </material>
#   <material name="Blue">
#   <color rgba="[0.5, 0.7, 1.0] 1.0"/>
link_colors = [[0.2, 0.2, 0.2], [0.5, 0.7, 1.0], [0.5, 0.7, 1.0], [1.0, 0.42, 0.04], [0.5, 0.7, 1.0], [0.5, 0.7, 1.0], [1.0, 0.42, 0.04], [0.2, 0.2, 0.2]]
link_entities = []
for link_num in range(len(link_meshes)):    
    if link_num==0: # base
        link_state = p.getBasePositionAndOrientation(bodyUniqueId=kukaId)      
        pos_world = link_state[0]
        rot_world = link_state[1]
    else: # link
        link_state = p.getLinkState(bodyUniqueId=kukaId, linkIndex=link_num-1)        
        pos_world = add_tuple(link_state[4], (0, 0, 0.1)) # world position of the URDF link frame
        rot_world = link_state[5] # world orientation of the URDF link frame

    link_entity = nvisii.entity.create(
        name=f"link_entity_{link_num}",
        mesh = link_meshes[link_num],
        transform = nvisii.transform.create(f"link_entity_{link_num}"),
        material = nvisii.material.create(f"link_entity_{link_num}")
    )
    link_entity.get_transform().set_position(pos_world)
    # link_entity.get_transform().set_rotation(rot_shift(rot_world)) # nvisii quat expects w as the first argument
    link_entity.get_transform().set_rotation(rot_world) 
    link_entity.get_material().set_base_color(link_colors[link_num])
    link_entity.get_material().set_roughness(0.1)   
    link_entity.get_material().set_specular(1)   
    link_entity.get_material().set_sheen(1)

    link_entities.append(link_entity)



# Lets run the simulation for joint alignment. 
label_paths = sorted(glob(os.path.join(opt.inputf, '*.json')))
with open(label_paths[opt.idx]) as json_file:
    label = json.load(json_file)
targetJointAngles = label['objects'][0]['joint_angles'] # goal for joint angle
target_keypoints = label['objects'][0]['projected_keypoints'] # goal for 2d joint keypoints [7, 2]
target_keypoints = np.array([target_keypoints[i][j] for i in range(len(target_keypoints)) for j in range(2)])  # [14]
# print('target_keypoints: ', target_keypoints)
jointAngles = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
# eps = 1e-6 # epsilon for Jacobian approximation
eps = np.linspace(1e-5, 1e-5, 7)
iterations = 100 # This value can be adjusted.
for iter in range(iterations):    
    
    # get joint 2d keypoint from 3d points and camera model
    keypoints = get_joint_keypoints_from_angles(jointAngles, opt, camera_name = 'camera')
    # print('keypoints: ', keypoints)
    # Jacobian approximation: keypoint rate (변화량)
    Jacobian = np.zeros((numJoints*2, numJoints))
    for col in range(numJoints):
        eps_array = np.zeros(numJoints)
        eps_array[col] = eps[col]
        keypoints_eps = get_joint_keypoints_from_angles(jointAngles+eps_array, opt, camera_name = 'camera')
        Jacobian[:,col] = (keypoints_eps - keypoints)/np.repeat(eps, 2, axis=0)
    dy = np.array(target_keypoints - keypoints)
    dx = np.linalg.pinv(Jacobian)@dy
    jointAngles += dx # all joint angle update

    for j in range(numJoints):
        p.resetJointState(bodyUniqueId=kukaId,
                        jointIndex=j,
                        targetValue=(jointAngles[j]),
                        )    
    p.stepSimulation()
    link_world_state = get_joint_world_position(kukaId)   # [num_link, position, rotation]
    for link_num in range(len(link_world_state)):       
        # get the nvisii entity for that object
        obj_entity = link_entities[link_num]
        obj_entity.get_transform().set_position(link_world_state[link_num][0])        
        obj_entity.get_transform().set_rotation(link_world_state[link_num][1]) 
    
    keypoints = get_joint_keypoints_from_angles(jointAngles, opt, camera_name = 'camera')
    # print(f'iteration: {str(iter).zfill(5)}/{str(iterations).zfill(5)}')



    # get_my_keypoints(camera_entity=camera, robotId=kukaId, joint_world_position=joint_world_position, opt=opt)
    if make_joint_sphere:
        create_joint_markers(joint_entities, link_world_state[1:][0])

    nvisii.render_to_file(
        width=int(opt.width), 
        height=int(opt.height), 
        samples_per_pixel=int(opt.spp),
        file_path=f"{opt.outf}/{str(iter).zfill(5)}.png"
    )
    iter_image_path = f"{opt.outf}/{str(iter).zfill(5)}.png"
    target_image_path = f"{opt.inputf}/{str(opt.idx).zfill(5)}.png"
    save_keypoint_visualize_image(iter_image_path, target_image_path, keypoints.reshape(7,2), target_keypoints.reshape(7,2), iter)
    criteria = np.abs( np.linalg.norm(dx) / np.linalg.norm(jointAngles) )
    print('iter: {}, criteria: {}'.format(iter, criteria))
    if criteria < 1e-1:
        eps *= 0.9
    if criteria < 1e-2:
        break

# export_to_ndds_file(
# filename = f"{opt.inputf}/{str(i).zfill(5)}.json",
# jointPositions = joint_world_position,
# jointStates = joint_states,
# width=opt.width, 
# height=opt.height, 
# camera_struct = camera_struct_look_at,
# )

    

p.disconnect()
nvisii.deinitialize()
framerate = str(iter/10)
subprocess.call(['ffmpeg', '-y', '-framerate', '2', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 'output.mp4'], cwd=os.path.realpath(opt.outf))
