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

opt = lambda : None
# opt.nb_objects = 2
opt.spp = 100
opt.width = 500
opt.height = 500 
opt.noise = False
opt.frame_freq = 8
opt.nb_frames = 10000
opt.type = 'train'
opt.outf = f'annotation/{opt.type}'
opt.random_objects = True
opt.nb_objs = 30

make_joint_sphere = False

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

def create_joint_markers(joint_entiies, jointPositions):    
    for i_p, position in enumerate(jointPositions):
        joint_entiies[i_p].get_transform().set_position(position)
        joint_entiies[i_p].get_transform().set_scale((0.1, 0.1, 0.1))
        joint_entiies[i_p].get_material().set_base_color((0.1,0.9,0.08))  
        joint_entiies[i_p].get_material().set_roughness(0.7) 

def get_joint_keypoints(jointPositions, width, height, camera_name = 'camera'):
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
        keypoints.append([p_image[0]*width, p_image[1]*height])
        points_cam.append([p_cam[0],p_cam[1],p_cam[2]])        
    return keypoints, points_cam

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

def set_random_objects(obj):
    obj.get_transform().set_position((
        uniform(1,2),
        uniform(-1,1),
        uniform(0,2)
    ))

    obj.get_transform().set_rotation((
        uniform(0,1), # X 
        uniform(0,1), # Y
        uniform(0,1), # Z
        uniform(0,1)  # W
    ))

    s = uniform(0.05,0.10)
    obj.get_transform().set_scale((
        s,s,s
    ))  

    rgb = colorsys.hsv_to_rgb(
        uniform(0,1),
        uniform(0.7,1),
        uniform(0.7,1)
    )

    obj.get_material().set_base_color(rgb)

    mat = obj.get_material()
    
    # Some logic to generate "natural" random materials
    material_type = np.random.choice(range(3))
    
    # Glossy / Matte Plastic
    if material_type == 0:  
        if np.random.choice(range(3)): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
    
    # Metallic
    if material_type == 1:  
        mat.set_metallic(uniform(0.9,1))
        if np.random.choice(range(3)): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
    
    # Glass
    if material_type == 2:  
        mat.set_transmission(uniform(0.9,1))
        
        # controls outside roughness
        if np.random.choice(range(3)): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
        
        # controls inside roughness
        if np.random.choice(range(3)): mat.set_transmission_roughness(uniform(.9, 1))
        else           : mat.set_transmission_roughness(uniform(.0,.1))

    mat.set_sheen(uniform(0,1)) # <- soft velvet like reflection near edges
    mat.set_clearcoat(uniform(0,1)) # <- Extra, white, shiny layer. Good for car paint.    
    if np.random.choice(range(2)): mat.set_anisotropic(uniform(0.9,1)) # elongates highlights 

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


# Lets create a random scene. 
if opt.random_objects:
    # First lets pre-load some mesh components.
    nvisii.mesh.create_sphere('m_0')
    nvisii.mesh.create_torus_knot('m_1')
    nvisii.mesh.create_teapotahedron('m_2')
    nvisii.mesh.create_box('m_3')
    nvisii.mesh.create_capped_cone('m_4')
    nvisii.mesh.create_capped_cylinder('m_5')
    nvisii.mesh.create_capsule('m_6')
    nvisii.mesh.create_cylinder('m_7')
    nvisii.mesh.create_disk('m_8')
    nvisii.mesh.create_dodecahedron('m_9')
    nvisii.mesh.create_icosahedron('m_10')
    nvisii.mesh.create_icosphere('m_11')
    nvisii.mesh.create_rounded_box('m_12')
    nvisii.mesh.create_spring('m_13')
    nvisii.mesh.create_torus('m_14')
    nvisii.mesh.create_tube('m_15')
    random_obj_entities = []
    for i in range(opt.nb_objs):
        name = f'random_obj_{i}'
        obj = nvisii.entity.create(
        name = name,
        transform = nvisii.transform.create(name),
        material = nvisii.material.create(name)
        )        
        mesh_id = np.random.choice(range(16))
        mesh = nvisii.mesh.get(f'm_{mesh_id}')
        obj.set_mesh(mesh)
        random_obj_entities.append(obj)

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



# Lets run the simulation for a few steps. 
for i in tqdm(range(int(opt.nb_frames))):
    steps_per_frame = math.ceil( 1.0 / (seconds_per_step * frames_per_second) )
    for _ in range(steps_per_frame):
        # p.stepSimulation()

        # robot joint pose setting
        t += dt
        if opt.type == 'train' and i%100 == 0: # refresh frequencies at every 100 frames
            signs = [np.random.choice([-1, 1]) for _ in range(numJoints)]
            freqs = np.random.rand(numJoints) + 0.2 # 0.2~1.2 [Hz]
        targetJointPoses = [clamping(lower_limit[k], 1.57*signs[k]*np.sin(freqs[k]*t), upper_limit[k]) for k in range(numJoints)]

        for j in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=targetJointPoses[j],
                                    targetVelocity=0,
                                    force=5000,
                                    positionGain=0.1,
                                    velocityGain=0.5)
        
        p.stepSimulation()

    joint_world_position = []
    
    # Lets update the pose of the objects in nvisii 
    for link_num in range(len(link_meshes)):
        # get the pose of the objects
        if link_num==0: # base
            link_state = p.getBasePositionAndOrientation(bodyUniqueId=kukaId)      
            pos_world = link_state[0]
            rot_world = link_state[1]
        else: # link
            link_state = p.getLinkState(bodyUniqueId=kukaId, linkIndex=link_num-1)
            pos_world = add_tuple(link_state[4], (0, 0, 0.1)) # world position of the URDF link frame
            rot_world = link_state[5] # world orientation of the URDF link frame
            joint_world_position.append(pos_world)   # (link position is identical to joint position)
        
        # get the nvisii entity for that object
        obj_entity = link_entities[link_num]
        obj_entity.get_transform().set_position(pos_world)        
        # obj_entity.get_transform().set_rotation(rot_shift(rot_world)) # nvisii quat expects w as the first argument
        obj_entity.get_transform().set_rotation(rot_world) 
    # print(f'rendering frame {str(i).zfill(5)}/{str(opt.nb_frames).zfill(5)}')

    # get joint states
    joint_states = []
    for j in range(numJoints):
        state = p.getJointState(kukaId, j)
        joint_states.append([state[0], state[1]])

    # get_my_keypoints(camera_entity=camera, robotId=kukaId, joint_world_position=joint_world_position, opt=opt)
    if make_joint_sphere:
        create_joint_markers(joint_entities, joint_world_position)

    if opt.random_objects:
        for obj_entity in random_obj_entities:
            set_random_objects(obj_entity)

    nvisii.render_to_file(
        width=int(opt.width), 
        height=int(opt.height), 
        samples_per_pixel=int(opt.spp),
        file_path=f"{opt.outf}/{str(i).zfill(5)}.png"
    )

    export_to_ndds_file(
    filename = f"{opt.outf}/{str(i).zfill(5)}.json",
    jointPositions = joint_world_position,
    jointStates = joint_states,
    width=opt.width, 
    height=opt.height, 
    camera_struct = camera_struct_look_at,
    )

    

p.disconnect()
nvisii.deinitialize()

subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))
