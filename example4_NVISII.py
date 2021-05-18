import os 
import nvisii
import random
import colorsys
import subprocess 
import math
import pybullet as p 
# import pybullet_data
import numpy as np

opt = lambda : None
opt.nb_objects = 2
opt.spp = 100
opt.width = 500
opt.height = 500 
opt.noise = False
opt.frame_freq = 8
opt.nb_frames = 100 #300
opt.outf = 'results'
opt.random_color = True


# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

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
camera.get_transform().look_at(
    at = (0,0,1),
    up = (0,0,1),
    eye = (3,0,1),
)
nvisii.set_camera_entity(camera)

# cam_proj_matrix = nvisii.entity.get('camera').get_camera().get_projection()
cam_proj_matrix = camera.get_camera().get_projection()
print(cam_proj_matrix)
# exit(0)
# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0
physicsClient = p.connect(p.GUI) # non-graphical version
# p.setGravity(0,0,-10)

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

gray_wall_color = nvisii.texture.create_from_file("gray_wall", "./content/photos_2020_5_11_fst_gray-wall-grunge.jpg")
gradient_color = nvisii.texture.create_from_file("gradient", "./content/gradient.png")

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
floor.get_material().set_base_color_texture(gray_wall_color)
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
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF("plane.urdf", [0, 0, -0.3])
kukaId = p.loadURDF("urdfs/kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0.0], [0, 0, 0, 1])

numJoints = p.getNumJoints(kukaId)

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
    link_entity.get_transform().set_rotation(rot_world)
    
    if opt.random_color: # random link color
        # Material setting
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0,1),
            random.uniform(0.7,1),
            random.uniform(0.7,1)
        )

        link_entity.get_material().set_base_color(rgb)

        obj_mat = link_entity.get_material()
        r = random.randint(0,2)

        # This is a simple logic for more natural random materials, e.g.,  
        # mirror or glass like objects
        if r == 0:  
            # Plastic / mat
            obj_mat.set_metallic(0)  # should 0 or 1      
            obj_mat.set_transmission(0)  # should 0 or 1      
            obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
        if r == 1:  
            # metallic
            obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
            obj_mat.set_transmission(0)  # should 0 or 1      
        if r == 2:  
            # glass
            obj_mat.set_metallic(0)  # should 0 or 1      
            obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

        if r > 0: # for metallic and glass
            r2 = random.randint(0,1)
            if r2 == 1: 
                obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
            else:
                obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1 
        
    else: # original color
        link_entity.get_material().set_base_color(link_colors[link_num])
        link_entity.get_material().set_roughness(0.1)   
        link_entity.get_material().set_specular(1)   
        link_entity.get_material().set_sheen(1)

    link_entities.append(link_entity)


# # lets create a bunch of objects 
# mesh = nvisii.mesh.create_teapotahedron('mesh')

# # set up for pybullet - here we will use indices for 
# # objects with holes 
# vertices = mesh.get_vertices()
# indices = mesh.get_triangle_indices()

# ids_pybullet_and_nvisii_names = []

# for i in range(opt.nb_objects):
#     name = f"mesh_{i}"
#     obj= nvisii.entity.create(
#         name = name,
#         transform = nvisii.transform.create(name),
#         material = nvisii.material.create(name)
#     )
#     obj.set_mesh(mesh)

#     # transforms
#     pos = nvisii.vec3(
#         random.uniform(-4,4),
#         random.uniform(-4,4),
#         random.uniform(2,5)
#     )
#     rot = nvisii.normalize(nvisii.quat(
#         random.uniform(-1,1),
#         random.uniform(-1,1),
#         random.uniform(-1,1),
#         random.uniform(-1,1),
#     ))
#     s = random.uniform(0.2,0.5)
#     scale = (s,s,s)

#     obj.get_transform().set_position(pos)
#     obj.get_transform().set_rotation(rot)
#     obj.get_transform().set_scale(scale)

#     # pybullet setup 
#     pos = [pos[0],pos[1],pos[2]]
#     rot = [rot[0],rot[1],rot[2],rot[3]]
#     scale = [scale[0],scale[1],scale[2]]

#     obj_col_id = p.createCollisionShape(
#         p.GEOM_MESH,
#         vertices = vertices,
#         meshScale = scale,
#         # if you have static object like a bowl
#         # this allows you to have concave objects, but 
#         # for non concave object, using indices is 
#         # suboptimal, you can uncomment if you want to test
#         # indices =  indices,  
#     )
    
#     p.createMultiBody(
#         baseCollisionShapeIndex = obj_col_id,
#         basePosition = pos,
#         baseOrientation= rot,
#         baseMass = random.uniform(0.5,2)
#     )       

#     # to keep track of the ids and names 
#     ids_pybullet_and_nvisii_names.append(
#         {
#             "pybullet_id":obj_col_id, 
#             "nvisii_id":name
#         }
#     )

    # # Material setting
    # rgb = colorsys.hsv_to_rgb(
    #     random.uniform(0,1),
    #     random.uniform(0.7,1),
    #     random.uniform(0.7,1)
    # )

    # obj.get_material().set_base_color(rgb)

    # obj_mat = obj.get_material()
    # r = random.randint(0,2)

    # # This is a simple logic for more natural random materials, e.g.,  
    # # mirror or glass like objects
    # if r == 0:  
    #     # Plastic / mat
    #     obj_mat.set_metallic(0)  # should 0 or 1      
    #     obj_mat.set_transmission(0)  # should 0 or 1      
    #     obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
    # if r == 1:  
    #     # metallic
    #     obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
    #     obj_mat.set_transmission(0)  # should 0 or 1      
    # if r == 2:  
    #     # glass
    #     obj_mat.set_metallic(0)  # should 0 or 1      
    #     obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

    # if r > 0: # for metallic and glass
    #     r2 = random.randint(0,1)
    #     if r2 == 1: 
    #         obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
    #     else:
    #         obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  

# Lets run the simulation for a few steps. 
for i in range (int(opt.nb_frames)):
    steps_per_frame = math.ceil( 1.0 / (seconds_per_step * frames_per_second) )
    for j in range(steps_per_frame):
        p.stepSimulation()

        # robot joint pose setting
        t += dt
        targetJointPoses = [clamping(lower_limit[k], signs[k]*np.sin(freqs[k]*t), upper_limit[k]) for k in range(numJoints)]

        for l in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                    jointIndex=l,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=targetJointPoses[l],
                                    targetVelocity=0,
                                    force=5000,
                                    positionGain=0.1,
                                    velocityGain=0.5)


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
        
        # get the nvisii entity for that object
        obj_entity = link_entities[link_num]
        obj_entity.get_transform().set_position(pos_world)
        obj_entity.get_transform().set_rotation(rot_world) 

    print(f'rendering frame {str(i).zfill(5)}/{str(opt.nb_frames).zfill(5)}')

    nvisii.render_to_file(
        width=int(opt.width), 
        height=int(opt.height), 
        samples_per_pixel=int(opt.spp),
        file_path=f"{opt.outf}/{str(i).zfill(5)}.png"
    )
    

p.disconnect()
nvisii.deinitialize()

# subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))
