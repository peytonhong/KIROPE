import os
from pickle import HIGHEST_PROTOCOL 
import nvisii
import random
import colorsys
import subprocess 
import math
import pybullet as p
# import pybullet_data
import numpy as np
import simplejson as json
from torch.nn.functional import hinge_embedding_loss
from tqdm import tqdm
from glob import glob
import cv2
import csv

class DigitalTwin():

    def __init__(self, urdf_path="../urdfs/kuka_iiwa/model.urdf", 
                    mesh_path="../urdfs/kuka_iiwa/meshes", 
                    headless=False,
                    save_images=False,
                    ):
        
        self.opt = lambda : None
        # self.opt.nb_objects = 2
        self.opt.spp = 100
        self.opt.width = 500
        self.opt.height = 500 
        self.opt.noise = False
        self.opt.frame_freq = 8
        self.opt.nb_frames = 10000
        self.opt.inputf = '../annotation/test_no_rand_obj'
        self.opt.outf = 'joint_alignment'
        self.opt.idx = 999

        self.make_joint_sphere = False
        self.save_images = save_images
        
        # # # # # # # # # # # # # # # # # # # # # # # # #
        if os.path.isdir(self.opt.outf):
            print(f'folder {self.opt.outf}/ exists')
            existing_files = glob(f'{self.opt.outf}/*')
            for f in existing_files:
                os.remove(f)
        else:
            os.mkdir(self.opt.outf)
            print(f'created folder {self.opt.outf}/')
            
        # # # # # # # # # # # # # # # # # # # # # # # # #

        # show an interactive window, and use "lazy" updates for faster object creation time 
        nvisii.initialize(headless=headless, lazy_updates=True)

        # if not self.opt.noise is True: 
        nvisii.enable_denoiser()

        # Create a camera
        camera = nvisii.entity.create(
            name = "camera",
            transform = nvisii.transform.create("camera"),
            camera = nvisii.camera.create_from_fov(
                name = "camera", 
                field_of_view = 0.85,
                aspect = float(self.opt.width)/float(self.opt.height)
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



        # Setup bullet physics stuff
        self.seconds_per_step = 1.0 / 240.0
        self.frames_per_second = 30.0
        # physicsClient = p.connect(p.GUI) # graphical version
        self.physicsClient_main = p.connect(p.DIRECT) # non-graphical version
        self.physicsClient_jpnp = p.connect(p.DIRECT) # non-graphical version


        # create a collision shape that is a convex hull
        obj_col_id = p.createCollisionShape(
            p.GEOM_MESH,
            vertices = vertices,
            meshScale = scale,
            physicsClientId=self.physicsClient_main
        )

        # create a body without mass so it is static
        p.createMultiBody(
            baseCollisionShapeIndex = obj_col_id,
            basePosition = pos,
            baseOrientation= rot,
            physicsClientId=self.physicsClient_main
        )    

        # lets create a robot
        self.robotId_main = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.physicsClient_main)
        self.robotId_jpnp = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.physicsClient_jpnp)
        p.resetBasePositionAndOrientation(self.robotId_main, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId=self.physicsClient_main)
        p.resetBasePositionAndOrientation(self.robotId_jpnp, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId=self.physicsClient_jpnp)

        self.numJoints = p.getNumJoints(self.robotId_main, physicsClientId=self.physicsClient_main)

        if self.make_joint_sphere:
            # create joint entities to validate joint position (sphere shape)
            self.joint_entities = []
            for j in range(self.numJoints):
                sphere = nvisii.entity.create(
                name=f"sphere{j}",
                mesh = nvisii.mesh.create_sphere(f"sphere{j}"),
                transform = nvisii.transform.create(f"sphere{j}"),
                material = nvisii.material.create(f"sphere{j}")
                )
                self.joint_entities.append(sphere)


        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient_main)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient_main)

        jointInfo = p.getJointInfo(self.robotId_main, 0, physicsClientId=self.physicsClient_main)
        lower_limit = [p.getJointInfo(self.robotId_main, i, physicsClientId=self.physicsClient_main)[8] for i in range(self.numJoints)]
        upper_limit = [p.getJointInfo(self.robotId_main, i, physicsClientId=self.physicsClient_main)[9] for i in range(self.numJoints)]

        self.jointAngles_main = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32) # main robot
        self.jointAngles_jpnp = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32) # Joint PnP robot
        self.jointVelocities_main = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        # load robot link meshes from obj file
        obj_list = [f'{mesh_path}/link_{i}.obj' for i in range(self.numJoints+1)]
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
        
        
        link_world_state = self.get_joint_world_position(self.robotId_main, self.physicsClient_main)   # [num_link, position, rotation]
        self.link_entities = []
        # create link entity
        for link_num in range(len(link_world_state)):
            link_entity = nvisii.entity.create(
                name=f"link_entity_{link_num}",
                mesh = link_meshes[link_num],
                transform = nvisii.transform.create(f"link_entity_{link_num}"),
                material = nvisii.material.create(f"link_entity_{link_num}")
            )
            link_entity.get_transform().set_position(link_world_state[link_num][0])
            link_entity.get_transform().set_rotation(link_world_state[link_num][1]) 
            link_entity.get_material().set_base_color(link_colors[link_num])
            link_entity.get_material().set_roughness(0.1)   
            link_entity.get_material().set_specular(1)   
            link_entity.get_material().set_sheen(1)

            self.link_entities.append(link_entity)

        # Kalman filter variables
        nj = self.numJoints
        self.dT = self.frames_per_second
        self.A = np.vstack( (np.hstack((np.eye(nj), np.eye(nj)*self.dT)), 
                            np.hstack((np.zeros((nj,nj)), np.eye(nj))))) # [2*numJoints, 2*numJoints]
        self.P = np.eye(2*nj)
        Q_upper = np.eye(nj)*0.1     # angle noise
        Q_lower = np.eye(nj)*0.0001 # angular velocity noise
        self.Q = np.vstack( (np.hstack((Q_upper, np.zeros((nj,nj)))), 
                            np.hstack((np.zeros((nj,nj)), Q_lower)))) # [2*numJoints, 2*numJoints]
        self.R = np.eye(nj)
        
        self.X = np.hstack((self.jointAngles_main, self.jointVelocities_main)).reshape(-1,1) #[2*numJoints, 1]
        self.H = np.hstack((np.eye(nj), np.zeros((nj,nj)))) # [numJoints, 2*numJoints]

        self.jpnp_max_iterations = 100
        self.iters = 0 # iteration steps for forward function calls (iters is used to name the saved image file)

        with open('./utils/dt_debug.csv', 'w') as  f: 
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['iter', 'angle_error', 'joint_angles_gt', 'jointAngle_command', 'self.jointAngles_main' , 'jointAngles_jpnp'])


    def forward(self, target_keypoints, joint_angles_gt):       

        # J-PnP -> Kalman Filter -> main simulation update -> find new keypoint
        
        target_keypoints = target_keypoints*[self.opt.width, self.opt.height] # expand keypoint range to full width, height
        # Lets run the simulation for joint alignment. 
        self.jointAngles_jpnp = self.jointAngles_main
        target_keypoints = np.array([target_keypoints[i][j] for i in range(len(target_keypoints)) for j in range(2)])  # [14]
        
        # Joint PnP
        # self.jointAngles_jpnp, angle_error, retry, iter = self.joint_pnp_with_LM(target_keypoints, self.jointAngles_jpnp)
        self.jointAngles_jpnp, angle_error, retry, iter = self.joint_pnp(target_keypoints, self.X[:self.numJoints].reshape(-1))
        print('angle error: ', angle_error, retry, iter)
        # Kalman filter        
        self.jointAngles_main = self.jointAngles_jpnp # valid measurement value with less than criteria    
        if angle_error < 5.:
            self.R = np.eye(self.numJoints)*0.0001
        else:
            self.R = np.eye(self.numJoints)*1 # invalid measurement

        # Update Kalman filter for self.jointAngles_main
        K = self.P @ np.transpose(self.H) @ np.linalg.inv(self.H @ self.P @ np.transpose(self.H) + self.R)
        
        self.X = self.X + K @ (self.jointAngles_main.reshape(-1,1) - self.H @ self.X)
        
        self.P = (np.eye(2*self.numJoints) - K @ self.H) @ self.P
        # filtered joint angle
        self.jointAngles_main = self.X[:self.numJoints].reshape(-1)
        self.jointVelocities_main = self.X[self.numJoints:].reshape(-1)
        self.X = self.A @ self.X
        self.P = self.A @ self.P @ np.transpose(self.A) + self.Q
        
        jointAngle_command = self.jointAngles_main
        
        # PyBullet main simulation update
        steps_per_frame = math.ceil( 1.0 / (self.seconds_per_step * self.frames_per_second) ) # 8 steps per frame
        for _ in range(steps_per_frame):
            for j in range(self.numJoints):
                p.setJointMotorControl2(bodyIndex=self.robotId_main,
                                        jointIndex=j,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.jointAngles_main[j],
                                        # targetVelocity=self.jointVelocities_main[j],
                                        # targetPosition=joint_angles_gt[j],
                                        targetVelocity=0,
                                        force=1000,
                                        positionGain=0.1,
                                        velocityGain=0.1,
                                        physicsClientId=self.physicsClient_main
                                        )
            p.stepSimulation(physicsClientId=self.physicsClient_main)

        # joint angles from main simulator
        jointStates = p.getJointStates(self.robotId_main, range(self.numJoints), physicsClientId=self.physicsClient_main)
        self.jointAngles_main = np.array([jointStates[i][0] for i in range(len(jointStates))])

        keypoints = self.get_joint_keypoints_from_angles(self.jointAngles_main, self.robotId_main, self.physicsClient_main, self.opt, camera_name = 'camera')
        keypoints = keypoints.reshape(-1,2)[:, ::-1] # [7, 2] (h, w)
        keypoints /= [self.opt.height, self.opt.width]

        self.save_debug_file([iter, angle_error, joint_angles_gt, jointAngle_command, self.jointAngles_main, self.jointAngles_jpnp])
        self.iters += 1

        return keypoints

    def save_result_image(self, jointAngles_jpnp, target_keypoints):        
        # reset robot joint with new angles
        for j in range(self.numJoints):
            p.resetJointState(bodyUniqueId=self.robotId_jpnp,
                            jointIndex=j,
                            targetValue=(jointAngles_jpnp[j]),
                            physicsClientId=self.physicsClient_jpnp
                            )    
        p.stepSimulation(physicsClientId=self.physicsClient_jpnp)
        link_world_state = self.get_joint_world_position(self.robotId_jpnp, self.physicsClient_jpnp)   # [num_link, position, rotation]
        for link_num in range(len(link_world_state)):       
            # get the nvisii entity for that object
            obj_entity = self.link_entities[link_num]
            obj_entity.get_transform().set_position(link_world_state[link_num][0])
            obj_entity.get_transform().set_rotation(link_world_state[link_num][1])
        
        
        # print(f'iteration: {str(iter).zfill(5)}/{str(iterations).zfill(5)}')



        # get_my_keypoints(camera_entity=camera, robotId=robotId_main, joint_world_position=joint_world_position, self.opt=self.opt)
        
        if self.make_joint_sphere:
            self.create_joint_markers(self.joint_entities, link_world_state[1:][0])

        nvisii.render_to_file(
            width=int(self.opt.width), 
            height=int(self.opt.height), 
            samples_per_pixel=int(self.opt.spp),
            file_path=f"{self.opt.outf}/{str(iter).zfill(5)}.png"
        )
        iter_image_path = f"{self.opt.inputf}/{str(iter).zfill(5)}.png"        
        print('iter_image_path', iter_image_path)
        target_image_path = f"{self.opt.inputf}/{str(self.opt.idx).zfill(5)}.png"
        keypoints = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
        self.save_keypoint_visualize_image(iter_image_path, target_image_path, keypoints.reshape(7,2), target_keypoints.reshape(7,2), iter)

    def joint_pnp(self, target_keypoints, jointAngles_jpnp):
        jointAngles_init = jointAngles_jpnp.copy()        

        for retry in range(1):
            eps = np.linspace(1e-6, 1e-6, self.numJoints)
            if not retry == 0:
                angle_noise = np.random.randn(self.numJoints)*0.01
                # jointAngles_jpnp = jointAngles_init + angle_noise # reset jointAngles_jpnp with noise
                jointAngles_jpnp = jointAngles_jpnp + angle_noise # reset jointAngles_jpnp with noise

            for iter in range(self.jpnp_max_iterations):
                # get joint 2d keypoint from 3d points and camera model
                keypoints = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
                            
                # Jacobian approximation: keypoint rate (변화량)
                Jacobian = np.zeros((self.numJoints*2, self.numJoints))
                for col in range(self.numJoints):
                    eps_array = np.zeros(self.numJoints)
                    eps_array[col] = eps[col]
                    keypoints_eps = self.get_joint_keypoints_from_angles(jointAngles_jpnp+eps_array, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
                    Jacobian[:,col] = (keypoints_eps - keypoints)/np.repeat(eps, 2, axis=0)
                
                dy = np.array(target_keypoints - keypoints)
                dx = np.linalg.pinv(Jacobian)@dy
                jointAngles_jpnp += dx # all joint angle update
                jointAngles_jpnp = self.angle_wrapper(jointAngles_jpnp)
                # criteria = np.abs( np.linalg.norm(dx) / np.linalg.norm(self.jointAngles_jpnp) )
                criteria = np.linalg.norm(dx)
                if criteria < 1e-1:
                    eps *= 0.9
                if criteria < 1e-3:
                    break
            
            angle_error = np.linalg.norm(jointAngles_init*180/np.pi - jointAngles_jpnp*180/np.pi)/self.numJoints
            
            if angle_error < 1.0:
                break
            
        if self.save_images:
            self.save_result_image(jointAngles_jpnp, target_keypoints)
        
        return jointAngles_jpnp, angle_error, retry, iter


    def joint_pnp_with_LM(self, target_keypoints, jointAngles_jpnp):

        jointAngles_init = jointAngles_jpnp.copy()
        eps = np.linspace(1e-6, 1e-6, self.numJoints)
        
        for retry in range(10):
            lam = 0.08
            if not retry == 0:
                angle_noise = np.random.randn(self.numJoints)*0.01
                # jointAngles_jpnp = jointAngles_init + angle_noise # reset jointAngles_jpnp with noise
                jointAngles_jpnp = jointAngles_jpnp + angle_noise # reset jointAngles_jpnp with noise                
            for iter in range(self.jpnp_max_iterations):
                # get joint 2d keypoint from 3d points and camera model
                keypoints = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
                            
                # Jacobian approximation: keypoint rate (변화량)
                Jacobian = np.zeros((self.numJoints*2, self.numJoints))
                for col in range(self.numJoints):
                    eps_array = np.zeros(self.numJoints)
                    eps_array[col] = eps[col]
                    keypoints_eps = self.get_joint_keypoints_from_angles(jointAngles_jpnp+eps_array, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
                    Jacobian[:,col] = (keypoints_eps - keypoints)/np.repeat(eps, 2, axis=0)
                # dy = np.array(target_keypoints - keypoints)
                # dx = np.linalg.pinv(Jacobian)@dy
                # self.jointAngles_jpnp += dx # all joint angle update
                # if iter == 0:
                #     lam = np.mean(np.diag(np.transpose(Jacobian)@Jacobian))*1e-3         # LM Algorithm
                dy = np.array(keypoints - target_keypoints).reshape(-1,1)
                dx = - np.linalg.inv(np.transpose(Jacobian)@Jacobian + lam*np.eye(self.numJoints)) @ np.transpose(Jacobian) @ dy # LM Algorithm
                dx = dx.reshape(-1)
                jointAngles_new = jointAngles_jpnp + dx # all joint angle update                
                keypoints_new = self.get_joint_keypoints_from_angles(jointAngles_new, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
                
                if np.linalg.norm(target_keypoints - keypoints_new) < np.linalg.norm(target_keypoints - keypoints): # accepted
                    jointAngles_jpnp += dx
                    jointAngles_jpnp = self.angle_wrapper(jointAngles_jpnp)
                    lam /= 10
                else:
                    lam *= 10
                    # continue
                criteria = np.abs( np.linalg.norm(dx) / np.linalg.norm(self.jointAngles_jpnp) )
                
                # if criteria < 1e-1:
                #     eps *= 0.9
                if criteria < 1e-2:
                    break

            angle_error = ((jointAngles_init*180/np.pi - jointAngles_jpnp*180/np.pi)**2).mean()
            if angle_error < 1.:             
                break
                
                

        if self.save_images:
            self.save_result_image(jointAngles_jpnp, target_keypoints)
        
        return jointAngles_jpnp, angle_error, retry, iter

    def angle_wrapper(self, angles):
        # wrapping angles to have vales between [-pi, pi)
        return np.arctan2(np.sin(angles), np.cos(angles))

    def save_debug_file(self, data):
        iter, angle_error, joint_angles_gt, jointAngle_command, jointAngles_main, jointAngles_jpnp = data
        with open('./utils/dt_debug.csv', 'a') as  f: 
            writer = csv.writer(f, delimiter=',')
            message = np.hstack((iter, 
                                angle_error, 
                                np.array(joint_angles_gt).reshape(-1), 
                                np.array(jointAngle_command).reshape(-1), 
                                np.array(jointAngles_main).reshape(-1), 
                                np.array(jointAngles_jpnp).reshape(-1),
                                ))
            writer.writerow(message)


    def save_keypoint_visualize_image(self, iter_image_path, target_image_path, iter_keypoints, target_keypoints, iter):
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
        

    def create_joint_markers(self, joint_entiies, jointPositions):    
        for i_p, position in enumerate(jointPositions):
            joint_entiies[i_p].get_transform().set_position(position)
            joint_entiies[i_p].get_transform().set_scale((0.1, 0.1, 0.1))
            joint_entiies[i_p].get_material().set_base_color((0.1,0.9,0.08))  
            joint_entiies[i_p].get_material().set_roughness(0.7) 

    def get_joint_keypoints(self, jointPositions, opt, camera_name = 'camera'):
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


    def get_my_keypoints(self, camera_entity, robotId, joint_world_position, opt):
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

    def get_joint_world_position(self, bodyUniqueId, physicsClientId):
        # Lets update the pose of the objects in nvisii 
        link_world_state = []
        for link_num in range(p.getNumJoints(bodyUniqueId, physicsClientId=physicsClientId)+1):
            # get the pose of the objects
            if link_num==0: # base
                link_state = p.getBasePositionAndOrientation(bodyUniqueId=bodyUniqueId, physicsClientId=physicsClientId)      
                pos_world = link_state[0]
                rot_world = link_state[1]
            else: # link
                link_state = p.getLinkState(bodyUniqueId=bodyUniqueId, linkIndex=link_num-1, physicsClientId=physicsClientId)
                pos_world = self.add_tuple(link_state[4], (0, 0, 0.1)) # world position of the URDF link frame
                rot_world = link_state[5] # world orientation of the URDF link frame
            link_world_state.append([pos_world, rot_world])   # (link position is identical to joint position) [8, 2]
        return np.array(link_world_state)

    def get_joint_keypoints_from_angles(self, jointAngles, bodyUniqueId, physicsClientId, opt, camera_name = 'camera'):
        cam_matrix = nvisii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
        cam_proj_matrix = nvisii.entity.get(camera_name).get_camera().get_projection()
        
        for j in range(self.numJoints):
            p.resetJointState(bodyUniqueId=bodyUniqueId,
                            jointIndex=j,
                            targetValue=(jointAngles[j]),
                            physicsClientId=physicsClientId
                            )
        p.stepSimulation(physicsClientId=physicsClientId)

        link_world_state = self.get_joint_world_position(bodyUniqueId, physicsClientId)   # [num_link, position, rotation]
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
            # keypoints.append([p_image[0], p_image[1]])
            points_cam.append([p_cam[0],p_cam[1],p_cam[2]])            
        
        keypoints = np.array([keypoints[i][j] for i in range(len(keypoints)) for j in range(2)])  # [14]
        
        return keypoints


    def clamping(self, min, val, max):
    # returns clamped value between min and max
        return sorted([min, val, max])[1]

    def rot_shift(self, rot):
        # shift quaternion array from [x,y,z,w] (PyBullet) to [w,x,y,z] (NVISII)
        return [rot[-1], rot[0], rot[1], rot[2]]

    def add_tuple(self, a, b):
        assert len(a)==len(b), "Size of two tuples are not matched!"
        return tuple([sum(x) for x in zip(a, b)])



    
    def __del__(self):
        p.disconnect(physicsClientId=self.physicsClient_main)
        p.disconnect(physicsClientId=self.physicsClient_jpnp)
        nvisii.deinitialize()
        # framerate = str(iter/10)
        # subprocess.call(['ffmpeg', '-y', '-framerate', '2', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 'output.mp4'], cwd=os.path.realpath(self.opt.outf))
