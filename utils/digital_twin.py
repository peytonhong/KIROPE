import os
from pickle import HIGHEST_PROTOCOL
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
from scipy.spatial.transform import Rotation as R

class DigitalTwin():

    def __init__(self, urdf_path=None, 
                    dataset=None,
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
        # for OpenCV based camera parameters (used to calculate keypoints)
        self.cam_K = dataset.cam_K
        self.cam_R_1 = dataset.cam_R_1
        self.cam_R_2 = dataset.cam_R_2

        # for OpenGL based camera parameters (used to capture image by p.getCameraImage)
        camera_struct_look_at = dataset.camera_struct_look_at_1
        fov = dataset.fov_1
        self.cam_intrinsic = p.computeProjectionMatrixFOV(fov=fov, # [view angle in degree]
                                            aspect=self.opt.width/self.opt.height,
                                            nearVal=0.1,
                                            farVal=100,
                                            )
        self.cam_extrinsic_1 = p.computeViewMatrix(cameraEyePosition=camera_struct_look_at['eye'],
                                        cameraTargetPosition=camera_struct_look_at['at'],
                                        cameraUpVector=camera_struct_look_at['up'],
                                        )

        # Setup bullet physics stuff
        self.seconds_per_step = 1.0 / 240.0
        self.frames_per_second = 30.0

        # physicsClient = p.connect(p.GUI) # graphical version
        self.physicsClient_main = p.connect(p.DIRECT) # non-graphical version
        self.physicsClient_jpnp = p.connect(p.DIRECT) # non-graphical version

        # lets create a robot
        self.robotId_main = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.physicsClient_main)
        self.robotId_jpnp = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.physicsClient_jpnp)
        p.resetBasePositionAndOrientation(self.robotId_main, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId=self.physicsClient_main)
        p.resetBasePositionAndOrientation(self.robotId_jpnp, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId=self.physicsClient_jpnp)

        self.numJoints = p.getNumJoints(self.robotId_main, physicsClientId=self.physicsClient_main)

        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient_main)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient_main)

        jointInfo = p.getJointInfo(self.robotId_main, 0, physicsClientId=self.physicsClient_main)
        lower_limit = [p.getJointInfo(self.robotId_main, i, physicsClientId=self.physicsClient_main)[8] for i in range(self.numJoints)]
        upper_limit = [p.getJointInfo(self.robotId_main, i, physicsClientId=self.physicsClient_main)[9] for i in range(self.numJoints)]

        self.jointAngles_main = np.zeros(self.numJoints) # main robot
        self.jointAngles_jpnp = np.zeros(self.numJoints) # Joint PnP robot
        self.jointVelocities_main = np.zeros(self.numJoints)

        # Kalman filter variables
        nj = self.numJoints
        self.dT = 1/self.frames_per_second
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

        with open('./utils/dt_debug.csv', 'w') as  f: 
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['iter', 'angle_error', 'joint_angles_gt', 'jointAngle_command', 'self.jointAngles_main' , 'jointAngles_jpnp'])


    def forward(self, target_keypoints_1, target_keypoints_2, joint_angles_gt):       

        # J-PnP -> Kalman Filter -> main simulation update -> find new keypoint
        
        target_keypoints_1 = target_keypoints_1*[self.opt.width, self.opt.height] # expand keypoint range to full width, height
        target_keypoints_2 = target_keypoints_2*[self.opt.width, self.opt.height] # expand keypoint range to full width, height
        # Lets run the simulation for joint alignment. 
        self.jointAngles_jpnp = self.jointAngles_main
        target_keypoints_1 = [target_keypoints_1[i][j] for i in range(len(target_keypoints_1)) for j in range(2)]  # [12]
        target_keypoints_2 = [target_keypoints_2[i][j] for i in range(len(target_keypoints_2)) for j in range(2)]  # [12]
        target_keypoints = np.array(target_keypoints_1 + target_keypoints_2)

        # Joint PnP
        # self.jointAngles_jpnp, angle_error, iter = self.joint_pnp(target_keypoints, self.X[:self.numJoints].reshape(-1), self.cam_K, self.cam_R_1, self.cam_R_2)
        self.jointAngles_jpnp, angle_error, iter = self.joint_pnp(target_keypoints, np.zeros(self.numJoints), self.cam_K, self.cam_R_1, self.cam_R_2)
        
        # Kalman filter        
        self.jointAngles_main = self.jointAngles_jpnp # valid measurement value with less than criteria    
        # if angle_error < 5.:
        #     self.R = np.eye(self.numJoints)*0.0001
        # else:
        #     self.R = np.eye(self.numJoints)*1 # invalid measurement
        # self.R = np.eye(self.numJoints)*100000
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
                                        targetPosition=self.jointAngles_main[j],             # 여기 main으로 바꿔놔야 함 (KF작동조건은main)
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

        # keypoints = self.get_joint_keypoints_from_angles(self.jointAngles_main, self.robotId_main, self.physicsClient_main, self.opt, self.cam_K, self.cam_R_1)
        keypoints = self.get_joint_keypoints_from_angles(self.jointAngles_jpnp, self.robotId_main, self.physicsClient_main, self.opt, self.cam_K, self.cam_R_1)
        keypoints = keypoints.reshape(-1,2)[:, ::-1] # [6, 2] (h, w)
        keypoints /= [self.opt.height, self.opt.width]

        self.save_debug_file([iter, angle_error, joint_angles_gt, jointAngle_command, self.jointAngles_main, self.jointAngles_jpnp])

        return keypoints

    def joint_pnp(self, target_keypoints, jointAngles_jpnp, cam_K, cam_R_1, cam_R_2):
        jointAngles_init = jointAngles_jpnp.copy()
        
        eps = np.linspace(1e-6, 1e-6, self.numJoints)
        iterations = 100 # This value can be adjusted.
        for iter in range(iterations): #self.jpnp_max_iterations
            # get joint 2d keypoint from 3d points and camera model
            keypoints_1 = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, cam_K, cam_R_1)
            keypoints_2 = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, cam_K, cam_R_2)
            keypoints = np.vstack((keypoints_1, keypoints_2)).reshape(-1) # [24]
            # Jacobian approximation: keypoint rate (변화량)
            Jacobian = np.zeros((self.numJoints*2*2, self.numJoints)) # [24, 6]
            for col in range(self.numJoints):
                eps_array = np.zeros(self.numJoints)
                eps_array[col] = eps[col]
                keypoints_eps_1 = self.get_joint_keypoints_from_angles(jointAngles_jpnp+eps_array, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, cam_K, cam_R_1)
                keypoints_eps_2 = self.get_joint_keypoints_from_angles(jointAngles_jpnp+eps_array, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, cam_K, cam_R_2)
                keypoints_eps = np.vstack((keypoints_eps_1, keypoints_eps_2)).reshape(-1) # [24]
                Jacobian[:,col] = (keypoints_eps - keypoints)/np.repeat(eps, 4, axis=0)
            
            dy = np.array(target_keypoints - keypoints)
            dx = np.linalg.pinv(Jacobian)@dy
            jointAngles_jpnp += dx # all joint angle update

            jointAngles_jpnp = self.angle_wrapper(jointAngles_jpnp)
            # criteria = np.abs( np.linalg.norm(dx) / np.linalg.norm(self.jointAngles_jpnp) )
            criteria = np.linalg.norm(dy)               
            # if criteria < 1.:
            #     eps *= 0.99
            if criteria < 1e-3:
                break
        # angle_error = np.linalg.norm(jointAngles_init[:-1]*180/np.pi - jointAngles_jpnp[:-1]*180/np.pi)/self.numJoints
        angle_error = np.linalg.norm(np.sin(jointAngles_init[:-1]) - np.sin(jointAngles_jpnp[:-1]))/(self.numJoints-1)
        angle_error = np.arcsin(angle_error)*180/np.pi
        print('angle_error: ', angle_error)
        # print(jointAngles_jpnp*180/np.pi)
        return jointAngles_jpnp, angle_error, iter


    # def joint_pnp_with_LM(self, target_keypoints, jointAngles_jpnp):

    #     jointAngles_init = jointAngles_jpnp.copy()
    #     eps = np.linspace(1e-6, 1e-6, self.numJoints)
    #     for retry in range(1):
    #         lam = 0.08
    #         if not retry == 0:
    #             angle_noise = np.random.randn(self.numJoints)*0.01
    #             # jointAngles_jpnp = jointAngles_init + angle_noise # reset jointAngles_jpnp with noise
    #             jointAngles_jpnp = jointAngles_jpnp + angle_noise # reset jointAngles_jpnp with noise                
    #         for iter in range(self.jpnp_max_iterations):
    #             # get joint 2d keypoint from 3d points and camera model
    #             keypoints = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
                            
    #             # Jacobian approximation: keypoint rate (변화량)
    #             Jacobian = np.zeros((self.numJoints*2, self.numJoints))
    #             for col in range(self.numJoints):
    #                 eps_array = np.zeros(self.numJoints)
    #                 eps_array[col] = eps[col]
    #                 keypoints_eps = self.get_joint_keypoints_from_angles(jointAngles_jpnp+eps_array, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
    #                 Jacobian[:,col] = (keypoints_eps - keypoints)/np.repeat(eps, 2, axis=0)
    #             # dy = np.array(target_keypoints - keypoints)
    #             # dx = np.linalg.pinv(Jacobian)@dy
    #             # self.jointAngles_jpnp += dx # all joint angle update
    #             # if iter == 0:
    #             #     lam = np.mean(np.diag(np.transpose(Jacobian)@Jacobian))*1e-3         # LM Algorithm
    #             dy = np.array(keypoints - target_keypoints).reshape(-1,1)
    #             dx = - np.linalg.inv(np.transpose(Jacobian)@Jacobian + lam*np.eye(self.numJoints)) @ np.transpose(Jacobian) @ dy # LM Algorithm
    #             dx = dx.reshape(-1)
    #             jointAngles_new = jointAngles_jpnp + dx # all joint angle update                
    #             keypoints_new = self.get_joint_keypoints_from_angles(jointAngles_new, self.robotId_jpnp, self.physicsClient_jpnp, self.opt, camera_name = 'camera')
                
    #             if np.linalg.norm(target_keypoints - keypoints_new) < np.linalg.norm(target_keypoints - keypoints): # accepted
    #                 jointAngles_jpnp += dx
    #                 jointAngles_jpnp = self.angle_wrapper(jointAngles_jpnp)
    #                 lam /= 10
    #             else:
    #                 lam *= 10
    #                 # continue
    #             # criteria = np.abs( np.linalg.norm(dx) / np.linalg.norm(self.jointAngles_jpnp) )
    #             criteria = np.linalg.norm(dy)
    #             # if criteria < 1e-1:
    #             #     eps *= 0.9
    #             if criteria < 1e-3:
    #                 break
    #         angle_error = ((jointAngles_init*180/np.pi - jointAngles_jpnp*180/np.pi)**2).mean()
    #         if angle_error < 1.:             
    #             break
        
                

    #     if self.save_images:
    #         self.save_result_image(jointAngles_jpnp, target_keypoints)
        
    #     return jointAngles_jpnp, angle_error, retry, iter

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

    def get_my_keypoints(self, cam_K, cam_R, bodyUniqueId, physicsClientId, joint_world_position, opt):
        # get 2d keypoints from 3d positions using camera K, R matrix (2021.06.30, Hyosung Hong)    
        numJoints = p.getNumJoints(bodyUniqueId=bodyUniqueId, physicsClientId=physicsClientId)
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

    def get_joint_keypoints_from_angles(self, jointAngles, bodyUniqueId, physicsClientId, opt, cam_K, cam_R):
        for j in range(len(jointAngles)):
            p.resetJointState(bodyUniqueId=bodyUniqueId,
                            jointIndex=j,
                            targetValue=(jointAngles[j]),
                            physicsClientId=physicsClientId,
                            )
        p.stepSimulation(physicsClientId=physicsClientId)

        # get joint states
        joint_world_position = []
        for link_num in range(len(jointAngles)):    
            link_state = p.getLinkState(bodyUniqueId=bodyUniqueId, linkIndex=link_num, physicsClientId=physicsClientId)
            pos_world = list(link_state[4])
            rot_world = link_state[5] # world orientation of the URDF link frame        
            if link_num == 4:
                rot_mat = R.from_quat(rot_world).as_matrix()
                offset = np.array([0,-0.04,0.08535])
                pos_world = rot_mat.dot(offset) + pos_world
            if link_num == 5:
                rot_mat = R.from_quat(rot_world).as_matrix()
                offset = np.array([0.0,0.0619,0])
                pos_world = rot_mat.dot(offset) + pos_world
            joint_world_position.append(pos_world)        
        keypoints = self.get_my_keypoints(cam_K, cam_R, bodyUniqueId=bodyUniqueId, physicsClientId=physicsClientId, joint_world_position=joint_world_position, opt=opt)
        return keypoints # [numJoints, 2]


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
        # framerate = str(iter/10)
        # subprocess.call(['ffmpeg', '-y', '-framerate', '2', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 'output.mp4'], cwd=os.path.realpath(self.opt.outf))
