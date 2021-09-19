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

class DigitalTwin():

    def __init__(self, urdf_path=None):
        
        self.opt = lambda : None
        self.width = 640
        self.height = 480 
                    
        # camera parameters
        self.cam_K_1 = None
        self.cam_K_2 = None
        self.cam_RT_1 = None
        self.cam_RT_2 = None
        self.distortion_1 = None
        self.dotortion_2 = None
        self.fov = 43*np.pi/180 # Intel Realsense L515

        # Setup bullet physics stuff
        self.seconds_per_step = 1.0 / 240.0
        self.frames_per_second = 30.0

        # physicsClient = p.connect(p.GUI) # graphical version
        self.physicsClient_main = p.connect(p.DIRECT) # non-graphical version
        self.physicsClient_jpnp = p.connect(p.DIRECT) # non-graphical version

        # lets create a robot
        self.robotId_main = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.physicsClient_main)
        self.robotId_jpnp = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.physicsClient_jpnp)
        basePose = p.getQuaternionFromEuler([0,0,np.pi])
        p.resetBasePositionAndOrientation(self.robotId_main, [0.4, -0.15, 0.0], basePose, physicsClientId=self.physicsClient_main)
        p.resetBasePositionAndOrientation(self.robotId_jpnp, [0.4, -0.15, 0.0], basePose, physicsClientId=self.physicsClient_jpnp)

        self.numJoints = p.getNumJoints(self.robotId_main, physicsClientId=self.physicsClient_main)

        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient_main)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient_main)

        # jointInfo = p.getJointInfo(self.robotId_main, 0, physicsClientId=self.physicsClient_main)
        # lower_limit = [p.getJointInfo(self.robotId_main, i, physicsClientId=self.physicsClient_main)[8] for i in range(self.numJoints)]
        # upper_limit = [p.getJointInfo(self.robotId_main, i, physicsClientId=self.physicsClient_main)[9] for i in range(self.numJoints)]

        self.jointAngles_main = np.zeros(self.numJoints) # main robot
        self.jointAngles_jpnp = np.zeros(self.numJoints) # Joint PnP robot
        self.jointVelocities_main = np.zeros(self.numJoints)
        self.jointAngles_jpnp_old = np.zeros(self.numJoints)
        self.keypoints_1 = np.zeros((self.numJoints, 2))
        self.keypoints_2 = np.zeros((self.numJoints, 2))
        # Kalman filter variables
        nj = self.numJoints  # exclude end-effector
        # self.dT = 1/self.frames_per_second
        self.dT = 0.1
        self.A = np.vstack( (np.hstack((np.eye(nj), np.eye(nj)*self.dT)), 
                            np.hstack((np.zeros((nj,nj)), np.eye(nj))))) # [2*numJoints, 2*numJoints]
        self.P = np.eye(2*nj)
        Q_upper = np.eye(nj)#*0.1     # angle noise
        Q_lower = np.eye(nj)#*0.0001 # angular velocity noise
        self.Q = np.vstack( (np.hstack((Q_upper, np.zeros((nj,nj)))), 
                            np.hstack((np.zeros((nj,nj)), Q_lower)))) # [2*numJoints, 2*numJoints]
        self.R = np.eye(nj)
        
        self.X = np.zeros((2*nj, 1)) #[2*numJoints, 1]
        self.H = np.hstack((np.eye(nj), np.zeros((nj,nj)))) # [numJoints, 2*numJoints]

        self.jpnp_max_iterations = 30

        with open('./utils/dt_debug.csv', 'w', newline='') as  f: 
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['iter', 'angle_cos_error', 
                        'angle_gt_1','angle_gt_2','angle_gt_3','angle_gt_4','angle_gt_5', 'angle_gt_6', 
                        'angle_cmd_1', 'angle_cmd_2', 'angle_cmd_3', 'angle_cmd_4', 'angle_cmd_5', 'angle_cmd_6', 
                        'angle_main_1', 'angle_main_2', 'angle_main_3', 'angle_main_4', 'angle_main_5', 'angle_main_6', 
                        'angle_jpnp_1', 'angle_jpnp_2', 'angle_jpnp_3', 'angle_jpnp_4', 'angle_jpnp_5', 'angle_jpnp_6', 
                        ])


    def forward(self, target_keypoints_and_confidences_1, target_keypoints_and_confidences_2, sampled_batch):

        # J-PnP -> Kalman Filter -> main simulation update -> find new keypoint
        
        target_keypoints_1, confidences_1 = target_keypoints_and_confidences_1 # tuple
        target_keypoints_2, confidences_2 = target_keypoints_and_confidences_2 # tuple
        # target_keypoints_1 = target_keypoints_1*[self.width, self.height] # expand keypoint range to full width, height
        # target_keypoints_2 = target_keypoints_2*[self.width, self.height] # expand keypoint range to full width, height
        # Lets run the simulation for joint alignment. 
        # self.jointAngles_jpnp = self.jointAngles_main
        for i in range(self.numJoints):
            # if confidence score of a keypoint is below threshold, load previously estimated keypoint.
            if confidences_1[i] < 0.9:
                target_keypoints_1[i] = self.keypoints_1[i]
            if confidences_2[i] < 0.9: 
                target_keypoints_2[i] = self.keypoints_2[i]
        
        target_keypoints_1 = [target_keypoints_1[i][j] for i in range(len(target_keypoints_1)) for j in range(2)]  # flatten to [12,]
        target_keypoints_2 = [target_keypoints_2[i][j] for i in range(len(target_keypoints_2)) for j in range(2)]  # flatten to [12,]
        target_keypoints = np.array(target_keypoints_1 + target_keypoints_2) # [24,]

        # update camera parameters
        self.cam_K_1 = sampled_batch['cam_K_1'][0].numpy()
        self.cam_K_2 = sampled_batch['cam_K_2'][0].numpy()
        self.cam_RT_1 = sampled_batch['cam_RT_1'][0].numpy()
        self.cam_RT_2 = sampled_batch['cam_RT_2'][0].numpy()
        self.distortion_1 = sampled_batch['distortion_1'][0].numpy()
        self.distortion_2 = sampled_batch['distortion_2'][0].numpy()
        joint_angles_gt = sampled_batch['joint_angles'][0].numpy()

        # Joint PnP
        self.jointAngles_jpnp_old = self.jointAngles_jpnp
        self.jointAngles_jpnp, iter = self.joint_pnp(target_keypoints, self.X[:self.numJoints].reshape(-1))
        # self.jointAngles_jpnp, iter = self.joint_pnp(target_keypoints, np.zeros(self.numJoints))
        self.jointAngles_jpnp[-1] = 0 # set zero angle for end-effector since it is not observable.
        
        self.jointAngles_jpnp = self.regulate_angle(self.jointAngles_jpnp, self.jointAngles_jpnp_old)
        
        angle_cos_error = np.linalg.norm(np.cos(self.jointAngles_jpnp) - np.cos(self.jointAngles_jpnp_old))
        # Kalman filter                
        # self.jointAngles_main = self.jointAngles_jpnp # valid measurement value with less than criteria    
        
        K = self.P @ np.transpose(self.H) @ np.linalg.inv(self.H @ self.P @ np.transpose(self.H) + self.R)
        if angle_cos_error > 3*np.pi/180:
            K = np.zeros_like(K)
            self.X[self.numJoints:] = 0 # zero velocity
        self.X = self.X + K @ (self.jointAngles_jpnp.reshape(-1,1) - self.H @ self.X)
        self.P = (np.eye(2*self.numJoints) - K @ self.H) @ self.P
        # filtered joint angle
        self.X = self.A @ self.X
        self.P = self.A @ self.P @ np.transpose(self.A) + self.Q
        self.jointAngles_main = self.X[:self.numJoints].reshape(-1)
        self.jointVelocities_main = self.X[self.numJoints:].reshape(-1)
        # print(KF, self.jointAngles_jpnp[4]*180/np.pi, self.X[4]*180/np.pi)    
        
        jointAngle_command = self.jointAngles_main
        
        # PyBullet main simulation update
        # steps_per_frame = math.ceil( 1.0 / (self.seconds_per_step * self.frames_per_second) ) # 8 steps per frame
        steps_per_frame = 1
        for _ in range(steps_per_frame):
            for j in range(self.numJoints - 1):
                p.setJointMotorControl2(bodyIndex=self.robotId_main,
                                        jointIndex=j,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointAngle_command[j],             # 여기 main으로 바꿔놔야 함 (KF작동조건은main)
                                        targetVelocity=self.jointVelocities_main[j],
                                        # targetPosition=joint_angles_gt[j],
                                        # targetVelocity=0,
                                        force=500,
                                        positionGain=0.5,
                                        velocityGain=0.5,
                                        physicsClientId=self.physicsClient_main
                                        )
            p.stepSimulation(physicsClientId=self.physicsClient_main)

        # joint angles from main simulator
        jointStates = p.getJointStates(self.robotId_main, range(self.numJoints), physicsClientId=self.physicsClient_main)
        self.jointAngles_main = np.array([jointStates[i][0] for i in range(len(jointStates))])

        self.keypoints_1 = self.get_joint_keypoints_from_angles(self.jointAngles_main, self.robotId_main, self.physicsClient_main, self.cam_K_1, self.cam_RT_1, self.distortion_1)
        self.keypoints_2 = self.get_joint_keypoints_from_angles(self.jointAngles_main, self.robotId_main, self.physicsClient_main, self.cam_K_2, self.cam_RT_2, self.distortion_2)
        # keypoints = self.get_joint_keypoints_from_angles(self.jointAngles_jpnp, self.robotId_main, self.physicsClient_main, cam_K_1, cam_RT_1)        
        # keypoints /= [self.width, self.height] # normalize

        self.save_debug_file([iter, angle_cos_error, joint_angles_gt, jointAngle_command, self.jointAngles_main, self.jointAngles_jpnp])

        return self.keypoints_1, self.keypoints_2

    def joint_pnp(self, target_keypoints, jointAngles_jpnp):
        jointAngles_init = jointAngles_jpnp.copy()
        eps = 1e-6
        # eps = np.linspace(1e-6, 1e-6, self.numJoints)
        for iter in range(self.jpnp_max_iterations): #self.jpnp_max_iterations
            # get joint 2d keypoint from 3d points and camera model
            keypoints_1 = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.cam_K_1, self.cam_RT_1, self.distortion_1)
            keypoints_2 = self.get_joint_keypoints_from_angles(jointAngles_jpnp, self.robotId_jpnp, self.physicsClient_jpnp, self.cam_K_2, self.cam_RT_2, self.distortion_2)
            keypoints = np.vstack((keypoints_1, keypoints_2)).reshape(-1) # [24]
            # Jacobian approximation: keypoint rate (변화량)
            Jacobian = np.zeros((self.numJoints*2*2, self.numJoints)) # [24, 6]
            for col in range(self.numJoints):
                eps_array = np.zeros(self.numJoints)
                eps_array[col] = eps
                keypoints_eps_1 = self.get_joint_keypoints_from_angles(jointAngles_jpnp+eps_array, self.robotId_jpnp, self.physicsClient_jpnp, self.cam_K_1, self.cam_RT_1, self.distortion_1)
                keypoints_eps_2 = self.get_joint_keypoints_from_angles(jointAngles_jpnp+eps_array, self.robotId_jpnp, self.physicsClient_jpnp, self.cam_K_2, self.cam_RT_2, self.distortion_2)
                keypoints_eps = np.vstack((keypoints_eps_1, keypoints_eps_2)).reshape(-1) # [24]
                Jacobian[:,col] = (keypoints_eps - keypoints)/eps
            
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
        # print(jointAngles_jpnp*180/np.pi)
        return jointAngles_jpnp, iter


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
    #         angle_cos_error = ((jointAngles_init*180/np.pi - jointAngles_jpnp*180/np.pi)**2).mean()
    #         if angle_cos_error < 1.:             
    #             break
        
                

    #     if self.save_images:
    #         self.save_result_image(jointAngles_jpnp, target_keypoints)
        
    #     return jointAngles_jpnp, angle_cos_error, retry, iter

    def regulate_angle(self, angles_new, angles_old):
        # modify new angles to have near value of old angles
        # example: 179.41(old) -> -179.67(new) -> 180.32(modified)
        
        # print(angles_old[4]*180/np.pi, angles_new[4]*180/np.pi)
        angle_old_regulated = np.mod(angles_old + np.pi, 2*np.pi) - np.pi        
        for i in range(len(angles_new)):
            if (angles_new[i] - angle_old_regulated[i]) < -np.pi:
                diff = (angles_new[i] - angle_old_regulated[i]) + 2*np.pi
            elif (angles_new[i] - angle_old_regulated[i]) > np.pi:
                diff = (angles_new[i] - angle_old_regulated[i]) - 2*np.pi
            else:
                diff = (angles_new[i] - angle_old_regulated[i])            
            angles_new[i] = angles_old[i] + diff
        # print(angles_new[4]*180/np.pi)

        return angles_new

    def angle_wrapper(self, angles):
        # wrapping angles to have vales between [-pi, pi)
        return np.arctan2(np.sin(angles), np.cos(angles))

    def save_debug_file(self, data):
        iter, angle_cos_error, joint_angles_gt, jointAngle_command, jointAngles_main, jointAngles_jpnp = data
        with open('./utils/dt_debug.csv', 'a', newline='') as  f: 
            writer = csv.writer(f, delimiter=',')
            message = np.hstack((iter, 
                                angle_cos_error, 
                                np.array(joint_angles_gt).reshape(-1), 
                                np.array(jointAngle_command).reshape(-1), 
                                np.array(jointAngles_main).reshape(-1), 
                                np.array(jointAngles_jpnp).reshape(-1),
                                ))
            writer.writerow(message)

    def get_my_keypoints(self, cam_K, cam_RT, joint_world_position):
        # get 2d keypoints from 3d positions using camera K, R matrix (2021.06.30, Hyosung Hong)        
        jointPositions = np.zeros((self.numJoints, 3))
        jointKeypoints = np.zeros((self.numJoints, 2))
        for l in range(self.numJoints):
            jointPosition = np.array(list(joint_world_position[l])+[1.]).reshape(4,1)
            jointKeypoint = cam_K@cam_RT@jointPosition
            jointKeypoint /= jointKeypoint[-1]
            jointKeypoint[0] = self.width - jointKeypoint[0]   # OpenGL convention for left-right mirroring
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

    def get_joint_keypoints_from_angles(self, jointAngles, bodyUniqueId, physicsClientId, cam_K, cam_RT, distortion):
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
            if link_num == 0: # sholder
                rot_mat = p.getMatrixFromQuaternion(rot_world)
                rot_mat = np.array(rot_mat).reshape(3,3)
                offset = np.array([0,0,0])
                pos_world = rot_mat.dot(offset) + pos_world
            if link_num == 1: # upper_arm
                rot_mat = p.getMatrixFromQuaternion(rot_world)
                rot_mat = np.array(rot_mat).reshape(3,3)
                offset = np.array([0,0,0.1198])
                pos_world = rot_mat.dot(offset) + pos_world
            if link_num == 2: # fore_arm
                rot_mat = p.getMatrixFromQuaternion(rot_world)
                rot_mat = np.array(rot_mat).reshape(3,3)
                offset = np.array([0,0,0.025])
                pos_world = rot_mat.dot(offset) + pos_world
            if link_num == 3: # wrist 1
                rot_mat = p.getMatrixFromQuaternion(rot_world)
                rot_mat = np.array(rot_mat).reshape(3,3)
                offset = np.array([0,0,-0.085])
                pos_world = rot_mat.dot(offset) + pos_world
            if link_num == 4: # wrist 2
                rot_mat = p.getMatrixFromQuaternion(rot_world)
                rot_mat = np.array(rot_mat).reshape(3,3)
                offset = np.array([0,-0.045,0])
                pos_world = rot_mat.dot(offset) + pos_world
            if link_num == 5: # wrist 3
                rot_mat = p.getMatrixFromQuaternion(rot_world)
                rot_mat = np.array(rot_mat).reshape(3,3)
                offset = np.array([0,0,0])
                pos_world = rot_mat.dot(offset) + pos_world
            joint_world_position.append(pos_world)        
        joint_world_position = np.array(joint_world_position)
        # keypoints = self.get_my_keypoints(cam_K, cam_RT, joint_world_position=joint_world_position)
        
        rvecs = cv2.Rodrigues(cam_RT[:,:-1])[0]
        tvecs = cam_RT[:,-1]
        keypoints, jacobian = cv2.projectPoints(joint_world_position, rvecs, tvecs, cam_K, distortion)
        return keypoints.squeeze() # [numJoints, 2]


    def clamping(self, min, val, max):
    # returns clamped value between min and max
        return sorted([min, val, max])[1]

    def rot_shift(self, rot):
        # shift quaternion array from [x,y,z,w] (PyBullet) to [w,x,y,z] (NVISII)
        return [rot[-1], rot[0], rot[1], rot[2]]

    def add_tuple(self, a, b):
        assert len(a)==len(b), "Size of two tuples are not matched!"
        return tuple([sum(x) for x in zip(a, b)])

    def zero_joint_state(self):
        jointAngles = [0]*self.numJoints
        for j in range(len(jointAngles)):
            p.resetJointState(bodyUniqueId=self.robotId_main,
                            jointIndex=j,
                            targetValue=(jointAngles[j]),
                            physicsClientId=self.physicsClient_main,
                            )
        p.stepSimulation(physicsClientId=self.physicsClient_main)

    
    def __del__(self):
        p.disconnect(physicsClientId=self.physicsClient_main)
        p.disconnect(physicsClientId=self.physicsClient_jpnp)
        # framerate = str(iter/10)
        # subprocess.call(['ffmpeg', '-y', '-framerate', '2', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 'output.mp4'], cwd=os.path.realpath(self.opt.outf))
