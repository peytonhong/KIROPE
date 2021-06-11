import re
from numpy.core.defchararray import find
import torch
from torch import nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F


# x = np.linspace(-10, 10, 100)
# a_true = 1.
# b_true = 1.
# y_true = a_true*x + b_true
# y_true += np.random.randn(len(x))

# param = torch.nn.parameter.Parameter
# a_est = param(torch.zeros(1))
# b_est = param(torch.zeros(1))

# x = torch.tensor(x, requires_grad=False)
# y_true = torch.tensor(y_true, requires_grad=False)
# optimizer = optim.Adam([a_est, b_est], lr=0.1)

# for _ in range(10):
#     optimizer.zero_grad()
#     y_est = a_est*x + b_est
#     loss = F.mse_loss(y_est, y_true)
#     loss.backward()
#     print(loss.item())
#     optimizer.step()
#     print(a_est.grad, b_est.grad)
    

def get_joint_world_position(DH_FK):
    positions = torch.empty(len(DH_FK), 3)
    for i in range(len(DH_FK)):
        positions[i] = DH_FK[i][:3, 3]
    return positions

def get_my_keypoints(cam_K, cam_R, joint_world_position):
    # get 2d keypoints from 3d positions using camera K, R matrix (2021.05.03, Hyosung Hong)
    
    numJoints = len(joint_world_position)
    
    jointPositions = torch.empty(numJoints, 4)
    jointKeypoints = torch.empty(numJoints, 3)
    for i in range(numJoints):
        jointPosition = torch.ones(4)
        jointPosition[:3] = joint_world_position[i]
        jointPosition = jointPosition.reshape(4,1)        
        jointKeypoint = cam_K@cam_R@jointPosition
        jointKeypoint = jointKeypoint / jointKeypoint[-1]
        jointPositions[i] = jointPosition.flatten()
        jointKeypoints[i] = jointKeypoint.flatten()
    # print('jointPositions: ', np.array(jointPositions))
    # print('jointKeypoints: ', np.array(jointKeypoints))

    return jointKeypoints

def get_DH_matrix(theta):
    DH_arrays = torch.empty(len(theta), 4, 4)
    DH_FK = torch.empty(len(theta), 4, 4) # forward kinematics
    DH_parameters = []
    DH_parameters.append([-np.pi/2, theta[0], 0, 0.36])
    DH_parameters.append([np.pi/2, theta[1], 0, 0])
    DH_parameters.append([np.pi/2, theta[2], 0, 0.42])
    DH_parameters.append([-np.pi/2, theta[3], 0, 0])
    DH_parameters.append([-np.pi/2, theta[4], 0, 0.4])
    DH_parameters.append([np.pi/2, theta[5], 0, 0])
    DH_parameters.append([0, theta[6], 0, 0.081])

    for i in range(len(theta)):
        alpha, theta, a, d = DH_parameters[i]
        alpha = torch.tensor(alpha).type(torch.FloatTensor)
        a = torch.tensor(a).type(torch.FloatTensor)
        d = torch.tensor(d).type(torch.FloatTensor)
        DH = torch.empty(4,4)
        DH[0] = torch.tensor([torch.cos(theta), -torch.cos(alpha)*torch.sin(theta), torch.sin(alpha)*torch.sin(theta), a*torch.cos(theta)], requires_grad=True)
        DH[1] = torch.tensor([torch.sin(theta), torch.cos(alpha)*torch.cos(theta), -torch.sin(alpha)*torch.cos(theta), a*torch.sin(theta)], requires_grad=True)
        DH[2] = torch.tensor([0., torch.sin(alpha), torch.cos(alpha), d])
        DH[3] = torch.tensor([0., 0., 0., 1.])
        
        DH_arrays[i] = DH
        if i==0:
            FK = DH
        else:
            FK = FK@DH
        DH_FK[i] = FK

    return DH_arrays, DH_FK

def find_joint_keypoint(theta, cam_K, cam_R):
    DH_arrays, DH_FK = get_DH_matrix(theta)
    joint_world_position = get_joint_world_position(DH_FK)
    joint_keypoints = get_my_keypoints(cam_K, cam_R, joint_world_position)
    return joint_keypoints

cam_K = torch.tensor([[552.38470459, 0, 250],
                    [0, 552.38470459, 250],
                    [0, 0, 1]]).type(torch.FloatTensor)
cam_R = torch.tensor([[0, 1, 0, 0],
                    [0, 0, 1, -1],
                    [1, 0, 0, -3]]).type(torch.FloatTensor)

theta = np.zeros(7)
theta_target = np.array([0.9463700376368033,
                0.7750844989066339,
                0.12738267609709314,
                0.600720637547934,
                0.7831442711582647,
                0.24207542142262517,
                -1.0625684014695445])

# DH_arrays, DH_FK = get_DH_matrix(theta)

# joint_world_position = get_joint_world_position(DH_FK)
# joint_keypoints = get_my_keypoints(cam_K, cam_R, joint_world_position)
param = torch.nn.parameter.Parameter
theta = param(torch.tensor(theta).type(torch.FloatTensor))
theta_target = torch.tensor(theta_target)
joint_keypoints = find_joint_keypoint(theta, cam_K, cam_R) # initial keypoint

joint_keypoints_target = find_joint_keypoint(theta_target, cam_K, cam_R)
joint_keypoints_target = joint_keypoints_target.clone().detach()
optimizer = optim.Adam([theta], lr=1)
for _ in range(10):
    optimizer.zero_grad()
    joint_keypoints = find_joint_keypoint(theta, cam_K, cam_R)
    loss = F.mse_loss(joint_keypoints, joint_keypoints_target)
    loss.backward()
    print(loss.item(), theta)
    optimizer.step()
print(loss.item())
print(joint_keypoints.data)
print(joint_keypoints_target.data)