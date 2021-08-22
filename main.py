# -*- coding: utf-8 -*-
"""kirope.py

Kinematics Guided Robot Pose Estimation with Monocular Camera. (KIROPE)

Written by Hyosung Hong
"""

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import cv2
# import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_load import RobotDataset
from kirope_model import KIROPE_Attention, KIROPE_Transformer, ResnetSimple
from utils.digital_twin import DigitalTwin

def str2bool(v):
    # Converts True or False for argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


"""parsing and configuration"""
def argparse_args():  
  desc = "Pytorch implementation of 'Kinematics Guided Robot Pose Estimation with Monocular Camera (KIROPE)'"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('command', help="'train' or 'evaluate'")
  parser.add_argument('--num_epochs', default=100, type=int, help="The number of epochs to run")
  parser.add_argument('--batch_size', default=1, type=int, help="The number of batchs for each epoch")
  parser.add_argument('--digital_twin', default=False, type=str2bool, help="True: run digital twin simulation on testing")
  parser.add_argument('--resume', default=False, type=str2bool, help="True: Load the trained model and resume training")  
  
  return parser.parse_args()



def train(args, model, dataset, device, optimizer):

    model.train()

    train_loss_sum = 0
    num_trained_data = 0
    # weights = [1.0, 10.0, 40.0, 60.0, 80.0, 90.0, 100.0]
    # weights = [1.0, 2.0, 5.0, 7.0, 8.0, 9.0, 10.0]
    # weights = np.array(weights)/np.sum(weights) # normalize

    for _, sampled_batch in enumerate(tqdm(dataset, desc=f"Training with batch size ({args.batch_size})")):
        image_1 = sampled_batch['image_1'] # tensor [N, 3, 480, 640]
        image_2 = sampled_batch['image_2'] # tensor [N, 3, 480, 640]
        # state_embeddings = sampled_batch['state_embeddings'] # [N, 6, 100, 100]
        gt_belief_maps_1 = sampled_batch['belief_maps_1'] # [N, 6, 480, 640]
        gt_belief_maps_2 = sampled_batch['belief_maps_2'] # [N, 6, 480, 640]
        # pe = sampled_batch['positional_encoding'] # [N, 6, 256]
        # joint_angles = sampled_batch['joint_angles'] 
        # joint_velocities = sampled_batch['joint_velocities']
        # joint_states = sampled_batch['joint_states'] # [N, 6, 2]
        # keypoints_GT_1 = sampled_batch['keypoints_GT_1'] # [N, 6, 2]
        # keypoints_GT_2 = sampled_batch['keypoints_GT_2'] # [N, 6, 2]
        # image_path = sampled_batch['image_path']
        stacked_image_1 = sampled_batch['stacked_image_1'] # [N, 9, 480, 640]
        stacked_image_2 = sampled_batch['stacked_image_2'] # [N, 9, 480, 640]
        
        # stacking cam1, cam2 data to look like batch size == 2, but originally batch size should be 1.
        image = torch.vstack((stacked_image_1, stacked_image_2)) # [2N, 9, 480, 640]
        gt_belief_maps = torch.vstack((gt_belief_maps_1, gt_belief_maps_2)) # [N, 6, 480, 640]
        # pe = torch.vstack((pe, pe))
        # projected_keypoints = torch.vstack((keypoints_GT_1, keypoints_GT_2))
        image, gt_belief_maps = image.to(device), gt_belief_maps.to(device)
        # pe, projected_keypoints = pe.to(device), projected_keypoints.to(device)
        # projected_keypoints = projected_keypoints.to(device)
        # stacked_images, gt_belief_maps = stacked_images.to(device), gt_belief_maps.to(device)
        optimizer.zero_grad()
        # output = model(image, gt_belief_maps, pe) # Transformer style model
        output = model(image) # ResNet model
        loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)
        loss.backward()
        train_loss_sum += loss.item()*args.batch_size*2
        num_trained_data += args.batch_size*2
        optimizer.step()
        break
    
    train_loss_sum /= num_trained_data

    return train_loss_sum

def test(args, model, dataset, device, digital_twin):

    model.eval()

    test_loss_sum = 0
    num_tested_data = 0
    
    with torch.no_grad():
        for _, sampled_batch in enumerate(tqdm(dataset, desc=f"Testing with batch size ({args.batch_size})")):
            image_1 = sampled_batch['image_1'] # tensor [N, 3, 480, 640]
            image_2 = sampled_batch['image_2'] # tensor [N, 3, 480, 640]
            # state_embeddings = sampled_batch['state_embeddings'] # [N, 6, 100, 100]
            gt_belief_maps_1 = sampled_batch['belief_maps_1'] # [N, 6, 480, 640]
            gt_belief_maps_2 = sampled_batch['belief_maps_2'] # [N, 6, 480, 640]
            # pe = sampled_batch['positional_encoding'] # [N, 6, 256]
            joint_angles_gt = sampled_batch['joint_angles'] 
            # joint_velocities = sampled_batch['joint_velocities']
            # joint_states = sampled_batch['joint_states'] # [N, 6, 2]        
            keypoints_GT_1 = sampled_batch['keypoints_GT_1'] # [N, 6, 2]
            keypoints_GT_2 = sampled_batch['keypoints_GT_2'] # [N, 6, 2]
            image_path_1 = sampled_batch['image_path_1']
            image_path_2 = sampled_batch['image_path_2']
            stacked_image_1 = sampled_batch['stacked_image_1'] # [N, 9, 480, 640]
            stacked_image_2 = sampled_batch['stacked_image_2'] # [N, 9, 480, 640]
            cam_K_1 = sampled_batch['cam_K_1']
            cam_K_2 = sampled_batch['cam_K_2']
            cam_RT_1 = sampled_batch['cam_RT_1']
            cam_RT_2 = sampled_batch['cam_RT_2']
            distortion_1 = sampled_batch['distortion_1']
            distortion_2 = sampled_batch['distortion_2']

            # stacking cam1, cam2 data to look like batch size == 2, but originally batch size should be 1.
            image = torch.vstack((stacked_image_1, stacked_image_2)) # [2N, 9, 480, 640]
            gt_belief_maps = torch.vstack((gt_belief_maps_1, gt_belief_maps_2))
            # pe = torch.vstack((pe, pe))

            image, gt_belief_maps = image.to(device), gt_belief_maps.to(device)
            # pe = pe.to(device)
            # stacked_images, gt_belief_maps = stacked_images.to(device), gt_belief_maps.to(device)
            
            # output = model(image, gt_belief_maps, pe) # [2N, 6, 2(h,w)]
            output = model(image)
            
            loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)
            
            test_loss_sum += loss.item()*args.batch_size*2
            num_tested_data += args.batch_size*2
            
            if args.digital_twin:
                # joint_angles_gt = [joint_angles_gt[i][0].item() for i in range(len(joint_angles_gt))]
                # pred_keypoints = digital_twin.forward(extract_keypoints_from_belief_maps(output['pred_belief_maps'][0].cpu().numpy()), joint_angles_gt[0]) # w, h
                pred_keypoints = digital_twin.forward(keypoints_GT_1[0], 
                                                        keypoints_GT_2[0], 
                                                        joint_angles_gt[0],
                                                        cam_K_1, cam_K_2, cam_RT_1, cam_RT_2, distortion_1, distortion_2) # w, h GT value input for validation
                visualize_result(image_path_1[0], 
                                pred_keypoints, 
                                keypoints_GT_1[0].numpy(),
                                is_kp_normalized=False)
            else:
                print('keypoints_GT_1', keypoints_GT_1.shape)
                visualize_result(image_path_1[0], 
                                extract_keypoints_from_belief_maps(output['pred_belief_maps'][0].cpu().numpy()), 
                                keypoints_GT_1[0],
                                is_kp_normalized=False)
                        
        # visualize_state_embeddings(state_embeddings[0].cpu().numpy())
        
        test_loss_sum /= num_tested_data

    return test_loss_sum


def create_batch_belief_map(image_resolution, keypoints, sigma=10, noise_std=0):
    '''
    This function is referenced from NVIDIA Dream/datasets.py
    
    image_resolution: image size (height x width)
    keypoints: list of keypoints to draw in a 7x2 tensor
    sigma: the size of the point
    noise_std: stddev of keypoint pixel level noise to improve regularization performance.
    
    returns a tensor of n_points x h x w with the belief maps
    '''
    
    # Input argument handling
    assert (
        len(image_resolution) == 2
    ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
        len(image_resolution)
    )
    image_height, image_width = image_resolution
    out = np.zeros((len(keypoints), image_height, image_width))

    w = int(sigma * 2)

    for i_point, point in enumerate(keypoints):
        pixel_u = int(point[0] + np.random.randn()*noise_std) # width axis
        pixel_v = int(point[1] + np.random.randn()*noise_std) # height axis
        array = np.zeros((image_height, image_width))

        # TODO makes this dynamics so that 0,0 would generate a belief map.
        if (
            pixel_u - w >= 0
            and pixel_u + w < image_width
            and pixel_v - w >= 0
            and pixel_v + w < image_height
        ):
            for i in range(pixel_u - w, pixel_u + w + 1):
                for j in range(pixel_v - w, pixel_v + w + 1):
                    array[j, i] = np.exp(
                        -(
                            ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                            / (2 * (sigma ** 2))
                        )
                    )
        out[i_point] = array

    return out

def extract_keypoints_from_belief_maps(belief_maps):    
    print("belief_maps.shape", belief_maps.shape)
    keypoints = []
    for i in range(len(belief_maps)):
        indices = np.where(belief_maps[i] == belief_maps[i].max())
        keypoints.append([indices[1][0], indices[0][0]]) # keypoint format: [w, h]
        # print(belief_maps[0][i].max())
        
    return keypoints

def save_belief_map_images(belief_maps, map_type):
    # belief_maps: [7, h, w]
    belief_maps = (belief_maps*255).astype(np.uint8)    
    for i in range(len(belief_maps)):        
        image = cv2.cvtColor(belief_maps[i].copy(), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f'visualize_{map_type}_belief_maps_{i}.png', image)

def visualize_result(image_paths, pred_keypoints, gt_keypoints, is_kp_normalized):
    # visualize the joint position prediction wih ground truth for one sample
    # pred_kps, gt_kps: [numJoints, 2(w,h order)]
    rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
    bgr_colors = rgb_colors[:, ::-1]
    image = cv2.imread(image_paths)
    if is_kp_normalized:
        height, width, channel = image.shape
        pred_keypoints = [[int(u*width), int(v*height)] for u, v in pred_keypoints]
        gt_keypoints = [[int(u*width), int(v*height)] for u, v in gt_keypoints]
    # pred_keypoints = extract_keypoints_from_belief_maps(pred_kps)     
    # gt_keypoints = extract_keypoints_from_belief_maps(gt_belief_maps)
    # save_belief_map_images(pred_kps, 'pred')
    # save_belief_map_images(gt_belief_maps, 'gt')
    image = image.copy()
    for i, (pred_keypoint, gt_keypoint) in enumerate(zip(pred_keypoints, gt_keypoints)):
        print('gt_keypoint', gt_keypoint)
        cv2.drawMarker(image, (int(pred_keypoint[0]), int(pred_keypoint[1])), color=bgr_colors[i].tolist(), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=1)
        cv2.circle(image, (int(gt_keypoint[0]), int(gt_keypoint[1])), radius=5, color=bgr_colors[i].tolist(), thickness=2)        
    cv2.imwrite(f'visualization_result/{image_paths[-9:]}', image)
    
def visualize_state_embeddings(state_embeddings):
    for i in range(len(state_embeddings)):
        file_name = f'keypoint_embedding_{i}.png'
        embedding = (state_embeddings[i]*255).astype(np.uint8)
        image = cv2.cvtColor(embedding, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(file_name, image.copy())





def main(args):
    """
    There are two main components:
    * a convolutional backbone - we use ResNet-50
    * a Transformer - we use the default PyTorch nn.TransformerEncoder, nn.TransformerDecoder
    """
    
    hidden_dim = 256 # fixed for state embiddings
    lr = 1e-4           # learning rate
    model_path = './checkpoints/model_best.pth.tar'
    # model_path = "checkpoints/attention_model/attention_normal.tar"

    # model = KIROPE_Attention(num_joints=6, hidden_dim=hidden_dim)
    # model = KIROPE_Transformer(num_joints=7, hidden_dim=hidden_dim)
    model = ResnetSimple(num_joints=6)

    if torch.cuda.is_available(): # for multi gpu compatibility
        device = 'cuda'        
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device) 
    else:
        device = 'cpu'
        model = nn.DataParallel(model).to(device)
    # print(model)

    # for param in model.module.backbone.parameters():
    #     param.requires_grad = False
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = RobotDataset(data_dir='annotation/real/test')
    test_dataset = RobotDataset(data_dir='annotation/real/test')
    train_iterator = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    test_iterator = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    DT = DigitalTwin(urdf_path="urdfs/ur3/ur3_gazebo.urdf", 
                    dataset=test_dataset,
                    )

    if args.command == 'train':
        # CHECKPOINT_DIR
        CHECKPOINT_DIR = 'checkpoints'
        try:
            os.mkdir(CHECKPOINT_DIR)
        except(FileExistsError):
            pass

        best_test_loss = float('inf')
        
        for e in range(args.num_epochs):
            train_loss = train(args, model, train_iterator, device, optimizer)           
            test_loss = test(args, model, test_iterator, device, DT) # include visulaization result checking
            summary_note = f'Epoch: {e:3d}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}'
            print(summary_note)
            if best_test_loss > test_loss:
                best_test_loss = test_loss
                torch.save(model, model_path)

            

    else: # evaluate mode        
        model = torch.load(model_path)
        test_loss = test(args, model, test_iterator, device, DT)
        print(f'Test Loss: {test_loss:.10f}')




if __name__ == '__main__':
    
    # parse arguments
    args = argparse_args()
    if args is None:
        exit()
    print(args)
    
    main(args)