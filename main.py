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
from kirope_model import KIROPE_Transformer, ResnetSimple

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
  parser.add_argument('--resume', default=False, type=str2bool, help="True: Load the trained model and resume training")  
  
  return parser.parse_args()



def train(args, model, dataset, device, optimizer):

    model.train()

    train_loss_sum = 0
    num_trained_data = 0
    weights = [1.0, 1.2, 1.5, 1.9, 2.4, 3.0, 3.7]
    for _, sampled_batch in enumerate(tqdm(dataset, desc=f"Training with batch size ({args.batch_size})")):
        image = sampled_batch['image'] # tensor [N, 3, 800, 800]
        keypoint_embeddings = sampled_batch['keypoint_embeddings'] # [N, 7, 256]
        gt_belief_maps = sampled_batch['belief_maps'] # [N, 7, 500, 500]
        # joint_angles = sampled_batch['joint_angles'] 
        # joint_velocities = sampled_batch['joint_velocities']
        # joint_states = sampled_batch['joint_states'] # [N, 7, 2]        
        # projected_keypoints = sampled_batch['projected_keypoints']        
        # image_path = sampled_batch['image_path']

        image, keypoint_embeddings, gt_belief_maps = image.to(device), keypoint_embeddings.to(device), gt_belief_maps.to(device)
        optimizer.zero_grad()
        output = model(image, keypoint_embeddings)
        # output = model(image)
        # loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)
        loss_0 = (F.mse_loss(output['pred_belief_maps'][:,0], gt_belief_maps[:,0]))*weights[0] # weighted loss by joints
        loss_1 = (F.mse_loss(output['pred_belief_maps'][:,1], gt_belief_maps[:,1]))*weights[1] # weighted loss by joints
        loss_2 = (F.mse_loss(output['pred_belief_maps'][:,2], gt_belief_maps[:,2]))*weights[2] # weighted loss by joints
        loss_3 = (F.mse_loss(output['pred_belief_maps'][:,3], gt_belief_maps[:,3]))*weights[3] # weighted loss by joints
        loss_4 = (F.mse_loss(output['pred_belief_maps'][:,4], gt_belief_maps[:,4]))*weights[4] # weighted loss by joints
        loss_5 = (F.mse_loss(output['pred_belief_maps'][:,5], gt_belief_maps[:,5]))*weights[5] # weighted loss by joints
        loss_6 = (F.mse_loss(output['pred_belief_maps'][:,6], gt_belief_maps[:,6]))*weights[6] # weighted loss by joints
        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
        loss.backward()
        train_loss_sum += loss.item()*len(sampled_batch)
        num_trained_data += len(sampled_batch)
        optimizer.step()
    
    train_loss_sum /= num_trained_data

    return train_loss_sum

def test(args, model, dataset, device):

    model.eval()

    test_loss_sum = 0
    num_tested_data = 0
    with torch.no_grad():
        for _, sampled_batch in enumerate(tqdm(dataset, desc=f"Testing with batch size ({args.batch_size})")):
            image = sampled_batch['image'] # tensor [N, 3, 800, 800]
            keypoint_embeddings = sampled_batch['keypoint_embeddings'] # [N, 7, 256]
            gt_belief_maps = sampled_batch['belief_maps'] # [N, 7, 500, 500]
            # joint_angles = sampled_batch['joint_angles'] 
            # joint_velocities = sampled_batch['joint_velocities']
            # joint_states = sampled_batch['joint_states'] # [N, 7, 2]        
            # projected_keypoints = sampled_batch['projected_keypoints']        
            image_path = sampled_batch['image_path']

            image, keypoint_embeddings, gt_belief_maps = image.to(device), keypoint_embeddings.to(device), gt_belief_maps.to(device)
            
            output = model(image, keypoint_embeddings)
            # output = model(image)
            
            loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)
            
            test_loss_sum += loss.item()*len(sampled_batch)
            num_tested_data += len(sampled_batch)
            
            visualize_result(image_path, output['pred_belief_maps'].cpu().numpy(), gt_belief_maps.cpu().numpy())
            break
        
        test_loss_sum /= num_tested_data

    return test_loss_sum


def extract_keypoints_from_belief_maps(belief_maps):    
    keypoints = []
    for i in range(len(belief_maps[0])):
        indices = np.where(belief_maps[0][i] == belief_maps[0][i].max())
        keypoints.append([indices[1][0], indices[0][0]]) # keypoint format: [w, h]
        # print(belief_maps[0][i].max())
        
    return keypoints

def save_belief_map_images(belief_maps, map_type):
    belief_maps = (belief_maps[0]*255).astype(np.uint8)    
    for i in range(len(belief_maps)):        
        image = cv2.cvtColor(belief_maps[i].copy(), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f'visualize_{map_type}_belief_maps_{i}.png', image)

def visualize_result(image_paths, pred_belief_maps, gt_belief_maps):
    # visualize the joint position prediction wih ground truth
    rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
    bgr_colors = rgb_colors[:, ::-1]
    image = cv2.imread(image_paths[0])
    pred_keypoints = extract_keypoints_from_belief_maps(pred_belief_maps)     
    gt_keypoints = extract_keypoints_from_belief_maps(gt_belief_maps)         
    save_belief_map_images(pred_belief_maps, 'pred')
    save_belief_map_images(gt_belief_maps, 'gt')
    image = image.copy()
    for i, (pred_keypoint, gt_keypoint) in enumerate(zip(pred_keypoints, gt_keypoints)):
        cv2.drawMarker(image, (int(pred_keypoint[0]), int(pred_keypoint[1])), color=bgr_colors[i].tolist(), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=1)
        cv2.circle(image, (int(gt_keypoint[0]), int(gt_keypoint[1])), radius=5, color=bgr_colors[i].tolist(), thickness=2)        
    cv2.imwrite('visualize_result.png', image)
    






def main(args):
    """
    There are two main components:
    * a convolutional backbone - we use ResNet-50
    * a Transformer - we use the default PyTorch nn.TransformerEncoder, nn.TransformerDecoder
    """
    
    hidden_dim = 256 # fixed for keypoint embiddings
    lr = 1.5e-4           # learning rate

    model = KIROPE_Transformer(num_joints=7, hidden_dim=hidden_dim)
    # model = ResnetSimple()

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

    robot_dataset = RobotDataset()
    train_iterator = DataLoader(dataset=robot_dataset, batch_size=args.batch_size, shuffle=True)

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
            summary_note = f'Epoch: {e:3d}, Train Loss: {train_loss:.10f}'
            print(summary_note)

            if best_test_loss > train_loss:
                best_test_loss = train_loss
                torch.save(model, './checkpoints/model_best.pth.tar')

            test_loss = test(args, model, train_iterator, device) # for visulaization result checking

    else: # evaluate mode
        model = torch.load('./checkpoints/model_best.pth.tar')
        test_loss = test(args, model, train_iterator, device)
        print(f'Test Loss: {test_loss:.10f}')




if __name__ == '__main__':
    
    # parse arguments
    args = argparse_args()
    if args is None:
        exit()
    print(args)
    
    main(args)