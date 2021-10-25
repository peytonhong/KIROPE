# -*- coding: utf-8 -*-
"""kirope.py

Kinematics Guided Robot Pose Estimation with Monocular Camera. (KIROPE)

Written by Hyosung Hong
"""

from numpy.core.numeric import ones_like
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import cv2
import subprocess 
# import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_load import RobotDataset
from kirope_model import KIROPE_Attention, KIROPE_Transformer, ResnetSimple
from utils.digital_twin import DigitalTwin
from utils.util_functions import create_belief_map, extract_keypoints_from_belief_maps, save_belief_map_images
from utils.util_functions import visualize_result_two_cams, visualize_result_robot_human_two_cams
from utils.util_functions import get_pck_score, get_add_score, save_metric_json
from glob import glob
import time
import json

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
  parser.add_argument('--aug', default=True, type=str2bool, help="True: image and keypoint augmentation")
  parser.add_argument('--resume', default=False, type=str2bool, help="True: Load the trained model and resume training")  
  
  return parser.parse_args()



def train(args, model, dataset_iterator, device, optimizer, digital_twin):

    model.train()

    train_loss_sum = 0
    num_trained_data = 0    

    for iter, sampled_batch in enumerate(tqdm(dataset_iterator, desc=f"Training with batch size ({args.batch_size})")):
        image_1 = sampled_batch['image_1'] # tensor [N, 3, 480, 640]
        image_2 = sampled_batch['image_2'] # tensor [N, 3, 480, 640]
        gt_belief_maps_1 = sampled_batch['belief_maps_1'] # [N, 6, 480, 640]
        gt_belief_maps_2 = sampled_batch['belief_maps_2'] # [N, 6, 480, 640]
        keypoints_GT_1 = sampled_batch['keypoints_GT_1'] # [N, 6, 2]
        keypoints_GT_2 = sampled_batch['keypoints_GT_2'] # [N, 6, 2]
        keypoints_GT_1 = [[keypoints_GT_1[i][0].item(), keypoints_GT_1[i][1].item()] for i in range(len(keypoints_GT_1))]
        keypoints_GT_2 = [[keypoints_GT_2[i][0].item(), keypoints_GT_2[i][1].item()] for i in range(len(keypoints_GT_2))]
        
        if iter == 0:
            pred_belief_maps_1 = torch.zeros_like(gt_belief_maps_1) # [N, 9, 480, 640]
            pred_belief_maps_2 = torch.zeros_like(gt_belief_maps_2) # [N, 9, 480, 640]
        
        image_beliefmap_stack_1 = torch.cat((image_1, pred_belief_maps_1), dim=1) # [N, 9, 480, 640]
        image_beliefmap_stack_2 = torch.cat((image_2, pred_belief_maps_2), dim=1) # [N, 9, 480, 640]

        # stacking cam1, cam2 data to look like batch size == 2, but originally batch size should be 1.
        image = torch.vstack((image_beliefmap_stack_1, image_beliefmap_stack_2)) # [2N, 9, 480, 640]
        gt_belief_maps = torch.vstack((gt_belief_maps_1, gt_belief_maps_2)) # [2N, 6, 480, 640]

        image, gt_belief_maps = image.to(device), gt_belief_maps.to(device)
        optimizer.zero_grad()
        output = model(image) # ResNet model
        loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)
        loss.backward()
        train_loss_sum += loss.item()*args.batch_size*2
        num_trained_data += args.batch_size*2
        optimizer.step()
        
        if args.digital_twin:                
            pred_kps_1, pred_kps_2 = digital_twin.forward(
                                        extract_keypoints_from_belief_maps(output['pred_belief_maps'][0].cpu().detach().numpy()), 
                                        extract_keypoints_from_belief_maps(output['pred_belief_maps'][1].cpu().detach().numpy()), 
                                        sampled_batch
                                        )
            pred_belief_maps_1 = torch.tensor(create_belief_map(image.shape[2:], pred_kps_1)).type(torch.FloatTensor).unsqueeze(0)
            pred_belief_maps_2 = torch.tensor(create_belief_map(image.shape[2:], pred_kps_2)).type(torch.FloatTensor).unsqueeze(0)
        else:
            pred_belief_maps_1 = output['pred_belief_maps'][0].unsqueeze(0).detach()
            pred_belief_maps_2 = output['pred_belief_maps'][1].unsqueeze(0).detach()

        if iter%100 == 0:
            save_belief_map_images(output['pred_belief_maps'][0].cpu().detach().numpy(), 'train_cam1')
        
        
        
    train_loss_sum /= num_trained_data

    return train_loss_sum

def test(args, model, dataset, device, digital_twin):

    model.eval()

    test_loss_sum = 0
    num_tested_data = 0
    pck_scores = []
    add_scores = []
    pck_thresholds = np.linspace(0, 20, 50) # thresholds to evaluate [pixel]
    add_thresholds = np.linspace(0, 0.100, 50) # thresholds to evaluate [m]
    
    files = glob('visualization_result/*.jpg')
    for f in files:
        os.remove(f)
    DL_time_array = []
    DT_time_array = []
    # angle_save = []
    with torch.no_grad():
        for iter, sampled_batch in enumerate(tqdm(dataset, desc=f"Testing with batch size ({args.batch_size})")):
            time_begin = time.time()
            image_1 = sampled_batch['image_1'] # tensor [N, 3, 480, 640]
            image_2 = sampled_batch['image_2'] # tensor [N, 3, 480, 640]
            gt_belief_maps_1 = sampled_batch['belief_maps_1'] # [N, 6, 480, 640]
            gt_belief_maps_2 = sampled_batch['belief_maps_2'] # [N, 6, 480, 640]     
            keypoints_GT_1 = sampled_batch['keypoints_GT_1'] # [N, 6, 2]
            keypoints_GT_2 = sampled_batch['keypoints_GT_2'] # [N, 6, 2]
            keypoints_GT_1 = [[keypoints_GT_1[i][0].item(), keypoints_GT_1[i][1].item()] for i in range(len(keypoints_GT_1))]
            keypoints_GT_2 = [[keypoints_GT_2[i][0].item(), keypoints_GT_2[i][1].item()] for i in range(len(keypoints_GT_2))]
            cam_K_1  = np.array(sampled_batch['cam_K_1'])
            cam_RT_1 = np.array(sampled_batch['cam_RT_1'])
            cam_K_2  = np.array(sampled_batch['cam_K_2'])
            cam_RT_2 = np.array(sampled_batch['cam_RT_2'])

            if iter == 0:
                pred_belief_maps_1 = torch.zeros_like(gt_belief_maps_1) # [N, 9, 480, 640]
                pred_belief_maps_2 = torch.zeros_like(gt_belief_maps_2) # [N, 9, 480, 640]
        
            image_beliefmap_stack_1 = torch.cat((image_1, pred_belief_maps_1), dim=1) # [N, 9, 480, 640]
            image_beliefmap_stack_2 = torch.cat((image_2, pred_belief_maps_2), dim=1) # [N, 9, 480, 640]

            # stacking cam1, cam2 data to look like batch size == 2, but originally batch size should be 1.
            image = torch.vstack((image_beliefmap_stack_1, image_beliefmap_stack_2)) # [2N, 9, 480, 640]
            gt_belief_maps = torch.vstack((gt_belief_maps_1, gt_belief_maps_2))

            image, gt_belief_maps = image.to(device), gt_belief_maps.to(device)
            
            output = model(image)            
            loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)            
            test_loss_sum += loss.item()*args.batch_size*2
            num_tested_data += args.batch_size*2
            
            DL_time = time.time() - time_begin
            DL_time_array.append(DL_time)
            
            
            if args.digital_twin:                
                time_begin = time.time()
                pred_kps_1, pred_kps_2 = digital_twin.forward(
                                            extract_keypoints_from_belief_maps(output['pred_belief_maps'][0].cpu().detach().numpy()), 
                                            extract_keypoints_from_belief_maps(output['pred_belief_maps'][1].cpu().detach().numpy()), 
                                            sampled_batch
                                            )
                # pred_kps_1, pred_kps_2 = digital_twin.forward(
                #                             tuple(keypoints_GT_1, np.ones_like(keypoints_GT_1)),
                #                             tuple(keypoints_GT_2, np.ones_like(keypoints_GT_2)),
                #                             sampled_batch
                #                             )
                DT_time = time.time() - time_begin            
                DT_time_array.append(DT_time)
                
                pred_belief_maps_1 = torch.tensor(create_belief_map(image.shape[2:], pred_kps_1)).type(torch.FloatTensor).unsqueeze(0)
                pred_belief_maps_2 = torch.tensor(create_belief_map(image.shape[2:], pred_kps_2)).type(torch.FloatTensor).unsqueeze(0)
                visualize_result_robot_human_two_cams(
                                sampled_batch['image_path_1'][0], 
                                pred_kps_1, 
                                keypoints_GT_1,
                                sampled_batch['image_path_2'][0], 
                                pred_kps_2, 
                                keypoints_GT_2,
                                digital_twin,
                                cam_K_1[0], cam_RT_1[0], cam_K_2[0], cam_RT_2[0],
                                is_kp_normalized=False,
                                iter=iter
                                )
                add_score = get_add_score(digital_twin.jointWorldPosition_pred, digital_twin.jointWorldPosition_gt, add_thresholds)                
                add_scores.append(add_score)

                # angle_save.append((digital_twin.jointAngles_main*180/np.pi).tolist())
                
            else:
                pred_kps_1, _ = extract_keypoints_from_belief_maps(output['pred_belief_maps'][0].cpu().numpy())
                pred_kps_2, _ = extract_keypoints_from_belief_maps(output['pred_belief_maps'][1].cpu().numpy())
                visualize_result_robot_human_two_cams(
                                sampled_batch['image_path_1'][0],
                                pred_kps_1,
                                keypoints_GT_1,
                                sampled_batch['image_path_2'][0],
                                pred_kps_2,
                                keypoints_GT_2,
                                digital_twin,
                                cam_K_1[0], cam_RT_1[0], cam_K_2[0], cam_RT_2[0],
                                is_kp_normalized=False
                                )
            pck_score = get_pck_score(pred_kps_1, keypoints_GT_1, pck_thresholds)
            
            pck_scores.append(pck_score)            

            if iter == 488:
                save_belief_map_images(output['pred_belief_maps'][0].cpu().detach().numpy(), 'test_cam1')
                save_belief_map_images(output['pred_belief_maps'][1].cpu().detach().numpy(), 'test_cam2')
            # if iter == 280:
            #     save_belief_map_images(output['pred_belief_maps'][1].cpu().detach().numpy(), 'test_cam2')
            #     break

        # with open('visualization_result/metrics/angle_save_KF.json','w') as json_file:
        #     json.dump(angle_save, json_file)

        # visualize_state_embeddings(state_embeddings[0].cpu().numpy())
        
        test_loss_sum /= num_tested_data
        pck_scores = np.mean(pck_scores, axis=0)        
        save_metric_json(pck_thresholds, pck_scores, 'PCK')
        
        print(f'Deep Learning time: {np.mean(DL_time_array):.3f}s', )
        
        if args.digital_twin:
            add_scores = np.mean(add_scores, axis=0)
            save_metric_json(add_thresholds, add_scores, 'ADD')

            print(f'Digital Twin time: {np.mean(DT_time_array):.3f}s')
            DLDT_time = np.mean(DL_time+DT_time)
            print(f'Total time: {DLDT_time:.3f}s, {1/DLDT_time:.3f}FPS')
            
    return test_loss_sum


def main(args):
    """
    There are two main components:
    * a convolutional backbone - we use ResNet-50
    * a Transformer - we use the default PyTorch nn.TransformerEncoder, nn.TransformerDecoder
    """
    
    # hidden_dim = 256 # fixed for state embiddings
    lr = 1e-4           # learning rate
    model_path = './checkpoints/model_best.pth.tar'
    # model_path = "checkpoints/attention_model/attention_normal.tar"

    # model = KIROPE_Attention(num_joints=6, hidden_dim=hidden_dim)
    # model = KIROPE_Transformer(num_joints=7, hidden_dim=hidden_dim)
    if torch.cuda.is_available(): # for multi gpu compatibility
        device = 'cuda'
    else:
        device = 'cpu'

    if args.resume:
        model = torch.load(model_path)
    else:
        model = ResnetSimple(num_joints=6)
        if torch.cuda.is_available(): # for multi gpu compatibility
            device = 'cuda'        
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # print(model)

    # for param in model.module.backbone.parameters():
    #     param.requires_grad = False
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = RobotDataset(data_dir='annotation/real/validation')
    test_dataset = RobotDataset(data_dir='annotation/real/test')
    train_iterator = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    test_iterator = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    DT_train = DigitalTwin(urdf_path="urdfs/ur3/ur3_gazebo_no_limit.urdf")
    DT_test = DigitalTwin(urdf_path="urdfs/ur3/ur3_gazebo_no_limit.urdf")

    if args.command == 'train':
        # CHECKPOINT_DIR
        CHECKPOINT_DIR = 'checkpoints'
        try:
            os.mkdir(CHECKPOINT_DIR)
        except(FileExistsError):
            pass

        best_test_loss = float('inf')
        
        for e in range(args.num_epochs):
            train_loss = train(args, model, train_iterator, device, optimizer, DT_train)           
            test_loss = test(args, model, test_iterator, device, DT_test) # include visulaization result checking
            summary_note = f'Epoch: {e:3d}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}'
            print(summary_note)
            # subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', r"stacked_%04d.jpg",  f'videos/epoch_{e:04d}.gif'], cwd=os.path.realpath('visualization_result'))
            if best_test_loss > test_loss:
                best_test_loss = test_loss
                torch.save(model, model_path)
            DT_train.zero_joint_state()
            DT_test.zero_joint_state()

    else: # evaluate mode        
        model = torch.load(model_path)
        test_loss = test(args, model, test_iterator, device, DT_test)
        print(f'Test Loss: {test_loss:.10f}')




if __name__ == '__main__':
    
    # parse arguments
    args = argparse_args()
    if args is None:
        exit()
    print(args)
    
    main(args)