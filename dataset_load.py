import numpy as np
import json
from glob import glob
import os
import cv2
import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as T
from utils.gaussian_position_encoding import gaussian_state_embedding, positional_encoding
from utils.util_functions import create_belief_map
class RobotDataset(Dataset):
    """ Loading Robot Dataset for Pose Estimation"""

    def __init__(self, data_dir='annotation/real/train'):#, embed_dim=256):
        self.image_paths_1 = sorted(glob(os.path.join(data_dir, '*/cam1/*.jpg')))
        self.label_paths_1 = sorted(glob(os.path.join(data_dir, '*/cam1/*.json')))
        self.image_paths_2 = sorted(glob(os.path.join(data_dir, '*/cam2/*.jpg')))
        self.label_paths_2 = sorted(glob(os.path.join(data_dir, '*/cam2/*.json')))
        # standard PyTorch mean-std input image normalization
        self.image_transform = T.Compose([
            # T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths_1)

    def __getitem__(self, idx):
        image_path_1 = self.image_paths_1[idx]
        image_path_2 = self.image_paths_2[idx]
        image_1 = Image.open(image_path_1).convert('RGB') # [w, h]
        image_2 = Image.open(image_path_2).convert('RGB') # [w, h]
        w, h = image_1.size        
        image_1 = self.image_transform(image_1) # after T.ToTensor() [3, h, w]
        image_2 = self.image_transform(image_2) # after T.ToTensor() [3, h, w]
        with open(self.label_paths_1[idx]) as json_file:
            label_1 = json.load(json_file)
        with open(self.label_paths_2[idx]) as json_file:
            label_2 = json.load(json_file)
        joint_angles = np.array(label_1['object']['joint_angles'])
        projected_keypoints_wh_1 = np.array(label_1['object']['joint_keypoints']) #[6, 2(w,h)]
        projected_keypoints_wh_2 = np.array(label_2['object']['joint_keypoints']) #[6, 2(w,h)]
        
        belief_maps_1 = torch.tensor(create_belief_map((h, w), projected_keypoints_wh_1, noise_std=0)).type(torch.FloatTensor) # [6, h,w]
        belief_maps_2 = torch.tensor(create_belief_map((h, w), projected_keypoints_wh_2, noise_std=0)).type(torch.FloatTensor) # [6, h,w]
        belief_maps_1_noise = torch.tensor(create_belief_map((h, w), projected_keypoints_wh_1, sigma=10, noise_std=2)).type(torch.FloatTensor) # [6, h,w]
        belief_maps_2_noise = torch.tensor(create_belief_map((h, w), projected_keypoints_wh_2, sigma=10, noise_std=2)).type(torch.FloatTensor) # [6, h,w]
        img_belief_1 = torch.cat((image_1, belief_maps_1_noise), dim=0) # [9, h, w]
        img_belief_2 = torch.cat((image_2, belief_maps_2_noise), dim=0) # [9, h, w]
        img_belief_stack = torch.cat((img_belief_1, img_belief_2), dim=0) # [18, h, w]

        # stacked_image_1 = torch.cat((image_1, belief_maps_1), dim=0) # [9, 800, 800]
        # stacked_image_2 = torch.cat((image_2, belief_maps_2), dim=0) # [9, 800, 800]
        cam_K_1  = np.array(label_1['camera']['camera_intrinsic'])
        cam_K_2  = np.array(label_2['camera']['camera_intrinsic'])
        cam_RT_1 = np.array(label_1['camera']['camera_extrinsic'])
        cam_RT_2 = np.array(label_2['camera']['camera_extrinsic']) 
        distortion_1 = np.array(label_1['camera']['camera_distortion'])
        distortion_2 = np.array(label_2['camera']['camera_distortion'])

        sample = {
            'image_1': image_1, 
            'image_2': image_2, 
            'joint_angles': joint_angles, 
            # 'joint_velocities': joint_velocities, 
            # 'joint_states': joint_states, 
            'belief_maps_1': belief_maps_1,  # [6, 480, 640]
            # 'belief_maps_noise_1': belief_maps_noise_1, # [6, 480, 640]
            'belief_maps_2': belief_maps_2,  # [6, 480, 640]
            # 'belief_maps_noise_2': belief_maps_noise_2, # [6, 480, 640]
            'keypoints_GT_1': projected_keypoints_wh_1,  # [6, 2(w,h)]
            'keypoints_GT_2': projected_keypoints_wh_2,  # [6, 2(w,h)]
            # 'state_embeddings': state_embeddings.flatten(1),
            'image_path_1': image_path_1,
            'image_path_2': image_path_2,
            # 'stacked_image_1': stacked_image_1,
            # 'stacked_image_2': stacked_image_2,
            # 'positional_encoding': pe,
            'image_beliefmap_stack': img_belief_stack,
            'cam_K_1': cam_K_1,
            'cam_K_2': cam_K_2,
            'cam_RT_1': cam_RT_1,
            'cam_RT_2': cam_RT_2,
            'distortion_1': distortion_1,
            'distortion_2': distortion_2,
            }
        return sample
