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
        # self.image_resize_800 = T.Resize(800)
        # self.image_resize_16 = T.Resize(16)
        # self.embed_dim = embed_dim
        # Camera parameters
        with open(self.label_paths_1[0]) as json_file:
            label_1 = json.load(json_file)
        with open(self.label_paths_2[0]) as json_file:
            label_2 = json.load(json_file)
        self.cam_K  = np.array(label_1['camera_data']['camera_intrinsics'])
        self.cam_R_1 = np.array(label_1['camera_data']['camera_extrinsics'])
        self.cam_R_2 = np.array(label_2['camera_data']['camera_extrinsics'])
        self.camera_struct_look_at_1 = {'at': label_1['camera_data']['camera_look_at']['at'],
                                'up':  label_1['camera_data']['camera_look_at']['up'],
                                'eye': label_1['camera_data']['camera_look_at']['eye'],
                                }
        self.camera_struct_look_at_2 = {'at': label_2['camera_data']['camera_look_at']['at'],
                                'up':  label_2['camera_data']['camera_look_at']['up'],
                                'eye': label_2['camera_data']['camera_look_at']['eye'],
                                }
        self.fov_1 = label_1['camera_data']['fov']
        self.fov_2 = label_2['camera_data']['fov']

    def __len__(self):
        return len(self.image_paths_1)

    def __getitem__(self, idx):
        # image = cv2.imread(self.image_paths[idx]) # [h, w, c(BGR)]
        # image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)
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
        joint_angles = label_1['objects']['joint_angles']
        # joint_velocities = label_1['objects']['joint_velocities']
        # joint_velocities[-1] = 0 # end-effector joint velocity set to zero (because it is too much fast)
        # joint_states = (np.stack([joint_angles, joint_velocities], axis=1))
        projected_keypoints_wh_1 = label_1['objects']['projected_keypoints'] #[6, 2(w,h)]
        projected_keypoints_wh_2 = label_2['objects']['projected_keypoints'] #[6, 2(w,h)]
        # numJoints = len(projected_keypoints_wh_1)

        belief_maps_1 = torch.tensor(self.create_belief_map((h, w), projected_keypoints_wh_1, noise_std=0)).type(torch.FloatTensor) # [6, h,w]        
        # belief_maps_1 = self.resize_belief_map(belief_maps_1, self.image_resize_16)
        # belief_maps_noise_1 = torch.tensor(self.create_belief_map((h, w), projected_keypoints_wh_1, noise_std=5)).type(torch.FloatTensor) # [6, h,w]        
        # belief_maps_noise_1 = self.resize_belief_map(belief_maps_noise_1, self.image_resize_16)

        belief_maps_2 = torch.tensor(self.create_belief_map((h, w), projected_keypoints_wh_2, noise_std=0)).type(torch.FloatTensor) # [6, h,w]        
        # belief_maps_2 = self.resize_belief_map(belief_maps_2, self.image_resize_16)
        # belief_maps_noise_2 = torch.tensor(self.create_belief_map((h, w), projected_keypoints_wh_2, noise_std=5)).type(torch.FloatTensor) # [6, h,w]        
        # belief_maps_noise_2 = self.resize_belief_map(belief_maps_noise_2, self.image_resize_16)
        
        # state_embeddings = torch.tensor(gaussian_state_embedding(joint_states, self.embed_dim)).type(torch.FloatTensor) # [6, num_features]
        # _, num_features = state_embeddings.shape
        # w_feature = np.sqrt(num_features).astype(np.uint8)
        # state_embeddings = state_embeddings.reshape(6, w_feature, w_feature)
        # image_1 = self.image_resize_800(image_1) # [3, 800, 800]
        # image_2 = self.image_resize_800(image_2) # [3, 800, 800]
        # stacked_images = torch.cat((image_1, self.image_resize_800(state_embeddings)), dim=0) # [10, 800, 800]        
        # pe = positional_encoding(256, numJoints) # [6, 256]
        # projected_keypoints_hw_norm_1 = torch.tensor(np.array(projected_keypoints_wh_1)[:, ::-1].copy()).type(torch.FloatTensor) / torch.tensor([h, w]) # [6, 2(h,w)]
        # projected_keypoints_hw_norm_2 = torch.tensor(np.array(projected_keypoints_wh_2)[:, ::-1].copy()).type(torch.FloatTensor) / torch.tensor([h, w]) # [6, 2(h,w)]

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
            # 'projected_keypoints_1': projected_keypoints_hw_norm_1,  # [6, 2(h,w)]
            # 'projected_keypoints_2': projected_keypoints_hw_norm_2,  # [6, 2(h,w)]
            # 'state_embeddings': state_embeddings.flatten(1),
            'image_path_1': image_path_1,
            'image_path_2': image_path_2,
            # 'stacked_images': stacked_images,
            # 'positional_encoding': pe,
            }
        return sample


    def resize_belief_map(self, belief_maps, T_resize):
        belief_maps = T_resize(belief_maps)
        for i in range(len(belief_maps)):
            belief_maps[i] /= belief_maps[i].max() # peak value normalize to have 1 for the peak pixel
        return belief_maps


    def create_belief_map(self, image_resolution, keypoints, sigma=10, noise_std=0):
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