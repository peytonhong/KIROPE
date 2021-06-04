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

    def __init__(self, data_dir='annotation/train', embed_dim=256):        
        self.image_paths = sorted(glob(os.path.join(data_dir, '*.png')))
        self.label_paths = sorted(glob(os.path.join(data_dir, '*.json')))
        # standard PyTorch mean-std input image normalization
        self.image_transform = T.Compose([
            # T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.image_resize_800 = T.Resize(800)
        self.image_resize_16 = T.Resize(16)
        self.embed_dim = embed_dim


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = cv2.imread(self.image_paths[idx]) # [h, w, c(BGR)]
        # image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB') # [w, h]
        w, h = image.size        
        image = self.image_transform(image) # after T.ToTensor() [3, h, w]
        with open(self.label_paths[idx]) as json_file:
            label = json.load(json_file)
        joint_angles = label['objects'][0]['joint_angles']
        joint_velocities = label['objects'][0]['joint_velocities']
        joint_velocities[-1] = 0 # end-effector joint velocity set to zero (because it is too much fast)
        joint_states = (np.stack([joint_angles, joint_velocities], axis=1))
        projected_keypoints_wh = label['objects'][0]['projected_keypoints'] #[7, 2(w,h)]
        belief_maps = torch.tensor(self.create_belief_map((h, w), projected_keypoints_wh, noise_std=0)).type(torch.FloatTensor) # [7, h,w]        
        belief_maps = self.resize_belief_map(belief_maps, self.image_resize_16)
        belief_maps_noise = torch.tensor(self.create_belief_map((h, w), projected_keypoints_wh, noise_std=5)).type(torch.FloatTensor) # [7, h,w]        
        belief_maps_noise = self.resize_belief_map(belief_maps_noise, self.image_resize_16)
        state_embeddings = torch.tensor(gaussian_state_embedding(joint_states, self.embed_dim)).type(torch.FloatTensor) # [7, num_features]
        _, num_features = state_embeddings.shape
        w_feature = np.sqrt(num_features).astype(np.uint8)
        state_embeddings = state_embeddings.reshape(7, w_feature, w_feature)
        image = self.image_resize_800(image) # [3, 800, 800]
        stacked_images = torch.cat((image, self.image_resize_800(state_embeddings)), dim=0) # [10, 800, 800]        
        pe = positional_encoding(256, 7) # [7, 256]
        projected_keypoints_hw_norm = torch.tensor(np.array(projected_keypoints_wh)[:, ::-1].copy()).type(torch.FloatTensor) / torch.tensor([h, w]) # [7, 2(h,w)]
        sample = {
            'image': image, 
            'joint_angles': joint_angles, 
            'joint_velocities': joint_velocities, 
            'joint_states': joint_states, 
            'belief_maps': belief_maps,  # [7, 16, 16]
            'belief_maps_noise': belief_maps_noise, # [7, 16, 16]
            'projected_keypoints': projected_keypoints_hw_norm,  # [7, 2(h,w)]
            'state_embeddings': state_embeddings.flatten(1),
            'image_path': image_path,
            'stacked_images': stacked_images,
            'positional_encoding': pe,
            }
        return sample


    def resize_belief_map(self, belief_maps, T_resize):
        belief_maps = T_resize(belief_maps)
        for i in range(len(belief_maps)):
            belief_maps[i] /= belief_maps[i].max() # peak value normalize to have 1 for the peak pixel
        return belief_maps


    def create_belief_map(self, image_resolution, pointsBelief, sigma=10, noise_std=0):
        '''
        This function is referenced from NVIDIA Dream/datasets.py
        
        image_resolution: image size (width x height)
        pointsBelief: list of points to draw in a 7x2 tensor
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
        out = np.zeros((len(pointsBelief), image_height, image_width))

        w = int(sigma * 2)

        for i_point, point in enumerate(pointsBelief):
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