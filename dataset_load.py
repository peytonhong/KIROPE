import numpy as np
import json
from glob import glob
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T

class RobotDataset(Dataset):
    """ Loading Robot Dataset for Pose Estimation"""

    def __init__(self, data_dir='annotation'):        
        self.image_paths = sorted(glob(os.path.join(data_dir, '*.png')))
        self.label_paths = sorted(glob(os.path.join(data_dir, '*.json')))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = cv2.imread(self.image_paths[idx])
        # image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        image = Image.open(self.image_paths[idx]).convert('RGB')

        with open(self.label_paths[idx]) as json_file:
            label = json.load(json_file)
        joint_angles = label['objects'][0]['joint_angles']
        joint_velocities = label['objects'][0]['joint_velocities']
        joint_states = torch.tensor((np.stack([joint_angles, joint_velocities], axis=1)))
        projected_keypoints = label['objects'][0]['projected_keypoints']        
        belief_maps = self.create_belief_map(image.size, projected_keypoints)        
        
        sample = {'image': image, 'joint_angles': joint_angles, 'joint_velocities': joint_velocities, 'joint_states': joint_states, 'belief_maps': belief_maps}
        return sample





    def create_belief_map(self, image_resolution, pointsBelief, sigma=2):
        '''
        This function is referenced from NVIDIA Dream/datasets.py
        
        image_resolution: image size (width x height)
        pointsBelief: list of points to draw in a 7x2 tensor
        sigma: the size of the point
        
        returns a tensor of n_points x h x w with the belief maps
        '''
        
        # Input argument handling
        assert (
            len(image_resolution) == 2
        ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
            len(image_resolution)
        )
        image_width, image_height = image_resolution
        image_transpose_resolution = (image_height, image_width)
        out = np.zeros((len(pointsBelief), image_height, image_width))

        w = int(sigma * 2)

        for i_point, point in enumerate(pointsBelief):
            pixel_u = int(point[0])
            pixel_v = int(point[1])
            array = np.zeros(image_transpose_resolution)

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