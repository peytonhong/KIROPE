import numpy as np
import json
from glob import glob
import os
import cv2
import torch
from torch.utils.data import Dataset
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import imageio
from PIL import Image
import torchvision.transforms as T
from utils.gaussian_position_encoding import gaussian_state_embedding, positional_encoding
from utils.util_functions import create_belief_map
class RobotDataset(Dataset):
    """ Loading Robot Dataset for Pose Estimation"""

    def __init__(self, data_dir='annotation/real/train', augmentation=False):#, embed_dim=256):
        self.image_paths_1 = sorted(glob(os.path.join(data_dir, '*/cam1/*.jpg')))
        self.label_paths_1 = sorted(glob(os.path.join(data_dir, '*/cam1/*.json')))
        self.image_paths_2 = sorted(glob(os.path.join(data_dir, '*/cam2/*.jpg')))
        self.label_paths_2 = sorted(glob(os.path.join(data_dir, '*/cam2/*.json')))
        self.image_paths_all = self.image_paths_1 + self.image_paths_2
        self.label_paths_all = self.label_paths_1 + self.label_paths_2
        # standard PyTorch mean-std input image normalization
        self.image_transform = T.Compose([
            # T.Resize(800),
            T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        coco_dir = 'annotation/coco/val2017'
        self.coco_paths = sorted(glob(os.path.join(coco_dir, '*.jpg')))
        self.augmentation = augmentation
        self.aug_rate = 0.5 # 50%

    def __len__(self):
        return len(self.image_paths_1)

    def __getitem__(self, idx):        
        image_path_1 = self.image_paths_1[idx]
        image_path_2 = self.image_paths_2[idx]
        image_path_all = self.image_paths_all[idx]
        image_1 = Image.open(image_path_1).convert('RGB') # [w, h]
        image_2 = Image.open(image_path_2).convert('RGB') # [w, h]
        image_all = Image.open(image_path_all).convert('RGB') # [w, h]
        w, h = image_1.size        
        image_1 = self.image_transform(image_1) # after T.ToTensor() [3, h, w]
        image_2 = self.image_transform(image_2) # after T.ToTensor() [3, h, w]
        image_all = self.image_transform(image_all) # after T.ToTensor() [3, h, w]
        with open(self.label_paths_1[idx]) as json_file:
            label_1 = json.load(json_file)
        with open(self.label_paths_2[idx]) as json_file:
            label_2 = json.load(json_file)
        with open(self.label_paths_all[idx]) as json_file:
            label_all = json.load(json_file)
        joint_angles = np.array(label_1['object']['joint_angles'])
        keypoints_1 = np.array(label_1['object']['joint_keypoints']) #[6, 2(w,h)]
        keypoints_2 = np.array(label_2['object']['joint_keypoints']) #[6, 2(w,h)]
        keypoints_all = np.array(label_all['object']['joint_keypoints']) #[6, 2(w,h)]

        if self.augmentation and np.random.rand() < self.aug_rate:
            image_aug, keypoint_aug, aug_param = self.image_keypoint_augmentation(image_path_all, keypoints_all)
            image_all = self.image_transform(image_aug)
            keypoints_all = keypoint_aug

        belief_maps_1 = torch.tensor(create_belief_map((h, w), keypoints_1, noise_std=0)).type(torch.FloatTensor) # [6, h,w]
        belief_maps_2 = torch.tensor(create_belief_map((h, w), keypoints_2, noise_std=0)).type(torch.FloatTensor) # [6, h,w]
        belief_maps_all = torch.tensor(create_belief_map((h, w), keypoints_all, noise_std=0)).type(torch.FloatTensor) # [6, h,w]
        
        # belief_maps_1_noise = torch.tensor(create_belief_map((h, w), keypoints_1, sigma=10, noise_std=10)).type(torch.FloatTensor) # [6, h,w]
        # belief_maps_2_noise = torch.tensor(create_belief_map((h, w), keypoints_2, sigma=10, noise_std=10)).type(torch.FloatTensor) # [6, h,w]
        belief_maps_all_noise = torch.tensor(create_belief_map((h, w), keypoints_all, sigma=10, noise_std=10)).type(torch.FloatTensor) # [6, h,w]
        img_belief_1 = torch.cat((image_1, belief_maps_1), dim=0) # [9, h, w]
        img_belief_2 = torch.cat((image_2, belief_maps_2), dim=0) # [9, h, w]
        # img_belief_stack = torch.cat((img_belief_1, img_belief_2), dim=0) # [18, h, w]
        img_belief_all = torch.cat((image_all, belief_maps_all_noise), dim=0) # [9, h, w]
        # stacked_image_1 = torch.cat((image_1, belief_maps_1), dim=0) # [9, 800, 800]
        # stacked_image_2 = torch.cat((image_2, belief_maps_2), dim=0) # [9, 800, 800]
        # stacked_image = torch.cat((image_1, image_2), dim=0) # [6, h, w]
        # stacked_beliefmap = torch.cat((belief_maps_1, belief_maps_2), dim=0) # [12, h, w]
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
            'keypoints_GT_1': keypoints_1,  # [6, 2(w,h)]
            'keypoints_GT_2': keypoints_2,  # [6, 2(w,h)]
            # 'state_embeddings': state_embeddings.flatten(1),
            'image_path_1': image_path_1,
            'image_path_2': image_path_2,
            # 'stacked_image_1': stacked_image_1,
            # 'stacked_image_2': stacked_image_2,
            # 'stacked_image': stacked_image,
            # 'stacked_beliefmap': stacked_beliefmap,
            # 'positional_encoding': pe,
            # 'image_beliefmap_stack': img_belief_stack,
            'cam_K_1': cam_K_1,
            'cam_K_2': cam_K_2,
            'cam_RT_1': cam_RT_1,
            'cam_RT_2': cam_RT_2,
            'distortion_1': distortion_1,
            'distortion_2': distortion_2,
            'image_all': image_all,
            'belief_maps_all': belief_maps_all, # [6, 480, 640]
            'img_belief_all': img_belief_all,   # [9, 480, 640]
            'img_belief_1': img_belief_1,       # [9, 480, 640]
            'img_belief_2': img_belief_2,       # [9, 480, 640]
            }
        return sample

    def image_keypoint_augmentation(self, image_path, keypoint):
        # image and keypoint augmentation
        # order: scale -> rotate -> translate_xy
        
        image = imageio.imread(image_path)
        height, width = image.shape[:2]
        coco_image = imageio.imread(np.random.choice(self.coco_paths))
        coco_image = ia.imresize_single_image(coco_image, (height, width))
        if len(coco_image.shape) == 2:
            coco_image = cv2.cvtColor(coco_image, cv2.COLOR_GRAY2BGR)

        kps_wh = np.array(keypoint)
        kps = [Keypoint(x=kps_wh[i][0], y=kps_wh[i][1]) for i in range(len(kps_wh))]

        kpsoi = KeypointsOnImage(kps, shape=(height, width))
        kps_is_valid = False
        while not kps_is_valid:
            rot_angle = np.random.choice(np.arange(-90, 90, 1))
            scale_val = np.random.choice(np.arange(0.8, 1.2, 0.2))
            rotate = iaa.Affine(rotate=rot_angle)
            scale = iaa.Affine(scale=scale_val)

            image_aug, kpsoi_aug = scale(image=image, keypoints=kpsoi)
            image_aug, kpsoi_aug = rotate(image=image_aug, keypoints=kpsoi_aug)
            kps_x_min = np.min(kpsoi_aug.to_xy_array()[:,0])
            kps_x_max = np.max(kpsoi_aug.to_xy_array()[:,0])
            kps_y_min = np.min(kpsoi_aug.to_xy_array()[:,1])
            kps_y_max = np.max(kpsoi_aug.to_xy_array()[:,1])
            pixel_margin = 5
            trans_x_min = int(-kps_x_min + pixel_margin)
            trans_x_max = int(width - kps_x_max - pixel_margin)
            trans_y_min = int(-kps_y_min + pixel_margin)
            trans_y_max = int(height - kps_y_max - pixel_margin)
            trans_x = np.random.choice(np.arange(trans_x_min, trans_x_max, 1))
            trans_y = np.random.choice(np.arange(trans_y_min, trans_y_max, 1))
            translate = iaa.Affine(translate_px={"x": trans_x, "y": trans_y})
            image_aug, kpsoi_aug = translate(image=image_aug, keypoints=kpsoi_aug)
            
            kps_flag_buffer = []
            for keypoint in kpsoi_aug:
                if keypoint.x > 0 and keypoint.x < width:
                    if keypoint.y > 0 and keypoint.y < height:
                        kps_flag_buffer.append(True)
            if len(kps_flag_buffer) == len(kpsoi_aug):
                kps_is_valid = True

        image_aug[np.where(image_aug==0)] = coco_image[np.where(image_aug==0)]
        hue_saturation = iaa.AddToHueAndSaturation((-50, 50))
        image_aug = hue_saturation(image=image_aug)
        
        rotation_scale_translate = (rot_angle, scale_val, trans_x, trans_y)
        
        return image_aug, kpsoi_aug.to_xy_array(), rotation_scale_translate