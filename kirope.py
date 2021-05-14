# -*- coding: utf-8 -*-
"""kirope.py

Kinematics Guided Robot Pose Estimation with Monocular Camera. (KIROPE)

Hyosung Hong
"""

from re import I
from PIL import Image
# import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

import numpy as np
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset_load import RobotDataset


class KIROPE(nn.Module):
    """
    KIROPE implementation.

    """
    def __init__(self, num_joints, hidden_dim=256, nheads=2,
                 num_encoder_layers=7, num_decoder_layers=7):
        super().__init__()

        self.num_noints = num_joints

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        # self.transformer = nn.Transformer(
        #     hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nheads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nheads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        # self.linear_class = nn.Linear(hidden_dim, num_classes)
        # self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=20*20) # [1, 7, 400] -> after transpose: [1, 7, 20, 20]
        self.dconv1 = nn.ConvTranspose2d(in_channels=self.num_noints, out_channels=self.num_noints, kernel_size=3, stride=4, padding=1, output_padding=1) # [1, 7, 100, 100]
        self.dconv2 = nn.ConvTranspose2d(in_channels=self.num_noints, out_channels=self.num_noints, kernel_size=3, stride=4, padding=1, output_padding=1) # [1, 7, 500, 500]
        

    def forward(self, images, keypoint_embeddings):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)    # [1, 256, 25, 25]  from original shape [1, 3, 800, 800] : feature size reduced by 1/32
        
        
        
        # propagate through the transformer
        # h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),     # [H*W, 1, 256] == [850, 1, 256]
        #                      self.query_pos.unsqueeze(1)).transpose(0, 1)   # [100, 1, 256]
        # h.shape == [1, 100, 256]
        h = h.flatten(2).permute(2, 0, 1)
        # Transformer encoder without positional encoding
        h = self.transformer_encoder(h)  # [625, 1, 256]
        x = self.transformer_decoder(keypoint_embeddings.unsqueeze(1), h) # [7, 1, 256]

        x = x.transpose(1, 0, 2) # [1, 7, 256]
        x = self.fc_out(x)
        x = F.relu(self.dconv1(x))
        x = F.relu(self.dconv2(x))

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_belief_maps': x}


"""
There are two main components:
* a convolutional backbone - we use ResNet-50 in this demo
* a Transformer - we use the default PyTorch nn.TransformerEncoder, nn.TransformerDecoder
"""

hidden_dim=256

model = KIROPE(num_classes=7, hidden_dim=hidden_dim)
# state_dict = torch.hub.load_state_dict_from_url(
#     url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
#     map_location='cpu', check_hash=True)
# model.load_state_dict(state_dict)
model.eval()


"""
KIROPE uses standard ImageNet normalization.
"""

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

idx = 0

robot_dataset = RobotDataset()

image = robot_dataset[idx]['image']
# joint_states = robot_dataset[idx]['joint_states']
keypoint_embeddings = robot_dataset[idx]['keypoint_embeddings'] # [num_joints, hiddem_dim] [7, 256]
belief_map = robot_dataset[idx]['belief_maps']

train_iterator = DataLoader(dataset=robot_dataset, batch_size=1, shuffle=False)

for i, sampled_batch in enumerate(train_iterator):
    image = sampled_batch['image'] # tensor [N, 3, 800, 800]
    joint_angles = sampled_batch['joint_angles'] 
    joint_velocities = sampled_batch['joint_velocities']
    joint_states = sampled_batch['joint_states'] # [N, 7, 2]
    belief_maps = sampled_batch['belief_maps'] # [N, 7, 500, 500]
    projected_keypoints = sampled_batch['projected_keypoints']
    keypoint_embeddings = sampled_batch['keypoint_embeddings'] # [N, 7, 256]
    image_path = sampled_batch['image_path']
    
    break

result = model(image, keypoint_embeddings)

print(result['pred_belief_maps'].shape)
