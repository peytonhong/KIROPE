# -*- coding: utf-8 -*-
"""kirope.py

Kinematics Guided Robot Pose Estimation with Monocular Camera. (KIROPE)

Hyosung Hong
"""

from PIL import Image
# import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
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
    def __init__(self, num_classes, hidden_dim=256, nheads=2,
                 num_encoder_layers=7, num_decoder_layers=7):
        super().__init__()

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

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, images, joint_states):
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
        
        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([    
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),    # [H, W, 128]
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),    # [H, W, 128]
        ], dim=-1).flatten(0, 1).unsqueeze(1)                   # [H*W, 1, 256] == [850, 1, 256]
        
        # propagate through the transformer
        # h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),     # [H*W, 1, 256] == [850, 1, 256]
        #                      self.query_pos.unsqueeze(1)).transpose(0, 1)   # [100, 1, 256]
        # h.shape == [1, 100, 256]
        h = h.flatten(2).permute(2, 0, 1)
        # Transformer encoder without positional encoding
        h = self.transformer_encoder(h)  # [625, 1, 256]
        
        h = self.transformer_decoder(joint_states.unsqueeze(1), h)


        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_belief_maps': h}

"""Let's put everything together in a `detect` function:"""
def detect(im, joint_states, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0) # [1, 3, 800, 1066] image resize from transform function
    
    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img, joint_states)

    return outputs


"""
There are two main components:
* a convolutional backbone - we use ResNet-50 in this demo
* a Transformer - we use the default PyTorch nn.TransformerEncoder, nn.TransformerDecoder
"""

model = KIROPE(num_classes=7)
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
joint_states = robot_dataset[idx]['joint_states']
belief_map = robot_dataset[idx]['belief_maps']

train_iterator = DataLoader(dataset=robot_dataset, batch_size=4, shuffle=False)

# pil_img = Image.fromarray(image) # [500, 500]

result = detect(image, joint_states, model, transform)

print(result['pred_belief_maps'])
