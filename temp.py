import re
import torch
from torch import nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import torchvision.transforms as T
# row_embed = torch.nn.Embedding(50, 256)
# col_embed = torch.nn.Embedding(50, 256)
# h, w = 20, 30
# i = torch.arange(w)
# j = torch.arange(h)
# x_embed = col_embed(i)
# y_embed = row_embed(j)
# print(x_embed.unsqueeze(0).repeat(h, 1, 1).shape)
# print(y_embed.unsqueeze(1).repeat(1, w, 1).shape)
# pos = torch.cat([x_embed.unsqueeze(0).repeat(h, 1, 1),
#                 y_embed.unsqueeze(1).repeat(1, w, 1),
#                 ], dim=-1)
# print(pos.permute(2,0,1).unsqueeze(0).repeat(4, 1, 1, 1).shape)

# x1 = x_embed[1].detach().numpy().reshape(16,16)

def create_belief_map(image_resolution, pointsBelief, sigma=10, noise_std=0):
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

def save_belief_map_images(belief_maps, map_type):
    # belief_maps: [7, h, w]
    belief_maps = (belief_maps*255).astype(np.uint8)    
    for i in range(len(belief_maps)):        
        image = cv2.cvtColor(belief_maps[i].copy(), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f'visualization_result/visualize_{map_type}_belief_maps_{i}.png', image)


with open('annotation/test_many_obj/00001.json') as json_file:
    label = json.load(json_file)

projected_keypoints_wh = label['objects'][0]['projected_keypoints'] #[7, 2(w,h)]


belief_maps = torch.tensor(create_belief_map((500,500), projected_keypoints_wh, noise_std=0)).type(torch.FloatTensor)
image_resize_100 = T.Resize(100)
image_resize_16 = T.Resize(16)

belief_maps = image_resize_16(belief_maps)
print(belief_maps.max())
print(belief_maps.shape)
for i in range(len(belief_maps)):
    belief_maps[i] /= belief_maps[i].max()

save_belief_map_images(belief_maps.numpy(), 'gt')
# print(belief_maps[0])
print(belief_maps[1])