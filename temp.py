import numpy as np
import cv2
from torch.utils.data import DataLoader
from dataset_load import RobotDataset
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from utils.gaussian_position_encoding import gaussian_position_encoding

dataset = RobotDataset()
image = dataset[0]['image']
belief_map = dataset[0]['belief_maps']
# cv2.imshow('image', image)
# cv2.waitKey(0)
# for i in range(len(belief_map)):
#     print(i, belief_map[i].shape)
#     cv2.imshow('belief_map', belief_map[i])
#     cv2.waitKey(0)
joint_angles = dataset[0]['joint_angles']
joint_velocities = dataset[0]['joint_velocities']

# print('joint_angles: ', joint_angles, joint_angles)
# print('joint_velocities: ', joint_velocities, joint_velocities)
# joint_states = np.stack([joint_angles, joint_velocities], axis=1)
joint_states = dataset[0]['joint_states']
print('joint_states: ', joint_states.shape, joint_states)

train_iterator = DataLoader(dataset=dataset, batch_size=4, shuffle=False)

# img = Image.open('annotation/00000.png')
# print(img)
# img_np = np.asarray(img)
# print(img_np)
# img_pil = Image.fromarray(img_np)
# print(img_pil)
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# print(np.asarray(image)[:,:,3])

image_transform = transform(image)
# print(image_transform.shape)
# print('image.size', image.size)



joint_states = dataset[10]['joint_states']
print('joint_states: ', joint_states.shape, joint_states)

pos = gaussian_position_encoding(joint_states.numpy())

for i in range(len(pos)):    
    img = (pos[i]*255).reshape(16,16).astype(np.uint8)
    cv2.imshow('pos', img)    
    cv2.waitKey(0)

