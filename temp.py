import numpy as np
import cv2
from torch.utils.data import DataLoader
from dataset_load import RobotDataset
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from utils.gaussian_position_encoding import gaussian_state_embedding, positional_encoding

# dataset = RobotDataset()
# image = dataset[0]['image']
# belief_map = dataset[0]['belief_maps']
# # cv2.imshow('image', image)
# # cv2.waitKey(0)
# # for i in range(len(belief_map)):
# #     print(i, belief_map[i].shape)
# #     cv2.imshow('belief_map', belief_map[i])
# #     cv2.waitKey(0)
# joint_angles = dataset[0]['joint_angles']
# joint_velocities = dataset[0]['joint_velocities']

# # print('joint_angles: ', joint_angles, joint_angles)
# # print('joint_velocities: ', joint_velocities, joint_velocities)
# # joint_states = np.stack([joint_angles, joint_velocities], axis=1)
# joint_states = dataset[0]['joint_states']
# # print('joint_states: ', joint_states.shape, joint_states)



# # img = Image.open('annotation/00000.png')
# # print(img)
# # img_np = np.asarray(img)
# # print(img_np)
# # img_pil = Image.fromarray(img_np)
# # print(img_pil)
# transform = T.Compose([
#     T.Resize(800),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # print(np.asarray(image)[:,:,3])

# # image_transform = transform(image)
# # print(image_transform.shape)
# # print('image.size', image.size)



# joint_states = dataset[10]['joint_states']
# # print('joint_states: ', joint_states.shape, joint_states)

# pos = gaussian_state_embedding(joint_states.numpy())

# # for i in range(len(pos)):    
# #     img = (pos[i]*255).reshape(16,16).astype(np.uint8)
# #     cv2.imshow('pos', img)    
# #     cv2.waitKey(0)

# train_iterator = DataLoader(dataset=dataset, batch_size=2, shuffle=False)

# for i, sampled_batch in enumerate(train_iterator):
#     image = sampled_batch['image'] # [N, 3, 800, 800]
#     joint_angles = sampled_batch['joint_angles']
#     joint_velocities = sampled_batch['joint_velocities']
#     joint_states = sampled_batch['joint_states'] # [N, 7, 2]
#     belief_maps = sampled_batch['belief_maps'] # [N, 7, 500, 500]
#     projected_keypoints = sampled_batch['projected_keypoints']
#     keypoint_embeddings = sampled_batch['keypoint_embeddings'] # [N, 7, 256]
#     image_path = sampled_batch['image_path']

#     print('image.shape', image.shape)
#     # print('joint_angles.shape', joint_angles.shape)
#     # print('joint_velocities.shape', joint_velocities.shape)
#     print('joint_states.shape', joint_states.shape)
#     print('belief_maps.shape', belief_maps.shape)
#     # print('projected_keypoints.shape', projected_keypoints.shape)
#     print('keypoint_embeddings.shape', keypoint_embeddings.shape)
    
    
#     break


# print('keypoint_embeddings: ', keypoint_embeddings.shape)
# angles = np.arange(-180, 180, 1)*np.pi/180

# for i in range(len(angles)):
#     state_embeddings = gaussian_state_embedding([[angles[i], 1]])
#     image = state_embeddings[0].reshape(100,100)
#     image = (image*255).astype(np.uint8)
#     image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
#     cv2.imwrite(f'visualization_result/{str(i).zfill(5)}.png', image)

pe = pos_encoding(256, 7)
print(pe.min(), pe.max())
