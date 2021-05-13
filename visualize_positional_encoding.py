import numpy as np
import cv2
from dataset_load import RobotDataset
from tqdm import tqdm
from utils.gaussian_position_encoding import gaussian_position_encoding


video_name = 'visualize_positional_encoding.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 30.0, (572, 500))
dataset = RobotDataset()

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 0)
thickness = 2
for i in tqdm(range(len(dataset))):
    # i=0
    image_path = dataset[i]['image_path']
    projected_keypoints = dataset[i]['projected_keypoints']
    joint_states = dataset[i]['joint_states']
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for keypoint in projected_keypoints:
        cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), radius=5, color=(0,255,0), thickness=2)


    # print('joint_states: ', joint_states.shape, joint_states)

    pos = gaussian_position_encoding(joint_states.numpy())


    for j in range(len(pos)):    
        if j==0:
            img = (pos[j]*255).reshape(16,16).astype(np.uint8)
            continue
        img = np.vstack(((pos[j]*255).reshape(16,16).astype(np.uint8), img))
    img_resize = cv2.resize(img, (72, 500), interpolation = cv2.INTER_AREA)
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_GRAY2RGB)
    image_aug = np.hstack((image, img_resize))
    cv2.putText(image_aug, 'Joint 1:', (370, 480), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_aug, 'Joint 2:', (370, 407), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_aug, 'Joint 3:', (370, 334), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_aug, 'Joint 4:', (370, 261), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_aug, 'Joint 5:', (370, 188), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_aug, 'Joint 6:', (370, 115), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_aug, 'Joint 7:', (370,  42), font, fontScale, color, thickness, cv2.LINE_AA)
    video.write(image_aug)

# cv2.imwrite('visualize_positional_encoding.png', image_aug)



