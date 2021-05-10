import os
import numpy as numpy
import cv2
import json
from tqdm import tqdm

data_dir = 'annotation'
video_name = 'annotation_result.mp4'

images = [img for img in os.listdir(data_dir) if img.endswith(".png")]
num_images = len(images)
image = cv2.imread(os.path.join(data_dir, images[0]), cv2.IMREAD_COLOR)
height, width, channel = image.shape

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

for i in tqdm(range(num_images)):
    # load an image
    image_file = '{:05d}.png'.format(i)
    image_path = os.path.join(data_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # load keypoints
    with open(os.path.join(data_dir, '{:05d}.json'.format(i))) as f:
        annotation = json.load(f)
    keypoints = annotation['objects'][0]['projected_keypoints']

    for keypoint in keypoints:
        cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), radius=5, color=(0,255,0), thickness=2)
    
    video.write(image)

#     cv2.imshow('image', image)

#     cv2.waitKey(0)
# cv2.destroyAllWindows()