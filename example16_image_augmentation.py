import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import numpy as np 
import imageio
import json
import os
from glob import glob

import torchvision.transforms as T
image_transform = T.Compose([
            # T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

data_dir='annotation/real/test'
coco_dir='annotation/coco/val2017'
image_paths_1 = sorted(glob(os.path.join(data_dir, '*/cam1/*.jpg')))
label_paths_1 = sorted(glob(os.path.join(data_dir, '*/cam1/*.json')))
coco_paths = sorted(glob(os.path.join(coco_dir, '*.jpg')))

idx = 0
image = imageio.imread(image_paths_1[idx])
height, width = image.shape[:2]
coco_image = imageio.imread(np.random.choice(coco_paths))
coco_image = ia.imresize_single_image(coco_image, (height, width))

with open(label_paths_1[idx]) as json_file:
    label_1 = json.load(json_file)

kps_wh_1 = label_1['object']['joint_keypoints'] #[6, 2(w,h)]
kps_wh_1 = np.array(kps_wh_1)
kps = [Keypoint(x=kps_wh_1[i][0], y=kps_wh_1[i][1]) for i in range(len(kps_wh_1))]

kpsoi = KeypointsOnImage(kps, shape=image.shape)
kps_is_found = False
while not kps_is_found:
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
        kps_is_found = True

image_aug[np.where(image_aug==0)] = coco_image[np.where(image_aug==0)]
hue_saturation = iaa.AddToHueAndSaturation((-50, 50))
image_aug = hue_saturation(image=image_aug)

translate_inv = iaa.Affine(translate_px={"x": -trans_x, "y": -trans_y})
rotate_inv = iaa.Affine(rotate=-rot_angle)
scale_inv = iaa.Affine(scale=1/scale_val)
image_recovered, kps_recovered = translate_inv(image=image_aug, keypoints=kpsoi_aug)
image_recovered, kps_recovered = rotate_inv(image=image_recovered, keypoints=kps_recovered)
image_recovered, kps_recovered = scale_inv(image=image_recovered, keypoints=kps_recovered)

ia.imshow(
    np.hstack([
        kpsoi.draw_on_image(image, size=7),
        kpsoi_aug.draw_on_image(image_aug, size=7),
        kps_recovered.draw_on_image(image_recovered, size=7)
    ])
)