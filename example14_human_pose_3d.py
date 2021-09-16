# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.

Perform 2 camera based 2d keypoint inference and 3d triangulation using 
DLT(Direct Linear Transfrom) to get 3d human pose.
"""

from posixpath import join
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob
import cv2
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/human_pose/output_directory',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--im-or-folder',
        dest='im_or_folder', help='image or folder of images', default=None
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)

def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

def get_annotation_path(data_path, file_number):
    anntation_path = data_path + 'human_pose' + '/' + str(file_number).zfill(4) + '.json'
    return anntation_path

def make_annotation_file(
    filename = "tmp.json", #this has to include path as well        
    joint_keypoints_1 = [], # 2D keypoints
    joint_keypoints_2 = [], # 2D keypoints
    joint_3d_positions = [], # 3D joint position
    ):

    dict_out = {
                'joint_keypoints_1': joint_keypoints_1,
                'joint_keypoints_2': joint_keypoints_2,
                'joint_3d_positions': joint_3d_positions,
            }
    
    with open(filename, 'w') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=False)

def get_keypoint(predictor, image):    
    outputs = predictor(image)['instances'].to('cpu')
    has_bbox = False
    if outputs.has('pred_boxes'):
        bbox_tensor = outputs.pred_boxes.tensor.numpy()
        if len(bbox_tensor) > 0:
            has_bbox = True
            scores = outputs.scores.numpy()[:, None]
            bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
    if has_bbox:
        kps = outputs.pred_keypoints.numpy()
        kps_xy = kps[:, :, :2]
        kps_prob = kps[:, :, 2:3]
        kps_logit = np.zeros_like(kps_prob) # Dummy
        # kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
        kps = np.concatenate((kps_xy, kps_prob), axis=2) # [1, 17, 3], 17:(joints), 3:(x,y,score)
        kps = kps[0] # [17,3]
        # kps = kps.transpose(0, 2, 1) 
    else:
        kps = []
        bbox_tensor = []
    return kps

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def main(args):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    predictor = DefaultPredictor(cfg)
    
    data_path = 'annotation/real/test/20210819_025345_human/'
    im_list_1 = sorted(glob.glob(data_path + 'cam1/*.jpg'))
    im_list_2 = sorted(glob.glob(data_path + 'cam2/*.jpg'))
    json_list_1 = sorted(glob.glob(data_path + 'cam1/*.json'))
    json_list_2 = sorted(glob.glob(data_path + 'cam2/*.json'))
    with open(json_list_1[0], 'r') as json_file:
        annotation_1 = json.load(json_file)
    with open(json_list_2[0], 'r') as json_file:
        annotation_2 = json.load(json_file)
    cam_K_1 = np.array(annotation_1['camera']['camera_intrinsic'])
    cam_RT_1 = np.array(annotation_1['camera']['camera_extrinsic'])
    cam_K_2 = np.array(annotation_2['camera']['camera_intrinsic'])
    cam_RT_2 = np.array(annotation_2['camera']['camera_extrinsic'])
    P_1 = cam_K_1 @ cam_RT_1
    P_2 = cam_K_2 @ cam_RT_2
    # Full joint information: 17 joints 
    # [nose, L-eye, R-eye, L-ear, R-ear, L-shoulder, R-shoulder, L-elbow, R-elbow, L-wrist, R-wrist, L-pelvis, R-pelvis, L-knee, R-knee, L-ankle, R-ankle]
    joint_interest = [0,1,2,5,6,7,8,9,10] # [nose, L-eye, R-eye, L-shoulder, R-shoulder, L-elbow, R-elbow, L-ankle, R-ankle]
    
    
    for frame_i, (image_path_1, image_path_2) in enumerate(zip(im_list_1, im_list_2)):
        image_1 = cv2.imread(image_path_1)
        image_2 = cv2.imread(image_path_2)
        t = time.time()
        kps_1 = get_keypoint(predictor, image_1)
        kps_2 = get_keypoint(predictor, image_2)
        if len(kps_1) > 0 and len(kps_2) > 0: # if keypoints are detected in both images
            kps_1 = kps_1[joint_interest] # [9,3] joints with interest (3: x, y, score)        
            kps_2 = kps_2[joint_interest] # [9,3] joints with interest (3: x, y, score)        
            joints_3d = []
            for kp_1, kp_2 in zip(kps_1, kps_2):
                joint_3d = DLT(P_1, P_2, kp_1, kp_2)
                joints_3d.append(joint_3d)
            joints_3d = np.array(joints_3d)
            kps_1 = kps_1.tolist()
            kps_2 = kps_2.tolist()
            joints_3d = joints_3d.tolist()
            make_annotation_file(get_annotation_path(data_path, frame_i), kps_1, kps_2, joints_3d)
        else:
            kps_1 = []
            kps_2 = []
            joints_3d = []
        
        print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

if __name__ == '__main__':
    setup_logger()
    args = parse_args()
    main(args)
