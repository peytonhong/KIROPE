import numpy as np
import cv2
import os
import json

def create_belief_map(image_resolution, keypoints, sigma=4, noise_std=0):
    '''
    This function is referenced from NVIDIA Dream/datasets.py
    
    image_resolution: image size (height x width)
    keypoints: list of keypoints to draw in a numJointsx2 tensor
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
    out = np.zeros((len(keypoints), image_height, image_width))

    w = int(sigma * 2)
    
    for i_point, point in enumerate(keypoints):
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

def extract_keypoints_from_belief_maps(belief_maps):
    keypoints = []
    confidences = []
    for i in range(len(belief_maps)):
        indices = np.where(belief_maps[i] == belief_maps[i].max())
        keypoints.append([indices[1][0], indices[0][0]]) # keypoint format: [w, h]
        confidences.append(belief_maps[i].max()) # confidence score between 0 to 1
        # print(belief_maps[0][i].max())
    
    return (np.array(keypoints), np.array(confidences))

def save_belief_map_images(belief_maps, map_type):
    # belief_maps: [numJoints, h, w]
    belief_maps = (belief_maps*255).astype(np.uint8)    
    for i in range(len(belief_maps)):        
        image = cv2.cvtColor(belief_maps[i].copy(), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f'visualization_result/belief_maps/{map_type}_belief_maps_{i}.png', image)

def visualize_state_embeddings(state_embeddings):
    for i in range(len(state_embeddings)):
        file_name = f'keypoint_embedding_{i}.png'
        embedding = (state_embeddings[i]*255).astype(np.uint8)
        image = cv2.cvtColor(embedding, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(file_name, image.copy())

def visualize_result(image_paths, pred_keypoints, gt_keypoints, is_kp_normalized):
    # visualize the joint position prediction wih ground truth for one sample
    # pred_kps, gt_kps: [numJoints, 2(w,h order)]
    rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
    bgr_colors = rgb_colors[:, ::-1]
    image = cv2.imread(image_paths)
    if is_kp_normalized:
        height, width, channel = image.shape
        pred_keypoints = [[int(u*width), int(v*height)] for u, v in pred_keypoints]
        gt_keypoints = [[int(u*width), int(v*height)] for u, v in gt_keypoints]
    # pred_keypoints = extract_keypoints_from_belief_maps(pred_kps)     
    # gt_keypoints = extract_keypoints_from_belief_maps(gt_belief_maps)
    # save_belief_map_images(pred_kps, 'pred')
    # save_belief_map_images(gt_belief_maps, 'gt')
    image = image.copy()
    for i, (pred_keypoint, gt_keypoint) in enumerate(zip(pred_keypoints, gt_keypoints)):
        cv2.drawMarker(image, (int(pred_keypoint[0]), int(pred_keypoint[1])), color=bgr_colors[i].tolist(), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=1)
        cv2.circle(image, (int(gt_keypoint[0]), int(gt_keypoint[1])), radius=5, color=bgr_colors[i].tolist(), thickness=2)        
    cv2.imwrite(f'visualization_result/{image_paths[-8:]}', image)

def visualize_result_two_cams(image_paths_1, pred_keypoints_1, gt_keypoints_1, 
                            image_paths_2, pred_keypoints_2, gt_keypoints_2, 
                            is_kp_normalized):
    # visualize the joint position prediction wih ground truth for one sample
    # pred_kps, gt_kps: [numJoints, 2(w,h order)]
    rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
    bgr_colors = rgb_colors[:, ::-1]
    image_1 = cv2.imread(image_paths_1)
    image_2 = cv2.imread(image_paths_2)
    height, width, channel = image_1.shape
    if is_kp_normalized:
        pred_keypoints_1 = [[int(u*width), int(v*height)] for u, v in pred_keypoints_1]
        gt_keypoints_1 = [[int(u*width), int(v*height)] for u, v in gt_keypoints_1]
        pred_keypoints_2 = [[int(u*width), int(v*height)] for u, v in pred_keypoints_2]
        gt_keypoints_2 = [[int(u*width), int(v*height)] for u, v in gt_keypoints_2]
    image_1 = image_1.copy()
    image_2 = image_2.copy()
    for i, (pred_keypoint, gt_keypoint) in enumerate(zip(pred_keypoints_1, gt_keypoints_1)):
        cv2.drawMarker(image_1, (int(pred_keypoint[0]), int(pred_keypoint[1])), color=bgr_colors[i].tolist(), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
        cv2.circle(image_1, (int(gt_keypoint[0]), int(gt_keypoint[1])), radius=5, color=bgr_colors[i].tolist(), thickness=2)        
    for i, (pred_keypoint, gt_keypoint) in enumerate(zip(pred_keypoints_2, gt_keypoints_2)):
        cv2.drawMarker(image_2, (int(pred_keypoint[0]), int(pred_keypoint[1])), color=bgr_colors[i].tolist(), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
        cv2.circle(image_2, (int(gt_keypoint[0]), int(gt_keypoint[1])), radius=5, color=bgr_colors[i].tolist(), thickness=2) 
    
    image_stack = np.hstack((image_2, image_1))
    cv2.putText(image_stack, 'CAM1', (width+10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.putText(image_stack, 'CAM2', (10,30),       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.imwrite(f'visualization_result/stacked_{image_paths_1[-8:]}', image_stack)

def visualize_result_robot_human_two_cams(image_path_1, pred_keypoints_1, gt_keypoints_1, 
                            image_path_2, pred_keypoints_2, gt_keypoints_2, 
                            cam_K_1, cam_RT_1, cam_K_2, cam_RT_2,
                            is_kp_normalized):
    # visualize the joint position prediction wih ground truth for one sample
    # pred_kps, gt_kps: [numJoints, 2(w,h order)]
    rgb_colors = np.array([[87, 117, 144], [67, 170, 139], [144, 190, 109], [249, 199, 79], [248, 150, 30], [243, 114, 44], [249, 65, 68]]) # rainbow-like
    bgr_colors = rgb_colors[:, ::-1].tolist()
    image_1 = cv2.imread(image_path_1)
    image_2 = cv2.imread(image_path_2)
    height, width, channel = image_1.shape
    if is_kp_normalized:
        pred_keypoints_1 = [[int(u*width), int(v*height)] for u, v in pred_keypoints_1]
        gt_keypoints_1 = [[int(u*width), int(v*height)] for u, v in gt_keypoints_1]
        pred_keypoints_2 = [[int(u*width), int(v*height)] for u, v in pred_keypoints_2]
        gt_keypoints_2 = [[int(u*width), int(v*height)] for u, v in gt_keypoints_2]
    image_1 = image_1.copy()
    image_2 = image_2.copy()
    for i, (pred_keypoint, gt_keypoint) in enumerate(zip(pred_keypoints_1, gt_keypoints_1)):
        cv2.drawMarker(image_1, (int(pred_keypoint[0]), int(pred_keypoint[1])), color=bgr_colors[i], markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
        cv2.circle(image_1, (int(gt_keypoint[0]), int(gt_keypoint[1])), radius=5, color=bgr_colors[i], thickness=2)        
        # draw_lines_robot(image_1, pred_keypoints_1, bgr_colors)
    for i, (pred_keypoint, gt_keypoint) in enumerate(zip(pred_keypoints_2, gt_keypoints_2)):
        cv2.drawMarker(image_2, (int(pred_keypoint[0]), int(pred_keypoint[1])), color=bgr_colors[i], markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
        cv2.circle(image_2, (int(gt_keypoint[0]), int(gt_keypoint[1])), radius=5, color=bgr_colors[i], thickness=2) 
        # draw_lines_robot(image_2, pred_keypoints_2, bgr_colors)
    
    # draw human pose skeleton lines if it exists
    folder, file_name = os.path.split(image_path_1)
    parent_folder, cam_foler = os.path.split(folder)
    file_name = file_name[:4] + '.json'
    human_pose_path = os.path.join(parent_folder, 'human_pose', file_name)
    if os.path.exists(human_pose_path):
        with open(human_pose_path, 'r') as json_file:
            human_pose_json = json.load(json_file)
        human_pose_3d = np.array(human_pose_json['joint_3d_positions'])
        draw_lines_human(image_1, human_pose_3d, cam_K_1, cam_RT_1)
        draw_lines_human(image_2, human_pose_3d, cam_K_2, cam_RT_2)

    image_stack = np.hstack((image_2, image_1))
    cv2.putText(image_stack, 'CAM1', (width+10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.putText(image_stack, 'CAM2', (10,30),       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.imwrite(f'visualization_result/stacked_{image_path_1[-8:]}', image_stack)

def draw_lines_robot(image, keypoints, colormap):
    for i in range(len(keypoints)-1):
        cv2.line(image, (int(keypoints[i][0]), int(keypoints[i][1])), (int(keypoints[i+1][0]), int(keypoints[i+1][1])), color=colormap[i], thickness=2)
    
def draw_lines_human(image, p3d, cam_K, cam_RT):
    # draw human pose skeleton lines based on estimated 3d points
    
    # add neck position (center of shoulders)
    neck_x = (p3d[3][0] + p3d[4][0])/2
    neck_y = (p3d[3][1] + p3d[4][1])/2
    neck_z = (p3d[3][2] + p3d[4][2])/2    
    p3d = np.append(p3d, [[neck_x, neck_y, neck_z]], axis=0)
    
    keypoints = []
    for point_3d in p3d:
        point_3d = np.append(point_3d, [1])
        keypoint = cam_K @ cam_RT @ point_3d.reshape(-1,1)
        keypoint /= keypoint[-1]
        keypoints.append(keypoint)
        cv2.circle(image, (keypoint[0], keypoint[1]), radius=2, color=(0,255,0), thickness=2)
    joint_connections = [[0,1], [0,2], [0,11], [11,3], [11,4], [3,5], [4,6], [5,7], [6,8], [3,9], [4,10], [9,10]]
    keypoints = np.array(keypoints).reshape(-1,3)
    
    for _c in joint_connections:
        cv2.line(image, (int(keypoints[_c[0]][0]),int(keypoints[_c[0]][1])), (int(keypoints[_c[1]][0]),int(keypoints[_c[1]][1])), color=(0,255,0), thickness=2)
