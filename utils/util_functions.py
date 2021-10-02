import numpy as np
import cv2
import matplotlib.pyplot as plt
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

def visualize_state_embeddings(state_embeddings):
    for i in range(len(state_embeddings)):
        file_name = f'keypoint_embedding_{i}.png'
        embedding = (state_embeddings[i]*255).astype(np.uint8)
        image = cv2.cvtColor(embedding, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(file_name, image.copy())

def visualize_two_stacked_images(image_beliefmap_stack, image_path1, image_path2):
    image1 = cv2.imread(image_path1)
    beliefmap1 = (image_beliefmap_stack[3:9].transpose([1,2,0])*255).astype(np.uint8)
    image2 = cv2.imread(image_path2)
    beliefmap2 = (image_beliefmap_stack[12:18].transpose([1,2,0])*255).astype(np.uint8)    
    # image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)    
    height, width = image1.shape[:2]
    beliefmap1_single_channel = np.zeros((height, width), dtype=np.int)
    beliefmap2_single_channel = np.zeros((height, width), dtype=np.int)
    for i in range(beliefmap1.shape[-1]):
        beliefmap1_single_channel += beliefmap1[:,:,i]
        beliefmap2_single_channel += beliefmap2[:,:,i]
    # merge beliefmap in red channel    
    image1 = image1.astype(np.int)
    image2 = image2.astype(np.int)
    image1[:,:,2] += beliefmap1_single_channel
    image2[:,:,2] += beliefmap2_single_channel
    image1[:,:,2][np.where(image1[:,:,2]>255)] = 255
    image2[:,:,2][np.where(image2[:,:,2]>255)] = 255
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    image_hstack = np.hstack((image1, image2))
    cv2.imwrite(f'visualization_result/image_beliefmap_stack/{image_path1[-8:]}', image_hstack)

def visualize_single_stacked_images(image_beliefmap_stack, filenum):
    image1 = (image_beliefmap_stack[:3].transpose([1,2,0])*255).astype(np.uint8)
    beliefmap1 = (image_beliefmap_stack[3:9].transpose([1,2,0])*255).astype(np.uint8)  
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)    
    height, width = image1.shape[:2]
    beliefmap1_single_channel = np.zeros((height, width), dtype=np.int)
    for i in range(beliefmap1.shape[-1]):
        beliefmap1_single_channel += beliefmap1[:,:,i]
    # merge beliefmap in red channel    
    image1 = image1.astype(np.int)
    image1[:,:,2] += beliefmap1_single_channel
    image1[:,:,2][np.where(image1[:,:,2]>255)] = 255
    image1 = image1.astype(np.uint8)
    cv2.imwrite(f'visualization_result/image_beliefmap_stack/{filenum:04d}.jpg', image1)

def get_pck_score(kps_pred, kps_gt, thresholds):    
    # PCK: Percentage of Correct Keypoints with in threshold pixel
    pck = []
    pck_by_threshold = []
    for threshold in thresholds:
        for i in range(len(kps_gt)):
            error_2d = np.linalg.norm(kps_pred[i] - kps_gt[i])
            if error_2d < threshold:
                pck.append(True)
            else:
                pck.append(False)
        pck_by_threshold.append(np.mean(pck)) # percentage of Trues.
    return np.array(pck_by_threshold)

def get_add_score(pos_pred, pos_gt, thresholds):
    add = []
    add_by_threshold = []
    for threshold in thresholds:
        for i in range(len(pos_gt)):
            error_3d = np.linalg.norm(pos_pred[i] - pos_gt[i])
            if error_3d < threshold:
                add.append(True)
            else:
                add.append(False)    
        add_by_threshold.append(np.mean(add))
    return np.array(add_by_threshold)

def save_metric_json(thresholds, scores, metric_type):
    thresholds = np.array(thresholds)
    scores = np.array(scores)
    if metric_type == "PCK":
        unit = '[pixels]'
    elif metric_type == "ADD":
        unit = '[mm]'
        thresholds *= 1000 # unit [mm]
    output = np.vstack((thresholds, scores))
    with open('visualization_result/metrics/'+ metric_type + '_result.json', 'w') as json_file:
        json.dump(output.tolist(), json_file)
    plt.clf()
    plt.plot(thresholds, scores)
    plt.xlabel('threshold distance '+ unit)
    plt.ylabel(metric_type)
    plt.axis([0, thresholds[-1], 0, 1])
    plt.grid()
    plt.savefig('visualization_result/metrics/'+ metric_type + '_graph.png')