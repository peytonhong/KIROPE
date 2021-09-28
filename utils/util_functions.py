from unicodedata import digit
import numpy as np
import cv2
import os
import json
import pybullet as p
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import imageio
from glob import glob
import torchvision.transforms as T

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
                            digital_twin,
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
    
    robot_pos_3d = digital_twin.jointWorldPosition_pred
    
    # draw human pose skeleton lines if it exists
    folder, file_name = os.path.split(image_path_1)
    parent_folder, cam_foler = os.path.split(folder)
    file_name = file_name[:4] + '.json'
    human_pose_path = os.path.join(parent_folder, 'human_pose', file_name)
    collision_index = np.array([[],[]])
    if os.path.exists(human_pose_path):
        with open(human_pose_path, 'r') as json_file:
            human_pos_json = json.load(json_file)
        human_pos_3d = np.array(human_pos_json['joint_3d_positions'])
        collision_index = get_collision_index(robot_pos_3d, human_pos_3d, threshold=0.3)
        draw_lines_human(image_1, human_pos_3d, cam_K_1, cam_RT_1, collision_index)
        draw_lines_human(image_2, human_pos_3d, cam_K_2, cam_RT_2, collision_index)
    draw_lines_robot(image_1, robot_pos_3d, cam_K_1, cam_RT_1, collision_index)
    draw_lines_robot(image_2, robot_pos_3d, cam_K_2, cam_RT_2, collision_index)

    image_stack = np.hstack((image_2, image_1))
    cv2.putText(image_stack, 'CAM1', (width+10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.putText(image_stack, 'CAM2', (10,30),       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.imwrite(f'visualization_result/stacked_{image_path_1[-8:]}', image_stack)
    
    # get_robot_image(digital_twin, cam_K_2, cam_RT_2)
    

def get_collision_index(robot_pos_3d, human_pos_3d, threshold=0.3):
    # calculate Euclidean distance between each object's joints
    robot_collision_index = []
    human_collision_index = []
    for r in range(len(robot_pos_3d)):
        for h in range(len(human_pos_3d)):
            distance = np.linalg.norm(robot_pos_3d[r]-human_pos_3d[h]) # Euclidean distance
            if distance < threshold:
                robot_collision_index.append(r)
                human_collision_index.append(h)
    return np.stack((robot_collision_index, human_collision_index)) # [2, num_collisions]

def draw_lines_robot(image, p3d, cam_K, cam_RT, collision_index):
    keypoints = []
    for point_3d in p3d:
        point_3d = np.append(point_3d, [1])
        keypoint = cam_K @ cam_RT @ point_3d.reshape(-1,1)
        keypoint /= keypoint[-1]
        keypoints.append(keypoint)
    joint_connections = [[0,1],[1,2],[2,3],[3,4],[4,5]]
    keypoints = np.array(keypoints).reshape(-1,3)
    for _c in joint_connections:
        if _c[0] in collision_index[0,:] or _c[1] in collision_index[0,:]:
            color=(0,0,255)
        else:
            color=(0,255,0)
        cv2.line(image, (int(keypoints[_c[0]][0]),int(keypoints[_c[0]][1])), (int(keypoints[_c[1]][0]),int(keypoints[_c[1]][1])), color, thickness=2)
    
def draw_lines_human(image, p3d, cam_K, cam_RT, collision_index):
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
        cv2.circle(image, (keypoint[0], keypoint[1]), radius=2, color=(255,255,255), thickness=2)
    joint_connections = [[0,1], [0,2], [0,11], [11,3], [11,4], [3,5], [4,6], [5,7], [6,8], [3,9], [4,10], [9,10]]
    keypoints = np.array(keypoints).reshape(-1,3)
    
    for _c in joint_connections:
        if _c[0] in collision_index[1,:] or _c[1] in collision_index[1,:]:
            color=(0,0,255)
        else:
            color=(235,206,135)
        cv2.line(image, (int(keypoints[_c[0]][0]),int(keypoints[_c[0]][1])), (int(keypoints[_c[1]][0]),int(keypoints[_c[1]][1])), color, thickness=2)

def get_robot_image(digital_twin, cam_K, cam_RT):
    cam_K = cam_K.reshape(-1)
    fov = digital_twin.fov*0.5
    aspect = digital_twin.width/digital_twin.height
    nearVal = 0.1
    farVal = 10    
    cam_K_opengl = np.array([[1/(aspect*np.tan(fov/2)), 0, 0, 0],
                             [0, 1/np.tan(fov/2), 0, 0],
                             [0, 0, (nearVal+farVal)/(nearVal-farVal), 2*nearVal*farVal/(nearVal-farVal)],
                             [0, 0, -1, 0]]).reshape(-1)
    cam_intrinsic = p.computeProjectionMatrixFOV(fov=fov*180/np.pi, # [view angle in degree]
                                            aspect=digital_twin.width/digital_twin.height,
                                            nearVal=0.1,
                                            farVal=100,
                                            )
    cam_pos = -cam_RT[:,:3].transpose() @ cam_RT[:,-1]
    cam_RT = cam_RT.transpose().reshape(-1)
    
    camera_struct_look_at_1 = {
        'at':[0,0,0],
        'up':[0,0,1],
        'eye':cam_pos.tolist()
    }
    cam_extrinsic = p.computeViewMatrix(cameraEyePosition=camera_struct_look_at_1['eye'],
                                cameraTargetPosition=camera_struct_look_at_1['at'],
                                cameraUpVector=camera_struct_look_at_1['up'],
                                )
    print(cam_pos)
    print(cam_extrinsic)
    image_arr = p.getCameraImage(digital_twin.width,
                            digital_twin.height,
                            viewMatrix=cam_extrinsic,
                            projectionMatrix=cam_intrinsic,
                            shadow=1,
                            lightDirection=[1, 1, 1],
                            renderer=p.ER_BULLET_HARDWARE_OPENGL, #p.ER_TINY_RENDERER, #p.ER_BULLET_HARDWARE_OPENGL
                            physicsClientId=digital_twin.physicsClient_main
                            )


    image = np.array(image_arr[2]) # [height, width, 4]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.array(image_arr[4])
    cv2.imwrite('sample.jpg', image)
    print(mask.shape)

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

def image_keypoint_augmentation(image_path, keypoint):
    # image and keypoint augmentation
    # order: scale -> rotate -> translate_xy
    coco_dir='annotation/coco/val2017'
    coco_paths = sorted(glob(os.path.join(coco_dir, '*.jpg')))

    image = imageio.imread(image_path)
    height, width = image.shape[:2]
    coco_image = imageio.imread(np.random.choice(coco_paths))
    coco_image = ia.imresize_single_image(coco_image, (height, width))
    if len(coco_image.shape) == 2:
        coco_image = cv2.cvtColor(coco_image, cv2.COLOR_GRAY2BGR)

    kps_wh = np.array(keypoint)
    kps = [Keypoint(x=kps_wh[i][0], y=kps_wh[i][1]) for i in range(len(kps_wh))]

    kpsoi = KeypointsOnImage(kps, shape=(height, width))
    kps_is_valid = False
    while not kps_is_valid:
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
            kps_is_valid = True

    image_aug[np.where(image_aug==0)] = coco_image[np.where(image_aug==0)]
    hue_saturation = iaa.AddToHueAndSaturation((-50, 50))
    image_aug = hue_saturation(image=image_aug)
    image_transform = T.Compose([
            # T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    image_aug = image_transform(image_aug)
    rotation_scale_translate = (rot_angle, scale_val, trans_x, trans_y)
    
    return image_aug, kpsoi_aug.to_xy_array(), rotation_scale_translate

def augmentation_recovery(keypoint_aug, aug_param):
    # Recover the augmented keypoint
    # order: translate_xy -> rotate -> scale
    rot_angle, scale_val, trans_x, trans_y = aug_param
    translate_inv = iaa.Affine(translate_px={"x": -trans_x, "y": -trans_y})
    rotate_inv = iaa.Affine(rotate=-rot_angle)
    scale_inv = iaa.Affine(scale=1/scale_val)
    kps_recovered = translate_inv(keypoints=keypoint_aug)
    kps_recovered = rotate_inv(keypoints=kps_recovered)
    kps_recovered = scale_inv(keypoints=kps_recovered)

    return kps_recovered.to_xy_array()
