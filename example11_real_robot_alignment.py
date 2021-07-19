import numpy as np
import cv2
import glob
import json

# Function to draw the axis
# Draw axis function can also be used.
def draw(img, keypoints, imgpts):
    keypoint = tuple(keypoints[0].ravel())
    img = cv2.line(img, keypoint, tuple(imgpts[0].ravel()), (0, 0, 255), 2)
    img = cv2.line(img, keypoint, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv2.line(img, keypoint, tuple(imgpts[2].ravel()), (255, 0, 0), 2)
    return img


# Load camera parameter using File storage in OpenCV
cv_file = cv2.FileStorage("experiment_data/camera_parameters/calib_result_L515_hong.yaml", cv2.FILE_STORAGE_READ)
cam_K = cv_file.getNode("camera_matrix").mat()
distortion = cv_file.getNode("dist_coeff").mat()


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cbrow = 3
cbcol = 5

ref_points = np.zeros((cbrow * cbcol, 3), np.float32)
ref_points[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)*0.044

axis = np.float32([[0.088,0,0], [0,0.088,0], [0,0,0.088]]).reshape(-1,3)

with open('experiment_data/samples/references/references_hong.json', 'r') as json_file:
    load_data = json.load(json_file)

# print(np.array(load_data['ref_points']).shape)
# print(np.array(load_data['keypoints']).shape)
ref_points = np.array(load_data['ref_points'])
keypoints = np.array(load_data['keypoints'], dtype=np.float32)
ret = True
# for i, fname in enumerate(glob.glob('experiment_data/samples/*.png')):
fname = 'experiment_data/samples/00_1_Color.png'
img = cv2.imread(fname)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (cbcol,cbrow),None)

if ret == True:
    # keypoints2 = cv2.cornerSubPix(gray,keypoints,(11,11),(-1,-1),criteria)
    # print(keypoints)
    # Draw and display the keypoints
    img = cv2.drawChessboardCorners(img, (cbcol, cbrow), keypoints, ret)
    # Find the rotation and translation vectors.
    _,rvecs, tvecs, inliers = cv2.solvePnPRansac(ref_points, keypoints, cam_K, distortion)
    print(rvecs)
    print(tvecs)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_K, distortion)

    img = draw(img,keypoints,imgpts)
    cv2.imwrite(f'experiment_data/samples/align_result/pose_3.png', img)
    # cv2.imshow('img',img)
    # k = cv2.waitKey(0) & 0xff


cv2.destroyAllWindows()