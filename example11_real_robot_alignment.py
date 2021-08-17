import numpy as np
import cv2
import glob
import json
import pybullet as p


# Function to draw the axis
# Draw axis function can also be used.
def draw_result(img, keypoints, imgpts):
    keypoint = tuple(keypoints[0].ravel())
    img = cv2.line(img, keypoint, tuple(imgpts[0].ravel()), (0, 0, 255), 2)
    img = cv2.line(img, keypoint, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv2.line(img, keypoint, tuple(imgpts[2].ravel()), (255, 0, 0), 2)
    return img

physicsClient = p.connect(p.GUI) # non-graphical version
robotId = p.loadURDF("urdfs/ur3/ur3_new.urdf", [0, 0, 0], useFixedBase=True)

basePose = p.getQuaternionFromEuler([0,0,-180*np.pi/180])
p.resetBasePositionAndOrientation(robotId, [0.2, 0, 0.0], basePose) # robot base offset(20cm) from checkerboard
numJoints = p.getNumJoints(robotId)

# jointAngles = np.float32([0, -90, 0, -90, 0, 0])*np.pi/180          # No.1: Home position
# jointAngles = np.float32([90, -90, 90, -180, -90, 0])*np.pi/180   # No.2
# jointAngles = np.float32([0, 0, 0, 0, 0, 0])*np.pi/180            # zero angle

with open('annotation/dataset_experiment/20210812_210147/0040.json', 'r') as json_file:
    load_data = json.load(json_file)
jointAngles = load_data['joint_angles']

for j in range(numJoints):
    p.resetJointState(bodyUniqueId=robotId,
                    jointIndex=j,
                    targetValue=(jointAngles[j]),
                    )    
p.stepSimulation()

# get joint states
joint_world_position = []
for link_num in range(len(jointAngles)):    
    link_state = p.getLinkState(bodyUniqueId=robotId, linkIndex=link_num)
    pos_world = list(link_state[4])
    rot_world = link_state[5] # world orientation of the URDF link frame
    if link_num == 4:
        rot_mat = p.getMatrixFromQuaternion(rot_world)
        rot_mat = np.array(rot_mat).reshape(3,3)
        offset = np.array([0,-0.04,0.08535])
        pos_world = rot_mat.dot(offset) + pos_world
    if link_num == 5:
        rot_mat = p.getMatrixFromQuaternion(rot_world)
        rot_mat = np.array(rot_mat).reshape(3,3)
        offset = np.array([0.0,0.0619,0])
        pos_world = rot_mat.dot(offset) + pos_world
    joint_world_position.append(pos_world) 

joint_world_position = np.array(joint_world_position)
# print(joint_world_position)
# k = input("Press anykey to continue.")


# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cbrow = 3
cbcol = 5

# ref_points = np.zeros((cbrow * cbcol, 3), np.float32)
# ref_points[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)*0.044

axis = np.float32([[0.088,0,0], [0,0.088,0], [0,0,0.088]]).reshape(-1,3)

with open('annotation/dataset_experiment/20210812_210147/references/references_cam2.json', 'r') as json_file:
    load_data = json.load(json_file)

ref_points = np.array(load_data['ref_points'])
keypoints = np.array(load_data['keypoints'], dtype=np.float32)
# for i, fname in enumerate(glob.glob('experiment_data/samples/*.png')):
fname = 'annotation/dataset_experiment/20210812_210147/cam2/0040.jpg'
img = cv2.imread(fname)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, corners = cv2.findChessboardCorners(gray, (cbcol,cbrow),None)

# keypoints2 = cv2.cornerSubPix(gray,keypoints,(11,11),(-1,-1),criteria)
# print(keypoints)
# Draw and display the keypoints
img = cv2.drawChessboardCorners(img, (cbcol, cbrow), keypoints, True)

# Load camera parameter using File storage in OpenCV
cv_file = cv2.FileStorage("experiment_data/camera_parameters/calib_result_L515_ivcl_640x480.yaml", cv2.FILE_STORAGE_READ)
cam_K = cv_file.getNode("camera_matrix").mat()
distortion = cv_file.getNode("dist_coeff").mat()

# Find the rotation and translation vectors.
retVal, rvecs, tvecs, inliers = cv2.solvePnPRansac(ref_points, keypoints, cam_K, distortion)
print(retVal)
print(rvecs)
print(tvecs)
print(inliers)
# project 3D points to image plane
imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, cam_K, distortion)
# imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, cam_K, np.zeros_like(distortion))
# robot_joints = np.float32([[0.2, 0, 0], [0.2, 0, 0.1519]])
robot_kps, jacobian = cv2.projectPoints(joint_world_position, rvecs, tvecs, cam_K, distortion)
robot_kps = robot_kps.reshape(-1,2)
img = draw_result(img, keypoints.astype(np.int32), imgpts.astype(np.int32))
for keypoint in robot_kps:
    cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), radius=5, color=(0,255,0), thickness=2)
cv2.imwrite(f'experiment_data/experiment_result/temp_result.png', img)
# cv2.imshow('img',img)
k = cv2.waitKey(0) & 0xff


cv2.destroyAllWindows()