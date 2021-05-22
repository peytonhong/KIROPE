# Kinematics Guided Robot Pose Estimation with Monocular Camera

This work is intended to utilize the robot's Kinematics information for articulated robot's joint pose estimation.
Example1 ~ Example6 are those I have made to learn PyBullet and NVISII for automatic training data generation.
Those codes are referenced from basic tutorials of PyBullet and NVISII.

The main network includes Resnet-50 as a backbone for feature extraction from input images, followed by transpose convolution to generate keypoint belief maps. The output joint 2D keypoint is trained with ground truth keypoint belief maps which has the same size with the input image. The detected 2D keypoints are used to estimate the robot's joint angles by using J-PnP which is an iterative method simliar to conventional PnP to find the articulated joint angles.
### Contiributions of this work
* The robot kinematics information such as joint position and velocity is utilized along with RGB image to complement finding 2D keypoints of each joint.
* The Gaussian state embedding containing robot's joint angles and velocities. The stacked input RGB image and state embeddings can be naturally utilized in CNN.
* J-PnP: A variation of conventional PnP algorithm to find the robot's joint angles from corresponding 2D keypoints given that the position of camera and robot's base are fixed.
* A digital twin based architecture to deliver kinematics information into the proposed neural network. We can expect this arthitecture is robust finding joint keypoints even when the robot is occuluded by other objects.

### Network Architecture
![Architecture](https://github.com/peytonhong/KIROPE/blob/main/docs/network_architecture_new.png)

* Input data
  * RGB image (single)
  * Gaussian State Embeddings (joint pose and velocity from robot Kinematics analysis)
* Output
  * Robot joint 2D keypoints
* J-PnP (Joint PnP for Articulated Robot Joint Alignment) : 2D keypoint -> Joint angles
* Digital Twin: PyBullet based physics simulator
### At the moment, this repository is under consturction. More codes will be updated soon.
