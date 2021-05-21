# Kinematics Guided Robot Pose Estimation with Monocular Camera

This work is intended to utilize the robot's Kinematics information for articulated robot's joint pose estimation.
Example1 ~ Example6 are those I have made to learn PyBullet and NVISII for automatic training data generation.
Those codes are referenced from basic tutorials of PyBullet and NVISII.

The main network includes Resnet-50 as a backbone for feature extraction from input images, followed by transformer which is well known for handeling sequential data. The output joint 2D keypoint is trained with ground truth keypoint belief maps which has the same size with the input image.

### Network Architecture
![Architecture](https://github.com/peytonhong/KIROPE/blob/main/docs/network_architecture_2.png)

* Input data
  * RGB image (single or sequential)
  * Gaussian State Embeddings (joint pose and velocity from robot Kinematics)
* Output
  * Robot joint 2D keypoints
* J-PnP (PnP for Articulated Joint Alignment) : 2D keypoint -> Joint angles
* Digital Twin: PyBullet based physics simulator
### At the moment, this repository is under consturction. More codes will be updated soon.
