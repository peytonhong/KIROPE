# Kinematics driven Robot Pose Estimation with Monocular Camera

This work is intended to utilize the robot's Kinematics information for articulated robot's joint pose estimation.
Example1 ~ Example6 are those I have made to learn PyBullet and NVISII for automatic training data generation.
Those codes are referenced from basic tutorials of PyBullet and NVISII.

The main network comprises Resnet-50 which is used for extracting feature vectors from images, followed by transformer which is well known for handeling sequential data.

### Network Architecture
![Architecture](https://github.com/peytonhong/kirope/blob/main/docs/network_architecture.png)

* Input data
  * RGB image (single or sequential)
  * Robot Kinematics states (joint pose and velocity)
* Output
  * Robot joint 2D keypoints

### At the moment, this repository is under consturction. More codes will be updated soon.
