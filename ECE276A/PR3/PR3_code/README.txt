#####################################
###### README for ECE 276A PR3 ######
###### 	     Joseph Bell       ######
#####################################

Number of files: 2
File 1: hw3_main_part_ab.py
File 2: hw3_main_part_c.py


~~~~~~~~~~
 File 1 
~~~~~~~~~~

hw3_main_part_ab.py
~~~~~~~~~~~~~~~~~~~

This is the code used for parts A and B of project 3. Below are the functions and 
their descriptions. 

skew(vec): 
input: vec - np.array of shape (3,) 
return: skew symmetric matrix of input vector

u_hat(rv, v):
input: rv - rotational velocity np.array of shape (3,) v - linear velocity np.array of shape (3,)
return: u_hat matrix used for prediction step

u_curl_hat(rv, v):
input: input: rv - rotational velocity np.array of shape (3,) v - linear velocity np.array of shape (3,)
return: u_curl_hat matrix used for prediction step 3(not sure the legitimate name for this)

M_matrix3by4(K, b):
input: K - left camera intrinsic matrix that was provided, b - stereo camera baseline parameter that was provided
return: M matrix (in the simplified 3 by 4 shape) used for converting pixel coodinates to world coordinates

M_matrix4by4(K, b):
input: K - left camera intrinsic matrix that was provided, b - stereo camera baseline parameter that was provided
return: M matrix (in the non-simplified 4 by 4 shape) used for converting world coordinates to pixel coordinates
(note: 4 by 4 matrix was calculated as the 3 by 4 couldn't be used in a specific matrix calculation due to dimensions)

world_coordinate(K, b, feature, cam_T_imu, pose):
input: K - left camera intrinsic matrix that was provided, b - stereo camera baseline parameter that was provided,
cam_T_imu - extrinsic matrix from IMU to left camera in SE(3) that was provided, pose - current pose in SE(3), 
feature - a feature in pixel coordinates
return: feature converted to landmark in world coordinates

pixel_coordinate(K, b, w_coord, cam_T_imu, pose)
input: K - left camera intrinsic matrix that was provided, b - stereo camera baseline parameter that was provided,
w_coord - world coordinate of landmark with shape (3,), cam_T_imu - extrinsic matrix from IMU to left camera in 
SE(3) that was provided, pose - current pose in SE(3)
return: landmark converted to feature in pixel coordinates

Hjacobian(mu, K, b, pose, cam_T_imu):
input: mu - pose mean in SE(3), K - left camera intrinsic matrix that was provided, b - stereo camera baseline parameter that was provided,
cam_T_imu - extrinsic matrix from IMU to left camera in SE(3) that was provided
return: Jacobian matrix used for the mapping step




~~~~~~~~~~
 File 2
~~~~~~~~~~

hw3_main_part_c.py
~~~~~~~~~~~~~~~~~~~

This is the code used for part C of project 3. Below are the functions and 
their descriptions. This step stems from parts a and b so much of the code is 
the same. There is only one additional function implemented for the pose
update step.

Hjacobian2(m, K, b, pose, cam_T_imu):
input: m - landmark world coordinate with shape (3,), K - left camera intrinsic matrix that was provided, b - stereo camera baseline parameter that was provided,
cam_T_imu - extrinsic matrix from IMU to left camera in SE(3) that was provided
return: Jacobian matrix used for the pose update step
