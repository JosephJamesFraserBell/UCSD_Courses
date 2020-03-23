import numpy as np
from utils import *
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def skew(vec):
    skew_matrix = np.array([[0, -1*vec[2], vec[1]],
                            [vec[2], 0, -1*vec[0]],
                            [-1*vec[1], vec[0], 0]])
    return skew_matrix

def u_hat(rv, v):
    rv_skew = skew(rv)
    v = np.reshape(v, (3,1))
    padding = np.zeros((1,4))
    
    u_hat_matrix = np.append(rv_skew, v, axis=1)
    u_hat_matrix = np.append(u_hat_matrix, padding, axis=0)
    
    return u_hat_matrix

def u_curl_hat(rv, v):
    rv_skew = skew(rv)
    v_skew = skew(v)
    padding = np.zeros((3,3))
    
    top_half = np.append(rv_skew, v_skew, axis=1)
    bottom_half = np.append(padding, rv_skew, axis=1)
    u_curl_hat_matrix = np.append(top_half, bottom_half, axis=0)
    
    return u_curl_hat_matrix

def M_matrix3by4(K, b):
    M = np.array([[K[0,0], 0, K[0,2], 0],[0, K[1,1], K[1,2], 0],[0, 0, 0, K[0,0]*b]])
    return M

def M_matrix4by4(K, b):
    M = np.array([[K[0,0], 0, K[0,2], 0],[0, K[1,1], K[1,2], 0],[K[0,0], 0, K[0,2], -1*K[0,0]*b],[0, K[1,1], K[1,2], 0]])
    return M
#def world_coordinates(M, features, cam_T_imu, poses):
#    M_inv = np.linalg.inv(M)
#    imu_T_cam = np.linalg.inv(cam_T_imu)
#    world_coords = np.zeros((features.shape[0], features.shape[1], 1))
#
#    for i in range(poses.shape[2]):
#        world_T_imu = np.linalg.inv(poses[:,:,i])
#        pixels = features[:,:,i]
#        cam_coords = M_inv @ pixels
#        cam_coords = cam_coords/cam_coords[3,:]
#        imu_coords = imu_T_cam @ cam_coords
#        world_coord = world_T_imu @ imu_coords
#        world_coord = world_coord[:,:,None]
#        world_coords = np.append(world_coords, world_coord, axis=2)
#    
#    return world_coords[:,:,1:]

def world_coordinate(K, b, feature, cam_T_imu, pose):
    f_su = K[0,0]
    f_sv = K[1,1]
    cu = K[0,2]
    cv= K[1,2]
    
    ul = feature[0]
    vl = feature[1]
    
    d = feature[0] - feature[2]
    z = f_su*b/d
    y = (vl-cv)/f_sv*z
    x = (ul-cu)/f_su*z
    
    cam_coords = np.array([[x],[y],[z],[1]])
    world_coords = np.linalg.inv(cam_T_imu @ pose) @ cam_coords
    
    return world_coords

def pixel_coordinate(K, b, w_coord, cam_T_imu, pose): #M is the 3by4 version
    w_coord = np.append(w_coord,1)[:,None]
    M = M_matrix3by4(K,b)
    cam_coords = cam_T_imu @ pose @ w_coord
    cam_coords = cam_coords / cam_coords[2,0]
    pixel_coord_3by1 = M @ cam_coords
    
    ul = pixel_coord_3by1[0,0]
    vl = pixel_coord_3by1[1,0]
    d = pixel_coord_3by1[2,0]
    ur = ul-d
    
    pixel_coords = np.array([[ul],[vl],[ur],[vl]])

    return pixel_coords

def Hjacobian(mu, K, b, pose, cam_T_imu):
    P = np.identity(3)
    P = np.append(P, np.array([[0],[0],[0]]), axis=1)
    M = M_matrix4by4(K,b)
    mu = np.append(mu,1)[:,None]
    q = cam_T_imu @ pose @ mu
    q1= q[0,0]
    q2 = q[1,0]
    q3 = q[2,0]
    q4 = q[3,0]
    
    deriv_q = np.array([[1/q3, 0, -q1/q3*1/q3, 0],[0, 1/q3, -q2/q3*1/q3, 0],[0, 0, 0, 0],[0, 0, -q4/q3*1/q3, 1/q3]])
    
    H = M @ deriv_q @ cam_T_imu @ P.T
    
    return H
    
poses = np.zeros((4, 4, 1))
pose_covs = np.zeros((6, 6, 1))

mu_prior = np.identity(4)
cov_prior = np.ones((6,6))*0.1
time_step = 1
if __name__ == '__main__':
    filename = "./data/0027.npz"
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    mu_t = mu_prior
    cov_t = cov_prior

    ### PREDICTION ###
    for i in range(int(t.shape[1]/time_step - time_step)): #1106-1=1105 
        pt = i*time_step
        lt = t[0,pt]
        ct = t[0,pt+time_step]
        tau = ct-lt
        
        
        vel = linear_velocity[:,pt+time_step] 
        rot_vel = rotational_velocity[:,pt+time_step]
        twist = np.vstack((vel,rot_vel))
        
        u_hat_m = u_hat(rv=rot_vel, v=vel)
        u_c_hat_m = u_curl_hat(rv=rot_vel, v=vel)
        
        noise = 0
        mu_t1 = np.matmul(expm(-1*tau*u_hat_m), mu_t)
        cov_t1 = np.matmul(np.matmul(expm(-1*tau*u_c_hat_m), cov_t),expm(-1*tau*u_c_hat_m).T) + noise
        poses = np.append(poses, np.reshape(mu_t1, (4,4,1)), axis=2)
        pose_covs = np.append(pose_covs, np.reshape(cov_t1, (6,6,1)), axis=2)
        
        mu_t = mu_t1
        cov_t = cov_t1
        
    
    
    ### MAPPING ###
    
    obs_noise_var = 1
    obs_noise = np.random.normal(0, np.sqrt(obs_noise_var), 4)
    poses = poses[:,:,1:]
    pose_covs = pose_covs[:,:,1:]
    features = features[:,:,1:]
    num_features = features.shape[1]
    first_sighting = list(np.linspace(0,num_features-1,num_features, dtype=int))
    
    map_mu = np.zeros((3, features.shape[1])) #4,M
    map_cov = np.ones((3, 3, features.shape[1]))*0.1 #3,3,M
    
    for i in range(poses.shape[2]):
        pose = poses[:,:,i]
        for j in range(num_features):
            feature = features[:,j,i]
            if np.array_equal(feature, np.array([-1., -1., -1., -1.])): #feature not visible so skip this loop
                continue
            if j in first_sighting: #need to initialize position
                wc = world_coordinate(K, b, feature, cam_T_imu, pose)
                map_mu[:,j] = np.array([wc[0,0], wc[1,0], wc[2,0]])
                first_sighting.remove(j)
            
            mu = map_mu[:,j]
            sigma = map_cov[:,:,j]
            H = Hjacobian(mu, K, b, pose, cam_T_imu)
            
            K_gain = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + np.identity(4) * obs_noise_var)
            z_tilde = pixel_coordinate(K, b, mu, cam_T_imu, pose)
            z = feature + obs_noise
            map_mu[:,j] = mu + np.reshape(K_gain @ (z[:,None] - z_tilde), mu.shape)
            map_cov[:,:,j] = (np.identity(3) - K_gain @ H) @ sigma         
    
    world_T_imu = np.zeros(poses.shape)
    for i in range(poses.shape[2]):
        world_T_imu[:,:,i] = np.linalg.inv(poses[:,:,i])
    visualize_trajectory_2d(world_T_imu, path_name="trajectory", show_ori=True)
    
    x = map_mu[0,:]
    y = map_mu[1,:]
    plt.scatter(x, y, c='#98ff98', s=0.5)