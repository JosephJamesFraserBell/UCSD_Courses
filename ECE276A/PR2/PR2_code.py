# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:12:28 2020

@author: Joseph
"""

import load_data as ld
import p2_utils as p2util
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import cv2
from filterpy.monte_carlo import residual_resample

j0 = ld.get_joint("joint/train_joint4")
l0 = ld.get_lidar("lidar/train_lidar4")
r0 = ld.get_rgb("cam/RGB_0")
d0 = ld.get_depth("cam/DEPTH_0")
exIR_RGB = ld.getExtrinsics_IR_RGB()
IRCalib = ld.getIRCalib()
RGBCalib = ld.getRGBCalib()

# INPUT 
            # im              the map 
            # x_im,y_im       physical x,y positions of the grid map cells
            # vp(0:2,:)       occupied x,y positions from range sensor (in physical unit)  
            # xs,ys           physical x,y,positions you want to evaluate "correlation" 
            #
            # OUTPUT 
            # c               sum of the cell values of all the positions hit by range sensor



def mapCorrelation2(im, x_im, y_im, vp, xs, ys, map_res, grid_offset):
  nx = im.shape[0]
  ny = im.shape[1]
  yresolution = map_res
  xresolution = map_res
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round(y1/yresolution + grid_offset))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round(x1/xresolution + grid_offset))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr

def world_T_body_lidar2(neck_angle, head_angle, lidar_coords, pose, lidar_angles):
    com = 0.93
    b_translate_l = np.array([[0],[0],[0.33 + 0.15]])
    cartesian_coords = np.zeros((3, lidar_coords.shape[1]))
    world_x = pose[0,0]
    world_y = pose[0,1]
    world_angle = pose[0,2]
    
    
    lidar_x = np.multiply(lidar_coords, np.cos(lidar_angles))
    lidar_y = np.multiply(lidar_coords, np.sin(lidar_angles))
    
    cartesian_coords[0,:] = lidar_x
    cartesian_coords[1,:] = lidar_y
    
    #append row of 1s for homogenous
    cartesian_coords = np.append(cartesian_coords, np.ones((1,lidar_coords.shape[1])), axis=0) 
    
    b_yaw_l = np.array([[np.cos(neck_angle), -1*np.sin(neck_angle), 0],[np.sin(neck_angle), np.cos(neck_angle), 0],[0, 0, 1]])
    b_pitch_l = np.array([[np.cos(head_angle), 0, np.sin(head_angle)],[0, 1, 0],[-1*np.sin(head_angle), 0, np.cos(head_angle)]])
    #b_rot_l = b_pitch_l @ b_yaw_l
    b_rot_l = b_yaw_l @ b_pitch_l
        
    T = np.append(b_rot_l, b_translate_l, axis=1)
    T_lidar_to_body = np.append(T, np.array([[0, 0, 0, 1]]), axis=0)
        
    w_yaw_b = np.array([[np.cos(world_angle), -1*np.sin(world_angle), 0],[np.sin(world_angle), np.cos(world_angle), 0],[0, 0, 1]])
    w_translate_b = np.array([[world_x],[world_y],[com]])
    
    T2 = np.append(w_yaw_b, w_translate_b, axis=1)
    T_body_to_world = np.append(T2, np.array([[0, 0, 0, 1]]), axis=0)
        
    world_coords = np.matmul(T_body_to_world, np.matmul(T_lidar_to_body,cartesian_coords))
    
    return world_coords

def mapCorrelation3(lidar_scans, binary_map, map_res, grid_offset):
    lidar_scans = lidar_scans/map_res + grid_offset
    lidar_scans = lidar_scans.astype(int)
    lidarx = lidar_scans[0,:]
    lidary = lidar_scans[1,:]
    lidarx = lidarx[(lidarx>=0) & (lidarx < binary_map.shape[0])]
    lidary = lidary[(lidary>=0) & (lidary < binary_map.shape[1])]
    
    if lidarx.shape[0] < lidary.shape[0]:
        lidary = lidary[:lidarx.shape[0]]
    elif lidary.shape[0] < lidarx.shape[0]:
        lidarx = lidarx[:lidary.shape[0]]
    
    return np.sum(binary_map[lidarx,lidary])
  
    
def world_T_body_lidar(neck_angle, head_angle, lidar_coords, pose, lidar_angle):
    com = 0.93
    b_translate_l = np.array([[0],[0],[0.33 + 0.15]])
    cartesian_coords = np.zeros((3,1))

    world_x = pose[0,0]
    world_y = pose[0,1]
    world_angle = pose[0,2]
    
    
    lidar_x = lidar_coords*np.cos(lidar_angle)
    lidar_y = lidar_coords*np.sin(lidar_angle)
    
    cartesian_coords[0,0] = lidar_x
    cartesian_coords[1,0] = lidar_y
    
    #append row of 1s for homogenous
    cartesian_coords = np.append(cartesian_coords, np.ones((1,1)), axis=0) 
    
    b_yaw_l = np.array([[np.cos(neck_angle), -1*np.sin(neck_angle), 0],[np.sin(neck_angle), np.cos(neck_angle), 0],[0, 0, 1]])
    b_pitch_l = np.array([[np.cos(head_angle), 0, np.sin(head_angle)],[0, 1, 0],[-1*np.sin(head_angle), 0, np.cos(head_angle)]])
    #b_rot_l = b_pitch_l @ b_yaw_l
    b_rot_l = b_yaw_l @ b_pitch_l
    T = np.append(b_rot_l, b_translate_l, axis=1)
    T_lidar_to_body = np.append(T, np.array([[0, 0, 0, 1]]), axis=0)
        
    w_yaw_b = np.array([[np.cos(world_angle), -1*np.sin(world_angle), 0],[np.sin(world_angle), np.cos(world_angle), 0],[0, 0, 1]])
    w_translate_b = np.array([[world_x],[world_y],[com]])
    
    T2 = np.append(w_yaw_b, w_translate_b, axis=1)
    T_body_to_world = np.append(T2, np.array([[0, 0, 0, 1]]), axis=0)
        
    world_coords = T_body_to_world @ T_lidar_to_body @ cartesian_coords
    
    return world_coords

def initializeMap(lidar, grid_map, cell_dim, shift_origin):
    
    neck_angle = 0.0
    head_angle = 0.0
    pose = np.zeros((1,3))
    lidar_coords = lidar[0]['scan']
    num_of_lidar_coords = lidar_coords.shape[1]
    lidar_scans = np.zeros((3,1))
    for k in range(num_of_lidar_coords):#for each scan of full scan rotation
        lidar_coord = lidar_coords[0,k]
        if lidar_coord > 30: #skip this scan if it's greater than 29
            continue         #since probably nothing in the way - 30 is max
        lidar_angle = lidar_angles[k]
        world_coord = world_T_body_lidar(neck_angle, head_angle, lidar_coord, pose, lidar_angle)
        
        x = world_coord[0,0]
        y = world_coord[1,0]
        z = world_coord[2,0]
        l_scan = np.array([[x],[y],[z]])
        lidar_scans = np.append(lidar_scans, l_scan, axis=1)
        if z > 0.1:
            robot_x = pose[0,0]
            robot_y = pose[0,1]
            #bresenham of grid scaled coordinates
            r1 = p2util.bresenham2D(robot_x/cell_dim, robot_y/cell_dim, x/cell_dim, y/cell_dim)
            x_r1 = r1[0]
            y_r1 = r1[1]
            
            hit_x = r1[0][-1]
            hit_y = r1[1][-1]
            
            for m in range(len(x_r1)-1):
                free_grid_x = int(x_r1[m] + shift_origin)
                free_grid_y = int(y_r1[m] + shift_origin)
                grid_map[free_grid_y, free_grid_x] = grid_map[free_grid_y, free_grid_x] - np.log(4)
                    
            hit_grid_x = int(hit_x+shift_origin)
            hit_grid_y = int(hit_y+shift_origin)
            hit_coords = np.zeros((3,1))
            hit_coords[0,0] = hit_grid_x
            hit_coords[1,0] = hit_grid_y
            
            grid_map[hit_grid_y, hit_grid_x] = grid_map[hit_grid_y, hit_grid_x] + np.log(4)


    return grid_map

def update_log_odds_map(pose, index, lidar, grid_map, cell_dim, shift_origin, neck_angle, head_angle):

    lidar_coords = lidar[index]['scan']
    num_of_lidar_coords = lidar_coords.shape[1]
    lidar_scans = np.zeros((3,1))
    for k in range(num_of_lidar_coords):#for each scan of full scan rotation
        lidar_coord = lidar_coords[0,k]
        if lidar_coord > 30: #skip this scan if it's greater than 29
            continue         #since probably nothing in the way - 30 is max
        lidar_angle = lidar_angles[k]
        world_coord = world_T_body_lidar(neck_angle, head_angle, lidar_coord, pose, lidar_angle)
        
        x = world_coord[0,0]
        y = world_coord[1,0]
        z = world_coord[2,0]
        l_scan = np.array([[x],[y],[z]])
        lidar_scans = np.append(lidar_scans, l_scan, axis=1)
        if z > 0.1:
            robot_x = pose[0,0]
            robot_y = pose[0,1]
            #bresenham of grid scaled coordinates
            r1 = p2util.bresenham2D(robot_x/cell_dim, robot_y/cell_dim, x/cell_dim, y/cell_dim)
            x_r1 = r1[0]
            y_r1 = r1[1]
            
            hit_x = r1[0][-1]
            hit_y = r1[1][-1]
            if len(x_r1) > 500:
                continue
            for m in range(len(x_r1)-1):
                
                free_grid_x = int(x_r1[m] + shift_origin)
                free_grid_y = int(y_r1[m] + shift_origin)
                if free_grid_x < grid_map.shape[0] and free_grid_x >=0 and free_grid_y < grid_map.shape[1] and free_grid_y >=0:
                    grid_map[free_grid_y, free_grid_x] = grid_map[free_grid_y, free_grid_x] - np.log(4)

                    
            hit_grid_x = int(hit_x+shift_origin)
            hit_grid_y = int(hit_y+shift_origin)
            if hit_grid_x < grid_map.shape[0] and hit_grid_x >= 0 and hit_grid_y < grid_map.shape[1] and hit_grid_y >= 0:
                grid_map[hit_grid_y, hit_grid_x] = grid_map[hit_grid_y, hit_grid_x] + np.log(4)

    
    return grid_map
 
def resample(weights, particles):
    new_particles = np.zeros(particles.shape)
    sample_indices = residual_resample(weights)
    
    for i in range(len(sample_indices)):
        new_particles[i,:] = particles[sample_indices[i],:]
    return new_particles


########### Variables ##########################
num_of_scans = len(l0)
lidar_angles = np.linspace(-135,135,1081)*np.pi/180 # radian increments for lidar
head_ts = j0['ts']
num_head_ts = head_ts.shape[1]
neck_angles = j0['head_angles'][0,:]
head_angles = j0['head_angles'][1,:]
world_coords = np.ones((4,1))
poses = np.zeros((1,3))
step_size = 100
cell_dim = 0.05
map_dim = 40
grid_dim = int(map_dim/cell_dim)
grid_map = np.zeros((grid_dim, grid_dim))
grid_map_binary = np.zeros((grid_dim, grid_dim))

shift_origin = grid_map.shape[0]/2
pose = np.zeros((1,3))
hits = np.zeros((3,1))

num_particles = 199 #actual number is num_particles + 1
particles = np.array([[0.0,0.0,0.0]])
weights = [1/(num_particles+1)]*(num_particles+1)
temp_weights = [0]*(num_particles+1)
n_eff_l = []
neck_angle = 0.0
head_angle = 0.0
trajectoryx = []
trajectoryy = []
###############################################

###### Accumulating delta poses to get array of poses ######
for i in range(num_particles):
    particles = np.append(particles, np.array([[0.0,0.0,0.0]]), axis=0)

# intitializing grid mad
grid_map= initializeMap(l0, grid_map, cell_dim, shift_origin)
grid_map_binary = np.clip(grid_map, 0, 1)

for i in range(num_of_scans):
    lidar_ts = l0[i]['t'][0,0]
    pose = pose + l0[i]['delta_pose']
    poses = np.append(poses, pose, axis=0)
    
poses = poses[1:,:]

#go through each scan of lidar data
head_ts = j0['ts'][0]
print("Running...")
print(int(num_of_scans/step_size))
for i in range(int(num_of_scans/step_size)):
    
    lidar_ts = l0[i*step_size]['t'][0,0]
    ###################
    # PREDICTION STEP #
    ###################
    for n in range(num_particles+1):
        x_val = particles[n,0]
        y_val = particles[n,1]
        t_val = particles[n,2]
        noise = np.array([0.0,0.0,0.0])
        if abs(x_val)> 0:
            x_noise = np.random.normal(0,0.7,1)
            noise[0] = x_noise
        if abs(y_val) > 0 :
            y_noise = np.random.normal(0,0.7,1)
            noise[1] = y_noise
        if abs(t_val) > 0:
            z_noise = np.random.normal(0,0.2,1)
            noise[2] = z_noise
        
        particles[n,:] = poses[i*step_size,:] + noise

    lidar_coords = l0[i*step_size]['scan']
    num_of_lidar_coords = lidar_coords.shape[1]
    
    
    ########## Find matching joint time stamp ##########
    index = abs((head_ts - lidar_ts)).argmin()    
    neck_angle = neck_angles[index]
    head_angle = head_angles[index]
    ####################################################


     
    ###############
    # UPDATE STEP #
    ###############
    # GO THROUGH EACH PARTICLE AND CALCULATE CORRELATION     
    
    for r in range(num_particles+1):
        pose_p = np.reshape(particles[r,:], (1,3))
        lidar_scans = world_T_body_lidar2(neck_angle, head_angle, lidar_coords, pose_p, lidar_angles)
        c = mapCorrelation3(lidar_scans, grid_map_binary, cell_dim, shift_origin)
        temp_weights[r] = c
    

    
    ##### SOFT MAX #####
    temp_weights = temp_weights - np.max(temp_weights)
    weights = np.exp(temp_weights)
    weights = weights/np.sum(weights) #normalizing
    
    
    ### UPDATING BASED OFF BEST PARTICLE #####
    max_weight_index = np.argmax(weights)
    max_pose = np.reshape(particles[max_weight_index,:], (1,3))
    trajectoryx.append(int(max_pose[0,0]/cell_dim+shift_origin))
    trajectoryy.append(int(max_pose[0,1]/cell_dim+shift_origin))
    grid_map= update_log_odds_map(max_pose, i*step_size, l0, grid_map, cell_dim, shift_origin, neck_angle, head_angle)
    grid_map_binary = np.clip(grid_map, 0, 1)
    grid_map = np.clip(grid_map, -200, 200)
    if i*step_size%200 == 0:
        print(i*step_size)
        plt.imshow(grid_map, cmap='hot', origin='lower')
        plt.scatter(trajectoryx, trajectoryy, c='r', s=1.0)
        plt.show()
    
    
    ##### RESAMPLING ###### 
    weights_sq = [i**2 for i in weights]
    n_eff = (1/np.sum(weights_sq))
    n_eff_l.append(n_eff)
    if n_eff < 0.2*weights.shape[0]:
        #print("resampling")
        particles = resample(weights,particles)
        weights = [1/(num_particles+1)]*(num_particles+1)
