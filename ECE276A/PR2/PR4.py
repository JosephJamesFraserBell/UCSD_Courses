# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:12:28 2020

@author: Joseph
"""

import load_data as ld
import p2_utils as p2util
import numpy as np
import matplotlib.pyplot as plt

j0 = ld.get_joint("joint/train_joint0")
l0 = ld.get_lidar("lidar/train_lidar0")
r0 = ld.get_rgb("cam/RGB_0")
d0 = ld.get_depth("cam/DEPTH_0")
exIR_RGB = ld.getExtrinsics_IR_RGB()
IRCalib = ld.getIRCalib()
RGBCalib = ld.getRGBCalib()

def visualize_grid_map(grid_map):

    grid_map = grid_map - np.min(grid_map)
    grid_map = grid_map/np.max(grid_map)
    grid_map = grid_map * 255.0
    
    return grid_map
    
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
    b_rot_l = b_pitch_l @ b_yaw_l
        
    T = np.append(b_rot_l, b_translate_l, axis=1)
    T_lidar_to_body = np.append(T, np.array([[0, 0, 0, 1]]), axis=0)
        
    w_yaw_b = np.array([[np.cos(world_angle), -1*np.sin(world_angle), 0],[np.sin(world_angle), np.cos(world_angle), 0],[0, 0, 1]])
    w_translate_b = np.array([[world_x],[world_y],[com]])
    
    T2 = np.append(w_yaw_b, w_translate_b, axis=1)
    T_body_to_world = np.append(T2, np.array([[0, 0, 0, 1]]), axis=0)
        
    world_coords = T_body_to_world @ T_lidar_to_body @ cartesian_coords
    
    return world_coords

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
    b_rot_l = b_pitch_l @ b_yaw_l
        
    T = np.append(b_rot_l, b_translate_l, axis=1)
    T_lidar_to_body = np.append(T, np.array([[0, 0, 0, 1]]), axis=0)
        
    w_yaw_b = np.array([[np.cos(world_angle), -1*np.sin(world_angle), 0],[np.sin(world_angle), np.cos(world_angle), 0],[0, 0, 1]])
    w_translate_b = np.array([[world_x],[world_y],[com]])
    
    T2 = np.append(w_yaw_b, w_translate_b, axis=1)
    T_body_to_world = np.append(T2, np.array([[0, 0, 0, 1]]), axis=0)
        
    world_coords = T_body_to_world @ T_lidar_to_body @ cartesian_coords
    
    return world_coords

num_of_scans = len(l0)
lidar_angles = np.reshape(np.linspace(-135,135,1081)*np.pi/180,(1,1081)) # radian increments for lidar
head_ts = j0['ts']
num_head_ts = head_ts.shape[1]
neck_angles = j0['head_angles'][0,:]
head_angles = j0['head_angles'][1,:]
pose = np.zeros((1,3))
world_coords = np.ones((4,1))
neck_angle = 0.0
head_angle = 0.0
poses = np.zeros((1,3))
step_size = 100
head_ts = j0['ts'][0]
for i in range(num_of_scans):
    lidar_ts = l0[i]['t'][0,0]
    pose = pose + l0[i]['delta_pose']
    poses = np.append(poses, pose, axis=0)
    
poses = poses[1:,:]
print("scanning")
for i in range(int(num_of_scans/step_size)):
    lidar_ts = l0[i*step_size]['t'][0,0]
    pose = np.reshape(poses[i*step_size,:], (1,3))
    lidar_coords = l0[i*step_size]['scan']
    num_of_lidar_coords = lidar_coords.shape[1]
    
    index = abs((head_ts - lidar_ts)).argmin()
        
    neck_angle = neck_angles[index]
    head_angle = head_angles[index]
    
    for r in range(num_particles+1):
        pose_p = np.reshape(particles[r,:], (1,3))
        lidar_x = np.multiply(lidar_coords, np.cos(lidar_angles))
        lidar_y = np.multiply(lidar_coords, np.sin(lidar_angles))
        lidar_z = np.zeros((1,lidar_coords.shape[1]))
        lidar_scans = np.append(lidar_x, lidar_y, axis=0)
        lidar_scans = np.append(lidar_scans, lidar_z, axis=0)
        x_im = np.arange(0,grid_map_binary.shape[0], 1)
        y_im = np.arange(0,grid_map_binary.shape[1], 1)
        x_range = np.arange(pose_p[0,0]-0.2, pose_p[0,0]+0.25, 0.05)
        y_range = np.arange(pose_p[0,1]-0.2, pose_p[0,1]+0.25, 0.05)

        c = mapCorrelation2(grid_map_binary, x_im, y_im, lidar_scans, x_range, y_range, cell_dim, shift_origin)
        temp_weights[r] = np.max(c)
    
    world_coord = world_T_body_lidar2(neck_angle, head_angle, lidar_coords, pose, lidar_angles)
    world_coords = np.append(world_coords, world_coord, axis=1)
    





cell_dim = 0.05
map_dim = 30
grid_dim = int(map_dim/cell_dim)
grid_map = np.zeros((grid_dim, grid_dim))
shift_origin = grid_map.shape[0]/2
poses = poses[1:,:]
world_coords = world_coords[:,1:]
print("mapping")
print(int(poses.shape[0]/step_size))
try:
    for i in range(int(poses.shape[0]/step_size)):
        print(i)
        robot_x = poses[i*step_size,0]
        robot_y = poses[i*step_size,1]
        for j in range(i*1081,i*1081+1081):
            x = world_coords[0,j]
            y = world_coords[1,j]
            z = world_coords[2,j]
            if z > 1/10:
                r1 = p2util.bresenham2D(robot_x/cell_dim, robot_y/cell_dim, x/cell_dim, y/cell_dim)
                x_r1 = r1[0]
                y_r1 = r1[1]
                if len(x_r1) > 500:
                    continue
                hit_x = r1[0][-1]
                hit_y = r1[1][-1]
                
                for m in range(len(x_r1)):
                    if x_r1[m] + shift_origin < grid_map.shape[0] and y_r1[m] + shift_origin < grid_map.shape[1]:
                        free_grid_x = int(x_r1[m] + shift_origin)
                        free_grid_y = int(y_r1[m] + shift_origin)
                        grid_map[free_grid_x, free_grid_y] = grid_map[free_grid_x, free_grid_y] - np.log(4)
                    
                hit_grid_x = int(hit_x+shift_origin)
                hit_grid_y = int(hit_y+shift_origin)
                if hit_grid_x < grid_map.shape[0] and hit_grid_y < grid_map.shape[1]:
                    grid_map[hit_grid_x, hit_grid_y] = grid_map[hit_grid_x, hit_grid_y] + 2*np.log(4)
                
except IndexError:
    print("index error")
            




