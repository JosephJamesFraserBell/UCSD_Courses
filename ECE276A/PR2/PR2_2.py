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

grid_map, grid_map_binary = initializeMap(l0, grid_map, grid_map_binary, cell_dim, shift_origin)
grid_map = np.clip(grid_map, -20, 20)

for i in range(num_of_scans):
    lidar_ts = l0[i]['t'][0,0]
    pose = pose + l0[i]['delta_pose']
    poses = np.append(poses, pose, axis=0)
    
poses = poses[1:,:]
for i in range(int(num_of_scans/step_size)):
    lidar_ts = l0[i+step_size]['t'][0,0]
    pose = np.reshape(poses[i+step_size,:], (1,3))
    lidar_coords = l0[i+step_size]['scan']
    num_of_lidar_coords = lidar_coords.shape[1]
    
    for j in range(num_head_ts):
        head_ts = j0['ts'][0,j]
        if lidar_ts == head_ts:
            neck_angle = neck_angles[j]
            head_angle = head_angles[j]
            break
#    for k in range(num_of_lidar_coords):
#        lidar_coord = lidar_coords[0,k]
#        if lidar_coord > 29:
#            continue
#        lidar_angle = lidar_angles[0,k]
#        
#        world_coord = world_T_body_lidar(neck_angle, head_angle, lidar_coord, pose, lidar_angle)
#        world_coords = np.append(world_coords, world_coord, axis=1)
    world_coord = world_T_body_lidar2(neck_angle, head_angle, lidar_coords, pose, lidar_angles)
    world_coords = np.append(world_coords, world_coord, axis=1)
    





cell_dim = 0.05
map_dim = 30
grid_dim = int(map_dim/cell_dim)
grid_map = np.zeros((grid_dim, grid_dim))
shift_origin = grid_map.shape[0]/2
poses = poses[1:,:]
world_coords = world_coords[:,1:]
try:
    for i in range(poses.shape[0]):
        robot_x = poses[i,0]
        robot_y = poses[i,1]
        for j in range(i*1081,i*1081+1081):
            x = world_coords[0,j]
            y = world_coords[1,j]
            z = world_coords[2,j]
            if z > 1/100:
                r1 = p2util.bresenham2D(robot_x/cell_dim, robot_y/cell_dim, x/cell_dim, y/cell_dim)
                x_r1 = r1[0]
                y_r1 = r1[1]
                
                hit_x = r1[0][-1]
                hit_y = r1[1][-1]
                
                for m in range(len(x_r1)):
                    free_grid_x = int(x_r1[m] + shift_origin)
                    free_grid_y = int(y_r1[m] + shift_origin)
                    grid_map[free_grid_y, free_grid_x] = grid_map[free_grid_y, free_grid_x] - np.log(4)
                    
                hit_grid_x = int(hit_x+shift_origin)
                hit_grid_y = int(hit_y+shift_origin)
                
                grid_map[hit_grid_y, hit_grid_x] = grid_map[hit_grid_y, hit_grid_x] + 2*np.log(4)
except IndexError as e:
    print("index error")
            
            #grid_map[hit_grid_y, hit_grid_x] = 255
#grid_map = np.clip(grid_map, 0, np.max(grid_map))
#norm_grid_map = visualize_grid_map(grid_map)

#plt.imshow(grid_map, cmap='gray', origin='lower')


#x_poses = poses[:,0]
#x_poses = x_poses/cell_dim + shift_origin
#x_poses = [ int(x) for x in x_poses ]
#y_poses = poses[:,1]
#y_poses = y_poses/cell_dim + shift_origin
#y_poses = [ int(y) for y in y_poses ]
#plt.scatter(x_poses, y_poses, s=0.01, c='r')
#plt.show()

#x_vals = []
#y_vals = []
#for i in range(world_coords.shape[1]):
#    x = world_coords[0,i]
#    y = world_coords[1,i]
#    z = world_coords[2,i]
#    
#    if z > 1/100:
#        x_vals.append(x/cell_dim + shift_origin)
#        y_vals.append(y/cell_dim + shift_origin)
#plt.scatter(x_vals, y_vals)
#plt.show()
#plt.matshow(grid_map, cmap='gray', origin='lower')

#convert scan coordinates to world coordinates
# Center of Mass: kept at 0.93 meters
# Head: 33.0 cm above Center of Mass
# Lidar: 15.0 cm above Head
# Kinect: 7.0 cm above Head 

# visualize data

#ld.replay_lidar(l0[:500])
#ld.replay_rgb(r0[:20])
#ld.replay_depth(d0[:5])

