# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:13:17 2020

@author: joseph
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

rot = exIR_RGB['rgb_R_ir']
t = np.reshape(exIR_RGB['rgb_T_ir'], (1,3))
T = np.append(rot, t.T, axis=1)
T = np.append(T, np.array([[0,0,0,1]]), axis=0)

IRCalib = ld.getIRCalib()
RGBCalib = ld.getRGBCalib()

ir_fx = IRCalib['fc'][0]
ir_fy = IRCalib['fc'][1]
ir_cc = IRCalib['cc']

rgb_fx = RGBCalib['fc'][0]
rgb_fy = RGBCalib['fc'][1]
rgb_cc = RGBCalib['cc']

d_img = d0[0]['depth']
transforms = np.zeros((d_img.shape[0], d_img.shape[1], 3))
for i in range(d_img.shape[0]):
    for j in range(d_img.shape[1]):
        z = d_img[i,j]
        x = ((j - rgb_cc[0])*z)/rgb_fx
        y = ((i - rgb_cc[1])*z)/rgb_fy
        d_point = np.array([[x],[y],[z],[1]])
        transform = np.linalg.inv(T)@d_point
        transforms[i,j,0] = transform[0,0]
        transforms[i,j,1] = transform[1,0]
        transforms[i,j,2] = transform[2,0]

ir_coords = np.zeros((transforms.shape[0], transforms.shape[1], 2))
for i in range(transforms.shape[0]):
    for j in range(transforms.shape[1]):
        x = (transforms[i,j,0] * ir_fx/transforms[i,j,2]) + ir_cc[0]
        y = (transforms[i,j,1] * ir_fx/transforms[i,j,2]) + ir_cc[1]
        x=round(x)
        y=round(y)
        ir_coords[i,j,0] = x
        ir_coords[i,j,1] = y
        
def world_T_body_lidar(lidar_x, lidar_y):
    com = 0.93
    b_translate_l = np.array([[0],[0],[0.33 + 0.15]])
    cartesian_coords = np.zeros((3,1))
    pose=np.array([[0,0,0]])
    world_x = pose[0,0]
    world_y = pose[0,1]
    world_angle = pose[0,2]
    
    neck_angle = 0.0
    head_angle = 0.0
    
    
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

world_coords = np.zeros((ir_coords.shape[0], ir_coords.shape[1], 3))
for i in range(ir_coords.shape[0]):
    for j in range(ir_coords.shape[1]):
        lidar_x = ir_coords[i,j,0]
        lidar_y = ir_coords[i,j,1]
        world_coord = world_T_body_lidar(lidar_x, lidar_y)
        
        world_coords[i,j,0] = world_coord[0,0]
        world_coords[i,j,1] = world_coord[1,0]
        world_coords[i,j,2] = world_coord[2,0]



