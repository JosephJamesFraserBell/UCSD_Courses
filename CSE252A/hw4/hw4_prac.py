import numpy as np
#from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import cv2
import numpy as np
# from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import imageio
from scipy.signal import convolve

def calculate_fundamental_matrix(previous_pts, current_pts):
        fundamental_matrix, mask = cv2.findFundamentalMat(
            previous_pts,
            current_pts,
            cv2.FM_RANSAC
        )

        if fundamental_matrix is None or fundamental_matrix.shape == (1, 1):
            # dang, no fundamental matrix found
            raise Exception('No fundamental matrix found')
        elif fundamental_matrix.shape[0] > 3:
            # more than one matrix found, just pick the first
            fundamental_matrix = fundamental_matrix[0:3, 0:3]

        return np.matrix(fundamental_matrix) 

def compute_fundamental(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] 

        Returns:
        Fundamental Matrix (3x3)

    """
    
    """
    Your code here
    """
    

    u1=x2[0,:]
    v1=x2[1,:]
    u2=x1[0,:]
    v2=x1[1,:]
    
    A = np.array([[u2[0]*u1[0],u2[0]*v1[0],u2[0],v2[0]*u1[0],v2[0]*v1[0],v2[0],u1[0],v1[0],1]])
    for i in range(1,x1.shape[1]):
        B = np.array([[u2[i]*u1[i],u2[i]*v1[i],u2[i],v2[i]*u1[i],v2[i]*v1[i],v2[i],u1[i],v1[i],1]])
        A = np.vstack((A,B))
    

    U, S, V = np.linalg.svd(A)
    
    F1=V.T[:,-1].reshape((3,3))
    
    U2, S2, V2 = np.linalg.svd(F1)

    S2[2]=0
    S2 = np.diag(S2)
    
    temp = np.matmul(U2,S2)
    F = np.matmul(temp,V2.T)
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    # return your F matrix
    return F

def fundamental_matrix(x1,x2):
    # Normalization of the corner points is handled here
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)
    
    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))

    return F/F[2,2]

def compute_epipole(F):
    '''
    This function computes the epipoles for a given fundamental matrix 
    and corner point correspondences
    input:
    F--> Fundamental matrix
    output:
    e1--> corresponding epipole in image 1
    e2--> epipole in image2
    '''
    #your code here
   
    e1 = scipy.linalg.null_space(F)
    e2 = scipy.linalg.null_space(F.T)
    return e1,e2

def plot_epipolar_lines(img1,img2, cor1, cor2):
    """Plot epipolar lines on image given image, corners

    Args:
        img1: Image 1.
        img2: Image 2.
        cor1: Corners in homogeneous image coordinate in image 1 (3xn)
        cor2: Corners in homogeneous image coordinate in image 2 (3xn)

    """
    
    """
    Your code here:
    """
    
    #F = calculate_fundamental_matrix(cor1.T,cor2.T)
    F = fundamental_matrix(cor1,cor2)
    fig = plt.figure()
    plt.ylim(img1.shape[0],0)
    plt.xlim(0,img1.shape[1])
    x = np.linspace(0,img1.shape[1])
    plt.imshow(img1)
    plt.scatter(cor1[0,:],cor1[1,:])
    vals1 = np.zeros(cor2.shape)
    for i in range(cor2.shape[1]):
        vals1[:,i] = np.matmul(F,cor1[:,i])
    for i in range(vals1.shape[1]):
        y = -1*x*vals1[0,i]/vals1[1,i] - vals1[2,i]/vals1[1,i]
        plt.plot(x,y,'b')
    
    
    plt.show()
    
    fig = plt.figure()
    plt.ylim(img2.shape[0],0)
    plt.xlim(0,img2.shape[1])
    x = np.linspace(0,img2.shape[1])
    plt.imshow(img2)
    plt.scatter(cor2[0,:],cor2[1,:])
    vals2 = np.zeros(cor1.shape)
    for i in range(cor1.shape[1]):
        vals2[:,i] = np.matmul(F,cor2[:,i])
    for i in range(vals2.shape[1]):
        y = (-1*x*vals2[0,i] - vals2[2,i])/vals2[1,i]
        plt.plot(x,y,'b')

    plt.show()
    
import math
def compute_matching_homographies(e2, F, im2, points1, points2):
    
    '''This function computes the homographies to get the rectified images
    input:
    e2--> epipole in image 2
    F--> the Fundamental matrix (Think about what you should be passing F or F.T!)
    im2--> image2
    points1 --> corner points in image1
    points2--> corresponding corner points in image2
    output:
    H1--> Homography for image 1
    H2--> Homography for image 2
    '''
    # calculate H2
    width = im2.shape[1]
    height = im2.shape[0]

    T = np.identity(3)
    T[0][2] = -1.0 * width / 2
    T[1][2] = -1.0 * height / 2

    e = T.dot(e2)
    e1_prime = e[0]
    e2_prime = e[1]
    if e1_prime >= 0:
        alpha = 1.0
    else:
        alpha = -1.0

    R = np.identity(3)
    R[0][0] = alpha * e1_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[0][1] = alpha * e2_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[1][0] = - alpha * e2_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[1][1] = alpha * e1_prime / np.sqrt(e1_prime**2 + e2_prime**2)

    f = R.dot(e)[0]/R.dot(e)[2]
    G = np.identity(3)
    G[2][0] = - 1.0 / f

    H2 = np.linalg.inv(T).dot(G.dot(R.dot(T)))

    # calculate H1
    e_prime = np.zeros((3, 3))
    e_prime[0][1] = -e2[2]
    e_prime[0][2] = e2[1]
    e_prime[1][0] = e2[2]
    e_prime[1][2] = -e2[0]
    e_prime[2][0] = -e2[1]
    e_prime[2][1] = e2[0]

    v = np.array([1, 1, 1])
    M = e_prime.dot(F) + np.outer(e2, v)

    points1_hat = H2.dot(M.dot(points1.T)).T
    points2_hat = H2.dot(points2.T).T

    W = points1_hat / points1_hat[:, 2].reshape(-1, 1)
    b = (points2_hat / points2_hat[:, 2].reshape(-1, 1))[:, 0]

    # least square problem
    a1, a2, a3 = np.linalg.lstsq(W, b, rcond=None)[0]
    HA = np.identity(3)
    HA[0] = np.array([a1, a2, a3])

    H1 = HA.dot(H2).dot(M)
    return H1, H2

def computeH(source_points, target_points):
    # returns the 3x3 homography matrix such that:
    # np.matmul(H, source_points) = target_points
    # where source_points and target_points are expected to be in homogeneous
    # make sure points are 3D homogeneous
    assert source_points.shape[0]==3 and target_points.shape[0]==3
    #compute H^-1
    source_x1x2x3 = source_points[:,:3]
    source_x4 = source_points[:,-1:]
    source_x1x2x3_inv = np.linalg.inv(source_x1x2x3)
    source_lambdas = np.matmul(source_x1x2x3_inv,source_x4)
    diag_source_lambdas = np.array([[source_lambdas[0,0], 0, 0], [0, source_lambdas[1,0], 0], [0, 0, source_lambdas[2,0]]])
    H_inv1 = np.matmul(source_x1x2x3,diag_source_lambdas)
    #computer H^-2
    target_x1x2x3 = target_points[:,:3]
    target_x4 = target_points[:,-1:]
    target_x1x2x3_inv = np.linalg.inv(target_x1x2x3)
    target_lambdas = np.matmul(target_x1x2x3_inv,target_x4)
    diag_target_lambdas = np.array([[target_lambdas[0,0], 0, 0], [0, target_lambdas[1,0], 0], [0, 0, target_lambdas[2,0]]])
    H_inv2 = np.matmul(target_x1x2x3,diag_target_lambdas)
    
    H_mtx = np.matmul(H_inv2,np.linalg.inv(H_inv1))

    return  H_mtx

def to_homog(points): #here always remember that points is a 3x4 matrix
    return np.vstack((points, points[0].size * [1]))
    
# convert points from homogeneous to euclidian
def from_homog(points_homog):  
    return  points_homog / points_homog[-1,:]

    

def warp(source_img, H, target_size):

    assert target_size[2]==source_img.shape[2]
    
    target_img = np.zeros(target_size, dtype=int)
    
    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            x_coords = [i] 
            y_coords = [j]
            source_points = np.vstack((x_coords, y_coords))
            mp = from_homog(np.matmul(H,to_homog(source_points)))
            tx = int(math.floor(mp[0]))
            ty = int(math.floor(mp[1]))
            if tx < target_size[0] and tx > 0 and ty < target_size[1] and ty > 0:
                target_img[tx,ty] = source_img[i,j]

    return target_img

def image_rectification(im1,im2,points1,points2):
    '''this function provides the rectified images along with the new corner points as outputs for a given pair of 
    images with corner correspondences
    input:
    im1--> image1
    im2--> image2
    points1--> corner points in image1
    points2--> corner points in image2
    outpu:
    rectified_im1-->rectified image 1
    rectified_im2-->rectified image 2
    new_cor1--> new corners in the rectified image 1
    new_cor2--> new corners in the rectified image 2
    '''
    "your code here"
#####################################################################################
#####################################################################################
#TRIED USING INVERSE HOMOGRAPHY BUT STILL GETTING STRIATIONS AND EPIPOLAR LINES
#GOT MESSED UP - IT'S BEEN 5 HOURS AND IT TAKES 10 MINUTES TO RUN EACH CHANGE SO I WILL TAKE THE L
#ANY MERCY IS APPRECIATED HAVE A NICE DAY/EVENING
#####################################################################################
#####################################################################################
    F = calculate_fundamental_matrix(points1.T,points2.T)
    e1,e2 = compute_epipole(F)
    
    H1, H2 = compute_matching_homographies(e2/e2[2], F.T, im2, points2.T, points1.T)
    
    # rectified_im1 = warp(im1, H1, im1.shape)
    # rectified_im2 = warp(im2, H2, im2.shape)

    rectified_im1 = warp(im1, np.linalg.inv(H1), im1.shape)
    rectified_im2 = warp(im2, np.linalg.inv(H2), im2.shape)
    
    new_cor1 = np.zeros(points1.shape)
    new_cor2 = np.zeros(points2.shape)
    for i in range(points1.shape[1]):
        new_cor1[:,i] = np.matmul(H1,points1[:,i])
        new_cor2[:,i] = np.matmul(H2,points2[:,i])
#         new_cor1[:,i] = np.matmul(np.linalg.inv(H1),points1[:,i])
#         new_cor2[:,i] = np.matmul(np.linalg.inv(H2),points2[:,i])
                

    return rectified_im1,rectified_im2,new_cor1,new_cor2

def display_correspondence(img1, img2, corrs):
    """Plot matching result on image pair given images and correspondences

    Args:
        img1: Image 1.
        img2: Image 2.
        corrs: Corner correspondence

    """
    
    """
    Your code here.
    You may refer to the show_matching_result function
    """
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.hstack((img1, img2)), cmap='gray') 
    for p1, p2 in corrs:
        plt.scatter(p1[0], p1[1], s=35, edgecolors='b', facecolors='none')
        plt.scatter(p2[0]+img1.shape[0], p2[1], s=35, edgecolors='b', facecolors='none')
        plt.plot([p1[0], p2[0]+img1.shape[0]], [p1[1], p2[1]])

def correspondence_matching_epipole(img1, img2, corners1, F, R, NCCth):
    """Find corner correspondence along epipolar line.

    Args:
        img1: Image 1.
        img2: Image 2.
        corners1: Detected corners in image 1.
        F: Fundamental matrix calculated using given ground truth corner correspondences.
        R: NCC matching window radius.
        NCCth: NCC matching threshold.
    
    
    Returns:
        Matching result to be used in display_correspondence function

    """
    """
    Your code here.
    """
    matching = []
    for i in range(corners1.shape[0]):
        match_dict = dict()
        c1 = corners1[i,:]
        for j in range(img2.shape[1]):
            c2 = [c1[0],j]
            match = ncc_match(img1, img2, c1, c2, R)
            match_dict[match] = (tuple(c1), tuple(c2))
        max_match = max(match_dict.keys())
        matching.append(match_dict[max_match])
    return matching

I1 = imageio.imread("./p4/matrix/matrix0.png")
I2 = imageio.imread("./p4/matrix/matrix1.png")
cor1 = np.load("./p4/matrix/cor1.npy")
cor2 = np.load("./p4/matrix/cor2.npy")
I3 = imageio.imread("./p4/warrior/warrior0.png")
I4 = imageio.imread("./p4/warrior/warrior1.png")
cor3 = np.load("./p4/warrior/cor1.npy")
cor4 = np.load("./p4/warrior/cor2.npy")

# For matrix
rectified_im1,rectified_im2,new_cor1,new_cor2 = image_rectification(I1,I2,cor1,cor2)
F_new = fundamental_matrix(new_cor1, new_cor2)
plot_epipolar_lines(rectified_im1,rectified_im2,new_cor1,new_cor2)
nCorners = 10
# Choose your threshold
NCCth = 0.8
#decide the NCC matching window radius
R = 15
#detect corners using corner detector here, store in corners1
corners1 = corner_detect(rectified_im1, nCorners, smoothSTD, windowSize)
corrs = correspondence_matching_epipole(rectified_im1, rectified_im2, corners1, F_new, R, NCCth)
display_correspondence(rectified_im1, rectified_im2, corrs)

# For warrior
rectified_im3,rectified_im4,new_cor3,new_cor4 = image_rectification(I3,I4,cor3,cor4)
F_new2=fundamental_matrix(new_cor3, new_cor4)
plot_epipolar_lines(rectified_im3,rectified_im4,new_cor3,new_cor4)
# You may wish to change your NCCth and R for warrior here.
corners2 = corner_detect(rectified_im3, nCorners, smoothSTD, windowSize)
corrs = correspondence_matching_epipole(rectified_im3, rectified_im4, corners2, F_new2, R, NCCth)
display_correspondence(rectified_im3, rectified_im4, corrs)