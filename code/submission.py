"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper as hlpr
import math
from scipy.ndimage.filters import gaussian_filter

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):

    N = pts1.shape[0]
    A = np.zeros((N, 9))
    T = np.diag([1/M, 1/M, 1])
    aug_pts1 = np.hstack((pts1,np.ones((N,1))))
    aug_pts2 = np.hstack((pts2,np.ones((N,1))))
    for i in range(N):
        xm = aug_pts1[i].reshape((3,1))
        xmp = aug_pts2[i].reshape((3,1))
        xm = T @ xm
        xmp = T @ xmp
        A[i] = (xm @ np.transpose(xmp)).reshape((1,9))
    u, s, vh = np.linalg.svd(A, full_matrices = True)
    h = vh[-1,:]
    rank3F = h.reshape((3,3), order='F')
    # enforce rank 2
    u, s, vh = np.linalg.svd(rank3F, full_matrices = True)
    s[-1] = 0.0
    rank2F = u @ np.diag(s) @ vh
    F = np.transpose(T) @ rank2F @ T
    F = hlpr.refineF(F, pts1, pts2)

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    
    # Construct matrix A
    N = pts1.shape[0]
    A = np.zeros((N, 9))
    T = np.diag([1/M, 1/M, 1])
    aug_pts1 = np.hstack((pts1,np.ones((N,1))))
    aug_pts2 = np.hstack((pts2,np.ones((N,1))))
    for i in range(N):
        xm = aug_pts1[i].reshape((3,1))
        xmp = aug_pts2[i].reshape((3,1))
        xm = T @ xm
        xmp = T @ xmp
        A[i] = (xm @ np.transpose(xmp)).reshape((1,9))

    # Find 2 vectors span null space of A
    u, s, vh = np.linalg.svd(A, full_matrices = True)   # s with shape (7,), vh with shape (9,9)
    h1 = vh[-1,:]
    F1 = h1.reshape((3,3), order='F')
    h2 = vh[-2,:]
    F2 = h2.reshape((3,3), order='F')

    # Construct and solve polynomial eqn
    poly_coef = np.zeros(4)
    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    poly_coef[3] = fun(0)
    poly_coef[2] = 2*(fun(1)-fun(-1))/3-(fun(2)-fun(-2))/12
    poly_coef[1] = 0.5*fun(1)+0.5*fun(-1)-fun(0)
    poly_coef[0] = fun(1)-poly_coef.sum()
    alpha = np.roots(poly_coef)

    Farray = []
    for a in alpha:
        F = a * F1 + (1 - a) * F2
        u, s, vh = np.linalg.svd(F, full_matrices = True)
        s[-1] = 0.0
        F = u @ np.diag(s) @ vh
        F = np.transpose(T) @ F @ T
        # F = hlpr.refineF(F, pts1, pts2)
        Farray.append(F)
    Farray = np.array(Farray)
    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.transpose(K2) @ F @ K1
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    w = np.zeros((N, 4))    # in homogeneous coordinate
    for i in range(N):
        A = np.zeros((4, 4))
        A[1] = pts1[i,1]*C1[2]-C1[1]
        A[0] = pts1[i,0]*C1[2]-C1[0]
        A[3] = pts2[i,1]*C2[2]-C2[1]
        A[2] = pts2[i,0]*C2[2]-C2[0]
        u, s, vh = np.linalg.svd(A, full_matrices = True)
        w[i] = vh[-1,:]
    
    w = np.transpose(w)
    w = w/w[-1]
    w = np.transpose(w)

    reproj1 = C1 @ np.transpose(w)
    reproj1 = reproj1/reproj1[-1]
    reproj1 = np.transpose(reproj1)[:,:-1]
    reproj2 = C2 @ np.transpose(w)
    reproj2 = reproj2/reproj2[-1]
    reproj2 = np.transpose(reproj2)[:,:-1]
    err = np.sum((reproj1 - pts1)**2) + np.sum((reproj2 - pts2)**2)
    return w[:,:-1], err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    x1 = int(x1)
    y1 = int(y1)
    win_size = 15
    search_range = 40
    # get the epipolar line : lprime = [a, b, c], ax+by+cz=0, z=1, x=-(by+c)/a
    lprime = np.dot(F, np.array([[x1], [y1], [1]]))
    h, w, _ = im2.shape
    Y = np.array(range(y1-search_range, y1+search_range))
    X = np.round(-(lprime[1]*Y + lprime[2])/lprime[0]).astype(np.int)
    isValid = (X >= win_size//2) & (X < w - win_size//2) & (Y >= win_size//2) & (Y < h - win_size//2)
    x_pts, y_pts = X[isValid], Y[isValid]
    num_pts = x_pts.shape[0]

    min_err = float('inf')
    min_idx = 0
    patch1 = im1[int(y1-win_size/2) : int(y1+win_size/2), int(x1-win_size/2) : int(x1+win_size/2),:]
    for i in range(num_pts):
        patch2 = im2[int(y_pts[i]-win_size/2) : int(y_pts[i]+win_size/2), int(x_pts[i]-win_size/2) : int(x_pts[i]+win_size/2),:]
        error = np.sum(gaussian_filter((patch1-patch2)**2, sigma = 1))
        if error < min_err:
            min_err = error
            min_idx = i
    return x_pts[min_idx], y_pts[min_idx]