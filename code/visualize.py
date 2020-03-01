'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import submission as sub
import helper as hlpr
from findM2 import findM2

if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    data_select = np.load('../data/templeCoords.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    intrinsics_data = np.load('../data/intrinsics.npz')

    N = data['pts1'].shape[0]
    M = 640

    F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
    E = sub.essentialMatrix(F8, intrinsics_data["K1"], intrinsics_data["K2"])
    M2s = hlpr.camera2(E)
    M1 = np.hstack((np.eye(3), np.zeros((3,1))))
    M2 = findM2(M2s, intrinsics_data["K1"], intrinsics_data["K2"], data['pts1'], data['pts2'])
    C1 = intrinsics_data["K1"] @ M1
    C2 = intrinsics_data["K2"] @ M2

    pts1 = np.hstack([data_select['x1'], data_select['y1']])
    pts2 = np.zeros_like(pts1)
    for i, pt in enumerate(pts1):
        x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, pt[0], pt[1])
        pts2[i] = np.array([x2, y2])

    P, err = sub.triangulate(C1, pts1, C2, pts2)
    ax = plt.axes(projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], c = P[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # np.savez('q4_2.npz', F=F8, M1=M1, M2=M2, C1=C1, C2=C2)
