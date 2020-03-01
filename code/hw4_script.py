"""
Check the dimensions of function arguments
This is *not* a correctness check

Written by Chen Kong, 2018.
"""
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

    N = data['pts1'].shape[0]
    M = 640

    # 2.1
    F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
    assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'
    # hlpr.displayEpipolarF(im1, im2, F8)
    # np.savez('q2_1.npz', F=F8, M=M)
    print("q2-1:\n", F8)

    # 2.2
    # select 7 best corresponding pair
    aug_pts1 = np.hstack((data['pts1'],np.ones((N,1))))
    aug_pts2 = np.hstack((data['pts2'],np.ones((N,1))))
    xFx = np.abs(np.diag(aug_pts2 @ F8 @ np.transpose(aug_pts1)))
    best_idx = xFx.argsort()[:7]
    pts1 = data['pts1'][best_idx]
    pts2 = data['pts2'][best_idx]

    F7 = sub.sevenpoint(pts1, pts2, M)
    assert (len(F7) == 1) | (len(F7) == 3), 'sevenpoint returns length-1/3 list'

    for f7 in F7:
        assert f7.shape == (3, 3), 'seven returns list of 3x3 matrix'
        # hlpr.displayEpipolarF(im1, im2, f7)
    # np.savez('q2_2.npz', F=F7, M=M, pts1=pts1, pts2=pts2)
    print("q2-2:\n", F7)

    # 3.1
    intrinsics_data = np.load('../data/intrinsics.npz')
    # print(intrinsics_data.files)
    # [print(i) for i in intrinsics_data]
    E = sub.essentialMatrix(F8, intrinsics_data["K1"], intrinsics_data["K2"])
    print("q3-1:\n", E)
    M2s = hlpr.camera2(E)
    # print(M2s.shape, "\n", M2s[:,:,0])

    # 3.2
    M1 = np.hstack((np.eye(3), np.zeros((3,1))))
    M2 = findM2(M2s, intrinsics_data["K1"], intrinsics_data["K2"], data['pts1'], data['pts2'])
    C1 = intrinsics_data["K1"] @ M1
    C2 = intrinsics_data["K2"] @ M2

    P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
    assert P.shape == (N, 3), 'triangulate returns Nx3 matrix P'
    assert np.isscalar(err), 'triangulate returns scalar err'
    # np.savez('q3_3.npz', M2=M2, C2=C2, P=P)

    # 4.1
    print(data_select['x1'][0], data_select['y1'][0])
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, data_select['x1'][0], data_select['y1'][0])
    assert np.isscalar(x2) & np.isscalar(y2), 'epipolarCorrespondence returns x & y coordinates'
    
    # pts1, pts2 = hlpr.epipolarMatchGUI(im1, im2, F8)
    # np.savez('q4_1.npz', F=F8, pts1=pts1, pts2=pts2)
    # np.savez('q4_1.npz', F=F8, pts1=data['pts1'], pts2=data['pts2'])
    print('Format check passed.')
