'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission as sub
import helper as hlpr

def findM2(M2s, K1, K2, pts1, pts2):

    M1 = np.hstack((np.eye(3), np.zeros((3,1))))
    C1 = K1 @ M1
    n = M2s.shape[-1]
    for i in range(n):
        M2 = M2s[:,:,i]
        C2 = K2 @ M2
        P, err = sub.triangulate(C1, pts1, C2, pts2)
        
        pw = np.hstack((P[0],[1])).reshape((4,1))
        if((M2 @ pw)[-1]>=0 and P[0,-1]>=0):
            return M2
        # print("det:\t",np.linalg.det(M2[:,:-1]))
        # print("error: ", err)
        # print("\n\n")
    

