'''This file holds a few functions necessary for the optimization'''
import numpy as np
import trimesh


def get_cotanLap(mesh: trimesh.Trimesh):
    '''This function takes a mesh and returns the cotan Laplacian matrix, where each Cell ij is either:
     1: Falf of the sum of the cotan of the angles opposite to the edge ij if it exists and is on the inside of the surface of the mesh 
     2: The negative sum of the column i if i=j
     3: 0 otherwise'''

    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)

    # this already covers Case 3.
    lp_m = np.zeros((v.shape[0], v.shape[0]))

    # We iterate through all adjacent triangles to cover Case 2. We also update Case 3 alongside that
    pairs = mesh.face_adjacency

    for i,j in pairs:
        face_i = f[i]
        face_j = f[j]
        print(v[face_i], v[face_j])

        # get index of vertices that are the odd ones out
        equalit = np.zeros((3,3))
        for m in range(3):
            for n in range(3):
                equalit[m,n] = 1 if (face_i[m] == face_j[n]) else 0

        odd_i = np.argwhere(np.all(equalit == 0, axis = 1))[0][0]
        odd_j = np.argwhere(np.all(equalit == 0, axis = 0))[0][0]

        # get angle in face i:
        l = v[face_i[(odd_i + 1) % 3]] - v[face_i[odd_i]]
        r = v[face_i[(odd_i - 1) % 3]] - v[face_i[odd_i]]

        cot_sum = np.dot(l, r)/np.linalg.norm(np.cross(l,r))

        # get angle in face j
        l = v[face_j[(odd_j + 1) % 3]] - v[face_j[odd_j]]
        r = v[face_j[(odd_j - 1) % 3]] - v[face_j[odd_j]]

        # finally get the sum of which half will end up in the matrix
        cot_sum += np.dot(l, r)/np.linalg.norm(np.cross(l,r))
        fin_sum = cot_sum/2

        # We only got the indices of the of the "odd" vertices, but we actually care about the two vertices the triangles share
        ri, rj = face_i[(odd_i + 1) % 3], face_i[(odd_i - 1) % 3]
        lp_m[ri,rj] = lp_m[rj,ri] = fin_sum

        # Also update for Case 3
        lp_m[ri,ri] -= fin_sum
        lp_m[rj,rj] -= fin_sum

