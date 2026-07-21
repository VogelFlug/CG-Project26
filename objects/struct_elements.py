'''This file includes a few functions for creating symmetrical oddly-sized grids for structuring elements. All returns are numpy arrays'''
import numpy as np

tiny_sph = np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])

# med_sph = np.array([[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]],[[0,0,1,0,0]],[[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]])


def generate_cube(radius):
    return np.ones((2*radius-1,2*radius-1,2*radius-1))

def generate_sphere(radius):
    sphere = np.zeros((2*radius+1,2*radius+1,2*radius+1))
    x, y, z = np.indices((2*radius+1,2*radius+1,2*radius+1))

    # Euklidischer Abstand zum Mittelpunkt
    dist = np.sqrt((x - radius)**2 + (y - radius)**2 + (z-radius)**2)

    sphere[dist <= radius] = 1
    return sphere
