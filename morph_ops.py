import numpy as np

# Define the "changing element" via a grid and then perform the actual operation by putting the element over each voxel in the actual voxel grid it is in. 
# Probably very inefficient, but only needs to be done once before actual optimization so, eh, you win some you lose some

def dilation(v_grid: np.ndarray, object: np.ndarray):
    # Run through each voxel in the grid. If the voxel does not hold a zero (i.e. is on the inside or border of the surface),
    # put the object's voxel grid on top of this one and then check each voxel of the objects grid where its one. Mark their equivalence in the full grid as one as well
    return

def erosion(v_grid: np.ndarray, object: np.ndarray):
    # Run through each voxel in the grid. If the voxel does holds a zero put the object's voxel grid on top of this one
    # and then check each voxel of the objects grid where its one. Mark their equivalence in the full grid as zero 
    return

def opening(v_grid: np.ndarray, object: np.ndarray):
    # round out the corners on the inside
    erosion(v_grid, object)
    dilation(v_grid, object)
    return

def closing(v_grid: np.ndarray, object: np.ndarray):
    # Round out the corners on the outside (i.e. fill in the shape a bit)
    dilation(v_grid, object)
    erosion(v_grid, object)
    return