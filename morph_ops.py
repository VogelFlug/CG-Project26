import numpy as np

# Define the "changing element" via a grid and then perform the actual operation by putting the element over each voxel in the actual voxel grid it is in. 
# Probably very inefficient, but only needs to be done once before actual optimization so, eh, you win some you lose some

def layover(l_grid: np.ndarray, s_grid: np.ndarray, l_center: np.ndarray):
    '''Given a large grid, a small grid and the coordinates of a point on the large grid: center the small grid on top of the point in the large grid #
    and get ranges of what parts of the small grid still lie inside of the large grid
    
    Assumptions: Optimally this is done with two cube grids, even better if the smaller one has odd length, though neither of those are technically required
    
    Returns: 6-tuple (left, up, back, right, down, forward) where each value defines how far you are allowed to go in that direction.'''

    # Step one: get center of our object within the object. I use a bit of a weird calculation for this, which ensures that we always get the index of the center within the small grid
    s_dims = s_grid.shape
    s_center = np.array([s_dims[0]//2 + (s_dims[0] % 2 > 0) - 1, s_dims[1]//2 +(s_dims[1] % 2 > 0) - 1, s_dims[2]//2 + (s_dims[2] % 2 > 0) - 1])

    # Step two: how much do we shoot in the directions? First assume we have infinite space
    leftshot = s_center[0]
    upshot   = s_center[1]
    backshot = s_center[2]

    rightshot    = s_dims[0] - s_center[0] - 1
    downshot     = s_dims[1] - s_center[1] - 1
    frontshot = s_dims[2] - s_center[2] - 1

    # Step three: check how much of our shot we are actually allowed in that direction
    l_dims = l_grid.shape
    leftshot = leftshot - max(0, leftshot - l_center[0])
    upshot = upshot - max(0, leftshot - l_center[1])
    backshot = backshot - max(0, backshot - l_center[2])

    rightshot = rightshot + l_dims[0] - min(l_dims[0], rightshot + l_center[0])
    downshot = downshot + l_dims[1] - min(l_dims[1], downshot + l_center[0])
    frontshot = frontshot + l_dims[2] - min(l_dims[2], frontshot + l_center[0])

    return (leftshot, upshot, backshot, rightshot, downshot, frontshot)


def dilation(v_grid: np.ndarray, object: np.ndarray):
    '''Run through each voxel in the grid. If the voxel does not hold a zero (i.e. is on the inside or border of the surface),
    put the object's voxel grid on top of this one and then check each voxel of the objects grid where its one. Mark their equivalence in the full grid as one as well'''

    # use np.argwhere to directly get indices of all voxels that are inside the object
    indices = np.argwhere(v_grid > 0)
    for i in range(indices[0]):
        voxel = indices[i][0:3]
        voxelvalue = v_grid[voxel]
        print(voxel)
        print(voxelvalue)

        #now iterate over each element of our object 
    return

def erosion(v_grid: np.ndarray, object: np.ndarray):
    # Run through each voxel in the grid. If the voxel does holds a zero put the object's voxel grid on top of this one
    # and then check each voxel of the objects grid where its one. Mark their equivalence in the full grid as zero 
    # use np.argwhere to directly get indices of all voxels that are outside the object
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

