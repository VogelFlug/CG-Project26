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

    rightshot = s_dims[0] - s_center[0] - 1
    downshot  = s_dims[1] - s_center[1] - 1
    frontshot = s_dims[2] - s_center[2] - 1

    # Step three: check how much of our shot we are actually allowed in that direction
    l_dims = l_grid.shape
    leftshot = leftshot - max(0, leftshot - l_center[0])
    upshot   = upshot - max(0, upshot - l_center[1])
    backshot = backshot - max(0, backshot - l_center[2])

    rightshot = rightshot + l_dims[0] - max(l_dims[0], rightshot + l_center[0] + 1)
    downshot  = downshot + l_dims[1] - max(l_dims[1], downshot + l_center[1] + 1)
    frontshot = frontshot + l_dims[2] - max(l_dims[2], frontshot + l_center[2] + 1)

    return (leftshot, upshot, backshot, rightshot, downshot, frontshot),  s_grid[s_center[0]-leftshot:s_center[0]+rightshot+1, s_center[1]-upshot:s_center[1]+downshot+1, s_center[2]-backshot:s_center[2]+frontshot+1]


def dilation(v_grid: np.ndarray, object: np.ndarray):
    '''Run through each voxel in the grid. If the voxel does not hold a zero (i.e. is on the inside or border of the surface),
    put the object's voxel grid on top of this one and then check each voxel of the objects grid where its one. Mark their equivalence in the full grid as one as well
    
    Returns the dilated voxel grid'''

    # use np.argwhere to directly get indices of all voxels that are inside the object
    indices = np.argwhere(v_grid > 0)
    newgrid = v_grid.copy()
    for i in range(indices.shape[0]):
        voxel = indices[i][0:3]
        voxelvalue = v_grid[voxel[0],voxel[1],voxel[2]]
        # now get the range that we need to check
        (leftshot, upshot, backshot, rightshot, downshot, frontshot), tmp_object = layover(v_grid, object, voxel)
        #Finally: update each voxel in the range *if* the dilating object would "improve" the value
        newgrid[voxel[0]-leftshot:voxel[0]+rightshot+1, voxel[1]-upshot:voxel[1]+downshot+1, voxel[2]-backshot:voxel[2]+frontshot+1] = np.maximum(tmp_object*voxelvalue,  newgrid[voxel[0]-leftshot:voxel[0]+rightshot+1, voxel[1]-upshot:voxel[1]+downshot+1, voxel[2]-backshot:voxel[2]+frontshot+1])
    
    return newgrid

def alterosion(v_grid: np.ndarray, object: np.ndarray):
    '''Run through each voxel in the grid. If the voxel does holds a zero put the object's voxel grid on top of this one
    and then check each voxel of the objects grid where its one. Mark their equivalence in the full grid as zero

    Returns the eroded Voxel grid'''
    
    # use np.argwhere to directly get indices of all voxels that are outside the object
    indices = np.argwhere(v_grid < 1)
    newgrid = v_grid.copy()
    for i in range(indices.shape[0]):
        voxel = indices[i][0:3]
        voxelvalue = v_grid[voxel[0],voxel[1],voxel[2]]
        # now get the range that we need to check
        (leftshot, upshot, backshot, rightshot, downshot, frontshot), tmp_object = layover(v_grid, object, voxel)

        # get the programm to ignore the dead voxels in our object
        tmp_object = np.where(tmp_object == 0, 5, tmp_object)

        #Finally: update each voxel in the range *if* the eroding object would "worsen" the value
        newgrid[voxel[0]-leftshot:voxel[0]+rightshot+1, voxel[1]-upshot:voxel[1]+downshot+1, voxel[2]-backshot:voxel[2]+frontshot+1] = np.minimum(tmp_object*voxelvalue,  newgrid[voxel[0]-leftshot:voxel[0]+rightshot+1, voxel[1]-upshot:voxel[1]+downshot+1, voxel[2]-backshot:voxel[2]+frontshot+1])
    
    return newgrid


def erosion(v_grid: np.ndarray, object: np.ndarray):
    '''Run through each voxel in the grid. If the voxel does holds a zero put the object's voxel grid on top of this one
    and then check each voxel of the objects grid where its one. Mark their equivalence in the full grid as zero

    Returns the eroded Voxel grid
    
    TODO: Something about this is wrong and i dont know what...'''
    
    # use np.argwhere to directly get indices of all voxels that are outside the object
    indices = np.argwhere(v_grid > 0)
    newgrid = np.zeros(v_grid.shape, dtype = np.float64)

    # get it to be int
    eroder = np.array(object, dtype=int)
    # Get a grid where we snap all values > 0 to 1. We'll end up copying the values from the main grid instead
    fakegrid = np.where(v_grid > 0, 1, 0)
    for i in range(indices.shape[0]):
        voxel = indices[i][0:3]
        # now get the range that we need to check
        (leftshot, upshot, backshot, rightshot, downshot, frontshot), tmp_object = layover(v_grid, eroder, voxel)

        # Now we lay our object over our fakegrid and check if the object would "fit". If it does, let newgrid copy the original grid
        if(np.array_equal(np.bitwise_and(fakegrid[voxel[0]-leftshot:voxel[0]+rightshot+1, voxel[1]-upshot:voxel[1]+downshot+1, voxel[2]-backshot:voxel[2]+frontshot+1], tmp_object), tmp_object)):
            # to still semi maintain our border, if this is the case, let it only take the smallest value that isnt 0
            snippet = v_grid[voxel[0]-leftshot:voxel[0]+rightshot+1, voxel[1]-upshot:voxel[1]+downshot+1, voxel[2]-backshot:voxel[2]+frontshot+1] 
            #newgrid[voxel[0],voxel[1],voxel[2]] = v_grid[voxel[0],voxel[1],voxel[2]]#np.min(np.where(snippet == 0, 5, snippet))
            newgrid[voxel[0],voxel[1],voxel[2]] = np.min(np.where(snippet == 0, 5, snippet))
    
    return newgrid



def opening(v_grid: np.ndarray, object: np.ndarray):
    '''Round out the corners from the inside '''
    tmp1 = erosion(v_grid, object)
    tmp2 = dilation(tmp1, object)
    return tmp2


def closing(v_grid: np.ndarray, object: np.ndarray):
    '''Round out the corners from the outside (i.e. fill in the shape a bit)'''
    tmp1 = dilation(v_grid, object)
    tmp2 = erosion(tmp1, object)
    return tmp2





