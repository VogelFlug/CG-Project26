import dvx.torch as dvx
import numpy as np
import polyscope as ps

def register_grid(voxel_grid: np.ndarray):
    dims = voxel_grid.shape
    bound_low = (-3., -3., -3.)
    bound_high = (3., 3., 3.)

    ps_grid = ps.register_volume_grid("sample grid", dims, bound_low, bound_high)

    ps_grid.add_scalar_quantity("node scalar1", voxel_grid, 
                                defined_on='nodes', vminmax=(-1,1), enabled=True)

def register_as_point_clod(voxel_grid: np.ndarray):
    # alternative form of showing the voxel grid that removes the voxels that shouldnt be there

    idx = np.argwhere(voxel_grid > 0)
    values = voxel_grid[idx[:,0], idx[:,1], idx[:,2]]

    centers = idx.astype(float) + 0.5
    pc = ps.register_point_cloud("voxels", centers)
    pc.set_point_render_mode("quad")

    pc.set_radius(0.5, relative=False)

    pc.add_scalar_quantity(
        "value",
        values,
        vminmax=(0, 1),
        enabled=True
    )

