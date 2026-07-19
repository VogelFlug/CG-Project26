import dvx.torch as dvx
import torch
import numpy as np
import polyscope as ps
import morph_ops
import trimesh

ps.init()

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


mesh = trimesh.load("bunny.obj")
v = np.array(mesh.vertices) # type: ignore
f = np.array(mesh.faces) # type: ignore

vertices = torch.from_numpy(v)
vertices.requires_grad_(True)
faces    = torch.from_numpy(f)
 
occupancy = dvx.voxelize(64, vertices, faces)
occ_np = occupancy.detach().numpy()

# valuechecker = occ_np.copy()
# valuechecker[valuechecker == 0] = 0
# valuechecker[valuechecker == 1] = 0
# locs = np.argwhere(valuechecker > 0)
# print(occ_np[locs[0][0], locs[0][1], locs[0][2]])
register_grid(occ_np)
register_as_point_clod(occ_np)
ps.show()
