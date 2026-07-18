import dvx.torch as dvx
import torch
import numpy as np
import polyscope as ps
import morph_ops
import trimesh

def register_grid(voxel_grid: np.ndarray):
    ps.init()
    dims = voxel_grid.shape
    bound_low = (-3., -3., -3.)
    bound_high = (3., 3., 3.)

    ps_grid = ps.register_volume_grid("sample grid", dims, bound_low, bound_high)

    scalar_vals = voxel_grid
    ps_grid.add_scalar_quantity("node scalar1", scalar_vals, 
                                defined_on='nodes', vminmax=(-1., 1.), enabled=True)

    ps.show()


mesh = trimesh.load("bunny.obj")
v = np.array(mesh.vertices) # type: ignore
f = np.array(mesh.faces) # type: ignore

vertices = torch.from_numpy(v)
vertices.requires_grad_(True)
faces    = torch.from_numpy(f)
 
occupancy = dvx.voxelize(64, vertices, faces)
occ_np = occupancy.detach().numpy()

register_grid(occ_np)
