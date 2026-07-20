import dvx.torch as dvx
import torch
import numpy as np
import morph_ops as mo
import register as reg
import trimesh
import polyscope as ps

ps.init()
mesh = trimesh.load("bunny.obj")
v = np.array(mesh.vertices) # type: ignore
f = np.array(mesh.faces) # type: ignore

vertices = torch.from_numpy(v)
vertices.requires_grad_(True)
faces    = torch.from_numpy(f)
 
occupancy = dvx.voxelize(64, vertices, faces)
occ_np = occupancy.detach().numpy()
# reg.register_grid(occ_np, "Original grid")
reg.register_as_point_cloud(occ_np, "Original Grid cloud")

small_obj = np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])

# reg.register_grid(small_obj, "Small object")

# dil_obj = mo.dilation(occ_np,small_obj)

# reg.register_grid(mo.dilation(occ_np, small_obj), "Dilated grid")
# reg.register_as_point_cloud(dil_obj, "Dilated Grid cloud")


# ero_obj = mo.erosion(occ_np,small_obj)

# reg.register_grid(ero_obj, "Eroded grid")
# reg.register_as_point_cloud(ero_obj, "Eroded Grid cloud")

open_obj = mo.opening(occ_np,small_obj)

reg.register_grid(open_obj, "Opened grid")
reg.register_as_point_cloud(open_obj, "Opened Grid cloud")


close_obj = mo.opening(occ_np,small_obj)

reg.register_grid(close_obj, "Closed grid")
reg.register_as_point_cloud(close_obj, "Closed Grid cloud")

ps.show()
