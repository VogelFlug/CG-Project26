import dvx.torch as dvx
import torch
import numpy as np
import morph_ops
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
reg.register_grid(occ_np)
reg.register_as_point_clod(occ_np)
ps.show()
