import dvx.torch as dvx
import torch
import numpy as np
import polyscope as ps
import morph_ops
import trimesh

mesh = trimesh.load("bunny.obj");
v = np.array(mesh.vertices);
f = np.array(mesh.faces);

vertices = torch.from_numpy(v)
vertices.requires_grad_(True)
faces    = torch.from_numpy(f)
 
occupancy = dvx.voxelize(64, vertices, faces)
 
L = occupancy.sum()
L.backward()