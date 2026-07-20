import dvx.torch as dvx
import torch
import numpy as np
import trimesh
import polyscope as ps

import objects.struct_elements as os
import morph_ops as mo
import register as reg




def dilate_mesh(mesh: trimesh.Trimesh, struct_element: np.ndarray):
    '''An implementation of the main problem, which attempts to achieve dilation of the main mesh through the voxelization
    
    The current energy function is simply the Frobenius norm between the current voxelization and the goal'''
    step_size = 0.001

    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)

    vertices = torch.from_numpy(v)
    faces    = torch.from_numpy(f)

    # Now we get our first voxelization. We need this one to calculate the goal voxelization.
    first_occ = dvx.voxelize(64, vertices, faces)
    occ_np = first_occ.detach().numpy()
    # Get goal voxelization 
    goal_vox = torch.from_numpy(mo.dilation(occ_np, struct_element))

    # now for the actual voxelization. Get the first energy and then repeat 
    energy = torch.linalg.norm(first_occ - goal_vox)

    v_tens = torch.from_numpy(v)
    v_tens.requires_grad = True
    for i in range(1000):
        curr_occ = dvx.voxelize(64, v_tens, faces)
        energy = torch.linalg.norm(curr_occ - goal_vox)

        energy.backward()

        with torch.no_grad():
            v_tens -= step_size * v_tens.grad 

        print(energy)
        v_tens.grad.zero_() #type:ignore

    return vertices.detach().numpy(), faces.numpy()


if __name__ == '__main__':
    s_el = os.generate_sphere(1)
    mesh = trimesh.load_mesh("objects/bunny.obj")

    v,f = dilate_mesh(mesh, s_el)

    # now to visualize
    ps.init()
    ps_mesh = reg.register_mesh("OG Mesh", mesh)

    ps.register_surface_mesh("Result", v, f)

    ps.show()