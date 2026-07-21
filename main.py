import dvx.torch as dvx
import torch
import numpy as np
import trimesh
import polyscope as ps

import objects.struct_elements as os
import morph_ops as mo
import register as reg




def morph_mesh(mesh: trimesh.Trimesh, struct_element: np.ndarray, mode: str, step_size = 0.001, iterations = 1000):
    '''An implementation of the main problem, which attempts to achieve dilation of the main mesh through the voxelization
    
    The current energy function is simply the Frobenius norm between the current voxelization and the goal
    
    mode decides which morphological operation is performed on the base mesh to become the goal voxelization. Options are:
    "dilate": Dilation
    "erode": Erosion
    "close: Closing
    "open": Opening'''

    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)

    vertices = torch.from_numpy(v).clone()
    faces    = torch.from_numpy(f)

    # Now we get our first voxelization. We need this one to calculate the goal voxelization.
    first_occ = dvx.voxelize(64, vertices, faces)
    occ_np = first_occ.detach().numpy()

    # Get goal voxelization, depends on the four modes:
    if(mode == "dilate"):
        goal_vox = torch.from_numpy(mo.dilation(occ_np, struct_element))
    elif(mode == "erode"):
        goal_vox = torch.from_numpy(mo.erosion(occ_np, struct_element))
    elif(mode == "close"):
        goal_vox = torch.from_numpy(mo.closing(occ_np, struct_element))
    elif(mode == "open"):
        goal_vox = torch.from_numpy(mo.opening(occ_np, struct_element))
    else:
        raise Exception(mode + " is not a recognitiion of a morphological operation, maybe check your spelling?")

    v_tens = torch.from_numpy(v)
    v_tens.requires_grad = True
    for i in range(iterations):
        curr_occ = dvx.voxelize(64, v_tens, faces)
        energy = torch.linalg.norm(curr_occ - goal_vox)

        energy.backward()
    
        with torch.no_grad():
            if(v_tens.grad == None):
                raise Exception("No gradient was calculated, something went wrong")
            v_tens -= step_size * v_tens.grad  

        if(v_tens.grad == None):
                raise Exception("No gradient was calculated, something went wrong")   
        v_tens.grad.zero_()

    return v_tens.detach().numpy(), faces.numpy()


if __name__ == '__main__':
    mesh = trimesh.load_mesh("objects/bunny.obj")
    mesh.merge_vertices()

    ps.init()
    ps_mesh = reg.register_mesh("OG Mesh", mesh)

    operations = ["dilate", "erode", "close", "open"]

    #Create the structuring element
    s_el = os.generate_sphere(2)

    for op in operations:
        v,f = morph_mesh(mesh, s_el, op)

        # visualize the result of the current morphological operation
        ps.register_surface_mesh(op + " Result", v, f)

    ps.show()