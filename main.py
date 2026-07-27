import dvx.torch as dvx
import torch
import numpy as np
import trimesh
import polyscope as ps

import objects.struct_elements as os
import morph_ops as mo
import register as reg
import util



def morph_lap(mesh: trimesh.Trimesh, struct_element: np.ndarray, mode: str, step_size = 0.002, iterations = 100, adam = False, hyper = 0.01):
    '''An implementation of the main problem, which additionally implements the modifier to the step via the Laplacian matrix as seen in 
    "Large Steps in Inverse Rendering of Geometry [Nicolet et al. 2021]
    
    The current energy function is simply the Frobenius norm between the current voxelization and the goal
    
    mode decides which morphological operation is performed on the base mesh to become the goal voxelization. Options are:
    "dilate": Dilation
    "erode": Erosion
    "close: Closing
    "open": Opening

    adam: ignore for now is not yet implemented
    
    hyper is the value for how much the Laplacian weighs into the full solution'''


    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)

    vertices = torch.from_numpy(v).clone()
    faces    = torch.from_numpy(f)

    # Now we get our first voxelization. We need this one to calculate the goal voxelization.
    first_occ = dvx.voxelize(128, vertices, faces)
    occ_np = first_occ.detach().numpy()

    # Get goal voxelization, depends on the four modes:
    if(mode == "dilate"):
        goal_vox = torch.from_numpy(mo.dilation(occ_np, struct_element))
        reg.register_as_point_cloud(goal_vox.numpy(), "Goal Dilation")
    elif(mode == "erode"):
        goal_vox = torch.from_numpy(mo.erosion(occ_np, struct_element))
        #reg.register_as_point_cloud(goal_vox.numpy(), "Goal Erosion")
    elif(mode == "close"):
        goal_vox = torch.from_numpy(mo.closing(occ_np, struct_element))
        #reg.register_as_point_cloud(goal_vox.numpy(), "Goal Closing")
    elif(mode == "open"):
        goal_vox = torch.from_numpy(mo.opening(occ_np, struct_element))
        #reg.register_as_point_cloud(goal_vox.numpy(), "Goal Opening")
    else:
        raise Exception(mode + " is not a recognitiion of a morphological operation, maybe check your spelling?")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    goal_vox = goal_vox.to(device)
    faces = faces.to(device)

    # get the cotan Laplacian and the subsequent weighing
    lap = util.get_cotanLap(mesh)
    weigher = torch.from_numpy(np.identity(lap.shape[0]) + (hyper * lap)).to(device)

    # We won't actually be optimising through the vertices, we'll be optimising through the "weighed" vertices
    v_tens = torch.from_numpy(v).to(device)
    u = weigher @ v_tens
    u.requires_grad = True

    # Toggle between Adam or not
    if(adam == False):
        for i in range(iterations):
            # First, we get back to the actual vertices
            v_tens = torch.inverse(weigher) @ u

            # Get the voxelization of the current set of vertices
            curr_occ = dvx.voxelize(128, v_tens, faces)

            # Main energy function: "Penalize" difference between current voxelization and goal voxelization 
            mainenergy = torch.linalg.norm(curr_occ - goal_vox)

            # Self intersection energy: the voxelization returns the winding number, if its larger than 1 or smaller than 0, the mesh intersects itself, and we dont want that
            # has an epsilon due to floating point stuff
            si = torch.where((curr_occ < -1e-6) | (curr_occ > 1 + 1e-6), curr_occ, 0)
            si_energy = torch.linalg.norm(si)

            fullenergy = si_energy + mainenergy
            fullenergy.backward()
        
            with torch.no_grad():
                if(u.grad == None):
                    raise Exception("No gradient was calculated, something went wrong")
                u -= step_size * u.grad  

            if(u.grad == None):
                    raise Exception("No gradient was calculated, something went wrong")   
            u.grad.zero_()

    else:
        u = torch.nn.Parameter(u)
        optimizer = torch.optim.Adam([u], lr=step_size)
    
        for i in range(iterations):
            # First, we get back to the actual vertices
            v_tens = torch.inverse(weigher) @ u
            # Get the voxelization of the current set of vertices
            curr_occ = dvx.voxelize(128, v_tens, faces)
    
            # Main energy function: "Penalize" difference between current voxelization and goal voxelization 
            mainenergy = torch.linalg.norm(curr_occ - goal_vox)
    
            # Self intersection energy: the voxelization returns the winding number, if its larger than 1 or smaller than 0, the mesh intersects itself, and we dont want that
            # has an epsilon due to floating point stuff
            si = torch.where((curr_occ < -1e-6) | (curr_occ > 1 + 1e-6), curr_occ, 0)
            si_energy = si.pow(2).sum()
    
            fullenergy = si_energy + mainenergy
            fullenergy.backward()
        
            optimizer.step()
    
            if(u.grad == None):
                    raise Exception("No gradient was calculated, something went wrong")   
            u.grad.zero_()

    v_tens = torch.inverse(weigher) @ u
    return v_tens.cpu().detach().numpy(), faces.cpu().numpy()









def morph_mesh(mesh: trimesh.Trimesh, struct_element: np.ndarray, mode: str, step_size = 0.002, iterations = 100, adam = False):
    '''An implementation of the main problem, which attempts to achieve a morphological changing of the main mesh through the voxelization
    
    The current energy function is simply the Frobenius norm between the current voxelization and the goal
    
    mode decides which morphological operation is performed on the base mesh to become the goal voxelization. Options are:
    "dilate": Dilation
    "erode": Erosion
    "close: Closing
    "open": Opening
    
    Toggling adam to true, changes from regular gradient descent to an adam optimizer'''


    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)

    vertices = torch.from_numpy(v).clone()
    faces    = torch.from_numpy(f)

    # Now we get our first voxelization. We need this one to calculate the goal voxelization.
    first_occ = dvx.voxelize(128, vertices, faces)
    occ_np = first_occ.detach().numpy()

    # Get goal voxelization, depends on the four modes:
    if(mode == "dilate"):
        goal_vox = torch.from_numpy(mo.dilation(occ_np, struct_element))
        reg.register_as_point_cloud(goal_vox.numpy(), "Goal Dilation")
    elif(mode == "erode"):
        goal_vox = torch.from_numpy(mo.erosion(occ_np, struct_element))
        #reg.register_as_point_cloud(goal_vox.numpy(), "Goal Erosion")
    elif(mode == "close"):
        goal_vox = torch.from_numpy(mo.closing(occ_np, struct_element))
        #reg.register_as_point_cloud(goal_vox.numpy(), "Goal Closing")
    elif(mode == "open"):
        goal_vox = torch.from_numpy(mo.opening(occ_np, struct_element))
        #reg.register_as_point_cloud(goal_vox.numpy(), "Goal Opening")
    else:
        raise Exception(mode + " is not a recognitiion of a morphological operation, maybe check your spelling?")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    goal_vox = goal_vox.to(device)
    faces = faces.to(device)

    v_tens = torch.from_numpy(v).to(device)
    v_tens.requires_grad = True

    # Toggle between Adam or not
    if(adam == False):
        for i in range(iterations):
            # Get the voxelization of the current set of vertices
            curr_occ = dvx.voxelize(128, v_tens, faces)

            # Main energy function: "Penalize" difference between current voxelization and goal voxelization 
            mainenergy = torch.linalg.norm(curr_occ - goal_vox)

            # Self intersection energy: the voxelization returns the winding number, if its larger than 1 or smaller than 0, the mesh intersects itself, and we dont want that
            # has an epsilon due to floating point stuff
            si = torch.where((curr_occ < -1e-6) | (curr_occ > 1 + 1e-6), curr_occ, 0)
            si_energy = torch.linalg.norm(si)

            fullenergy = si_energy + mainenergy
            fullenergy.backward()
        
            with torch.no_grad():
                if(v_tens.grad == None):
                    raise Exception("No gradient was calculated, something went wrong")
                v_tens -= step_size * v_tens.grad  

            if(v_tens.grad == None):
                    raise Exception("No gradient was calculated, something went wrong")   
            v_tens.grad.zero_()

    else:
        v_tens = torch.nn.Parameter(v_tens)
        optimizer = torch.optim.Adam([v_tens], lr=step_size)
    
        for i in range(iterations):
            # Get the voxelization of the current set of vertices
            curr_occ = dvx.voxelize(128, v_tens, faces)
    
            # Main energy function: "Penalize" difference between current voxelization and goal voxelization 
            mainenergy = torch.linalg.norm(curr_occ - goal_vox)
    
            # Self intersection energy: the voxelization returns the winding number, if its larger than 1 or smaller than 0, the mesh intersects itself, and we dont want that
            # has an epsilon due to floating point stuff
            si = torch.where((curr_occ < -1e-6) | (curr_occ > 1 + 1e-6), curr_occ, 0)
            si_energy = si.pow(2).sum()
    
            fullenergy = si_energy + mainenergy
            fullenergy.backward()
        
            optimizer.step()
    
            if(v_tens.grad == None):
                    raise Exception("No gradient was calculated, something went wrong")   
            v_tens.grad.zero_()

    return v_tens.cpu().detach().numpy(), faces.cpu().numpy()













if __name__ == '__main__':
    mesh = trimesh.load_mesh("objects/bunny.obj")
    mesh.merge_vertices()
    ps.init()
    ps_mesh = reg.register_mesh("OG Mesh", mesh)

    operations = ["dilate","erode", "close", "open"]

    #Create the structuring element
    s_el = os.generate_sphere(2)

    for op in operations:
        v,f = morph_lap(mesh, s_el, op, adam=True)

        # visualize the result of the current morphological operation
        ps.register_surface_mesh(op + " Result", v, f)

    ps.show()