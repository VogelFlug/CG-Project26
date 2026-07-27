import numpy as np
import polyscope as ps
import trimesh

def register_grid(voxel_grid: np.ndarray, name: str):
    dims = voxel_grid.shape
    bound_low = (-3., -3., -3.)
    bound_high = (3., 3., 3.)

    ps_grid = ps.register_volume_grid(name, dims, bound_low, bound_high)

    ps_grid.add_scalar_quantity("node scalar1", voxel_grid, 
                                defined_on='nodes', vminmax=(-1,1), enabled=True)

def register_as_point_cloud(voxel_grid: np.ndarray, name: str, scale = 0.02):
    ''' alternative form of showing the voxel grid that removes the voxels that shouldnt be there. Scale scales the whole thing, as regularly it is far too big'''

    idx = np.argwhere(voxel_grid > 0)
    values = voxel_grid[idx[:,0], idx[:,1], idx[:,2]]

    centers = (idx.astype(float) + 0.5) * scale
    pc = ps.register_point_cloud(name, centers)
    pc.set_point_render_mode("quad")

    pc.set_radius(0.5*scale, relative=False)

    pc.add_scalar_quantity(
        "value",
        values,
        vminmax=(0, 1),
        enabled=True
    )

def register_mesh(name: str, mesh: trimesh.Trimesh):
    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)
    ps.register_surface_mesh(name, v, f)
