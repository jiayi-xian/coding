import torch
import numpy as np

def get_rays(H, W, K, c2w):
    # Create a meshgrid of pixel coordinates
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t() # shape (H, W)
    j = j.t() # shape (H, W)

    # Compute the ray directions in camera coordinates
    dirs = torch.stack([(i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    # This line performs a batched matrix-vector multiplication
    rays_d = torch.sum(dirs[..., np.newaxis, :3, :3] * c2w[:3, :3], -1)

    # Translate camera frame's origin to the world frame
    # This is the origin of all rays
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

# Example usage:
# Define hypothetical values for H, W, K, and c2w
H, W = 6, 4  # Image height and width
K = torch.tensor([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 0]], dtype=torch.float32)  # Intrinsic camera matrix
K = torch.tensor([[10, 0, 3.2, 0], [0, 10, 2.4, 0], [0, 0, 1, 0]], dtype=torch.float32)  # Intrinsic camera matrix
c2w = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [10, 20, 30, 1]], dtype=torch.float32)  # Camera-to-world transformation matrix

# Get the rays
rays_o, rays_d = get_rays(H, W, K, c2w)

# Print the results
print("Rays origin (rays_o):")
print(rays_o)
print("\nRays direction (rays_d):")
print(rays_d)