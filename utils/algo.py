import torch
import torch.nn.functional as F

def moving_average_smoothing(traj, window_size):
    """
    Apply moving average smoothing to a trajectory tensor.
    
    Args:
        traj: Tensor of size [B, T, 2], where B is batch size, T is the number of timesteps.
        window_size: Size of the moving average window (must be an odd integer).
    
    Returns:
        Smoothed trajectory tensor of size [B, T, 2].
    """
    if len(traj.shape) > 3:
        traj = traj.reshape(-1, traj.shape[-2], traj.shape[-1])

    # Ensure window_size is odd for symmetric padding
    assert window_size % 2 == 1, "window_size must be an odd integer."
    
    # Create a convolutional kernel
    kernel = torch.ones((1, 1, window_size)) / window_size
    
    # Apply padding (replication padding used here)
    traj_x = traj[:, :, 0].unsqueeze(1)  # [B, 1, T]
    traj_y = traj[:, :, 1].unsqueeze(1)  # [B, 1, T]

    smoothed_x = F.conv1d(traj_x, kernel.to(traj.device), padding=window_size // 2)
    smoothed_y = F.conv1d(traj_y, kernel.to(traj.device), padding=window_size // 2)
    
    # Combine the smoothed X and Y
    smoothed_traj = torch.stack([smoothed_x.squeeze(1), smoothed_y.squeeze(1)], dim=-1)  # [B, T+4*widow_size, 2]
    smoothed_traj[:, :window_size//2] = traj[:, :window_size//2]
    smoothed_traj[:, -window_size//2:] = traj[:, -window_size//2:]

    return smoothed_traj