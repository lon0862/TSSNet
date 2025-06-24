import torch

from typing import Any, List, Optional, Tuple, Union

def generate_clockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = torch.sin(angle)
    matrix[..., 1, 0] = -torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix

def generate_counterclockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = -torch.sin(angle)
    matrix[..., 1, 0] = torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix
    
def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta

def transform_point_to_local_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    point = point - position
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    return point

def transform_point_to_global_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    point = point + position
    return point
