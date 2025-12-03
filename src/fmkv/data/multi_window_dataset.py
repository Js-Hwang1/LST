"""
Multi-Window Dataset
====================

Dataset that groups multiple trajectory windows together for training.
This enables training with multiple super-tokens, making attention non-trivial.

Bug #19 Fix: Training with single super-token makes attention trivial (softmax
always outputs 1.0) and Jacobian is always zero. By training with multiple
windows, we simulate the real inference scenario with multiple compressed
super-tokens.
"""

from typing import Dict, List, Optional
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from fmkv.data.trajectory import TrajectoryWindow, load_trajectories


class MultiWindowDataset(Dataset):
    """
    Dataset that groups N consecutive windows together for training.
    
    This simulates the real inference scenario where the cache contains
    multiple compressed super-tokens from different windows.
    
    Each sample contains:
    - Multiple KV windows (N windows of window_size tokens each)
    - Future queries (queries that will attend to ALL windows)
    - The Sidecar compresses each window separately → N super-tokens
    - Loss is computed over attention with N super-tokens (non-trivial!)
    
    Example:
        >>> dataset = MultiWindowDataset.from_trajectories(
        ...     trajectories_path="./trajectories",
        ...     num_windows_per_sample=4,  # 4 windows → 4 super-tokens
        ...     window_size=64,
        ... )
        >>> 
        >>> sample = dataset[0]
        >>> print(sample["keys"].shape)      # (4, window_size, d_head)
        >>> print(sample["queries"].shape)   # (num_queries, d_head)
    """
    
    def __init__(
        self,
        trajectories: List[TrajectoryWindow],
        num_windows_per_sample: int = 4,
        window_size: int = 64,
        d_head: int = 128,
        shuffle_windows: bool = True,
    ):
        """
        Initialize multi-window dataset.
        
        Args:
            trajectories: List of trajectory windows
            num_windows_per_sample: Number of windows to group per sample
            window_size: Expected window size
            d_head: Head dimension
            shuffle_windows: Whether to shuffle windows when grouping
        """
        self.num_windows_per_sample = num_windows_per_sample
        self.window_size = window_size
        self.d_head = d_head
        
        # Group trajectories by layer for consistency
        self.trajectories_by_layer: Dict[int, List[TrajectoryWindow]] = {}
        for traj in trajectories:
            layer_idx = traj.layer_idx
            if layer_idx not in self.trajectories_by_layer:
                self.trajectories_by_layer[layer_idx] = []
            self.trajectories_by_layer[layer_idx].append(traj)
        
        # Create groups of N windows
        self.groups = []
        for layer_idx, layer_trajs in self.trajectories_by_layer.items():
            if shuffle_windows:
                layer_trajs = layer_trajs.copy()
                random.shuffle(layer_trajs)
            
            # Group into chunks of num_windows_per_sample
            for i in range(0, len(layer_trajs), num_windows_per_sample):
                window_group = layer_trajs[i:i + num_windows_per_sample]
                if len(window_group) == num_windows_per_sample:
                    self.groups.append(window_group)
        
        print(f"[MultiWindowDataset] Created {len(self.groups)} groups of "
              f"{num_windows_per_sample} windows each")
    
    @classmethod
    def from_trajectories(
        cls,
        trajectories_path: Path,
        **kwargs,
    ) -> "MultiWindowDataset":
        """
        Create dataset from saved trajectories.
        
        Args:
            trajectories_path: Path to saved trajectories
            **kwargs: Additional arguments for __init__
        
        Returns:
            MultiWindowDataset instance
        """
        trajectories = load_trajectories(trajectories_path)
        return cls(trajectories=trajectories, **kwargs)
    
    def __len__(self) -> int:
        return len(self.groups)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with multiple windows.
        
        Returns dict with:
            - keys: (num_windows, window_size, d_head) - multiple windows
            - values: (num_windows, window_size, d_head) - multiple windows
            - queries: (num_queries, d_head) - from the last window
            - layer_idx: int
        """
        windows = self.groups[idx]
        
        # Stack all windows
        keys_list = [w.keys for w in windows]
        values_list = [w.values for w in windows]
        
        # Use queries from the last window (they should attend to all windows)
        queries = windows[-1].future_queries
        layer_idx = windows[0].layer_idx
        
        sample = {
            "keys": torch.stack(keys_list, dim=0),  # (num_windows, window_size, d_head)
            "values": torch.stack(values_list, dim=0),  # (num_windows, window_size, d_head)
            "queries": queries,  # (num_queries, d_head)
            "layer_idx": torch.tensor(layer_idx),
        }
        
        return sample
    
    def get_collate_fn(self):
        """Get a collate function for batching."""
        def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            """Stack batch samples."""
            # batch is a list of dicts, each with:
            # keys: (num_windows, window_size, d_head)
            # values: (num_windows, window_size, d_head)
            # queries: (num_queries, d_head)
            
            return {
                "keys": torch.stack([s["keys"] for s in batch], dim=0),
                # → (batch, num_windows, window_size, d_head)
                "values": torch.stack([s["values"] for s in batch], dim=0),
                # → (batch, num_windows, window_size, d_head)
                "queries": torch.stack([s["queries"] for s in batch], dim=0),
                # → (batch, num_queries, d_head)
                "layer_idx": torch.stack([s["layer_idx"] for s in batch], dim=0),
            }
        
        return collate_fn
