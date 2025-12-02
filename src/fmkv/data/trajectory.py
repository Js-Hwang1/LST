"""
Trajectory Collection
=====================

Collect KV cache trajectories from frozen LLM forward passes.
These trajectories are used to train the Sidecar network.

The collector:
1. Runs the frozen LLM on text samples
2. Extracts KV states at each layer
3. Extracts "future queries" (hidden states that will attend to each window)
4. Saves trajectories for offline training
"""

import os
from typing import Dict, Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from fmkv.models.wrapper import ModelWrapper


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory collection."""
    
    # Window configuration
    window_size: int = 64
    stride: int = 32  # Overlap between windows
    
    # Future query sampling
    num_future_queries: int = 16
    future_query_range: int = 128  # Sample queries from next N positions
    
    # Layers to collect (None = all)
    target_layers: Optional[List[int]] = None
    
    # Collection limits
    max_samples: int = 10000
    max_seq_length: int = 4096
    min_seq_length: int = 256
    
    # Output
    output_dir: str = "./trajectories"
    save_every: int = 1000


@dataclass
class TrajectoryWindow:
    """A single window of KV states with future queries."""
    
    layer_idx: int
    window_idx: int
    
    # KV states for the window: (window_size, d_head)
    keys: torch.Tensor
    values: torch.Tensor
    
    # Future queries that will attend to this window: (num_queries, d_head)
    future_queries: torch.Tensor
    
    # Metadata
    position_offset: int  # Start position in original sequence
    sample_id: str


class TrajectoryCollector:
    """
    Collects KV trajectories from a frozen LLM for Sidecar training.
    
    Example:
        >>> wrapper = ModelWrapper.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> collector = TrajectoryCollector(wrapper, config)
        >>> 
        >>> # Collect from a text dataset
        >>> for batch in data_loader:
        ...     trajectories = collector.collect_batch(batch["input_ids"])
        ...     collector.save_trajectories(trajectories)
    """
    
    def __init__(
        self,
        model_wrapper: "ModelWrapper",
        config: TrajectoryConfig,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
    ):
        self.model = model_wrapper
        self.config = config
        self.tokenizer = tokenizer or model_wrapper.tokenizer
        
        # Determine target layers
        if config.target_layers is None:
            self.target_layers = list(range(model_wrapper.info.num_layers))
        else:
            self.target_layers = config.target_layers
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection statistics
        self.stats = {
            "samples_collected": 0,
            "windows_collected": 0,
            "tokens_processed": 0,
        }
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump({
                "window_size": config.window_size,
                "stride": config.stride,
                "num_future_queries": config.num_future_queries,
                "future_query_range": config.future_query_range,
                "target_layers": self.target_layers,
                "model_name": model_wrapper.info.name,
                "d_head": model_wrapper.info.head_dim,
                "num_heads": model_wrapper.info.num_heads,
            }, f, indent=2)
    
    @torch.no_grad()
    def collect_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> List[TrajectoryWindow]:
        """
        Collect trajectories from a batch of sequences.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask
            sample_ids: Optional IDs for tracking samples
        
        Returns:
            List of TrajectoryWindow objects
        """
        batch_size, seq_len = input_ids.shape
        
        if seq_len < self.config.min_seq_length:
            return []
        
        # Truncate to max length
        if seq_len > self.config.max_seq_length:
            input_ids = input_ids[:, :self.config.max_seq_length]
            seq_len = self.config.max_seq_length
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seq_len]
        
        # Generate sample IDs if not provided
        if sample_ids is None:
            sample_ids = [f"sample_{self.stats['samples_collected'] + i}" for i in range(batch_size)]
        
        # Forward pass with KV capture
        with self.model.capture_kv(self.target_layers):
            outputs = self.model.forward(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            
            kv_states = self.model.get_kv_states()
            hidden_states = outputs.hidden_states
        
        # Extract trajectories
        trajectories = []
        
        for layer_idx in self.target_layers:
            state = kv_states.get(layer_idx)
            if state is None or state.keys is None:
                continue
            
            # KV shape: (batch, heads, seq, d_head)
            keys = state.keys
            values = state.values
            
            # Get hidden states for this layer (for future queries)
            # hidden_states[layer_idx] shape: (batch, seq, hidden)
            layer_hidden = hidden_states[layer_idx]
            
            # Extract windows
            windows = self._extract_windows(
                layer_idx=layer_idx,
                keys=keys,
                values=values,
                hidden_states=layer_hidden,
                seq_len=seq_len,
                batch_size=batch_size,
                sample_ids=sample_ids,
            )
            
            trajectories.extend(windows)
        
        # Update stats
        self.stats["samples_collected"] += batch_size
        self.stats["windows_collected"] += len(trajectories)
        self.stats["tokens_processed"] += batch_size * seq_len
        
        return trajectories
    
    def _extract_windows(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_len: int,
        batch_size: int,
        sample_ids: List[str],
    ) -> List[TrajectoryWindow]:
        """Extract sliding windows from KV states."""
        windows = []
        window_size = self.config.window_size
        stride = self.config.stride
        num_queries = self.config.num_future_queries
        query_range = self.config.future_query_range
        
        # Iterate over window positions
        num_windows = (seq_len - window_size - query_range) // stride + 1
        
        for window_idx in range(max(num_windows, 0)):
            start = window_idx * stride
            end = start + window_size
            query_start = end
            query_end = min(end + query_range, seq_len)
            
            if query_end - query_start < num_queries:
                continue
            
            # For each sample in batch
            for b in range(batch_size):
                # Extract KV window (average over heads for now, or pick one head)
                # Using first head for simplicity - can modify for multi-head
                k_window = keys[b, 0, start:end, :].cpu()  # (window, d_head)
                v_window = values[b, 0, start:end, :].cpu()
                
                # Sample future queries from hidden states
                query_positions = torch.randperm(query_end - query_start)[:num_queries]
                query_positions = query_positions + query_start
                
                # Project hidden states to query dimension
                # In practice, need to apply Q projection - for now use hidden directly
                # This is a simplification; proper implementation would use Q projection
                future_q = hidden_states[b, query_positions, :].cpu()
                
                # Store the queries in head dimension format if needed
                # For now, store raw hidden states (will need Q projection during training)
                
                window = TrajectoryWindow(
                    layer_idx=layer_idx,
                    window_idx=window_idx,
                    keys=k_window,
                    values=v_window,
                    future_queries=future_q,
                    position_offset=start,
                    sample_id=sample_ids[b],
                )
                
                windows.append(window)
        
        return windows
    
    def save_trajectories(
        self,
        trajectories: List[TrajectoryWindow],
        filename: Optional[str] = None,
    ):
        """Save trajectories to disk."""
        if filename is None:
            filename = f"trajectories_{self.stats['windows_collected']}.pt"
        
        filepath = self.output_dir / filename
        
        # Convert to saveable format
        data = {
            "windows": [
                {
                    "layer_idx": t.layer_idx,
                    "window_idx": t.window_idx,
                    "keys": t.keys,
                    "values": t.values,
                    "future_queries": t.future_queries,
                    "position_offset": t.position_offset,
                    "sample_id": t.sample_id,
                }
                for t in trajectories
            ],
            "stats": self.stats.copy(),
        }
        
        torch.save(data, filepath)
        
        return filepath
    
    def collect_from_dataset(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        save_every: Optional[int] = None,
    ) -> Path:
        """
        Collect trajectories from a full dataset.
        
        Args:
            dataloader: DataLoader yielding input_ids
            max_samples: Maximum samples to collect
            save_every: Save checkpoint every N windows
        
        Returns:
            Path to saved trajectories
        """
        max_samples = max_samples or self.config.max_samples
        save_every = save_every or self.config.save_every
        
        all_trajectories = []
        checkpoint_idx = 0
        
        pbar = tqdm(dataloader, desc="Collecting trajectories")
        
        for batch in pbar:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
            else:
                input_ids = batch
                attention_mask = None
            
            # Move to model device
            device = self.model.info.device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            trajectories = self.collect_batch(input_ids, attention_mask)
            all_trajectories.extend(trajectories)
            
            pbar.set_postfix({
                "windows": len(all_trajectories),
                "samples": self.stats["samples_collected"],
            })
            
            # Checkpoint
            if len(all_trajectories) >= save_every:
                self.save_trajectories(
                    all_trajectories,
                    f"checkpoint_{checkpoint_idx}.pt",
                )
                all_trajectories = []
                checkpoint_idx += 1
            
            # Check limit
            if self.stats["samples_collected"] >= max_samples:
                break
        
        # Save remaining
        if all_trajectories:
            self.save_trajectories(all_trajectories, f"checkpoint_{checkpoint_idx}.pt")
        
        # Save final stats
        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)
        
        return self.output_dir


def load_trajectories(
    path: Union[str, Path],
) -> List[TrajectoryWindow]:
    """
    Load trajectories from disk.
    
    Supports multiple formats:
    - checkpoint_*.pt files (from TrajectoryCollector)
    - trajectories_*.pt files (from collect_trajectories.py script)
    """
    path = Path(path)
    
    if path.is_dir():
        # Load all trajectory files
        trajectories = []
        
        # Try multiple patterns
        patterns = ["checkpoint_*.pt", "trajectories_*.pt"]
        files_found = []
        
        for pattern in patterns:
            files_found.extend(sorted(path.glob(pattern)))
        
        if not files_found:
            print(f"Warning: No trajectory files found in {path}")
            print(f"  Looked for: {patterns}")
            print(f"  Available files: {list(path.glob('*.pt'))}")
            return trajectories
        
        print(f"Found {len(files_found)} trajectory files")
        
        for file in tqdm(files_found, desc="Loading trajectories"):
            data = torch.load(file, weights_only=False)
            
            for w in data.get("windows", []):
                # Handle different key formats
                traj = _parse_trajectory_window(w)
                if traj is not None:
                    trajectories.append(traj)
        
        return trajectories
    else:
        # Load single file
        data = torch.load(path, weights_only=False)
        return [_parse_trajectory_window(w) for w in data.get("windows", []) 
                if _parse_trajectory_window(w) is not None]


def _parse_trajectory_window(w: dict) -> Optional[TrajectoryWindow]:
    """Parse a trajectory window dict, handling different formats."""
    try:
        # Handle different key names for queries
        future_queries = w.get("future_queries")
        if future_queries is None:
            future_queries = w.get("queries")
        
        # Check if we have the required fields
        if "keys" not in w or "values" not in w:
            return None
        
        if future_queries is None:
            return None
        
        return TrajectoryWindow(
            layer_idx=w.get("layer_idx", 0),
            window_idx=w.get("window_idx", 0),
            keys=w["keys"],
            values=w["values"],
            future_queries=future_queries,
            position_offset=w.get("position_offset", w.get("window_start", 0)),
            sample_id=w.get("sample_id", "unknown"),
        )
    except Exception as e:
        print(f"Warning: Failed to parse trajectory window: {e}")
        return None

