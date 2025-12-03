"""
Sidecar Trainer
===============

Training loop for the Sidecar network with:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Wandb/Tensorboard logging
"""

import os
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from fmkv.sidecar import Sidecar, SidecarConfig
from fmkv.losses import ForceMatchingLoss
from fmkv.losses.force_matching import ForceMatchingLossConfig


@dataclass
class TrainingConfig:
    """Configuration for Sidecar training."""
    
    # Output
    output_dir: str = "./checkpoints"
    experiment_name: str = "fmkv"
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Loss
    force_matching_weight: float = 1.0
    consistency_weight: float = 0.1
    geometric_weight: float = 0.01
    
    # Regularization
    gradient_clip_norm: float = 1.0
    dropout: float = 0.1
    
    # Precision
    mixed_precision: bool = True
    dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    log_steps: int = 10
    wandb_project: Optional[str] = "fmkv"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Data
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 1e-4
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)


class SidecarTrainer:
    """
    Trainer for the Sidecar network.
    
    Handles:
    - Training loop with gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Checkpointing and resumption
    - Logging to wandb/tensorboard
    
    Example:
        >>> sidecar = Sidecar(sidecar_config)
        >>> trainer = SidecarTrainer(sidecar, train_config)
        >>> 
        >>> trainer.train(
        ...     train_dataloader=train_loader,
        ...     eval_dataloader=eval_loader,
        ... )
    """
    
    def __init__(
        self,
        sidecar: Sidecar,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.sidecar = sidecar
        self.config = config
        
        # Setup device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Convert Sidecar to device and dtype
        self.sidecar.to(device)
        
        # Convert to the specified dtype from config
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        target_dtype = dtype_map.get(sidecar.config.dtype, torch.float32)
        self.sidecar = self.sidecar.to(dtype=target_dtype)
        self.target_dtype = target_dtype  # Store for batch conversion
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump({
                "training": config.__dict__,
                "sidecar": sidecar.config.__dict__,
            }, f, indent=2, default=str)
        
        # Setup loss
        loss_config = ForceMatchingLossConfig(
            force_matching_weight=config.force_matching_weight,
            consistency_weight=config.consistency_weight,
        )
        self.loss_fn = ForceMatchingLoss(
            d_head=sidecar.config.d_head,
            config=loss_config,
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            sidecar.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler (will be initialized in train())
        self.scheduler = None
        
        # Mixed precision
        # Note: GradScaler only supports float16, not bfloat16
        use_grad_scaler = config.mixed_precision and config.dtype == "float16"
        self.scaler = GradScaler() if use_grad_scaler else None
        self.autocast_dtype = config.get_dtype()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Logging
        self.wandb_run = None
        self._init_logging()
    
    def _init_logging(self):
        """Initialize logging (wandb)."""
        if WANDB_AVAILABLE and self.config.wandb_project:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                config={
                    "training": self.config.__dict__,
                    "sidecar": self.sidecar.config.__dict__,
                },
                tags=self.config.wandb_tags,
            )
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        warmup_steps = min(self.config.warmup_steps, num_training_steps // 10)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            resume_from: Path to checkpoint to resume from
        """
        # Resume if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Calculate training steps
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        max_steps = min(self.config.max_steps, num_update_steps_per_epoch * 100)
        
        # Create scheduler
        self._create_scheduler(max_steps)
        
        # Training loop
        self.sidecar.train()
        
        progress_bar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc="Training",
        )
        
        running_loss = 0.0
        num_batches = 0
        
        while self.global_step < max_steps:
            self.epoch += 1
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss, metrics = self._training_step(batch)
                
                # Accumulate gradients
                loss = loss / self.config.gradient_accumulation_steps
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                running_loss += loss.item()
                num_batches += 1
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    if self.config.gradient_clip_norm > 0:
                        nn.utils.clip_grad_norm_(
                            self.sidecar.parameters(),
                            self.config.gradient_clip_norm,
                        )
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_steps == 0:
                        avg_loss = running_loss / num_batches
                        lr = self.scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })
                        
                        self._log_metrics({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": self.epoch,
                            **{f"train/{k}": v.item() if torch.is_tensor(v) else v 
                               for k, v in metrics.items()},
                        })
                        
                        running_loss = 0.0
                        num_batches = 0
                    
                    # Evaluation
                    if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self._evaluate(eval_dataloader)
                        self._log_metrics({"eval/loss": eval_loss})
                        
                        # Early stopping
                        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
                            self.best_eval_loss = eval_loss
                            self.early_stopping_counter = 0
                            self._save_checkpoint("best")
                        else:
                            self.early_stopping_counter += 1
                        
                        if self.early_stopping_counter >= self.config.early_stopping_patience:
                            print(f"Early stopping at step {self.global_step}")
                            break
                        
                        self.sidecar.train()
                    
                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(f"step_{self.global_step}")
                    
                    progress_bar.update(1)
                    
                    if self.global_step >= max_steps:
                        break
            
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                break
        
        progress_bar.close()
        
        # Final save
        self._save_checkpoint("final")
        
        if self.wandb_run:
            self.wandb_run.finish()
    
    def _training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute a single training step."""
        keys = batch["keys"]
        values = batch["values"]
        queries = batch["queries"]
        
        # Bug #19 Fix: Handle multi-window training
        # keys/values can be either:
        # - (batch, seq_len, d_head) - single window
        # - (batch, num_windows, seq_len, d_head) - multiple windows
        is_multi_window = keys.dim() == 4
        
        # Handle query dimension mismatch
        # Queries from hidden states may have different dim than keys
        q_dim = queries.size(-1)
        k_dim = keys.size(-1)
        
        if q_dim != k_dim:
            if q_dim > k_dim:
                queries = queries[..., :k_dim]
            else:
                queries = torch.nn.functional.pad(queries, (0, k_dim - q_dim))
        
        # Forward through Sidecar
        # Use autocast compatible with both old and new PyTorch APIs
        # Check PyTorch version to use correct API
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            # New API (PyTorch 2.0+): torch.amp.autocast(device_type="cuda", ...)
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.scaler is not None):
                if is_multi_window:
                    # Compress each window separately
                    k_cg, v_cg = self._compress_multi_window(keys, values)
                    # Compute dense attention over all original windows
                    keys_dense, values_dense = self._flatten_windows(keys, values)
                else:
                    k_cg, v_cg = self.sidecar.compress_cache(keys, values)
                    keys_dense, values_dense = keys, values
                
                # Compute loss
                loss, metrics = self.loss_fn(
                    queries=queries,
                    keys=keys_dense,
                    values=values_dense,
                    k_cg=k_cg,
                    v_cg=v_cg,
                )
        else:
            # Old API (PyTorch < 2.0): torch.cuda.amp.autocast(enabled=bool)
            # Note: old API doesn't support device_type or dtype parameters
            with autocast(enabled=self.scaler is not None):
                if is_multi_window:
                    k_cg, v_cg = self._compress_multi_window(keys, values)
                    keys_dense, values_dense = self._flatten_windows(keys, values)
                else:
                    k_cg, v_cg = self.sidecar.compress_cache(keys, values)
                    keys_dense, values_dense = keys, values
                
                # Compute loss
                loss, metrics = self.loss_fn(
                    queries=queries,
                    keys=keys_dense,
                    values=values_dense,
                    k_cg=k_cg,
                    v_cg=v_cg,
                )
        
        # Bug #14 & #15 Fix: Always compute and log output norms (outside autocast)
        # This helps us detect if Sidecar is outputting zeros
        with torch.no_grad():
            k_norm = k_cg.norm(dim=-1).mean().item()
            v_norm = v_cg.norm(dim=-1).mean().item()
            
            # Check for vanishing outputs
            if self.global_step % 100 == 0:
                if k_norm < 1e-6 or v_norm < 1e-6:
                    import warnings
                    warnings.warn(
                        f"Step {self.global_step}: Sidecar outputs near zero! "
                        f"k_cg norm: {k_norm:.6f}, v_cg norm: {v_norm:.6f}. "
                        f"This indicates vanishing gradients or bad initialization."
                    )
            
            # Add to metrics (always, not just at log_steps)
            metrics["output/k_cg_norm"] = k_norm
            metrics["output/v_cg_norm"] = v_cg_norm
        
        return loss, metrics
    
    def _compress_multi_window(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress multiple windows separately.
        
        Bug #19 Fix: Compresses each window into a super-token, resulting in
        multiple super-tokens that make attention non-trivial.
        
        Args:
            keys: (batch, num_windows, seq_len, d_head)
            values: (batch, num_windows, seq_len, d_head)
        
        Returns:
            k_cg: (batch, num_windows, d_head) - one super-token per window
            v_cg: (batch, num_windows, d_head)
        """
        batch_size, num_windows, seq_len, d_head = keys.shape
        
        # Reshape to (batch * num_windows, seq_len, d_head)
        keys_flat = keys.reshape(batch_size * num_windows, seq_len, d_head)
        values_flat = values.reshape(batch_size * num_windows, seq_len, d_head)
        
        # Compress each window
        k_cg_flat, v_cg_flat = self.sidecar.compress_cache(keys_flat, values_flat)
        
        # Reshape back to (batch, num_windows, d_head)
        k_cg = k_cg_flat.reshape(batch_size, num_windows, d_head)
        v_cg = v_cg_flat.reshape(batch_size, num_windows, d_head)
        
        return k_cg, v_cg
    
    def _flatten_windows(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten multiple windows into a single sequence for dense attention.
        
        Args:
            keys: (batch, num_windows, seq_len, d_head)
            values: (batch, num_windows, seq_len, d_head)
        
        Returns:
            keys_flat: (batch, num_windows * seq_len, d_head)
            values_flat: (batch, num_windows * seq_len, d_head)
        """
        batch_size, num_windows, seq_len, d_head = keys.shape
        keys_flat = keys.reshape(batch_size, num_windows * seq_len, d_head)
        values_flat = values.reshape(batch_size, num_windows * seq_len, d_head)
        return keys_flat, values_flat
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Run evaluation."""
        self.sidecar.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = self._move_batch_to_device(batch)
            
            keys = batch["keys"]
            values = batch["values"]
            queries = batch["queries"]
            
            # Handle dimension mismatch
            q_dim = queries.size(-1)
            k_dim = keys.size(-1)
            if q_dim != k_dim:
                if q_dim > k_dim:
                    queries = queries[..., :k_dim]
                else:
                    queries = torch.nn.functional.pad(queries, (0, k_dim - q_dim))
            
            k_cg, v_cg = self.sidecar.compress_cache(keys, values)
            loss, _ = self.loss_fn(queries, keys, values, k_cg, v_cg)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device and convert to target dtype."""
        result = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                # Move to device and convert dtype to match model
                result[k] = v.to(device=self.device, dtype=self.target_dtype)
            else:
                result[k] = v
        return result
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb."""
        if self.wandb_run:
            wandb.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        
        torch.save({
            "model_state_dict": self.sidecar.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__,
            "sidecar_config": self.sidecar.config.__dict__,
        }, checkpoint_path)
        
        # Manage checkpoint limit
        self._cleanup_checkpoints(checkpoint_dir)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.sidecar.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if checkpoint["scaler_state_dict"] and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        
        print(f"Resumed from checkpoint: {path} (step {self.global_step})")
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints to stay within limit."""
        checkpoints = sorted(
            checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()


def load_sidecar(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Sidecar:
    """
    Load a trained Sidecar from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to
    
    Returns:
        Loaded Sidecar model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    
    # Reconstruct config
    sidecar_config = SidecarConfig(**checkpoint["sidecar_config"])
    
    # Create model
    sidecar = Sidecar(sidecar_config)
    sidecar.load_state_dict(checkpoint["model_state_dict"])
    
    if device:
        sidecar.to(device)
    
    return sidecar

