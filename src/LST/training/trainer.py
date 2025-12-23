"""
LST Trainer
===========

Main training loop for Learned Super-Token compression.

Features:
    - Multi-objective loss (PPL + QPAA + Diversity)
    - Warmup for auxiliary losses
    - Gradient clipping
    - WandB logging
    - Checkpointing
"""

import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import CombinedLoss, LossWeights
from ..sidecar import SidecarPPL

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainerConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2000

    # Loss weights
    lambda_ppl: float = 1.0
    lambda_qpaa: float = 0.5
    lambda_diversity: float = 0.1
    qpaa_warmup_steps: int = 200  # Steps before QPAA is fully enabled
    num_probes: int = 8

    # Compression
    window_size: int = 8
    num_sink: int = 4
    num_recent: int = 8

    # Logging
    log_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 500

    # Paths
    output_dir: str = "./checkpoints/lst"

    # Reproducibility
    seed: int = 42


def set_seed(seed: int) -> None:
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LSTTrainer:
    """
    Trainer for LST sidecar networks.

    Implements multi-objective training with:
        - PPL loss for generation quality
        - Query-Probing (QPAA) for arbitrary query robustness
        - Diversity loss to prevent collapse

    Args:
        model: Frozen language model
        sidecar: Trainable sidecar network
        config: Training configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        wandb_project: Optional WandB project name
        run_name: Optional run name
    """

    def __init__(
        self,
        model: nn.Module,
        sidecar: SidecarPPL,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        wandb_project: str | None = None,
        run_name: str | None = None,
    ):
        self.model = model
        self.sidecar = sidecar
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_project = wandb_project

        # Device
        self.device = next(model.parameters()).device

        # Loss function
        loss_weights = LossWeights(
            ppl=config.lambda_ppl,
            query_probing=config.lambda_qpaa,
            diversity=config.lambda_diversity,
        )
        self.loss_fn = CombinedLoss(
            weights=loss_weights,
            num_probes=config.num_probes,
            warmup_steps=config.qpaa_warmup_steps,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            sidecar.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, step / config.warmup_steps) if config.warmup_steps > 0 else 1.0,
        )

        # WandB
        if WANDB_AVAILABLE and wandb_project:
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=vars(config),
            )

        # Get d_head from model
        if hasattr(model.config, "head_dim"):
            self.d_head = model.config.head_dim
        else:
            self.d_head = model.config.hidden_size // model.config.num_attention_heads

    def compress_cache(
        self,
        cache: list[tuple[Tensor, Tensor]],
    ) -> tuple[list[tuple[Tensor, Tensor]], Tensor | None]:
        """
        Compress KV cache with sidecar.

        Returns:
            Tuple of (compressed_cache, super_tokens_flat)
        """
        compressed = []
        all_super_tokens = []

        for k, v in cache:
            B, H, S, D = k.shape

            if S <= self.config.num_sink + self.config.num_recent + self.config.window_size:
                compressed.append((k, v))
                continue

            # Split: sink, middle, recent
            ks = k[:, :, : self.config.num_sink, :]
            vs = v[:, :, : self.config.num_sink, :]
            kr = k[:, :, -self.config.num_recent :, :]
            vr = v[:, :, -self.config.num_recent :, :]
            km = k[:, :, self.config.num_sink : -self.config.num_recent, :]
            vm = v[:, :, self.config.num_sink : -self.config.num_recent, :]

            M = km.shape[2]
            num_windows = M // self.config.window_size

            if num_windows == 0:
                compressed.append((k, v))
                continue

            # Reshape windows
            trim_len = num_windows * self.config.window_size
            km_windows = km[:, :, :trim_len, :].view(B, H, num_windows, self.config.window_size, D)
            vm_windows = vm[:, :, :trim_len, :].view(B, H, num_windows, self.config.window_size, D)
            km_leftover = km[:, :, trim_len:, :]
            vm_leftover = vm[:, :, trim_len:, :]

            # Flatten for sidecar
            km_flat = km_windows.reshape(-1, self.config.window_size, D)
            vm_flat = vm_windows.reshape(-1, self.config.window_size, D)
            kv_concat = torch.cat([km_flat, vm_flat], dim=-1)

            # Compress (with gradients)
            k_comp, v_comp = self.sidecar(kv_concat)

            # Store super-tokens for diversity loss
            super_tokens = torch.cat([k_comp, v_comp], dim=-1)  # (N, 2*D)
            all_super_tokens.append(super_tokens)

            # Reshape back
            k_comp = k_comp.view(B, H, num_windows, D)
            v_comp = v_comp.view(B, H, num_windows, D)

            # Concatenate
            k_new = torch.cat([ks, k_comp, km_leftover, kr], dim=2)
            v_new = torch.cat([vs, v_comp, vm_leftover, vr], dim=2)

            compressed.append((k_new, v_new))

        # Flatten all super-tokens for diversity loss
        if all_super_tokens:
            super_tokens_flat = torch.cat(all_super_tokens, dim=0)
        else:
            super_tokens_flat = None

        return compressed, super_tokens_flat

    def train_step(
        self,
        input_ids: Tensor,
        step: int,
    ) -> dict[str, float]:
        """
        Single training step.

        Args:
            input_ids: Token IDs (B, S)
            step: Current step number

        Returns:
            Loss dictionary
        """
        B, S = input_ids.shape

        # Split into prefix and suffix
        prefix_len = S // 2
        prefix = input_ids[:, :prefix_len]
        suffix = input_ids[:, prefix_len:]

        # Get dense cache from prefix (frozen)
        with torch.no_grad():
            outputs = self.model(prefix, use_cache=True)
            dense_cache = list(outputs.past_key_values)

        # Compress cache (with gradients through sidecar)
        self.sidecar.train()
        compressed_cache, super_tokens = self.compress_cache(dense_cache)

        # Compute combined loss
        total_loss, loss_dict = self.loss_fn(
            self.model,
            suffix,
            tuple(compressed_cache),
            tuple(dense_cache),
            super_tokens,
            step=step,
        )

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.sidecar.parameters(),
            self.config.grad_clip,
        )
        self.optimizer.step()
        self.scheduler.step()

        return loss_dict

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.sidecar.eval()
        all_losses = []

        for batch in self.val_loader:
            batch = batch.to(self.device)
            B, S = batch.shape

            prefix_len = S // 2
            prefix = batch[:, :prefix_len]
            suffix = batch[:, prefix_len:]

            outputs = self.model(prefix, use_cache=True)
            dense_cache = list(outputs.past_key_values)

            compressed_cache, super_tokens = self.compress_cache(dense_cache)

            _, loss_dict = self.loss_fn(
                self.model,
                suffix,
                tuple(compressed_cache),
                tuple(dense_cache),
                super_tokens,
            )
            all_losses.append(loss_dict)

        # Average losses (handle empty validation set)
        avg_losses = {}
        if not all_losses:
            logger.warning("No validation samples processed")
            return {"val/loss": 0.0}

        for key in all_losses[0]:
            avg_losses[f"val/{key}"] = np.mean([loss_d[key] for loss_d in all_losses])

        return avg_losses

    def save_checkpoint(self, step: int, path: str) -> None:
        """Save checkpoint."""
        torch.save(
            {
                "step": step,
                "model_state_dict": self.sidecar.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    "d_head": self.d_head,
                    "window_size": self.config.window_size,
                    "hidden_dim": self.sidecar.hidden_dim,
                    "num_encoder_layers": len(self.sidecar.encoder.layers),
                },
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def train(self) -> None:
        """Main training loop."""
        set_seed(self.config.seed)
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("Starting training...")
        logger.info(f"  Max steps: {self.config.max_steps}")
        logger.info(
            f"  Loss weights: PPL={self.config.lambda_ppl}, "
            f"QPAA={self.config.lambda_qpaa}, Div={self.config.lambda_diversity}"
        )

        global_step = 0
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.config.max_steps, desc="Training")

        while global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = batch.to(self.device)

            # Train step
            loss_dict = self.train_step(batch, global_step)

            global_step += 1
            pbar.update(1)

            # Logging
            if global_step % self.config.log_steps == 0:
                ppl = np.exp(loss_dict.get("ppl", 0))
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss_dict['total']:.4f}",
                    ppl=f"{ppl:.2f}",
                    qpaa=f"{loss_dict.get('qpaa', 0):.4f}",
                    lr=f"{lr:.2e}",
                )

                if WANDB_AVAILABLE and self.wandb_project:
                    wandb.log(
                        {
                            "train/loss": loss_dict["total"],
                            "train/ppl": ppl,
                            "train/ppl_loss": loss_dict.get("ppl", 0),
                            "train/qpaa_loss": loss_dict.get("qpaa", 0),
                            "train/diversity_loss": loss_dict.get("diversity", 0),
                            "train/lr": lr,
                            "step": global_step,
                        }
                    )

            # Validation
            if global_step % self.config.eval_steps == 0:
                val_losses = self.validate()
                if val_losses:
                    val_ppl = np.exp(val_losses.get("val/ppl", 0))
                    logger.info(
                        f"Step {global_step}: val_loss={val_losses.get('val/total', 0):.4f}, "
                        f"val_ppl={val_ppl:.2f}"
                    )

                    if WANDB_AVAILABLE and self.wandb_project:
                        val_losses["step"] = global_step
                        wandb.log(val_losses)

            # Checkpointing
            if global_step % self.config.save_steps == 0:
                ckpt_path = os.path.join(self.config.output_dir, f"step_{global_step}.pt")
                self.save_checkpoint(global_step, ckpt_path)

        pbar.close()

        # Save final
        final_path = os.path.join(self.config.output_dir, "final.pt")
        self.save_checkpoint(global_step, final_path)
        logger.info(f"Training complete! Final model: {final_path}")

        if WANDB_AVAILABLE and self.wandb_project:
            wandb.finish()
