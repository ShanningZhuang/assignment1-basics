# 1. IMPORTS AND SETUP
import hydra
import torch, numpy as np, os, time, logging
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from cs336_basics.data_loader import get_batch  # your modules
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.Loss import cross_entropy_loss
from cs336_basics.Optimizer import gradient_clipping
from tqdm import tqdm
import wandb


def update_learning_rate(optimizer, lr_schedule, iteration, cfg):
    """Update learning rate based on schedule."""
    new_lr = lr_schedule.get_lr(t=iteration)

    # Update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    return new_lr


def train_step(model, optimizer, lr_schedule, batch, iteration, cfg):
    """
    Perform one training step.

    Args:
        model: TransformerLM model
        optimizer: Optimizer (AdamW, SGD, etc.)
        batch: Tuple of (input_tokens, target_tokens)

    Returns:
        loss: Training loss for this batch
    """
    model.train()  # Set model to training mode

    # Unpack batch
    input_tokens, target_tokens = batch  # x1, x2

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass: get logits from model
    logits = model(input_tokens)  # Shape: [batch_size, context_length, vocab_size]

    # Reshape for loss computation
    # cross_entropy_loss expects: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten logits and targets
    logits_flat = logits.view(
        -1, vocab_size
    )  # [batch_size * context_length, vocab_size]
    targets_flat = target_tokens.view(-1)  # [batch_size * context_length]

    # Compute loss
    loss = cross_entropy_loss(logits_flat, targets_flat)

    # Backward pass
    loss.backward()

    # Optional: Gradient clipping (recommended for transformer training)
    if hasattr(cfg, "gradient_clip_norm") and cfg.gradient_clip_norm > 0:
        gradient_clipping(model.parameters(), cfg.gradient_clip_norm)

    # Optimizer step
    update_learning_rate(optimizer, lr_schedule, iteration, cfg)
    optimizer.step()

    return loss


def validate(model, val_data, cfg):
    """Run validation and return average loss."""
    model.eval()  # Set to evaluation mode
    total_loss = 0
    num_batches = min(
        cfg.val_batches, len(val_data) // (cfg.batch_size * cfg.context_length)
    )

    with torch.no_grad():  # Disable gradients for efficiency
        for _ in range(num_batches):
            batch = get_batch(val_data, cfg.batch_size, cfg.context_length, cfg.device)
            input_tokens, target_tokens = batch

            # Forward pass only
            logits = model(input_tokens)

            # Compute loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_tokens.view(-1)

            loss = cross_entropy_loss(logits_flat, targets_flat)
            total_loss += loss.item()

    return total_loss / num_batches


def log_metrics(metrics_dict):
    """Log metrics to all configured destinations."""
    # Console logging
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
    print(f"[{metrics_dict['iteration']}] {metrics_str}")

    # Weights & Biases
    wandb.log(metrics_dict)


# 6. MAIN TRAINING LOOP
@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    # 2. LOAD CONFIG
    # Print the full config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Access config values
    print(f"Training for {cfg.max_iterations} iterations")
    print(f"Model has {cfg.model.d_model} dimensions")
    print(f"Learning rate: {cfg.optimizer.lr}")

    # Setup device
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # 3. DATA LOADING
    # Setup data paths
    data_dir = Path(cfg.data_dir)
    train_data = np.memmap(data_dir / cfg.data.train_file, dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / cfg.data.val_file, dtype=np.uint16, mode="r")

    # 4. MODEL AND OPTIMIZER SETUP
    # Initialize model using Hydra's instantiate
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)

    # Initialize optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    lr_schedule = hydra.utils.instantiate(cfg.lr_schedule)

    # Logging setup
    wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg))

    # 5. TRAINING Loop
    # Training metrics tracking
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    for iteration in tqdm(range(cfg.max_iterations), desc="Training"):
        # Training step
        batch = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        loss = train_step(model, optimizer, lr_schedule, batch, iteration, cfg).item()
        train_losses.append(loss)
        # Logging
        if iteration % cfg.log_freq == 0:
            avg_train_loss = np.mean(train_losses[-cfg.log_freq :])
            log_dict = {
                "iteration": iteration,
                "train_loss": loss,
                "train_loss_recent": avg_train_loss,
                "train_ppl": np.exp(loss),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            log_metrics(log_dict)

        # Validation
        if iteration % cfg.val_freq == 0:
            val_loss = validate(model, val_data, cfg)
            val_losses.append(val_loss)
            val_log_dict = {
                "iteration": iteration,
                "val_loss": val_loss,
                "val_ppl": np.exp(val_loss),
            }
            log_metrics(val_log_dict)

        # Checkpointing
        if iteration % cfg.save_freq == 0:
            checkpoint_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
            save_checkpoint(
                model,
                optimizer,
                iteration,
                f"{checkpoint_dir}/checkpoint_{iteration}.pt",
            )

    wandb.finish()
    print("Training Finish")


# 7. ENTRY POINT
if __name__ == "__main__":
    train()
