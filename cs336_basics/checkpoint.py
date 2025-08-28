import os
import typing
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    Save model, optimizer, and iteration state to a checkpoint file.

    This function should dump all the state from the first three parameters into the
    file-like object out. You can use the state_dict method of both the model and the
    optimizer to get their relevant states and use torch.save(obj, out) to dump obj
    into out (PyTorch supports either a path or a file-like object here). A typical
    choice is to have obj be a dictionary, but you can use whatever format you want
    as long as you can load your checkpoint later.

    Args:
        model: The PyTorch model to save
        optimizer: The optimizer to save
        iteration: The current iteration number
        out: Output path or file-like object to save to
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a checkpoint and restore model and optimizer states.

    This function should load a checkpoint from src (path or file-like object), and
    then recover the model and optimizer states from that checkpoint. Your function
    should return the iteration number that was saved to the checkpoint. You can use
    torch.load(src) to recover what you saved in your save_checkpoint implementation,
    and the load_state_dict method in both the model and optimizers to return them
    to their previous states.

    Args:
        src: Source path or file-like object to load from
        model: The PyTorch model to restore state to
        optimizer: The optimizer to restore state to

    Returns:
        The iteration number that was saved in the checkpoint
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
