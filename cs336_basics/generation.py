"""
Text generation utilities for language models.

This module implements various sampling strategies for generating text from language models,
including temperature scaling and nucleus (top-p) sampling.
"""

import torch
from .Attention import softmax
from typing import Optional, Union


def temperature_scaled_softmax(
    logits: torch.Tensor, dim, temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply temperature scaling to logits and then softmax.

    Temperature scaling formula: softmax(logits / temperature)
    - Lower temperature (< 1.0): More peaked distribution (more deterministic)
    - Higher temperature (> 1.0): More uniform distribution (more random)
    - Temperature = 1.0: Standard softmax
    - Temperature → 0: Approaches greedy (argmax) selection

    Args:
        logits (torch.Tensor): Raw logits from the model, shape (..., vocab_size)
        temperature (float): Temperature parameter. Must be positive.

    Returns:
        torch.Tensor: Probability distribution after temperature scaling and softmax
    """
    # TODO: Implement temperature scaling
    # Hint: Scale the logits by dividing by temperature, then apply softmax
    # Don't forget to validate the temperature parameter
    if temperature == 0:
        y = torch.zeros_like(logits, device=logits.device)
        y.scatter_(dim, torch.argmax(logits, dim=dim, keepdim=True), 1)
        return y
    if temperature < 0:
        raise ValueError
    logits = logits / temperature
    return softmax(logits, dim)


def top_p_sampling(
    probs: torch.Tensor, p: float = 0.9, return_indices: bool = False
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply nucleus (top-p) sampling to a probability distribution.

    The mathematical definition from the assignment:
    P(x_{t+1} = i | q) = {
        q_i / Σ_{j∈V(p)} q_j   if i ∈ V(p)
        0                       otherwise
    }
    where V(p) is the smallest set of indices such that Σ_{j∈V(p)} q_j ≥ p.

    Algorithm:
    1. Sort probabilities in descending order
    2. Compute cumulative sum of sorted probabilities
    3. Find the cutoff where cumsum ≥ p (this gives you V(p))
    4. Keep only probabilities for indices in V(p)
    5. Renormalize so they sum to 1

    Args:
        probs (torch.Tensor): Probability distribution, shape (..., vocab_size).
                             Should be the output of softmax (i.e., already normalized).
        p (float): Nucleus parameter. Must be between 0 and 1. Higher values include
                  more tokens in the nucleus, lower values are more restrictive.
        return_indices (bool): If True, also return the indices that were kept in the nucleus.

    Returns:
        torch.Tensor: Modified probability distribution with low-probability tokens set to 0
                     and remaining probabilities renormalized.
        If return_indices=True, also returns torch.Tensor of shape (..., vocab_size) with
        boolean mask indicating which indices are in V(p).
    """
    # TODO: Implement nucleus (top-p) sampling
    #
    # Key steps:
    # 1. Validate that p is in the valid range (0, 1]
    # 2. Sort probabilities in descending order (use torch.sort)
    # 3. Compute cumulative sum of sorted probabilities
    # 4. Find which tokens to include in V(p) - the nucleus
    # 5. Create a new probability distribution with only nucleus tokens
    # 6. Renormalize the probabilities so they sum to 1
    # 7. Handle the return_indices flag if needed
    #
    # Special case: Always include at least the top token, even if its probability > p
    # This prevents issues when the highest probability token alone exceeds p.
    if not 0 < p <= 1:
        raise ValueError
    sorted_probs, sorted_indices = torch.sort(
        probs, dim=-1, descending=True
    )  # ..., vocab_size
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    zeros_shape = list(cumsum.shape)
    zeros_shape[-1] = 1
    cumsum = torch.cat(
        [torch.zeros(zeros_shape, device=cumsum.device), cumsum[..., :-1]], dim=-1
    )
    preserved_masks = torch.where(cumsum < p, 1, 0)
    # Scatter back to original order
    original_masks = torch.zeros_like(preserved_masks, device=probs.device)
    original_masks.scatter_(dim=-1, index=sorted_indices, src=preserved_masks)
    probs = probs * original_masks
    probs = probs / probs.sum(dim=-1, keepdim=True)
    if not return_indices:
        return probs
    else:
        return probs, original_masks.bool()


def sample_from_distribution(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """
    Sample token indices from a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution, shape (..., vocab_size)
        num_samples (int): Number of samples to draw

    Returns:
        torch.Tensor: Sampled token indices, shape (..., num_samples)
    """
    # Use torch.multinomial to sample from the probability distribution
    # multinomial expects shape (batch_size, vocab_size) so we need to handle arbitrary shapes
    original_shape = probs.shape[:-1]  # Everything except vocab_size
    vocab_size = probs.shape[-1]

    # Flatten all dimensions except the last one (vocab_size)
    probs_flat = probs.view(-1, vocab_size)  # (batch_size_flat, vocab_size)

    # Sample from each distribution in the flattened batch
    samples_flat = torch.multinomial(
        probs_flat, num_samples=num_samples, replacement=True
    )

    # Reshape back to the original batch dimensions plus num_samples
    samples = samples_flat.view(*original_shape, num_samples)

    return samples


def generate_next_token(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Generate the next token given a sequence of input tokens.

    Args:
        model: The language model (should output logits)
        input_ids (torch.Tensor): Input token sequence, shape (batch_size, seq_len)
        temperature (float): Temperature for scaling logits
        top_p (Optional[float]): If provided, apply nucleus sampling with this parameter
        do_sample (bool): If True, sample from the distribution. If False, use greedy decoding.

    Returns:
        torch.Tensor: Next token indices, shape (batch_size, 1)
    """
    model.eval()
    if not do_sample:
        temperature = 0
    probs = temperature_scaled_softmax(model(input_ids), -1, temperature)
    probs = probs if top_p is None else top_p_sampling(probs, top_p)
    next_token_samples = sample_from_distribution(probs, num_samples=1)[
        :, -1, :
    ]  # batch seq_len 1 -> batch 1 . Sample the last index
    return next_token_samples


def generate_text(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a sequence of tokens from a language model using autoregressive decoding.

    This function implements the standard language model decoding process:
    1. Start with input prompt
    2. Generate one token at a time
    3. Append each new token to the sequence
    4. Use the expanded sequence to predict the next token
    5. Repeat until max_new_tokens or EOS token

    Args:
        model: The language model
        input_ids (torch.Tensor): Input prompt tokens, shape (batch_size, seq_len)
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for sampling
        top_p (Optional[float]): Top-p parameter for nucleus sampling
        do_sample (bool): Whether to sample or use greedy decoding
        eos_token_id (Optional[int]): End-of-sequence token ID. Generation stops when this is sampled.
        pad_token_id (Optional[int]): Padding token ID for when sequences finish early

    Returns:
        torch.Tensor: Generated token sequence including the input prompt,
                     shape (batch_size, seq_len + num_generated_tokens)
    """
    # TODO: Implement autoregressive text generation
    #
    # Key components:
    # 1. Set model to evaluation mode
    # 2. Track which sequences in the batch are still generating (for EOS handling)
    # 3. Loop for max_new_tokens iterations:
    #    a. Generate next token using generate_next_token()
    #    b. Handle EOS tokens (stop generation for those sequences)
    #    c. Handle padding for finished sequences
    #    d. Concatenate new tokens to the sequence
    #    e. Break early if all sequences are done
    # 4. Return the complete generated sequences
    model.eval()
    batch_size = input_ids.shape[0]
    generate_mask = torch.ones(batch_size, device=input_ids.device).bool()
    for _ in range(max_new_tokens):
        active_indices = torch.where(generate_mask)
        masked_ids = input_ids[active_indices]
        masked_next_token = generate_next_token(
            model, masked_ids, temperature, top_p, do_sample
        )  # batch 1
        # Expand input_ids directly using scatter
        new_column = torch.full(
            (batch_size, 1),
            pad_token_id if pad_token_id is not None else 0,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        new_column[active_indices] = masked_next_token
        input_ids = torch.cat([input_ids, new_column], dim=-1)
        # After generating next_token, check for EOS
        if eos_token_id is not None:
            just_hit_eos = new_column.squeeze(-1) == eos_token_id
            generate_mask = generate_mask & ~just_hit_eos

        # Early termination if all sequences finished
        if not generate_mask.any():
            break
    return input_ids
