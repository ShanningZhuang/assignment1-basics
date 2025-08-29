"""
Tests for text generation utilities.

These tests will help you verify your implementation is working correctly.
Run with: python3 -m pytest tests/test_generation.py -v
"""

import torch
import pytest
import numpy as np
from cs336_basics.generation import (
    temperature_scaled_softmax,
    top_p_sampling,
    sample_from_distribution,
    generate_next_token,
    generate_text,
)


class MockTransformerLM(torch.nn.Module):
    """Mock transformer for testing generation functions."""

    def __init__(self, vocab_size: int, seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # Return random logits for testing
        return torch.randn(batch_size, seq_len, self.vocab_size)


def test_temperature_scaled_softmax():
    """Test temperature scaling works correctly."""
    vocab_size = 10
    logits = torch.randn(vocab_size)

    # Test default temperature (should be equivalent to normal softmax)
    probs_temp1 = temperature_scaled_softmax(logits, dim=-1, temperature=1.0)
    probs_normal = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(probs_temp1, probs_normal)

    # Test low temperature (should be more peaked)
    probs_low_temp = temperature_scaled_softmax(logits, dim=-1, temperature=0.1)

    # Low temperature should make the distribution more peaked
    # The max probability should be higher than with normal softmax
    assert probs_low_temp.max() > probs_normal.max()

    # Test high temperature (should be more uniform)
    probs_high_temp = temperature_scaled_softmax(logits, dim=-1, temperature=10.0)

    # High temperature should make the distribution more uniform
    # The entropy should be higher
    entropy_normal = -(probs_normal * torch.log(probs_normal + 1e-10)).sum()
    entropy_high_temp = -(probs_high_temp * torch.log(probs_high_temp + 1e-10)).sum()
    assert entropy_high_temp > entropy_normal

    # Test that probabilities sum to 1
    assert torch.allclose(probs_temp1.sum(), torch.tensor(1.0))
    assert torch.allclose(probs_low_temp.sum(), torch.tensor(1.0))
    assert torch.allclose(probs_high_temp.sum(), torch.tensor(1.0))

    # Test temperature = 0 (should behave like greedy/argmax)
    probs_temp_zero = temperature_scaled_softmax(logits, dim=-1, temperature=0.0)
    expected_greedy = torch.zeros_like(logits)
    expected_greedy[torch.argmax(logits)] = 1.0
    torch.testing.assert_close(probs_temp_zero, expected_greedy)

    # Test error for negative temperature
    with pytest.raises(ValueError):
        temperature_scaled_softmax(logits, dim=-1, temperature=-1.0)


def test_top_p_sampling_basic():
    """Test basic top-p sampling functionality."""
    # Create a simple probability distribution
    probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])

    # Test p=0.8 (should include first two tokens: 0.5 + 0.3 = 0.8)
    filtered_probs = top_p_sampling(probs, p=0.8)

    # Check that low probability tokens are zeroed out
    assert filtered_probs[2] == 0.0  # 0.1 should be excluded
    assert filtered_probs[3] == 0.0  # 0.05 should be excluded
    assert filtered_probs[4] == 0.0  # 0.05 should be excluded

    # Check that the remaining probabilities are renormalized
    assert torch.allclose(filtered_probs.sum(), torch.tensor(1.0))

    # Check that the relative proportions are maintained
    expected_ratio = 0.5 / 0.8  # Should be 0.625
    assert torch.allclose(filtered_probs[0], torch.tensor(expected_ratio), atol=1e-6)


def test_top_p_sampling_edge_cases():
    """Test edge cases for top-p sampling."""
    # Test p=1.0 (should include all tokens)
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    filtered_probs = top_p_sampling(probs, p=1.0)
    torch.testing.assert_close(filtered_probs, probs)

    # Test case where top token alone exceeds p
    probs = torch.tensor([0.9, 0.05, 0.03, 0.02])
    filtered_probs = top_p_sampling(probs, p=0.5)
    # Should still include the top token even though it exceeds p
    assert filtered_probs[0] > 0
    # Should be renormalized to sum to 1
    assert torch.allclose(filtered_probs.sum(), torch.tensor(1.0))

    # Test error for invalid p values
    with pytest.raises(ValueError):
        top_p_sampling(probs, p=0.0)
    with pytest.raises(ValueError):
        top_p_sampling(probs, p=1.5)


def test_top_p_sampling_batch():
    """Test top-p sampling with batched input."""
    batch_size = 3
    vocab_size = 5

    # Create batch of probability distributions
    probs = torch.softmax(torch.randn(batch_size, vocab_size), dim=-1)

    filtered_probs = top_p_sampling(probs, p=0.8)

    # Check shapes
    assert filtered_probs.shape == probs.shape

    # Check that each example sums to 1
    for i in range(batch_size):
        assert torch.allclose(filtered_probs[i].sum(), torch.tensor(1.0))

    # Test with return_indices
    filtered_probs, indices_mask = top_p_sampling(probs, p=0.8, return_indices=True)
    assert indices_mask.shape == probs.shape
    assert indices_mask.dtype == torch.bool

    # Check that the mask correctly identifies included tokens
    for i in range(batch_size):
        assert (filtered_probs[i] > 0).equal(indices_mask[i])


def test_sample_from_distribution():
    """Test sampling from probability distributions."""
    probs = torch.tensor([0.6, 0.3, 0.1])

    # Test single sample
    sample = sample_from_distribution(probs, num_samples=1)
    assert sample.shape == (1,)
    assert 0 <= sample[0] < len(probs)

    # Test multiple samples
    samples = sample_from_distribution(probs, num_samples=100)
    assert samples.shape == (100,)

    # Check that all samples are valid indices
    assert (samples >= 0).all()
    assert (samples < len(probs)).all()

    # Test batched sampling
    batch_probs = torch.stack([probs, probs])
    batch_samples = sample_from_distribution(batch_probs, num_samples=5)
    assert batch_samples.shape == (2, 5)


def test_generate_next_token():
    """Test next token generation."""
    vocab_size = 100
    seq_len = 10
    batch_size = 2

    model = MockTransformerLM(vocab_size, seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test greedy decoding
    next_tokens = generate_next_token(model, input_ids, do_sample=False)
    assert next_tokens.shape == (batch_size, 1)
    assert (next_tokens >= 0).all()
    assert (next_tokens < vocab_size).all()

    # Test sampling
    next_tokens = generate_next_token(model, input_ids, temperature=1.0, do_sample=True)
    assert next_tokens.shape == (batch_size, 1)

    # Test with top-p sampling
    next_tokens = generate_next_token(
        model, input_ids, temperature=1.0, top_p=0.9, do_sample=True
    )
    assert next_tokens.shape == (batch_size, 1)


def test_generate_text():
    """Test full text generation."""
    vocab_size = 50
    seq_len = 5
    batch_size = 2
    max_new_tokens = 10

    model = MockTransformerLM(vocab_size, seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test basic generation
    generated = generate_text(model, input_ids, max_new_tokens=max_new_tokens)

    expected_length = seq_len + max_new_tokens
    assert generated.shape == (batch_size, expected_length)

    # Check that the original input is preserved
    torch.testing.assert_close(generated[:, :seq_len], input_ids)

    # Test with EOS token
    eos_token_id = vocab_size - 1
    generated_with_eos = generate_text(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
    )

    # Generated sequence should be at most the expected length
    assert generated_with_eos.shape[1] <= expected_length

    # Test greedy decoding
    generated_greedy = generate_text(
        model, input_ids, max_new_tokens=max_new_tokens, do_sample=False
    )
    assert generated_greedy.shape == (batch_size, expected_length)


def test_top_p_mathematical_correctness():
    """Test that top-p sampling follows the mathematical definition correctly."""
    # Create a known probability distribution
    probs = torch.tensor([0.4, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])

    p = 0.7
    filtered_probs = top_p_sampling(probs, p=p)

    # Sort original probabilities to find the nucleus
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)

    # Find V(p) - the smallest set such that sum >= p
    # We include tokens where cumsum <= p, plus the boundary token
    nucleus_size = torch.sum(cumsum <= p).item()
    if nucleus_size < len(probs) and cumsum[nucleus_size] > p:
        nucleus_size += 1  # Include the boundary token

    # Check that exactly the right tokens are included
    included_count = torch.sum(filtered_probs > 0).item()

    # For this specific example with p=0.7:
    # 0.4 (cumsum=0.4) + 0.25 (cumsum=0.65) + 0.15 (cumsum=0.8)
    # The first two sum to 0.65 <= 0.7, and adding the third makes 0.8 > 0.7
    # So we should include the first 3 tokens
    assert included_count == 3

    # Check that the included tokens are the top ones
    assert filtered_probs[0] > 0  # 0.4
    assert filtered_probs[1] > 0  # 0.25
    assert filtered_probs[2] > 0  # 0.15
    assert filtered_probs[3] == 0  # 0.10 should be excluded

    # Check renormalization
    original_sum = probs[0] + probs[1] + probs[2]  # 0.4 + 0.25 + 0.15 = 0.8
    assert torch.allclose(filtered_probs.sum(), torch.tensor(1.0))
    assert torch.allclose(filtered_probs[0], probs[0] / original_sum)


def test_temperature_scaling_extreme_values():
    """Test temperature scaling with extreme values."""
    logits = torch.tensor([10.0, 5.0, 1.0, 0.1])

    # Temperature = 0 should be exactly greedy (one-hot at max)
    temp_zero = temperature_scaled_softmax(logits, dim=-1, temperature=0.0)
    expected_onehot = torch.zeros_like(logits)
    expected_onehot[0] = 1.0  # First element has highest logit (10.0)
    torch.testing.assert_close(temp_zero, expected_onehot)

    # Very low temperature should make it almost deterministic
    very_low_temp = temperature_scaled_softmax(logits, dim=-1, temperature=0.01)
    assert very_low_temp[0] > 0.99  # Should be almost all probability on max

    # Very high temperature should make it more uniform
    very_high_temp = temperature_scaled_softmax(logits, dim=-1, temperature=100.0)
    # Should be much more uniform than normal softmax
    normal_softmax = torch.softmax(logits, dim=-1)
    assert very_high_temp.std() < normal_softmax.std()


def test_top_p_sampling_single_token():
    """Test top-p sampling when only one token should be included."""
    # Create distribution where first token has very high probability
    probs = torch.tensor([0.95, 0.02, 0.02, 0.01])

    # With p=0.9, should only include the first token
    filtered_probs = top_p_sampling(probs, p=0.9)

    # Only first token should have non-zero probability
    assert filtered_probs[0] == 1.0
    assert filtered_probs[1] == 0.0
    assert filtered_probs[2] == 0.0
    assert filtered_probs[3] == 0.0


def test_top_p_sampling_all_tokens():
    """Test top-p sampling when all tokens should be included."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

    # With p=1.0, should include all tokens
    filtered_probs = top_p_sampling(probs, p=1.0)
    torch.testing.assert_close(filtered_probs, probs)


def test_sample_from_distribution_determinism():
    """Test that sampling is deterministic when given a one-hot distribution."""
    # Create one-hot distribution
    probs = torch.tensor([0.0, 1.0, 0.0, 0.0])

    # Should always sample index 1
    for _ in range(10):
        sample = sample_from_distribution(probs, num_samples=1)
        assert sample[0] == 1


if __name__ == "__main__":
    # Run some basic tests manually
    print("Running basic tests...")

    # Test if functions exist and have correct signatures
    try:
        # Test temperature scaling
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = temperature_scaled_softmax(logits, dim=-1, temperature=1.0)
        print("✓ temperature_scaled_softmax exists and runs")
    except NotImplementedError:
        print("✗ temperature_scaled_softmax not implemented yet")
    except Exception as e:
        print(f"✗ temperature_scaled_softmax error: {e}")

    try:
        # Test top-p sampling
        probs = torch.tensor([0.5, 0.3, 0.2])
        result = top_p_sampling(probs, p=0.8)
        print("✓ top_p_sampling exists and runs")
    except NotImplementedError:
        print("✗ top_p_sampling not implemented yet")
    except Exception as e:
        print(f"✗ top_p_sampling error: {e}")

    try:
        # Test sampling
        probs = torch.tensor([0.5, 0.3, 0.2])
        result = sample_from_distribution(probs, num_samples=1)
        print("✓ sample_from_distribution exists and runs")
    except NotImplementedError:
        print("✗ sample_from_distribution not implemented yet")
    except Exception as e:
        print(f"✗ sample_from_distribution error: {e}")

    print("\nRun 'python3 -m pytest tests/test_generation.py -v' for full test suite!")
