import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe_save():
    """
    Test that the BPE trainer can save vocab and merges in GPT-2 format
    and that the saved files match the reference files.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Import the BPETrainer to use the save method
    from cs336_basics.bpe_trainer import BPETrainer

    # Save the trained bpe to temporary files
    import tempfile
    import os

    save_path = FIXTURES_PATH / "train-bpe-result"
    BPETrainer.save(vocab, merges, save_path)

    # Load the saved files
    saved_vocab_path = f"{save_path}-vocab.json"
    saved_merges_path = f"{save_path}-merges.txt"

    with open(saved_vocab_path, "r", encoding="utf-8") as f:
        saved_vocab = json.load(f)

    with open(saved_merges_path, "r", encoding="utf-8") as f:
        saved_merges = f.read().strip().split("\n")

    # Load the reference files
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    with open(reference_vocab_path, "r", encoding="utf-8") as f:
        reference_vocab = json.load(f)

    with open(reference_merges_path, "r", encoding="utf-8") as f:
        reference_merges = f.read().strip().split("\n")

    # Compare the vocabularies
    assert (
        saved_vocab.keys() == reference_vocab.keys()
    ), "Saved vocabulary doesn't match reference vocabulary"

    # Compare the merges
    assert saved_merges == reference_merges, "Saved merges don't match reference merges"

    print("✓ Save method works correctly - files match reference files!")


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes(
                [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
            )
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [
        word for word in vocab.values() if word != b"<|endoftext|>"
    ]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
