import os
import json
from collections import Counter
import regex as re
from tqdm import tqdm


class BPETrainer:
    """
    A BPE Tokenizer that can be trained from a text file.
    It inherits encoding and decoding functionalities from the base Tokenizer class.
    """

    @staticmethod
    def _flatten_and_convert_to_bytes(value):
        """
        Recursively flattens a nested tuple of ints into a bytes object.
        If the value is an int, it's converted to bytes.
        If the value is already bytes, it's returned as is.
        """
        if isinstance(value, bytes):
            return value

        def _flatten_recursive(items):
            flat_list = []
            for item in items:
                if isinstance(item, tuple):
                    flat_list.extend(_flatten_recursive(item))
                else:
                    flat_list.append(item)
            return flat_list

        if isinstance(value, tuple):
            return bytes(_flatten_recursive(value))
        elif isinstance(value, int):
            return bytes([value])
        else:
            # Should not be reached with correct logic, but as a fallback.
            return value

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ):
        """
        Trains the BPE tokenizer from a text file.
        return vocab and merge and class itself
        """
        with open(input_path, "rb") as f:
            input_str = f.read().decode("utf-8")

        # 1. Vocabulary Initialization
        vocab = {i: bytes([i]) for i in range(256)}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")
        merges = []

        # 2. Corpus Pre-processing and Pre-tokenization
        special_tokens_pattern = (
            "|".join(re.escape(token) for token in special_tokens)
            if special_tokens
            else None
        )

        if special_tokens_pattern:
            chunks = re.split(f"({special_tokens_pattern})", input_str)
            text_chunks = [chunk for chunk in chunks if chunk not in special_tokens]
        else:
            text_chunks = [input_str]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretoken_counts = Counter()

        for chunk in text_chunks:
            if chunk.strip():
                pretokens = re.findall(PAT, chunk)
                pretoken_counts.update(pretokens)

        byte_pretoken_counts = {}
        for pretoken, count in pretoken_counts.items():
            byte_tuple = tuple(pretoken.encode("utf-8"))
            byte_pretoken_counts[byte_tuple] = count

        # 3. Merge
        for _ in tqdm(range(vocab_size - len(vocab)), desc="Training BPE"):
            byte_freq = Counter()
            for byte_pretoken, count in byte_pretoken_counts.items():
                for byte_pair in zip(byte_pretoken, byte_pretoken[1:]):
                    byte_freq.update({byte_pair: count})

            if not byte_freq:
                break

            most_frequent_pair = byte_freq.most_common(1)[0][0]
            max_freq = byte_freq[most_frequent_pair]
            tied_pairs = [pair for pair, freq in byte_freq.items() if freq == max_freq]

            if len(tied_pairs) > 1:
                byte_tied_pairs = [
                    (
                        self._flatten_and_convert_to_bytes(pair[0]),
                        self._flatten_and_convert_to_bytes(pair[1]),
                    )
                    for pair in tied_pairs
                ]
                best_pair = tied_pairs[byte_tied_pairs.index(max(byte_tied_pairs))]
            else:
                best_pair = tied_pairs[0]

            new_token = best_pair
            vocab[len(vocab)] = new_token
            merges.append(new_token)

            new_byte_pretoken_counts = {}
            for old_pretoken, count in byte_pretoken_counts.items():
                new_pretoken = list(old_pretoken)
                i = 0
                while i < len(new_pretoken) - 1:
                    if (new_pretoken[i], new_pretoken[i + 1]) == best_pair:
                        new_pretoken = (
                            new_pretoken[:i] + [new_token] + new_pretoken[i + 2 :]
                        )
                    else:
                        i += 1
                new_byte_pretoken_counts[tuple(new_pretoken)] = count
            byte_pretoken_counts = new_byte_pretoken_counts

        # handle the vocab format
        final_vocab = {}
        for token_id, token_value in vocab.items():
            final_vocab[token_id] = self._flatten_and_convert_to_bytes(token_value)

        # Flatten merges
        final_merges = []
        for p1, p2 in merges:
            b1 = self._flatten_and_convert_to_bytes(p1)
            b2 = self._flatten_and_convert_to_bytes(p2)
            final_merges.append((b1, b2))

        return final_vocab, final_merges


def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], BPETrainer]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """
    trainer = BPETrainer()
    vocab, merges = trainer.train(input_path, vocab_size, special_tokens, **kwargs)
    return vocab, merges, trainer
