import os
import json
from collections import Counter, defaultdict
import regex as re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from .common import gpt2_bytes_to_unicode


def pretokenize_chunk(args):
    """
    Worker function for multiprocessing pretokenization.

    Args:
        args: Tuple containing (input_path, start, end, special_tokens_pattern, PAT)

    Returns:
        Counter: pretoken counts for this chunk
    """
    input_path, start, end, special_tokens_pattern, PAT = args

    # Read the chunk from file
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

    # Handle special tokens if present
    if special_tokens_pattern:
        chunks = re.split(f"({special_tokens_pattern})", chunk_str)
        text_chunks = [
            chunk for chunk in chunks if not re.match(special_tokens_pattern, chunk)
        ]
    else:
        text_chunks = [chunk_str]

    # Pretokenize each text chunk
    pretoken_counts = Counter()
    for text_chunk in text_chunks:
        if text_chunk.strip():
            pretokens = re.findall(PAT, text_chunk)
            pretoken_counts.update(pretokens)

    return pretoken_counts


def find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    with open(file_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


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

    @staticmethod
    def save(vocab, merges, path_to_save):
        """Given the vocab and merges, save them into proper format like GPT2

        Args:
            vocab (dict[int, bytes]): The vocabulary mapping from token ID to token bytes
            merges (list[tuple[bytes, bytes]]): The BPE merges as pairs of bytes
            path_to_save (str): The base path to save the files (without extension)
                Will create two files: {path_to_save}.json and {path_to_save}.txt
        """
        # Get the byte-to-unicode mapping for GPT-2 format
        bytes_to_unicode = gpt2_bytes_to_unicode()

        # Convert vocab to GPT-2 format (string tokens -> token IDs)
        gpt2_vocab = {}
        for token_id, token_bytes in vocab.items():
            # Convert bytes to GPT-2 string representation
            token_str = "".join(bytes_to_unicode[b] for b in token_bytes)
            gpt2_vocab[token_str] = token_id

        # Save vocab as JSON file
        vocab_path = f"{path_to_save}-vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(gpt2_vocab, f, ensure_ascii=False)

        # Convert merges to GPT-2 format and save as text file
        merges_path = f"{path_to_save}-merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            for token1_bytes, token2_bytes in merges:
                # Convert both tokens to GPT-2 string representation
                token1_str = "".join(bytes_to_unicode[b] for b in token1_bytes)
                token2_str = "".join(bytes_to_unicode[b] for b in token2_bytes)
                f.write(f"{token1_str} {token2_str}\n")

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_workers=1,
        **kwargs,
    ):
        """
        Trains the BPE tokenizer from a text file.
        return vocab and merge
        """
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

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Use multiprocessing for pretokenization if num_workers > 1
        if num_workers > 1:
            # Find chunk boundaries based on special tokens
            split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
            boundaries = find_chunk_boundaries(input_path, num_workers, split_token)

            # Prepare arguments for worker processes
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((input_path, start, end, special_tokens_pattern, PAT))

            # Process chunks in parallel
            print(f"Processing {len(chunk_args)} chunks with {num_workers} workers...")
            with Pool(num_workers) as pool:
                chunk_results = list(
                    tqdm(
                        pool.imap(pretokenize_chunk, chunk_args),
                        total=len(chunk_args),
                        desc="Pretokenizing chunks",
                    )
                )

            # Combine results from all chunks
            pretoken_counts = Counter()
            for chunk_counter in chunk_results:
                pretoken_counts.update(chunk_counter)
        else:
            # Single-threaded processing (original implementation)
            with open(input_path, "rb") as f:
                input_str = f.read().decode("utf-8")

            if special_tokens_pattern:
                chunks = re.split(f"({special_tokens_pattern})", input_str)
                text_chunks = [chunk for chunk in chunks if chunk not in special_tokens]
            else:
                text_chunks = [input_str]

            pretoken_counts = Counter()
            for chunk in text_chunks:
                if chunk.strip():
                    pretokens = re.findall(PAT, chunk)
                    pretoken_counts.update(pretokens)

        byte_pretoken_counts = {}
        for pretoken, count in pretoken_counts.items():
            byte_tuple = tuple(pretoken.encode("utf-8"))
            byte_pretoken_counts[byte_tuple] = count

        byte_freq = Counter()
        byte_pair2pretoken = defaultdict(set)
        for byte_pretoken, count in byte_pretoken_counts.items():
            for byte_pair in zip(byte_pretoken, byte_pretoken[1:]):
                byte_freq.update({byte_pair: count})
                byte_pair2pretoken[byte_pair].add(byte_pretoken)

        # 3. Merge
        for _ in tqdm(range(vocab_size - len(vocab)), desc="Training BPE"):

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

            pretoken_changed = byte_pair2pretoken.pop(best_pair)
            byte_freq.pop(best_pair)

            for pretoken in pretoken_changed:
                if pretoken not in byte_pretoken_counts:
                    continue
                count = byte_pretoken_counts.pop(pretoken)
                new_pretoken = list(pretoken)
                i = 0
                while i < len(new_pretoken) - 1:
                    if (new_pretoken[i], new_pretoken[i + 1]) == best_pair:
                        if i > 0:
                            prev_token = new_pretoken[i - 1]
                            old_pair = (prev_token, best_pair[0])
                            byte_freq[old_pair] -= count
                            byte_pair2pretoken[old_pair].discard(pretoken)

                            new_pair = (prev_token, new_token)
                            byte_freq[new_pair] = byte_freq.get(new_pair, 0) + count

                        if i < len(new_pretoken) - 2:
                            next_token = new_pretoken[i + 2]
                            old_pair = (best_pair[1], next_token)
                            byte_freq[old_pair] -= count
                            byte_pair2pretoken[old_pair].discard(pretoken)

                            new_pair = (new_token, next_token)
                            byte_freq[new_pair] = byte_freq.get(new_pair, 0) + count

                        new_pretoken = (
                            new_pretoken[:i] + [new_token] + new_pretoken[i + 2 :]
                        )
                    else:
                        i += 1

                final_pretoken = tuple(new_pretoken)
                byte_pretoken_counts[final_pretoken] = count
                for i in range(len(new_pretoken) - 1):
                    pair = (new_pretoken[i], new_pretoken[i + 1])
                    if pair != best_pair:
                        byte_pair2pretoken[pair].add(final_pretoken)

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

        self.final_vocab = final_vocab
        self.final_merges = final_merges
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
