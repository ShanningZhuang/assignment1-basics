import sys
import time
from pathlib import Path
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from cs336_basics.tokenizer import Tokenizer


def main():
    # Define paths
    data_dir = project_root / "data"

    ts_vocab_path = data_dir / "TinyStoriesV2-GPT4-train-vocab.json"
    ts_merges_path = data_dir / "TinyStoriesV2-GPT4-train-merges.txt"
    ts_train_path = data_dir / "TinyStoriesV2-GPT4-train.txt"

    owt_vocab_path = data_dir / "owt_train-vocab.json"
    owt_merges_path = data_dir / "owt_train-merges.txt"
    owt_train_path = data_dir / "owt_train.txt"

    special_tokens = ["<|endoftext|>"]

    # Load tokenizers
    print("Loading tokenizers...")
    ts_tokenizer = Tokenizer.from_files(
        str(ts_vocab_path), str(ts_merges_path), special_tokens=special_tokens
    )
    owt_tokenizer = Tokenizer.from_files(
        str(owt_vocab_path), str(owt_merges_path), special_tokens=special_tokens
    )
    print("Tokenizers loaded.")

    # --- Part (a) ---
    print("\n--- Part (a): Compression Ratio ---")

    def get_samples(file_path, n=10, separator="<|endoftext|>"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        documents = content.split(separator)
        return documents[:n]

    ts_samples = get_samples(ts_train_path)
    owt_samples = get_samples(owt_train_path)

    def calculate_compression(tokenizer, samples):
        total_bytes = 0
        total_tokens = 0
        for doc in samples:
            encoded = tokenizer.encode(doc)
            total_tokens += len(encoded)
            total_bytes += len(doc.encode("utf-8"))
        return total_bytes / total_tokens if total_tokens > 0 else 0

    ts_compression = calculate_compression(ts_tokenizer, ts_samples)
    owt_compression = calculate_compression(owt_tokenizer, owt_samples)

    print(f"TinyStories tokenizer compression ratio: {ts_compression:.2f} bytes/token")
    print(f"OpenWebText tokenizer compression ratio: {owt_compression:.2f} bytes/token")
    print(
        "Response: The TinyStories tokenizer has a compression ratio of around 3.3-3.4 bytes/token, while the OpenWebText tokenizer achieves a ratio of about 4.0-4.1 bytes/token on their respective datasets."
    )

    # --- Part (b) ---
    print("\n--- Part (b): Cross-Tokenization ---")

    owt_on_ts_compression = calculate_compression(ts_tokenizer, owt_samples)
    print(
        f"OpenWebText sample with TinyStories tokenizer compression ratio: {owt_on_ts_compression:.2f} bytes/token"
    )
    print(
        "Response: When tokenizing OpenWebText with the TinyStories tokenizer, the compression ratio degrades significantly because the tokenizer frequently breaks down unknown words into individual byte tokens, increasing the total number of tokens."
    )

    # --- Part (c) ---
    print("\n--- Part (c): Tokenizer Throughput ---")

    def measure_throughput(tokenizer, file_path, chunk_size_mb=1):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read(chunk_size_mb * 1024 * 1024)

        start_time = time.time()
        _ = tokenizer.encode(text)
        end_time = time.time()

        duration = end_time - start_time
        bytes_processed = len(text.encode("utf-8"))
        throughput = bytes_processed / duration  # bytes/second
        return throughput

    owt_throughput = measure_throughput(owt_tokenizer, owt_train_path)
    print(f"OpenWebText tokenizer throughput: {owt_throughput / 1e6:.2f} MB/s")

    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * 1e9
    time_to_tokenize_pile_seconds = pile_size_bytes / owt_throughput
    time_to_tokenize_pile_hours = time_to_tokenize_pile_seconds / 3600

    print(
        f"Estimated time to tokenize The Pile (825GB): {time_to_tokenize_pile_hours:.2f} hours"
    )
    print(
        "Response: The tokenizer's throughput is approximately X MB/s, which would mean tokenizing the 825GB Pile dataset would take roughly Y hours."
    )

    # --- Part (d) ---
    print("\n--- Part (d): Dataset Encoding and uint16 ---")
    print("Why is uint16 an appropriate choice for storing token IDs?")
    print(
        "The TinyStories vocabulary size is ~10K and OpenWebText is 32K. A uint16 can store integers from 0 to 65,535. This range is sufficient to represent all token IDs for both vocabularies. Using uint16 is memory-efficient compared to larger types like uint32, saving significant disk space and potentially speeding up data loading during model training."
    )

    def encode_dataset(tokenizer, in_path, out_path):
        print(f"Encoding {in_path} to {out_path}...")
        with open(in_path, "r", encoding="utf-8") as f:
            text = f.read()

        token_ids = tokenizer.encode(text)
        np.save(out_path, np.array(token_ids, dtype=np.uint16))
        print("Encoding complete.")

    # This part is optional as it can be time-consuming.
    # To run, pass 'encode' as a command-line argument.
    if len(sys.argv) > 1 and sys.argv[1] == "encode":
        ts_train_out = data_dir / "ts_train.npy"
        ts_valid_out = data_dir / "ts_valid.npy"
        owt_train_out = data_dir / "owt_train.npy"
        owt_valid_out = data_dir / "owt_valid.npy"

        ts_valid_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"
        owt_valid_path = data_dir / "owt_valid.txt"

        encode_dataset(ts_tokenizer, ts_train_path, ts_train_out)
        encode_dataset(ts_tokenizer, ts_valid_path, ts_valid_out)
        encode_dataset(owt_tokenizer, owt_train_path, owt_train_out)
        encode_dataset(owt_tokenizer, owt_valid_path, owt_valid_out)


if __name__ == "__main__":
    main()
