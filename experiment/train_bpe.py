import json
import time
from cs336_basics.bpe import run_train_bpe


def train_bpe(
    input_path
):
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(start_time - end_time)
    
if __name__ == "__main__":
    train_bpe('data/TinyStoriesV2-GPT4-train.txt')