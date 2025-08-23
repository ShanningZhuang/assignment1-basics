import cProfile
import pstats
import json
import time
from cs336_basics.bpe_trainer import run_train_bpe


def train_bpe(input_path):
    # --- Profiling starts here ---
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        vocab, merges, trainer = run_train_bpe(
            input_path=input_path,
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
        )
        trainer.save(vocab, merges, input_path.split(".")[0])
    finally:
        profiler.disable()
        # --- Profiling ends here ---

        # Print the profiling stats
        print("BPE Training Profile Results:")
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(20)  # Print the top 20 time-consuming functions


if __name__ == "__main__":
    train_bpe("data/owt_train.txt")
