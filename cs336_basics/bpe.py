import os
from collections import Counter
import regex as re
from tqdm import tqdm


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


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    print("1. load the input_path")
    # 1. load the input_path
    with open(input_path, "rb") as f:
        strings = f.read().decode("utf-8")
    # 2. train_bpe
    vocab, merges = train_bpe(strings, vocab_size, special_tokens)
        
    return vocab, merges

def train_bpe(
    input_str: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """
        Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    ## TODO
    """
    We don't have to re-tokenize all the text, we only need to update those who is the most frequent appeared.
    So my idea is maintain a dict with {pair:(count,words)}.
    """
    
    # 1. Vocabulary Initialization
    vocab = {i: bytes([i]) for i in range(256)}
    # vocab = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    merges = []
    
    # 2. Corpus Pre-processing and Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Split input string by special tokens and tokenize
    # Create regex pattern for special tokens
    special_tokens_pattern = '|'.join(re.escape(token) for token in special_tokens) if special_tokens else None
    
    # Split by special tokens to get independent chunks
    if special_tokens_pattern:
        chunks = re.split(f'({special_tokens_pattern})', input_str)
        text_chunks = [chunk for chunk in chunks if chunk not in special_tokens]
    else:
        text_chunks = [input_str]
    
    # Pre-tokenize text chunks and count frequencies
    pretoken_counts = Counter()
    
    for chunk in text_chunks:
        if chunk.strip():  # Skip empty chunks
            # Use regex to split chunk into pre-tokens
            pretokens = re.findall(PAT, chunk)
            pretoken_counts.update(pretokens)
    
    # For each item in pretoken_counts, convert it into tuple[bytes]
    byte_pretoken_counts = {}
    for pretoken, count in pretoken_counts.items():
        byte_tuple = tuple(pretoken.encode('utf-8'))
        byte_pretoken_counts[byte_tuple] = count
        
    # 3. Merge
    
    for _ in tqdm(range(vocab_size - len(vocab)), desc="Training BPE"):
        # 1) Count Byte Pairs
        byte_freq = Counter()
        # Iterate through the current pre-token frequency map (the one with byte-tuple keys).
        for byte_pretoken, count in byte_pretoken_counts.items():
            # For each pre-token, find all adjacent pairs of bytes/tokens.
            # Zip will stops when the shorter so that its result is the byte_pair
            for byte_pair in zip(byte_pretoken, byte_pretoken[1:]):
                
                byte_freq.update({byte_pair:count})
        # 2) Find the Best Pair
        # Find the byte pair with the highest frequency count
        # If there's a tie in frequency, choose the pair that is lexicographically greatest
        most_frequent_pair = byte_freq.most_common(1)[0][0] if byte_freq else None
        if most_frequent_pair is None:
            return vocab, []
        
        # Handle ties by finding all pairs with the same max frequency and choosing lexicographically greatest
        max_freq = byte_freq[most_frequent_pair]
        tied_pairs = [pair for pair, freq in byte_freq.items() if freq == max_freq]
        if len(tied_pairs) > 1:
            byte_tied_pairs = [(_flatten_and_convert_to_bytes(pair[0]),_flatten_and_convert_to_bytes(pair[1])) for pair in tied_pairs] # lexicographically greatest rule, strange but in the pdf
            best_pair = tied_pairs[byte_tied_pairs.index(max(byte_tied_pairs))]
        else:
            best_pair = tied_pairs[0]
        # 3) Perform the Merge
        new_token = best_pair
        # if type(best_pair[0]) != int or type(best_pair[1]) != int:
        #     pass
        vocab[len(vocab)] = new_token
        merges.append(new_token)
        
        # 4) Update Pre-tokens
        # Create a new pre-token frequency map.
        # Iterate through the old map and, for each pre-token, replace all occurrences of the "best pair" with the new merged token. For example, `(..., b's', b't', ...)` becomes `(..., b'st', ...)`. This updated map will be used in the next iteration.
        new_byte_pretoken_counts = {}
        
        for old_pretoken, count in byte_pretoken_counts.items():
            # Convert tuple to list for easier manipulation
            new_pretoken = list(old_pretoken)
            
            # Replace all occurrences of the best pair with the new merged token
            i = 0
            while i < len(new_pretoken) - 1:
                if (new_pretoken[i], new_pretoken[i + 1]) == best_pair:
                    # Replace the pair with the new merged token
                    # new_pretoken = new_pretoken[:i] + [vocab[len(vocab)-1]] + new_pretoken[i + 2:]
                    new_pretoken = new_pretoken[:i] + [new_token] + new_pretoken[i + 2:]
                else:
                    i += 1
            
            # Convert back to tuple and add to new map
            new_byte_pretoken_counts[tuple(new_pretoken)] = count
        
        # Update the pretoken counts for next iteration
        byte_pretoken_counts = new_byte_pretoken_counts
        
    # handle the vocab format
    # Convert vocab from dict[int, tuple] to dict[int, bytes]
    final_vocab = {}
    for token_id, token_value in vocab.items():
        final_vocab[token_id] = _flatten_and_convert_to_bytes(token_value)
    vocab = final_vocab

    # Flatten merges
    final_merges = []
    for p1, p2 in merges:
        b1 = _flatten_and_convert_to_bytes(p1)
        b2 = _flatten_and_convert_to_bytes(p2)
        final_merges.append((b1, b2))
    merges = final_merges
    
    return vocab, merges
    
def train_bpe_parrellelize():
    pass