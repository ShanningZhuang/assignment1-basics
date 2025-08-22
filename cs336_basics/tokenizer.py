from heapq import merge
from typing import Any
import regex as re
from .common import gpt2_bytes_to_unicode
import json


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab: dict[int, bytes] - The tokenizer vocabulary
            merges: list[tuple[bytes, bytes]] - BPE merges
            special_tokens: list[str] | None = None - Optional list of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # Create the reverse mapping for vocab {tokens: id}
        self.vocab_inv = {token: i for i, token in self.vocab.items()}
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges.

        Args:
            vocab_filepath: str - Path to vocabulary file
            merges_filepath: str - Path to merges file
            special_tokens: list[str] | None = None - Optional list of special tokens
        """
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes(
                [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
            )
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return get_tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text: str - Input text to encode

        Returns:
            list[int] - Sequence of token IDs
        """
        # 1. Split by Special Tokens
        special_tokens = self.special_tokens
        vocab = self.vocab
        vocab_inv = self.vocab_inv
        merges = self.merges
        merges_dict = self.merges_dict

        if special_tokens:
            special_pattern = "|".join(
                re.escape(k) for k in sorted(special_tokens, key=len, reverse=True)
            )
            special_chunks = re.split(f"({special_pattern})", text)
        else:
            special_chunks = [text]

        # The result special_chunks is the list of ['words sdfa dsf ', '<special_tokens>','sdfa', 'fdsaf. dsfa']
        # 2. encoding
        # For each chunk if it is special_token then find it in vocab, if not then encode it using merges and vocab
        encoded_tokens = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for chunk in special_chunks:
            if special_tokens is not None and chunk in special_tokens:
                encoded_tokens.append(vocab_inv[chunk.encode("utf-8")])
            else:
                # Use PAT to iteratively handle the chunk (Pre-tokenize)
                for match in re.finditer(PAT, chunk):
                    word_str = match.group()
                    word_bytes = [bytes([b]) for b in word_str.encode("utf-8")]
                    can_merge = True
                    while can_merge:

                        can_merge = False
                        merge_indexs = []
                        # convert word to list of bytes
                        if len(word_bytes) == 1:
                            break
                        # match the merges
                        for bytes_pair in zip(word_bytes, word_bytes[1:]):
                            # in a word will have many matches, we should find the first possible merge rule
                            merge_indexs.append(
                                merges_dict.get(bytes_pair, len(merges) + 1)
                            )
                        # Merge the min indexs
                        min_merge_index = min(merge_indexs)
                        for i in range(len(word_bytes) - 1):
                            if (
                                merge_indexs[i] == min_merge_index
                                and merge_indexs[i] != len(merges) + 1
                            ):
                                word_bytes = (
                                    word_bytes[:i]
                                    + [word_bytes[i] + word_bytes[i + 1]]
                                    + word_bytes[i + 2 :]
                                )
                                can_merge = True
                                break
                    for token_bytes in word_bytes:
                        encoded_tokens.append(vocab_inv[token_bytes])
        return encoded_tokens

    def encode_iterable(self, iterable) -> Any:
        """Given an iterable of strings, return a generator that lazily yields token IDs.

        This is required for memory-efficient tokenization of large files that we cannot
        directly load into memory.

        Args:
            iterable - An iterable of strings (e.g., a Python file handle)

        Returns:
            Generator that yields token IDs
        """
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids: list[int] - Sequence of token IDs to decode

        Returns:
            str - Decoded text
        """
        vocab = self.vocab
        decoded_bytes = b""
        for id in ids:
            decoded_bytes += vocab[id]
        decoded_str = decoded_bytes.decode("utf-8", errors="replace")
        return decoded_str
