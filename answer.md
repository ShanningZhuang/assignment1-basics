Problem (unicode1):

1) Null
2) repr(x): developer-facing, shows escape codes for non-printable characters (\x00, \n, \t, etc.).

print(x): user-facing, shows the actual character (if printable) or nothing (if invisible control).
3) >>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring

Problem (unicode2):

1) Training a tokenizer on **UTF-8 bytes** is preferred because UTF-8 is the de facto web/text standard, ASCII-compatible, and produces a compact, unambiguous byte stream where every character has a unique encoding without endianness issues—unlike UTF-16/UTF-32, which are less space-efficient for common text and introduce surrogate pairs or wasted bytes. This ensures consistency, efficiency, and broad compatibility across languages and platforms.

2) ❌ Incorrect function
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


For ASCII (e.g., "hello"), each character is one byte, so it accidentally works.

For multi-byte characters (Japanese, emoji, etc.), splitting destroys the encoding.

3) Example two-byte sequence
b'\xC0\xAF'

Explanation

This is an overlong encoding of the character / (U+002F); overlong forms are explicitly forbidden in UTF-8, so this byte sequence is invalid and cannot decode to any Unicode character(s).

