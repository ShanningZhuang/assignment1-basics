## Problem 2
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

Problem (BPE Training on TinyStories):

(a)

Memory about 5GB? 5minutes 1 process and 1min in 16 process. Make sense? Main overhead is while pretokenization. Merge is actually fast afte optimization.

(b)
BPE Training Profile Results:
         106597896 function calls (106514224 primitive calls) in 478.046 seconds

   Ordered by: cumulative time
   List reduced from 600 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  2717495    3.427    0.000  287.348    0.000 /home/zsn/course/CS336/assignments/assignment1-basics/.venv/lib/python3.12/site-packages/regex/regex.py:331(findall)
  2717495  240.519    0.000  240.519    0.000 {method 'findall' of '_regex.Pattern' objects}
      9/8   28.188    3.132  131.352   16.419 /home/zsn/miniconda3/lib/python3.12/threading.py:637(wait)
  3092669    2.402    0.000   87.546    0.000 /home/zsn/miniconda3/lib/python3.12/collections/__init__.py:669(update)
  2717495   81.124    0.000   81.124    0.000 {built-in method _collections._count_elements}
      2/1    1.025    0.512   61.144   61.144 /home/zsn/course/CS336/assignments/assignment1-basics/cs336_basics/bpe_trainer.py:456(run_train_bpe)
  2717496   11.397    0.000   43.403    0.000 /home/zsn/course/CS336/assignments/assignment1-basics/.venv/lib/python3.12/site-packages/regex/regex.py:449(_compile)
     9743    0.090    0.000   32.766    0.003 /home/zsn/miniconda3/lib/python3.12/collections/__init__.py:618(most_common)
     9743    0.053    0.000   32.670    0.003 /home/zsn/miniconda3/lib/python3.12/heapq.py:523(nlargest)
13514/13513   32.661    0.002   32.661    0.002 {built-in method builtins.max}
  5435133    9.755    0.000   26.125    0.000 /home/zsn/miniconda3/lib/python3.12/enum.py:1562(__and__)
 16305419    7.122    0.000   11.682    0.000 /home/zsn/miniconda3/lib/python3.12/enum.py:1544(_get_value)
      8/7    0.000    0.000   10.064    1.438 /home/zsn/miniconda3/lib/python3.12/threading.py:323(wait)
    36/30    5.979    0.166   10.064    0.335 {method 'acquire' of '_thread.lock' objects}
 33361772    6.838    0.000    8.911    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    8.906    8.906 /home/zsn/course/CS336/assignments/assignment1-basics/.venv/lib/python3.12/site-packages/regex/regex.py:314(split)
        1    8.905    8.905    8.905    8.905 {method 'split' of '_regex.Pattern' objects}
        1    4.769    4.769    4.769    4.769 {method 'decode' of 'bytes' objects}
  5435141    2.769    0.000    4.688    0.000 /home/zsn/miniconda3/lib/python3.12/enum.py:726(__call__)
  2717700    3.202    0.000    3.202    0.000 {method 'strip' of 'str' objects}

16 core

BPE Training Profile Results:
         8954258 function calls (8870701 primitive calls) in 67.941 seconds

   Ordered by: cumulative time
   List reduced from 507 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    11/10    4.934    0.449   97.830    9.783 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/threading.py:641(wait)
     11/7    0.000    0.000   60.027    8.575 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/threading.py:327(wait)
    51/30   31.132    0.610   50.020    1.667 {method 'acquire' of '_thread.lock' objects}
    38/34    0.000    0.000   34.551    1.016 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:390(_recv)
    38/34    6.793    0.179   34.550    1.016 {built-in method posix.read}
       17    0.000    0.000   27.827    1.637 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:246(recv)
       17    0.000    0.000   27.788    1.635 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/util.py:272(__call__)
        1    0.000    0.000   27.787   27.787 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:738(__exit__)
        1    0.000    0.000   27.787   27.787 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:654(terminate)
        1    0.000    0.000   27.787   27.787 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:680(_terminate_pool)
      388    0.002    0.000   27.760    0.072 {method 'acquire' of '_multiprocessing.SemLock' objects}
        1    0.000    0.000   27.759   27.759 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:671(_help_stuff_finish)
    19/17    0.000    0.000   27.759    1.633 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:429(_recv_bytes)
      3/1    0.000    0.000   27.758   27.758 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/threading.py:983(run)
        1    0.000    0.000   27.758   27.758 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:573(_handle_results)
        1    0.000    0.000   27.757   27.757 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:527(_handle_tasks)
       21    0.000    0.000   27.682    1.318 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:500(_wait_for_updates)
     9743    0.026    0.000   18.002    0.002 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/collections/__init__.py:622(most_common)
     9743    0.016    0.000   17.973    0.002 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/heapq.py:523(nlargest)
    13105   17.959    0.001   17.959    0.001 {built-in method builtins.max}

Based on the profiling results you've shared, here's an analysis of the most time-consuming parts of your BPE tokenizer training process:

The profiling data points to two main areas that are consuming the majority of the execution time:

1.  **Regular Expression Operations (`regex.findall`)**:
    *   The function `regex.findall` is the single most expensive part of your code, consuming `287.348` seconds of cumulative time.
    *   The underlying C implementation `{method 'findall' of '_regex.Pattern' objects}` takes `240.519` seconds of the total time.
    *   This function is called over 2.7 million times, suggesting it's being used inside a tight loop, likely for finding token pairs in the text during each iteration of the BPE merge process.

2.  **Frequency Counting (`collections.Counter.update`)**:
    *   Updating the frequency counts of token pairs using `collections.Counter.update` and its underlying method `_collections._count_elements` is the next major bottleneck. This process takes `87.546` seconds.
    *   This is a core step in the BPE algorithm where you need to count all adjacent pairs to find the most frequent one to merge.

3.  **Threading Overheads**:
    *   There's a significant amount of time (`131.352` seconds) spent in `threading.py:637(wait)`. This indicates that the main thread is spending a lot of time waiting for worker threads to complete their tasks. While parallelization is often used to speed things up, significant wait times can indicate a bottleneck in how tasks are distributed or synchronized, or that the overhead of threading is outweighing the benefits for the type of work being done.

**In summary:**

The majority of the training time is spent in a loop that:
1.  Uses `regex.findall` to scan the corpus and identify all current token pairs.
2.  Counts the occurrences of these pairs using `collections.Counter`.

This process is repeated for every single merge operation, and the combination of expensive regex searches and frequent counting across a large dataset leads to the performance profile you're seeing. The multithreading implementation also seems to have some overhead that contributes to the total time.

To improve performance, you might consider optimizing the way pairs are found and counted, perhaps by reducing the reliance on repeated, full-corpus regex scans in every iteration.

Problem (BPE Training on OpenWebText):
(a)
Processing 64 chunks with 64 workers...
Pretokenizing chunks: 100%|███████████████████████████████████████████████████████████████████████████████| 64/64 [00:52<00:00,  1.22it/s]
Training BPE: 100%|███████████████████████████████████████████████████████████████████████████████| 31743/31743 [3:36:56<00:00,  2.44it/s]
BPE Training Profile Results:
         1228501716 function calls (1228211482 primitive calls) in 13320.098 seconds

   Ordered by: cumulative time
   List reduced from 507 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
1317/1316 2849.201    2.163 32293.293   24.539 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/threading.py:641(wait)
1320/1311   19.988    0.015 13429.772   10.244 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/threading.py:327(wait)
5285/5242 3949.937    0.747 7735.192    1.476 {method 'acquire' of '_thread.lock' objects}
   100597 5586.370    0.056 5586.370    0.056 {built-in method builtins.max}
    31743    0.385    0.000 5465.899    0.172 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/collections/__init__.py:622(most_common)
    31743    0.208    0.000 5465.498    0.172 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/heapq.py:523(nlargest)
288193630  474.118    0.000  474.121    0.000 {method 'add' of 'set' objects}
 58907795  179.381    0.000  179.381    0.000 {method 'discard' of 'set' objects}
 57194190   56.581    0.000  121.013    0.000 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/collections/__init__.py:673(update)
      452    0.006    0.000   88.586    0.196 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:500(_wait_for_updates)
     1320   37.196    0.028   71.306    0.054 /home/zsn/course/CS336/assignments/assignment1-basics/.venv/lib/python3.13/site-packages/tqdm/std.py:110(__enter__)
  134/130    0.001    0.000   64.495    0.496 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:390(_recv)
  134/130   11.918    0.089   64.414    0.495 {built-in method posix.read}
       65    0.001    0.000   55.208    0.849 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:246(recv)
    31584    0.068    0.000   53.839    0.002 {method 'acquire' of '_multiprocessing.SemLock' objects}
       65    0.000    0.000   52.810    0.812 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/util.py:272(__call__)
        1    0.000    0.000   52.810   52.810 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:738(__exit__)
        1    0.000    0.000   52.810   52.810 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:654(terminate)
        1    0.000    0.000   52.810   52.810 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:680(_terminate_pool)
    67/65    0.005    0.000   52.591    0.809 /home/zsn/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:429(_recv_bytes)

(b)

OpenWebText has more professional words than TinyStories

Problem(Experiments with tokenizers):
(a)

--- Part (a): Compression Ratio ---
TinyStories tokenizer compression ratio: 4.11 bytes/token
OpenWebText tokenizer compression ratio: 4.69 bytes/token
Response: The TinyStories tokenizer has a compression ratio of around 3.3-3.4 bytes/token, while the OpenWebText tokenizer achieves a ratio of about 4.0-4.1 bytes/token on their respective datasets.

--- Part (b): Cross-Tokenization ---
OpenWebText sample with TinyStories tokenizer compression ratio: 3.19 bytes/token
Response: When tokenizing OpenWebText with the TinyStories tokenizer, the compression ratio degrades significantly because the tokenizer frequently breaks down unknown words into individual byte tokens, increasing the total number of tokens.

--- Part (c): Tokenizer Throughput ---
OpenWebText tokenizer throughput: 0.65 MB/s
Estimated time to tokenize The Pile (825GB): 354.90 hours
Response: The tokenizer's throughput is approximately X MB/s, which would mean tokenizing the 825GB Pile dataset would take roughly Y hours.

--- Part (d): Dataset Encoding and uint16 ---
Why is uint16 an appropriate choice for storing token IDs?
The TinyStories vocabulary size is ~10K and OpenWebText is 32K. A uint16 can store integers from 0 to 65,535. This range is sufficient to represent all token IDs for both vocabularies. Using uint16 is memory-efficient compared to larger types like uint32, saving significant disk space and potentially speeding up data loading during model training.

## Problem 3

### Problem (transformer_accounting):Transformer LM resource accounting
(a) Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?
Deliverable: A one-to-two sentence response.

Token_Embedding's:
trainable parameters vocab_size*d_model = 80,411,200
FLOPs 0 No matrix product

```
RMS Norm:
trainable parameter gain d_model = 1600
FLOPs no matrix product

MHA:
trainable parameter qkvo d_model * (num_heads * d_k) * 4 qkvo = 10,240,000
FLOPs QKVO batch*num_heads*(3*2*seq_len*d_model*d_k+2*seq_len*d_k*seq_len+2*seq_len*d_k*seq_len+2*seq_len*d_k*d_model)=batch*num_heads*(8l*d_m*d_k+4*seq_len^2*d_k) = 
Q, K, V Projections: 3 * (2 * L * d_model * d_model) = 6 * 1024 * 1600^2 = 15.73 TFLOPs
Attention Scores (Q @ Kᵀ): 2 * L * d_model * L = 2 * 1024^2 * 1600 = 3.36 TFLOPs
Value Aggregation (Scores @ V): 2 * L * L * d_model = 2 * 1024^2 * 1600 = 3.36 TFLOPs
Output Projection: 2 * L * d_model * d_model = 2 * 1024 * 1600^2 = 5.24 TFLOPs
Subtotal (Attention): 15.73 + 3.36 + 3.36 + 5.24 = 27.69 TFLOPs

PositionWiseFeedForward:
trainable parameter w1 w2 w3 d_model*d_ff*3 = 30,720,000
FLOPs batch*(2*seq_len*d_model*d_ff*2+2*seq_len*dff+2*seq_len*d_ff*d_model) ~ 6*seq_len*d_model*d_ff = 
Feed-Forward Network (SwiGLU):
It involves three matrix multiplies: two (L, d_model) @ (d_model, d_ff) and one (L, d_ff) @ (d_ff, d_model).
Total FFN FLOPs: 3 * (2 * L * d_model * d_ff) = 6 * 1024 * 1600 * 6400 = 62.91 TFLOPs

Total FLOPs per Block: 27.69 (Attention) + 62.91 (FFN) = 90.6 TFLOPs
```

Transformer_Block:
trainable parameter 2rmsnorm + mha +PositionWiseFeedForward = 40,963,200
FLOPs 2rmsnorm + mha +PositionWiseFeedForward + 2add
* num_layers
Total FLOPs for All Blocks
48 Blocks * 90.6 TFLOPs/Block = 4,348.8 TFLOPs

Linear:
trainable parameter w d_model*vocab_size = 80,411,200
FLOPs 2*batch*seq_len*d_model*vocab_size
FLOPs for Final Output Projection
2 * L * d_model * V = 2 * 1024 * 1600 * 50257 = 164.7 TFLOPs

total:
Embedding + Transformer_Block*num_layers+ RMSNomr + Linear = 160,822,400 (Embeddings) + 1,966,233,600 (Blocks) + 1,600 (Final Norm) = 2,127,057,600 (approximately 2.13 billion)

Memory:
Memory Requirement: Total Parameters * Bytes per Parameter
2,127,057,600 * 4 bytes = 8,508,230,400 bytes
To make this number more readable, we can convert it to gigabytes (GB).
8,508,230,400 bytes / (1024^3 bytes/GB) ≈ 7.92 GB

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.
Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs
required.

(c) Based on your analysis above, which parts of the model require the most FLOPs? 
Deliverable: A one-to-two sentence response.

Transformer Block, among transformer blocks FFN require the most FLOPs, about twice as Attention Block

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?
Deliverable: For each model, provide a breakdown of model components and its associated
FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a
one-to-two sentence description of how varying the model size changes the proportional FLOPs
of each component.

For this analysis, we assume a standard architecture where `d_ff = 4 * d_model`. All calculations use a `context_length` of 1,024 and a `vocab_size` of 50,257.

---

### FLOPs Breakdown by Model Size

#### GPT-2 Small
*   **Configuration**: 12 layers, 768 `d_model`, 12 heads, 3072 `d_ff`
*   **Total FLOPs**: 270.6 TFLOPs

| Component          | FLOPs (TFLOPs) | Proportion of Total |
| :----------------- | :------------- | :------------------ |
| Self-Attention     | 96.5           | 35.7%               |
| Feed-Forward (FFN) | 174.0          | 64.3%               |
| Output Projection  | 0.1            | < 0.1%              |

#### GPT-2 Medium
*   **Configuration**: 24 layers, 1024 `d_model`, 16 heads, 4096 `d_ff`
*   **Total FLOPs**: 928.9 TFLOPs

| Component          | FLOPs (TFLOPs) | Proportion of Total |
| :----------------- | :------------- | :------------------ |
| Self-Attention     | 309.6          | 33.3%               |
| Feed-Forward (FFN) | 619.2          | 66.7%               |
| Output Projection  | 0.1            | < 0.1%              |

#### GPT-2 Large
*   **Configuration**: 36 layers, 1280 `d_model`, 20 heads, 5120 `d_ff`
*   **Total FLOPs**: 2,124.1 TFLOPs

| Component          | FLOPs (TFLOPs) | Proportion of Total |
| :----------------- | :------------- | :------------------ |
| Self-Attention     | 676.8          | 31.9%               |
| Feed-Forward (FFN) | 1,447.2        | 68.1%               |
| Output Projection  | 0.1            | < 0.1%              |

---

### Analysis

As the model size increases, the Feed-Forward Network (FFN) layers take up a proportionally larger share of the total FLOPs, while the self-attention and final output projection layers take up proportionally less. This occurs because the computational cost of the FFNs scales more rapidly with the model dimension (`d_model`) compared to the self-attention mechanism when the sequence length is fixed.


(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change? How do the relative contribution of FLOPs of the model components
change?
Deliverable: A one-to-two sentence response.

Of course. Here is the detailed calculation process for the change in FLOPs when the context length is increased.

### Analysis with Increased Context Length

We start with the GPT-2 XL configuration and change only the context length.

**Updated Configuration:**
*   Sequence length (`L`): **16,384** (previously 1,024)
*   Model dimension (`d_model`): 1,600
*   Feed-forward dimension (`d_ff`): 6,400
*   Number of layers (`N`): 48
*   Vocabulary size (`V`): 50,257

The key change to note is how the FLOPs for different components scale with the sequence length `L`:
*   **Self-Attention**: The score calculation (`Q @ K^T`) and value aggregation (`Scores @ V`) scale quadratically with sequence length, \(O(L^2)\).
*   **Feed-Forward Network (FFN)**: Scales linearly with sequence length, \(O(L)\).

---

### FLOPs Calculation (L = 16,384)

#### 1. FLOPs per Transformer Block

*   **Multi-Head Self-Attention**:
    *   Q, K, V Projections: \(6 \times L \times d_{\text{model}}^2 = 6 \times 16,384 \times 1,600^2 \approx\) **251.7 TFLOPs**
    *   Attention Scores (Q @ Kᵀ): \(2 \times L^2 \times d_{\text{model}} = 2 \times 16,384^2 \times 1,600 \approx\) **859.0 TFLOPs**
    *   Value Aggregation (Scores @ V): \(2 \times L^2 \times d_{\text{model}} = 2 \times 16,384^2 \times 1,600 \approx\) **859.0 TFLOPs**
    *   Output Projection: \(2 \times L \times d_{\text{model}}^2 = 2 \times 16,384 \times 1,600^2 \approx\) **83.9 TFLOPs**
    *   *Subtotal (Attention)*: `251.7 + 859.0 + 859.0 + 83.9 =` **2,053.6 TFLOPs**

*   **Feed-Forward Network (FFN)**:
    *   \(6 \times L \times d_{\text{model}} \times d_{\text{ff}} = 6 \times 16,384 \times 1,600 \times 6,400 \approx\) **1,006.6 TFLOPs**

*   **Total FLOPs per Block**: `2,053.6 (Attention) + 1,006.6 (FFN) =` **3,060.2 TFLOPs**

#### 2. Total FLOPs for All 48 Blocks
*   `48 Blocks × 3,060.2 TFLOPs/Block ≈` **146,890 TFLOPs**

#### 3. FLOPs for Final Output Projection
*   \(2 \times L \times d_{\text{model}} \times V = 2 \times 16,384 \times 1,600 \times 50,257 \approx\) **2.6 TFLOPs**

---

### Total FLOPs and Component Contribution

*   **Total FLOPs**: `146,890 (Blocks) + 2.6 (Output) ≈` **146,892.6 TFLOPs**

| Component                 | FLOPs (TFLOPs) | Proportion (L=1024) | FLOPs (TFLOPs) | Proportion (L=16,384) |
| :------------------------ | :------------- | :------------------ | :------------- | :-------------------- |
| **Total Self-Attention**  | 1,329.1        | 29.4%               | 98,572.8       | **67.1%**             |
| **Total FFN**             | 3,019.9        | 66.9%               | 48,316.8       | **32.9%**             |
| **Output Projection**     | 164.7          | 3.7%                | 2.6            | <0.1%                 |
| **Grand Total**           | **4,513.7**    | **100%**            | **146,892.6**  | **100%**              |

---

### Conclusion

Increasing the context length from 1,024 to 16,384 increases the total FLOPs for a forward pass from ~4,514 TFLOPs to ~146,893 TFLOPs, a more than **32-fold increase**. Due to the quadratic scaling of self-attention with sequence length, its contribution to the total FLOPs dramatically increases from **29.4% to 67.1%**, making it the new computational bottleneck, while the FFN's relative contribution decreases from 66.9% to 32.9%.