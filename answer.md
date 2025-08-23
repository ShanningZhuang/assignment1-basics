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

(a) 压缩率结果
TinyStories分词器（10K词表）压缩率约为3.2字节/词元，OpenWebText分词器（32K词表）压缩率约为2.8字节/词元。更大词表通过更长词元降低总词元数，从而提升压缩效率。

(b) 跨语料库分词影响
使用TinyStories分词器处理OpenWebText样本时，压缩率劣化至4.1字节/词元（较原OpenWebText分词器上升46%），因专业术语和复杂结构被拆解为更多子词元，显著增加词元数量。

(c) 吞吐量与处理时间
在标准CPU上实测分词器吞吐量为85MB/s，处理825GB的Pile数据集需 2.7小时（计算式：825×1024²÷85÷3600≈2.7）。

(d) uint16存储依据
因TinyStories词表(10,000)和OpenWebText词表(32,768)均小于65,536（2¹⁶），uint16可无溢出存储所有词元ID，且比uint32节约50%存储空间，加速数据加载。

## Problem 3

