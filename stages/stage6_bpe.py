"""
🔆 TEJA — Stage 6: BPE Tokenizer from Scratch
===============================================
Built from zero. Trained to shine.
Created by Sabith, Nilambur, Kerala.

STAGE 6 GOAL:
    Until now we used `tiktoken` — a library that handles tokenization
    for us. In Stage 6 we throw it away and BUILD OUR OWN BPE tokenizer
    from scratch using nothing but pure Python.

    This is NOT a toy. The exact same algorithm powers:
        - GPT-2 (vocab=50,257)
        - GPT-4 (vocab=100,277)
        - LLaMA (vocab=32,000)
        - Claude (vocab=~100K)

    After Stage 6, you will understand EXACTLY how every character of
    text gets converted into numbers before entering the model.

    No GPU needed. Runs on your local machine. Pure Python.

=======================================================================
CONCEPT 1: WHY NOT CHARACTER-LEVEL?
=======================================================================

    Stage 1-4 used character-level tokenization:
        "hello" → [h, e, l, l, o] → [35, 24, 39, 39, 46]

    Problems:
    1. SHORT CONTEXT: Each token = 1 character.
       block_size=256 → only 256 characters ≈ 2-3 sentences.

    2. WASTEFUL: Common words cost many tokens.
       "therefore" = 9 tokens.  The model learns to spell, not think.

    3. IGNORES WORD STRUCTURE: The model must figure out from scratch
       that "run", "running", "runner" are related.

=======================================================================
CONCEPT 2: WHY NOT WORD-LEVEL?
=======================================================================

    The opposite approach — one token per word:
        "hello world" → ["hello", "world"] → [1024, 5432]

    Problems:
    1. HUGE VOCABULARY: English has ~170,000 words.
       That's a massive embedding table.

    2. OUT-OF-VOCABULARY: "Nilambur" isn't in the dictionary.
       Unknown words just become <UNK> — information lost.

    3. MORPHOLOGY LOST: "run" and "running" are completely different
       tokens with no shared representation.

=======================================================================
CONCEPT 3: BYTE PAIR ENCODING — THE SWEET SPOT
=======================================================================

    BPE was originally a DATA COMPRESSION algorithm (1994).
    Applied to NLP tokenization by Sennrich et al. (2015).
    GPT-2 (2019) popularized it for large language models.

    Core idea: start with bytes, merge common pairs greedily.

    STEP 1: Encode text as UTF-8 bytes
        "hello" → [104, 101, 108, 108, 111]
        Every possible text = sequence of bytes 0-255
        Initial vocabulary = 256 tokens (one per byte value)

    STEP 2: Count all adjacent pairs
        In "hello": (104,101), (101,108), (108,108), (108,111)
        Across ALL text: which pair appears most often?

    STEP 3: Merge the most frequent pair
        Suppose (101, 32) = "e " (letter e followed by space)
        appears 5,000 times. Merge it into a new token: 256.
        Replace every occurrence of (101, 32) with 256.

    STEP 4: Repeat
        Now count pairs again. Most frequent → new token 257.
        Keep going until vocab_size is reached.

    RESULT: A vocabulary of common subwords:
        Token 256 = "e "   (e + space — common word ending)
        Token 300 = "the"  (very common English word)
        Token 400 = " in"  (space + "in")
        Token 800 = " the" (space + "the" — most common in English)

    ENCODING NEW TEXT:
        Apply the learned merges in order (earliest first) to greedily
        compress the byte sequence. The result is a short token sequence.

    KEY INSIGHT:
        Common substrings get merged early → become single tokens.
        Rare strings stay as multiple byte-level tokens.
        This naturally handles ALL languages and even code.

=======================================================================
CONCEPT 4: COMPRESSION = CONTEXT
=======================================================================

    BPE is essentially a compression algorithm.

    With block_size=256:
        Character-level: 256 chars ≈ 2-3 sentences
        BPE (vocab=512):  256 tokens ≈ 3-4 sentences
        BPE (vocab=50K):  256 tokens ≈ 800-1000 chars ≈ full paragraph

    More compression → more context per forward pass → smarter model.
    This is why GPT-4's 100K vocab is better than GPT-2's 50K vocab.

=======================================================================
LET'S BUILD IT!  🔆
=======================================================================
"""

import os
import sys
import json
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from teja.utils import print_banner

print_banner("BPE Tokenizer from Scratch", 6)


# ========================================================================
# CORE BPE FUNCTIONS
# These two functions are the ENTIRE algorithm. Everything else is
# just bookkeeping around them.
# ========================================================================

def get_stats(ids):
    """
    Count frequency of every adjacent pair in ids.

    Example:
        ids = [104, 101, 108, 108, 111]   ("hello" as bytes)
        returns: {(104,101):1, (101,108):1, (108,108):1, (108,111):1}

    We use zip(ids, ids[1:]) to get all consecutive pairs efficiently.
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Replace every occurrence of `pair` in `ids` with `idx`.

    This is a single BPE merge step.

    Example:
        ids  = [104, 101, 108, 108, 111]
        pair = (108, 108)   ← most frequent pair
        idx  = 256          ← new token id
        returns: [104, 101, 256, 111]   ← (108,108) → 256

    We scan left to right. When we find the pair, emit idx and skip
    both elements. Otherwise emit the current element normally.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# ========================================================================
# BPE TOKENIZER CLASS
# ========================================================================

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer — built from scratch.

    Attributes:
        merges : dict[(int, int) → int]
            The learned merge rules. Maps a pair of token ids to the
            new merged token id. ORDER MATTERS — merges are applied
            in the order they were learned during training.

        vocab  : dict[int → bytes]
            Maps every token id to its byte representation.
            Used for decoding: token_id → bytes → string.

    Usage:
        tok = BPETokenizer()
        tok.train(text, vocab_size=512)
        ids = tok.encode("Hello, world!")
        txt = tok.decode(ids)
    """

    def __init__(self):
        self.merges = {}   # (int, int) → int
        self.vocab  = {}   # int → bytes

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train(self, text, vocab_size, verbose=True):
        """
        Learn BPE merge rules from text.

        Algorithm:
            1. Encode text as UTF-8 bytes → list of ints (0-255)
            2. For each merge (vocab_size - 256 total merges):
               a. Count all adjacent pairs
               b. Find the most frequent pair
               c. Assign it a new token id (256, 257, 258, ...)
               d. Replace all occurrences of the pair with the new id
               e. Record the merge rule

        Args:
            text:       String to train on
            vocab_size: Target vocabulary size (must be >= 256)
            verbose:    Print each merge as it's learned
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 (256 base byte tokens)"

        # Step 1: text → bytes → list of ints
        ids = list(text.encode('utf-8'))
        original_length = len(ids)

        # Initial vocabulary: 256 tokens, one per byte value
        # vocab[i] = bytes([i])  means token i represents the byte value i
        self.vocab = {i: bytes([i]) for i in range(256)}

        num_merges = vocab_size - 256
        print(f"\nTraining BPE tokenizer:")
        print(f"  Text length: {original_length:,} bytes")
        print(f"  Target vocab size: {vocab_size}")
        print(f"  Number of merges: {num_merges}")
        print(f"  Initial tokens: {len(ids):,}")
        print()

        t0 = time.time()

        for i in range(num_merges):
            # Count all adjacent pairs in the current token sequence
            stats = get_stats(ids)
            if not stats:
                print(f"  No more pairs to merge after {i} merges.")
                break

            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            freq = stats[pair]

            # New token id: starts at 256, increases by 1 each merge
            idx = 256 + i

            # Replace all occurrences of pair with the new token
            ids = merge(ids, pair, idx)

            # Record the merge rule
            self.merges[pair] = idx

            # The new token's byte representation = concatenation of its two parts
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                try:
                    token_str = self.vocab[idx].decode('utf-8')
                except:
                    token_str = repr(self.vocab[idx])
                print(f"  merge {i+1:>4}/{num_merges}: "
                      f"({pair[0]:3}, {pair[1]:3}) → {idx} "
                      f"= {repr(token_str):<20} "
                      f"freq={freq:,}")

        elapsed = time.time() - t0
        compression = original_length / len(ids)
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"  Original: {original_length:,} tokens")
        print(f"  After BPE: {len(ids):,} tokens")
        print(f"  Compression ratio: {compression:.2f}x  "
              f"(each BPE token ≈ {compression:.1f} bytes)")

    # ------------------------------------------------------------------
    # ENCODING
    # ------------------------------------------------------------------

    def encode(self, text):
        """
        Convert text to a list of token ids using learned merge rules.

        Algorithm:
            1. Convert text to bytes → list of ints (0-255)
            2. Apply merge rules greedily, in the order they were learned
               (earliest merge = highest priority)
            3. Repeat until no more merges can be applied

        The key insight: we apply merges in TRAINING ORDER.
        The first merge learned (most frequent pair) gets applied first.
        This matches how the vocabulary was built.

        Example:
            text = "hello"
            bytes = [104, 101, 108, 108, 111]
            if (108, 108) → 256 was learned:
                → [104, 101, 256, 111]
            if (104, 101) → 257 was learned:
                → [257, 256, 111]
        """
        ids = list(text.encode('utf-8'))

        # Keep applying merges until none are possible
        while len(ids) >= 2:
            stats = get_stats(ids)

            # Among all current pairs, find the one with the LOWEST merge index
            # (i.e., the merge that was learned EARLIEST in training)
            # Use float('inf') as fallback for pairs not in merges dict
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))

            # If this pair isn't in our merge rules, we're done
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    # ------------------------------------------------------------------
    # DECODING
    # ------------------------------------------------------------------

    def decode(self, ids):
        """
        Convert a list of token ids back to text.

        Simple: look up each token id in vocab → get bytes → join → decode UTF-8.

        The vocab dict maps every token id (including merged tokens)
        to the bytes it represents. So decoding is just a lookup.
        """
        token_bytes = b''.join(self.vocab[i] for i in ids)
        return token_bytes.decode('utf-8', errors='replace')

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save(self, path):
        """Save tokenizer to a JSON file."""
        data = {
            'merges': {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()},
            'vocab':  {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Tokenizer saved to: {path}")

    @classmethod
    def load(cls, path):
        """Load tokenizer from a JSON file."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        tok = cls()
        tok.merges = {
            tuple(map(int, k.split(','))): v
            for k, v in data['merges'].items()
        }
        tok.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        print(f"Tokenizer loaded from: {path}")
        return tok

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def vocab_size(self):
        return len(self.vocab)


# ========================================================================
# DEMO: TRAIN ON SHAKESPEARE
# ========================================================================

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')
print(f"Loading text from: {data_path}")

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Text length: {len(text):,} characters")

# Train with a small vocab first (fast, educational)
# vocab_size = 256 base bytes + 256 merges = 512 total tokens
VOCAB_SIZE = 512

tok = BPETokenizer()
tok.train(text, vocab_size=VOCAB_SIZE, verbose=True)


# ========================================================================
# SECTION 1: VISUALIZE WHAT WAS LEARNED
# ========================================================================
print("\n" + "="*60)
print("  WHAT DID BPE LEARN?")
print("="*60)
print("\nFirst 20 merges (most frequent byte pairs in Shakespeare):\n")

for i, (pair, idx) in enumerate(list(tok.merges.items())[:20]):
    a = tok.vocab[pair[0]]
    b = tok.vocab[pair[1]]
    merged = tok.vocab[idx]
    try:
        a_str      = a.decode('utf-8')
        b_str      = b.decode('utf-8')
        merged_str = merged.decode('utf-8')
    except:
        a_str      = repr(a)
        b_str      = repr(b)
        merged_str = repr(merged)
    print(f"  {i+1:>3}. {repr(a_str):>10} + {repr(b_str):<10} → {repr(merged_str)}")

print("\nLast 5 merges (rarer combinations):\n")
for pair, idx in list(tok.merges.items())[-5:]:
    merged_str = tok.vocab[idx]
    try:
        merged_str = merged_str.decode('utf-8')
    except:
        merged_str = repr(merged_str)
    print(f"  token {idx}: {repr(merged_str)}")


# ========================================================================
# SECTION 2: ENCODING EXAMPLES
# ========================================================================
print("\n" + "="*60)
print("  ENCODING EXAMPLES")
print("="*60)

test_sentences = [
    "To be, or not to be, that is the question.",
    "Hello, world!",
    "the the the the the",
]

for sentence in test_sentences:
    char_tokens  = list(sentence)                    # character-level
    bpe_tokens   = tok.encode(sentence)              # our BPE
    decoded_back = tok.decode(bpe_tokens)            # verify round-trip

    print(f"\nText:      {repr(sentence)}")
    print(f"Chars:     {len(char_tokens):>3} tokens  {char_tokens[:10]}...")
    print(f"BPE:       {len(bpe_tokens):>3} tokens  {bpe_tokens[:10]}...")
    print(f"Decoded:   {repr(decoded_back)}")
    print(f"Lossless:  {'✓ YES' if decoded_back == sentence else '✗ NO'}")
    if len(char_tokens) > 0:
        ratio = len(char_tokens) / len(bpe_tokens)
        print(f"Compression: {ratio:.1f}x  "
              f"({len(char_tokens)} chars → {len(bpe_tokens)} tokens)")


# ========================================================================
# SECTION 3: COMPARE WITH TIKTOKEN
# ========================================================================
print("\n" + "="*60)
print("  COMPARISON: OUR BPE  vs  tiktoken (GPT-2)")
print("="*60)

try:
    import tiktoken
    gpt2_enc = tiktoken.get_encoding('gpt2')

    sample = "To be, or not to be, that is the question."

    our_ids  = tok.encode(sample)
    gpt2_ids = gpt2_enc.encode(sample)

    print(f"\nSample: {repr(sample)}\n")
    print(f"  Our BPE   (vocab={tok.vocab_size:,}):  "
          f"{len(our_ids):>3} tokens → {our_ids}")
    print(f"  tiktoken  (vocab=50,257):  "
          f"{len(gpt2_ids):>3} tokens → {gpt2_ids}")
    print(f"\n  tiktoken decoded: {repr(gpt2_enc.decode(gpt2_ids))}")
    print(f"\n  GPT-2's larger vocab → fewer tokens → more context per block!")

except ImportError:
    print("\n  (Install tiktoken to see comparison: pip install tiktoken)")


# ========================================================================
# SECTION 4: SAVE THE TOKENIZER
# ========================================================================
tok_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'teja_bpe.json')
tok.save(tok_path)

# Verify round-trip: save → load → encode → decode
tok2     = BPETokenizer.load(tok_path)
test_str = "Hello from Teja!"
assert tok2.decode(tok2.encode(test_str)) == test_str
print(f"Save/load verified ✓")


# ========================================================================
# SECTION 5: TRAIN A LARGER TOKENIZER
# ========================================================================
print("\n" + "="*60)
print("  SCALING UP: vocab_size = 1000")
print("="*60)

tok_1k = BPETokenizer()
tok_1k.train(text, vocab_size=1000, verbose=False)

sample = "To be, or not to be, that is the question."
print(f"\nSample: {repr(sample)}")
print(f"  vocab=256  (bytes only):  {len(list(sample.encode('utf-8'))):>3} tokens")
print(f"  vocab=512  (our Stage6):  {len(tok.encode(sample)):>3} tokens")
print(f"  vocab=1000 (larger):      {len(tok_1k.encode(sample)):>3} tokens")
try:
    import tiktoken
    gpt2 = tiktoken.get_encoding('gpt2')
    print(f"  vocab=50257 (GPT-2):      {len(gpt2.encode(sample)):>3} tokens")
except ImportError:
    pass
print("\n  Larger vocab = more compression = more context = smarter model!")


# ========================================================================
# SUMMARY
# ========================================================================
print(f"""
{'='*60}
  TEJA — Stage 6 Complete!
{'='*60}

  Tokenizer vocab:  {tok.vocab_size} tokens  (256 bytes + {tok.vocab_size-256} merges)
  Trained on:       Shakespeare ({len(text):,} chars)
  Saved to:         data/teja_bpe.json

  THE BPE ALGORITHM (just 2 core functions!):

    get_stats(ids):
      Count frequency of every adjacent pair.
      One line: dict(zip(ids, ids[1:]))

    merge(ids, pair, idx):
      Replace all occurrences of pair with idx.
      ~10 lines: scan left→right, replace when found.

    train() = call these two functions in a loop.
    encode() = apply merges greedily to new text.
    decode() = vocab lookup → bytes → string.

  COMPRESSION COMPARISON:
    vocab=256   (raw bytes)  → 1.0x compression
    vocab=512   (Stage 6)    → {len(list(text[:1000].encode()))}/{len(tok.encode(text[:1000]))}x on Shakespeare sample
    vocab=50257 (GPT-2)      → ~3-4x on English text

  WHAT COMES NEXT:
    Stage 7: Use OUR tokenizer to retrain the Stage 5 model
    Stage 8: Instruction fine-tuning → teach Teja to follow prompts
    Stage 9: Scale to hundreds of millions of parameters
{'='*60}
""")
