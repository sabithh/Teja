"""
🔆 TEJA — Stage 5: Real Data + BPE Tokenization + Scale
=========================================================
Built from zero. Trained to shine.
Created by Sabith, Nilambur, Kerala.

STAGE 5 GOAL:
    Take the COMPLETE GPT-2 architecture from Stage 4 and upgrade it
    in three dimensions simultaneously:

        1. DATA       — Shakespeare (1M chars) → OpenWebText (real web data)
        2. TOKENIZER  — Character-level (65 tokens) → BPE (50,257 tokens)
        3. SCALE      — 1.2M params → ~30M params

    Plus three new training techniques:
        4. WEIGHT TYING      — Share embedding and output weights
        5. COSINE LR DECAY   — Warmup + cosine schedule
        6. GRADIENT CLIPPING — Safety net against gradient explosions

    After Stage 5, Teja will generate COHERENT ENGLISH TEXT.
    This is the jump from "it looks like text" to "it reads like text."

=======================================================================
COLAB SETUP (run these cells first in Colab):
=======================================================================

    !pip install tiktoken datasets tqdm

    # Optional: mount Drive to persist checkpoints across sessions
    # from google.colab import drive
    # drive.mount('/content/drive')

    # Then clone the repo and run this file:
    # !git clone https://github.com/sabithh/teja
    # %cd teja
    # !python stages/stage5_scale.py

=======================================================================
CONCEPT 1: BPE TOKENIZATION
=======================================================================

    Until now, Teja used CHARACTER-LEVEL tokenization:

        Vocab: 65 unique characters (a-z, A-Z, punctuation, space, newline)
        "hello" → [h, e, l, l, o] → [35, 24, 39, 39, 46]  (5 tokens)

        Problems:
        1. SHALLOW CONTEXT: block_size=64 only covers ~64 characters
           (about half a sentence). The model can't see patterns that
           span multiple sentences.
        2. WASTEFUL: The model spends capacity learning to spell instead
           of learning language. "international" costs 13 tokens.

    SOLUTION: Byte Pair Encoding (BPE)

    BPE is an algorithm that builds a vocabulary of common SUBWORDS:

        Step 1: Start with all 256 bytes as the initial vocabulary
        Step 2: Count all adjacent pairs of tokens in the corpus
        Step 3: Merge the most frequent pair into a new single token
        Step 4: Repeat until vocab_size is reached

        Example merges (simplified):
          'h' 'e' → 'he'  (very common pair)
          'he' 'l' → 'hel' (if 'hel' is also common)
          'hel' 'lo' → 'hello' (final merged token)

    GPT-2 uses BPE with vocab_size = 50,257 tokens:

        "hello" → [hello]  →  [15496]          (1 token!)
        "international" → ["intern", "ational"] → 2 tokens
        " the" → [" the"] → [1115]             (space included in token)

    With block_size=256 BPE tokens, we see ~1,000 characters of context.
    That's full paragraphs. The model can learn long-range dependencies.

    We use `tiktoken` — OpenAI's fast, Rust-backed BPE tokenizer:

        import tiktoken
        enc = tiktoken.get_encoding("gpt2")  # GPT-2's exact tokenizer
        enc.encode("Hello, world!")  → [15496, 11, 995, 0]
        enc.decode([15496, 11, 995, 0]) → "Hello, world!"
        enc.n_vocab  → 50257

=======================================================================
CONCEPT 2: WEIGHT TYING
=======================================================================

    In our model, there are two places that involve the vocabulary:

    INPUT:  token_embedding_table  — maps token_id → vector
            Shape: [vocab_size × n_embd]  =  [50257 × 384]  ≈ 19M params

    OUTPUT: lm_head                — maps vector → logits over vocab
            Shape: [n_embd × vocab_size]  =  [384 × 50257]  ≈ 19M params

    These are TRANSPOSES of each other. The same semantic information
    (what each token means) is needed in both places.

    Weight Tying = make lm_head.weight = token_embedding_table.weight

    Benefits:
        EFFICIENCY:    Saves ~19M parameters (half the total!)
        CONSISTENCY:   One representation for each token, used both
                       when reading it (input) and predicting it (output)
        REGULARIZATION: Harder to overfit with tied weights

    In code:
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight  # tie

    GPT-2 does this. BERT does this. Most modern LLMs do this.

=======================================================================
CONCEPT 3: COSINE LR SCHEDULE WITH LINEAR WARMUP
=======================================================================

    Fixed LR (Stages 1-4) works for small models. At scale, a schedule
    gives meaningfully better results.

    Our schedule has two phases:

    PHASE 1 — LINEAR WARMUP (first 1,000 steps):

        LR goes from 0 → max_lr linearly.

        Why start at 0? At initialization, weights are random noise.
        A large LR on random gradients can "damage" the network before
        it's learned anything useful. Warmup lets the model orient
        itself with small, careful steps first.

    PHASE 2 — COSINE DECAY (remaining 49,000 steps):

        LR follows a cosine curve from max_lr → min_lr:

        LR(t) = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × t/T))

        t = current step in the decay phase
        T = total decay steps

        At t=0:   cos(0) = 1.0 → LR = max_lr   (start fast)
        At t=T/2: cos(π/2) = 0 → LR = midpoint (medium)
        At t=T:   cos(π) = -1 → LR = min_lr    (end slow)

        Cosine is smoother than linear decay — the learning rate
        changes slowly at the extremes and faster in the middle.
        This gives the model time to exploit what it's learned
        before the LR drops too low.

    Typical setup:
        max_lr = 3e-4
        min_lr = 3e-5  (= max_lr / 10)
        warmup_iters = 1000

=======================================================================
CONCEPT 4: GRADIENT CLIPPING
=======================================================================

    LayerNorm (Stage 4) prevents activations from exploding.
    Gradient clipping prevents GRADIENTS from exploding.

    During backprop, occasional batches can produce very large gradients
    (outlier data, unlucky parameter initialization at a certain step).
    These can cause catastrophic weight updates.

    Solution: clip the global gradient norm to a max value.

        # Before optimizer.step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    This computes the total gradient norm across all parameters:
        total_norm = sqrt(sum(p.grad ** 2 for all params))

    If total_norm > max_norm, it scales ALL gradients proportionally
    so their combined norm equals max_norm.

    If gradients are small (the normal case), this does nothing.
    If they spike (the rare catastrophic case), they're clipped.

    Basically free insurance — add it and forget about it.

=======================================================================
LET'S BUILD IT!  🔆
=======================================================================
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from teja.config import STAGE_5_CONFIG
from teja.utils import get_device, count_parameters, print_banner

# ========================
# REPRODUCIBILITY
# ========================
torch.manual_seed(1337)


# ========================
# HYPERPARAMETERS
# ========================
batch_size      = STAGE_5_CONFIG['batch_size']                   # 16
grad_accum      = STAGE_5_CONFIG['gradient_accumulation_steps']  # 4  → effective batch = 64
eval_batch_size = 4   # smaller batch for eval — logits (4×256×50257) = 205MB vs 820MB at batch=16
block_size    = STAGE_5_CONFIG['block_size']        # 256
learning_rate = STAGE_5_CONFIG['learning_rate']     # 3e-4
max_iters     = STAGE_5_CONFIG['max_iters']         # 50,000
eval_interval = STAGE_5_CONFIG['eval_interval']     # 1,000
eval_iters    = STAGE_5_CONFIG['eval_iters']        # 200
n_embd        = STAGE_5_CONFIG['n_embd']            # 384
n_head        = STAGE_5_CONFIG['n_head']            # 6
n_layer       = STAGE_5_CONFIG['n_layer']           # 6
dropout       = STAGE_5_CONFIG['dropout']           # 0.2

# Stage 5 additions
warmup_iters = 1_000   # linear warmup over first 1K steps
min_lr       = 3e-5    # minimum LR = learning_rate / 10
grad_clip    = 1.0     # clip gradients to this global norm
vocab_size   = 50_257  # GPT-2 BPE vocabulary size

# How many documents to pull from OpenWebText for data prep.
# Each document averages ~1,000-2,000 BPE tokens.
# 100K docs ≈ 150M tokens ≈ 300MB on disk.
# Reduce if you have limited storage or want faster prep.
NUM_TRAIN_DOCS = 100_000

head_size = n_embd // n_head   # 384 / 6 = 64

device = get_device()
print_banner("Real Data + BPE Tokenization + Scale", 5)


# ========================================================================
# SECTION 1: DATA PREPARATION
# ========================================================================
# This section runs ONCE and saves tokenized data to disk as binary files.
# On subsequent runs, it detects the files and skips this step.
#
# Binary format: numpy uint16 array, one token per element.
# uint16 covers 0..65535, which includes all 50,257 BPE tokens.
# At 2 bytes/token: 150M tokens = 300MB — fits easily in RAM or on Colab.
# ========================================================================

def prepare_openwebtext(data_dir):
    """
    Stream OpenWebText from HuggingFace, tokenize with tiktoken,
    and save train.bin + val.bin to data_dir.

    Requires: pip install tiktoken datasets tqdm
    Time: ~10-20 minutes on Colab (network + tokenization)
    Storage: ~300MB for 100K documents

    On subsequent runs, detects existing files and returns immediately.
    """
    train_bin = os.path.join(data_dir, 'train.bin')
    val_bin   = os.path.join(data_dir, 'val.bin')

    if os.path.exists(train_bin) and os.path.exists(val_bin):
        print(f"✓ Tokenized data already exists at {data_dir}")
        return

    print(f"\nPreparing OpenWebText dataset (this runs once)...")
    print(f"  Documents to collect: {NUM_TRAIN_DOCS:,}")
    print(f"  Expected output: ~300MB in {data_dir}/\n")

    # Lazy imports: only needed during data prep
    try:
        import tiktoken
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("ERROR: Missing dependencies for data preparation.")
        print("Run:  pip install tiktoken datasets tqdm")
        sys.exit(1)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # <|endoftext|> token id = 50256

    # Stream OpenWebText — no need to download all 40GB
    print("Streaming OpenWebText from HuggingFace...")
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    # Collect documents
    docs = []
    for doc in tqdm(dataset, total=NUM_TRAIN_DOCS, desc="Collecting docs"):
        docs.append(doc['text'])
        if len(docs) >= NUM_TRAIN_DOCS:
            break

    print(f"Collected {len(docs):,} documents.")

    # Split 95% train / 5% val
    split_idx   = int(0.95 * len(docs))
    train_texts = docs[:split_idx]
    val_texts   = docs[split_idx:]

    def tokenize_and_save(texts, path, desc):
        """Tokenize texts and save as uint16 binary."""
        all_tokens = []
        for text in tqdm(texts, desc=desc):
            # encode_ordinary skips special tokens in source text
            tokens = enc.encode_ordinary(text)
            tokens.append(eot)         # separate documents with <|endoftext|>
            all_tokens.extend(tokens)
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(path)
        print(f"  Saved {len(arr):,} tokens ({len(arr)/1e6:.1f}M) → {path}")
        return len(arr)

    os.makedirs(data_dir, exist_ok=True)
    n_train = tokenize_and_save(train_texts, train_bin, "Tokenizing train")
    n_val   = tokenize_and_save(val_texts,   val_bin,   "Tokenizing val")

    total = n_train + n_val
    print(f"\nDone! Total tokens: {total:,} ({total/1e6:.1f}M)")
    print(f"Files: {data_dir}/train.bin  ({n_train/1e6:.1f}M tokens)")
    print(f"       {data_dir}/val.bin    ({n_val/1e6:.1f}M tokens)")


# ========================================================================
# SECTION 2: LOAD DATA (Memory-Mapped)
# ========================================================================
# np.memmap reads from disk on demand — we never load the full file into RAM.
# For 300MB files this doesn't matter much, but it's the correct pattern
# for datasets in the GB-range (nanoGPT uses this approach).
# ========================================================================

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'openwebtext')
prepare_openwebtext(data_dir)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

print(f"\nDataset loaded:")
print(f"  Train: {len(train_data):,} tokens  ({len(train_data)/1e6:.1f}M)")
print(f"  Val:   {len(val_data):,} tokens  ({len(val_data)/1e6:.1f}M)")
print(f"  Vocab: {vocab_size:,} BPE tokens (GPT-2 tokenizer)")
print(f"  Context window: {block_size} tokens (~{block_size * 4} chars)\n")


def get_batch(split, bs=None):
    """Sample a random batch from train or val data."""
    if bs is None:
        bs = batch_size
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (bs,))
    x = torch.stack([torch.from_numpy(data[i    : i + block_size    ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + block_size + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, bs=eval_batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ========================================================================
# SECTION 3: LEARNING RATE SCHEDULER
# ========================================================================

def get_lr(step):
    """
    Cosine LR schedule with linear warmup.

    step < warmup_iters:  LR increases linearly  0 → learning_rate
    step >= warmup_iters: LR decays via cosine  learning_rate → min_lr
    """
    # Phase 1: linear warmup
    if step < warmup_iters:
        return learning_rate * step / warmup_iters

    # Phase 2: cosine decay
    progress = (step - warmup_iters) / (max_iters - warmup_iters)  # 0.0 → 1.0
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))          # 1.0 → 0.0
    return min_lr + coeff * (learning_rate - min_lr)


# ========================================================================
# MODEL — Same TejaGPT, larger + weight tying
# ========================================================================
# Architecture is IDENTICAL to Stage 4.
# What changes:
#   vocab_size: 65  →  50,257
#   n_embd:    128  →  384
#   n_head:      4  →  6
#   block_size:  64 →  256
#   + weight tying between lm_head and token_embedding_table
# ========================================================================

class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key    = nn.Linear(n_embd, head_size, bias=False)
        self.query  = nn.Linear(n_embd, head_size, bias=False)
        self.value  = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k   = self.key(x)
        q   = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network (GELU activation)."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: Pre-Norm + residual connections (GPT-2 style)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size  = n_embd // n_head
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)
        self.sa    = MultiHeadAttention(n_head, head_size)
        self.ffwd  = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual + pre-norm attention
        x = x + self.ffwd(self.ln2(x)) # residual + pre-norm FFN
        return x


class TejaGPT(nn.Module):
    """
    Teja GPT — Stage 5 version.

    Architecturally identical to Stage 4, but:
      - vocab_size = 50,257  (BPE tokens, not 65 characters)
      - n_embd = 384, n_head = 6, n_layer = 6, block_size = 256
      - Weight tying: token_embedding_table.weight == lm_head.weight
      - Proper weight initialization (std=0.02 like GPT-2)
    """

    def __init__(self):
        super().__init__()

        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)

        # lm_head projects hidden state → logits over vocabulary.
        # bias=False because weight-tied models don't need a bias here.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # ----------------------------------------------------------------
        # WEIGHT TYING
        # The embedding matrix (vocab_size × n_embd) and the output
        # projection matrix (n_embd × vocab_size) encode the same
        # information — what each token means. Sharing them saves ~19M
        # parameters and improves consistency. After this line:
        #     self.token_embedding_table.weight IS self.lm_head.weight
        # ----------------------------------------------------------------
        self.token_embedding_table.weight = self.lm_head.weight

        # GPT-2 weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with std=0.02 (GPT-2 style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                               # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x       = tok_emb + pos_emb                                             # (B, T, n_embd)
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)                                               # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive text generation.

        temperature: values > 1.0 increase randomness, < 1.0 make output
                     more deterministic. 1.0 = raw model distribution.
        top_k:       restrict sampling to the top-k most likely tokens.
                     Common values: 50, 100, 200.
        """
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float('-inf')

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx


# ========================================================================
# CREATE MODEL + OPTIMIZER
# ========================================================================
print("Creating TejaGPT (Stage 5 — Scaled)...")
model = TejaGPT()
model = model.to(device)
total_params = count_parameters(model)

print(f"\n   ARCHITECTURE COMPARISON:")
print(f"     Stage 4 — char vocab=65,    n_embd=128, block=64  →   1.2M params")
print(f"     Stage 5 — BPE  vocab=50257, n_embd=384, block=256 →  {total_params/1e6:.1f}M params")
print()

# AdamW with beta2=0.95 and weight_decay=0.1 (GPT-2 paper values)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)


# ========================================================================
# TRAINING LOOP
# ========================================================================
effective_batch = batch_size * grad_accum
print(f"Training for {max_iters:,} steps...")
print(f"   Arch:       {n_layer} blocks × {n_head} heads × {n_embd} dims")
print(f"   Context:    {block_size} BPE tokens (~{block_size * 4} chars)")
print(f"   Batch:      {batch_size} × {grad_accum} grad accum steps = {effective_batch} effective")
print(f"   Dropout:    {dropout}")
print(f"   LR:         {learning_rate} → {min_lr} (cosine, {warmup_iters} warmup steps)")
print(f"   Grad clip:  {grad_clip}\n")

t0             = time.time()
best_val_loss  = float('inf')
start_step     = 0

checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------------------------------------------------------
# RESUME FROM CHECKPOINT
# Checks Drive first (persists across sessions), then local.
# If found, restores model weights, optimizer state, and step number
# so training continues exactly where it left off.
# -----------------------------------------------------------------------
# Persistent locations — checked in order for resume, saved to all that exist
KAGGLE_CKPT = '/kaggle/working/teja_stage5_best.pt'           # survives Kaggle session end
DRIVE_CKPT  = '/content/drive/MyDrive/teja_checkpoints/teja_stage5_best.pt'  # Colab + Drive
LOCAL_CKPT  = os.path.join(checkpoint_dir, 'teja_stage5_best.pt')

resume_path = (
    KAGGLE_CKPT if os.path.exists(KAGGLE_CKPT) else
    DRIVE_CKPT  if os.path.exists(DRIVE_CKPT)  else
    LOCAL_CKPT  if os.path.exists(LOCAL_CKPT)  else
    None
)

if resume_path:
    print(f"Resuming from: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_step    = ckpt['step']
    best_val_loss = ckpt['val_loss']
    print(f"  Step: {start_step:,} | Best val loss: {best_val_loss:.4f}\n")
else:
    print("No checkpoint found — starting from scratch.\n")

for step in range(start_step, max_iters + 1):

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if step % eval_interval == 0:
        losses  = estimate_loss()
        elapsed = time.time() - t0
        lr_now  = get_lr(step)
        tokens_processed = step * effective_batch * block_size

        print(
            f"   step {step:>6,} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f} | "
            f"lr {lr_now:.2e} | "
            f"tokens {tokens_processed/1e6:.1f}M | "
            f"{elapsed:.0f}s"
        )

        # Save best checkpoint (local + Drive if mounted)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt_data = {
                'step':                 step,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss.item(),
                'train_loss':           losses['train'].item(),
                'vocab_size':           vocab_size,
                'n_embd':               n_embd,
                'n_head':               n_head,
                'n_layer':              n_layer,
                'block_size':           block_size,
                'total_params':         total_params,
            }
            torch.save(ckpt_data, LOCAL_CKPT)
            saved_to = [LOCAL_CKPT]
            # Kaggle: save to /kaggle/working/ — appears in Output tab, survives session end
            if os.path.exists('/kaggle/working'):
                torch.save(ckpt_data, KAGGLE_CKPT)
                saved_to.append('Kaggle Output')
            # Colab: save to Drive if mounted
            if os.path.exists(os.path.dirname(DRIVE_CKPT)):
                torch.save(ckpt_data, DRIVE_CKPT)
                saved_to.append('Google Drive')
            print(f"   ✓ Checkpoint saved → {', '.join(saved_to)} (step {step:,}, val {best_val_loss:.4f})")

    if step == max_iters:
        break

    # ------------------------------------------------------------------
    # Update learning rate (cosine schedule with warmup)
    # ------------------------------------------------------------------
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ------------------------------------------------------------------
    # Forward + backward with gradient accumulation
    #
    # Instead of one big batch (batch_size=64), we run grad_accum=4
    # smaller batches (batch_size=16) and SUM their gradients before
    # stepping. Mathematically identical to a single batch of 64 —
    # but uses 4× less VRAM at any one moment.
    #
    # We divide the loss by grad_accum so the gradient magnitude stays
    # the same as if we'd done one big batch.
    # ------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum):
        xb, yb   = get_batch('train')
        _, loss  = model(xb, yb)
        loss     = loss / grad_accum   # scale loss so gradients average correctly
        loss.backward()                # accumulates into .grad buffers

    # GRADIENT CLIPPING — clips total gradient norm to grad_clip (=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()


# ========================================================================
# GENERATE SAMPLE TEXT
# ========================================================================
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    print("\nGenerating sample text from trained model:")
    print("=" * 60)
    model.eval()

    # Prompt-conditioned generation
    prompts = ["The history of", "In recent years,", "Scientists have discovered"]
    for prompt in prompts:
        start_ids = enc.encode(prompt)
        context   = torch.tensor([start_ids], dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=200)
        text      = enc.decode(generated[0].tolist())
        print(f"\nPrompt: '{prompt}'")
        print(text)

    print("=" * 60)

except ImportError:
    print("(Install tiktoken to see generated text: pip install tiktoken)")


# ========================================================================
# SAVE FINAL CHECKPOINT
# ========================================================================
losses = estimate_loss()
final_path = os.path.join(checkpoint_dir, 'teja_stage5_final.pt')
torch.save({
    'step':             max_iters,
    'model_state_dict': model.state_dict(),
    'val_loss':         losses['val'].item(),
    'train_loss':       losses['train'].item(),
    'vocab_size':       vocab_size,
    'n_embd':           n_embd,
    'n_head':           n_head,
    'n_layer':          n_layer,
    'block_size':       block_size,
    'total_params':     total_params,
}, final_path)
print(f"\nCheckpoint saved: {final_path}")


# ========================================================================
# SUMMARY
# ========================================================================
print(f"""
{'='*60}
  TEJA — Stage 5 Complete!
{'='*60}

  Parameters:       {total_params:,}  ({total_params/1e6:.1f}M)
  Best val loss:    {best_val_loss:.4f}
  Final val loss:   {losses['val']:.4f}
  Dataset:          OpenWebText (~150M tokens)
  Tokenizer:        GPT-2 BPE  (vocab = {vocab_size:,})
  Device:           {device}

  WHAT CHANGED FROM STAGE 4 → STAGE 5:
  ┌─────────────────────────────────────────────────────┐
  │ Tokenizer:  char-level (65)  →  BPE (50,257)        │
  │ Data:       Shakespeare 1M chars → OWT 150M tokens  │
  │ vocab_size: 65          →  50,257                   │
  │ n_embd:     128         →  384                      │
  │ block_size: 64          →  256                      │
  │ Params:     ~1.2M       →  ~{total_params/1e6:.0f}M                      │
  │ LR:         fixed 1e-3  →  cosine 3e-4 → 3e-5      │
  │ + Weight tying (saves ~19M params)                  │
  │ + Gradient clipping (norm=1.0)                      │
  └─────────────────────────────────────────────────────┘

  FULL STAGE COMPARISON:
    Stage 1 (Bigram):           val ~2.47 |     4K params | chars
    Stage 2 (Single Attn):      val ~2.37 |     8K params | chars
    Stage 3 (No Residuals):     val ~3.36 |   209K params | UNSTABLE
    Stage 4 (TejaGPT):          val ~1.51 |   1.2M params | chars
    Stage 5 (Teja Scaled):      val ~{losses['val']:.2f} |  {total_params/1e6:.0f}M params | BPE ← HERE

  WHAT COMES NEXT:
    Stage 6: Build a BPE tokenizer from scratch (don't use tiktoken)
    Stage 7: Supervised fine-tuning for instruction following
    Stage 8: Scale to hundreds of millions of parameters
{'='*60}
""")
