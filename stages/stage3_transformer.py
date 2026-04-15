"""
🔆 TEJA — Stage 3: Multi-Head Attention + Feed-Forward (Mini Transformer)
==========================================================================
Built from zero. Trained to shine.
Created by Sabith, Nilambur, Kerala.

STAGE 3 GOAL:
    Build a proper (tiny) Transformer. In Stage 2, we had a single
    attention head — one "perspective" on the data. Now we add:
        1. MULTI-HEAD attention — multiple heads running in parallel
        2. FEED-FORWARD network — a small MLP after attention
        3. STACKED BLOCKS — repeat the whole thing N times for depth

    This is the architecture from "Attention Is All You Need" (2017),
    the paper that started the modern AI revolution.

=====================================================================
CONCEPT 1: MULTI-HEAD ATTENTION — Multiple Perspectives
=====================================================================

    In Stage 2, our single attention head learned ONE pattern of
    "which previous tokens to attend to." But language has MANY
    types of dependencies:

        "The CAT sat on the MAT" ← rhyming pattern
        "She went to THE STORE" ← article-noun relationship
        "John said HE was tired" ← pronoun-antecedent reference

    One head can't learn all these patterns simultaneously.

    Solution: Run MULTIPLE attention heads in PARALLEL, each with its
    own Q, K, V weight matrices. Each head independently learns different
    attention patterns.

    Then CONCATENATE all head outputs and project to a single vector.

    If n_embd = 64 and n_head = 4:
        Each head has head_size = 64 / 4 = 16
        Head 1 output: (B, T, 16) — maybe learns positional patterns
        Head 2 output: (B, T, 16) — maybe learns vowel-consonant patterns
        Head 3 output: (B, T, 16) — maybe learns word boundaries
        Head 4 output: (B, T, 16) — maybe learns other patterns

        Concatenate: (B, T, 64) ← all perspectives combined
        Project: Linear(64, 64) ← learned mixing of perspectives

    KEY INSIGHT: Multi-head attention is CHEAPER than one big head!
        1 big head with head_size=64: Q is (64,64) → 4096 params
        4 small heads with head_size=16: Q is (64,16)×4 → 4096 params
        Same parameter count, but 4 independent attention patterns!

=====================================================================
CONCEPT 2: FEED-FORWARD NETWORK (FFN) — The "Thinking" Layer
=====================================================================

    After the attention layer gathers information from other tokens,
    the FFN processes that information. Think of it as:

        Attention = "look around and gather relevant context"
        FFN       = "think about what you've gathered and process it"

    The FFN is surprisingly simple — just two linear layers with a
    ReLU activation in between:

        FFN(x) = Linear2(ReLU(Linear1(x)))

    Dimensions:
        Linear1: n_embd → 4 * n_embd     (expand)
        ReLU:    element-wise activation  (non-linearity)
        Linear2: 4 * n_embd → n_embd     (contract)

    The expansion factor 4× is from the original Transformer paper.
    The "expand then contract" pattern lets the model use a larger
    internal representation temporarily to do more complex computation.

    Why ReLU? It introduces non-linearity. Without it, two linear layers
    would collapse into one (linear × linear = linear). Non-linearity
    is essential for learning complex patterns.

    The FFN is applied to each position INDEPENDENTLY — it doesn't
    mix information across positions (that's attention's job).

=====================================================================
CONCEPT 3: TRANSFORMER BLOCK — The Repeatable Unit
=====================================================================

    A Transformer block combines attention + FFN into one unit:

        Block(x) = FFN(MultiHeadAttention(x))

    This is the fundamental repeatable building block. Stack N of them
    and you get a deeper model:

        Input → Block 1 → Block 2 → Block 3 → ... → Block N → Output

    Why stacking helps:
        Block 1: learns basic character patterns ("th", "he", "in")
        Block 2: uses Block 1's output to learn word patterns
        Block 3: uses word-level info to learn phrase patterns
        Block N: learns increasingly abstract, high-level patterns

    More blocks = more compositional reasoning, but also:
        - More parameters (more memory needed)
        - Harder to train (gradients can vanish)
        - Slower training (more computation per step)

    For Stage 3, we use n_layer=4 blocks, which is small but
    enough to demonstrate the power of depth.

    NOTE: We're NOT adding residual connections or layer norm yet —
    that's Stage 4. This stage shows WHY we'll need them.

=====================================================================
ARCHITECTURE OVERVIEW:
=====================================================================

    Input → Token Emb + Pos Emb → [Block × 4] → Linear → Logits

    Where each Block = MultiHeadAttention(4 heads) → FeedForward
    
    Parameters estimate:
        Token embedding:    65 × 64       =  4,160
        Position embedding: 32 × 64       =  2,048
        Per head (×4):      3 × (64×16)   =  3,072 per head × 4 = 12,288
        Per block proj out: 64 × 64 + 64  =  4,160
        Per block FFN:      64×256 + 256 + 256×64 + 64 = 33,088
        Per block total:    ~49,536
        4 blocks:           ~198,144
        LM head:            64 × 65 + 65  =  4,225
        Total:              ~220,000 parameters

    That's ~50x more than Stage 1!

=====================================================================
Let's build it! 🔆
=====================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from teja.config import STAGE_3_CONFIG
from teja.utils import get_device, count_parameters, print_banner


# ========================
# REPRODUCIBILITY
# ========================
torch.manual_seed(1337)


# ========================
# HYPERPARAMETERS
# ========================
batch_size    = STAGE_3_CONFIG['batch_size']      # 32
block_size    = STAGE_3_CONFIG['block_size']       # 32 (longer context now!)
learning_rate = STAGE_3_CONFIG['learning_rate']    # 1e-3
max_iters     = STAGE_3_CONFIG['max_iters']        # 10,000
eval_interval = STAGE_3_CONFIG['eval_interval']    # 500
eval_iters    = STAGE_3_CONFIG['eval_iters']       # 200
n_embd        = STAGE_3_CONFIG['n_embd']           # 64
n_head        = STAGE_3_CONFIG['n_head']           # 4 heads
n_layer       = STAGE_3_CONFIG['n_layer']          # 4 blocks
dropout       = STAGE_3_CONFIG['dropout']          # 0.0

# Derived
head_size = n_embd // n_head  # 64 / 4 = 16 per head

# Device
device = get_device()

# Banner
print_banner("Multi-Head Attention + Feed-Forward (Mini Transformer)", 3)


# ========================================================================
# DATA LOADING (reused from previous stages)
# ========================================================================
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Dataset: {len(text):,} chars | Vocab: {vocab_size}")
print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")
print(f"Config: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, block_size={block_size}")
print(f"Head size: {head_size} (n_embd / n_head = {n_embd}/{n_head})")


def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i : i + block_size]     for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ========================================================================
# ATTENTION HEAD (reused from Stage 2, now with dropout)
# ========================================================================
class Head(nn.Module):
    """
    One head of self-attention.

    Same as Stage 2, but now each head has a smaller head_size
    (n_embd / n_head) because multiple heads share the embedding space.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # Attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # Randomly zero out some attention weights

        # Weighted aggregation
        v = self.value(x)   # (B, T, head_size)
        out = wei @ v       # (B, T, head_size)
        return out


# ========================================================================
# MULTI-HEAD ATTENTION — NEW IN STAGE 3!
# ========================================================================
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.

    This is like having multiple "perspectives" on the data.
    Each head independently decides what to attend to, then their
    outputs are concatenated and projected.

    If n_embd = 64 and n_head = 4:
        Each head outputs (B, T, 16)
        Concatenated: (B, T, 64)
        Projected: (B, T, 64) via a linear layer

    Why projection after concatenation?
        The raw concatenation is just stacking. The projection
        lets the model learn how to MIX information across heads.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create `num_heads` independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Output projection: mix the outputs of all heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run each head in parallel, concatenate their outputs
        # Each head returns (B, T, head_size)
        # After concat: (B, T, num_heads * head_size) = (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Project the concatenated output
        out = self.dropout(self.proj(out))
        return out


# ========================================================================
# FEED-FORWARD NETWORK — NEW IN STAGE 3!
# ========================================================================
class FeedForward(nn.Module):
    """
    A simple feed-forward network: expand → ReLU → contract.

    Applied to each position independently (no cross-position mixing).
    This is the "thinking" step after attention has gathered context.

    Architecture:
        Input (n_embd) → Linear (4 * n_embd) → ReLU → Linear (n_embd) → Output

    The 4× expansion gives the model a larger "workspace" to compute
    intermediate representations before projecting back down.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),    # Expand: 64 → 256
            nn.ReLU(),                          # Non-linearity
            nn.Linear(4 * n_embd, n_embd),     # Contract: 256 → 64
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ========================================================================
# TRANSFORMER BLOCK — NEW IN STAGE 3!
# ========================================================================
class Block(nn.Module):
    """
    One Transformer block: Multi-Head Attention → Feed-Forward.

    This is the fundamental repeatable unit.
    Stack N blocks and you get a deeper model.

    Data flow:
        x → MultiHeadAttention(x) → FeedForward → output

    NOTE: No residual connections or LayerNorm yet — that's Stage 4.
    You'll notice that training can be slightly unstable without them,
    which is exactly WHY Stage 4 exists.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)   # Self-attention
        self.ffwd = FeedForward(n_embd)                    # Feed-forward

    def forward(self, x):
        x = self.sa(x)     # Multi-head self-attention
        x = self.ffwd(x)   # Feed-forward processing
        return x


# ========================================================================
# THE MINI TRANSFORMER MODEL
# ========================================================================
class MiniTransformer(nn.Module):
    """
    Stage 3: A proper (mini) Transformer language model.

    Architecture:
        Input → Token Emb + Pos Emb → [Block × n_layer] → Linear → Logits

    What's new vs Stage 2:
        - Multiple attention heads per layer (4 instead of 1)
        - Feed-forward network in each block
        - Multiple stacked blocks (4 layers of processing)
        - Much larger capacity (~220K params vs ~7.5K)

    This architecture is recognizable as a real Transformer,
    just missing residuals and LayerNorm (coming in Stage 4).
    """

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks — the core of the model
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )

        # Final language model head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding_table(idx)                              # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Pass through all Transformer blocks
        x = self.blocks(x)  # (B, T, n_embd)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop to block_size (position embeddings limit)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ========================================================================
# CREATE MODEL AND OPTIMIZER
# ========================================================================
print("\nCreating Mini Transformer (Stage 3)...")
model = MiniTransformer()
model = model.to(device)
total_params = count_parameters(model)

print(f"   Comparison: Stage 1 = 4,225 | Stage 2 = 7,553 | Stage 3 = {total_params:,}")
print(f"   That's {total_params / 4225:.0f}x more than Stage 1!\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ========================================================================
# GENERATE BEFORE TRAINING
# ========================================================================
print("Text from UNTRAINED Mini Transformer:")
print("-" * 50)
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(seed, max_new_tokens=200)[0].tolist()))
print("-" * 50)


# ========================================================================
# TRAINING LOOP
# ========================================================================
print(f"\nTraining for {max_iters:,} iterations...")
print(f"   Model: {n_layer} blocks, {n_head} heads, {n_embd} embedding dim")
print(f"   Batch: {batch_size} | Context: {block_size} chars")
print(f"   Learning rate: {learning_rate}\n")

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"   Step {iter:>5,}: train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"   Step {max_iters:>5,}: train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f}")
print(f"\n   Training complete!")


# ========================================================================
# GENERATE AFTER TRAINING
# ========================================================================
print("\nText from TRAINED Mini Transformer:")
print("=" * 60)
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(seed, max_new_tokens=500)[0].tolist())
print(generated)
print("=" * 60)


# ========================================================================
# VERIFICATION SUMMARY
# ========================================================================
print(f"""
{'='*60}
  TEJA -- Stage 3 Complete!
{'='*60}

  Parameters:       {total_params:,}
  Final train loss: {losses['train']:.4f}
  Final val loss:   {losses['val']:.4f}
  Device:           {device}

  STAGE COMPARISON:
    Stage 1 (Bigram):        val loss ~ 2.47 |   4,225 params
    Stage 2 (1-Head Attn):   val loss ~ 2.37 |   7,553 params
    Stage 3 (Mini Transf.):  val loss ~ {losses['val']:.2f} | {total_params:,} params

  ARCHITECTURE:
    - {n_layer} Transformer blocks stacked
    - {n_head} attention heads per block (head_size = {head_size})
    - Feed-forward with 4x expansion ({n_embd} -> {4*n_embd} -> {n_embd})
    - Context window: {block_size} characters

  WHAT WE LEARNED:
    - Multi-head attention captures multiple types of patterns simultaneously
    - Feed-forward networks process the gathered context
    - Stacking blocks enables hierarchical feature learning
    - Without residuals/LayerNorm, deep training can be UNSTABLE
      (this motivates Stage 4!)

  WHAT COMES NEXT (Stage 4):
    - RESIDUAL CONNECTIONS: x + f(x) instead of just f(x)
      (lets gradients flow directly through the network)
    - LAYER NORMALIZATION: stabilize activations at each block
      (prevents values from exploding or vanishing)
    - PRE-NORM formulation (GPT-2 style)
    - DROPOUT for regularization
    - This makes training MUCH more stable and reaches lower loss
    - After Stage 4, the architecture is COMPLETE (matches GPT-2)
{'='*60}
""")
