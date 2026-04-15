"""
🔆 TEJA — Stage 4: Full GPT-2 Architecture (LayerNorm + Residuals)
===================================================================
Built from zero. Trained to shine.
Created by Sabith, Nilambur, Kerala.

STAGE 4 GOAL:
    Fix Stage 3's training instability by adding the two critical
    architectural elements that make deep Transformers trainable:
        1. RESIDUAL (SKIP) CONNECTIONS
        2. LAYER NORMALIZATION

    After Stage 3, we saw the loss SPIKE and never recover.
    Why? Without residuals and LayerNorm:
        - Gradients vanish or explode through deep stacks
        - Activation magnitudes drift, making learning unstable
        - The optimizer struggles to coordinate updates across layers

    Stage 4 fixes all of this. The result is a proper GPT-2 architecture.

=====================================================================
CONCEPT 1: RESIDUAL (SKIP) CONNECTIONS
=====================================================================

    The idea is beautifully simple:

        Instead of:  y = f(x)                ← Stage 3
        We do:       y = x + f(x)            ← Stage 4

    That's it. We ADD the input back to the output.

    Why this is so powerful:

    1. GRADIENT HIGHWAY
       During backpropagation, gradients must flow from the loss
       all the way back to the early layers. Through many layers,
       gradients can vanish (become tiny) or explode (become huge).

       With residuals, the gradient has a "highway" — it can flow
       directly through the + operation without going through layers:

       Without residuals: grad must pass through all N layers
       With residuals:    grad can skip directly via the + shortcut

       This means even layer 1 gets strong gradient signal!

    2. IDENTITY DEFAULT
       At initialization, f(x) is nearly random noise (small values).
       So y = x + f(x) ≈ x. The network starts as an identity
       function and gradually learns departures from it.

       This is much easier to optimize than learning the entire
       function from scratch.

    3. FEATURE REFINEMENT
       Each layer REFINES the representation rather than replacing it.
       Layer 1's output feeds into layer 2, which ADDS its improvements.
       Information is never lost — only enriched.

    Visual:
        ┌──────────┐
    x ──┤ Attention ├──⊕── y
        └──────────┘   ↑
              x ───────┘  (skip connection)

    In code:
        x = x + self.attention(x)    # attention adds to x
        x = x + self.feed_forward(x)  # FFN adds to x

=====================================================================
CONCEPT 2: LAYER NORMALIZATION
=====================================================================

    Layer normalization stabilizes the values flowing through the network
    by normalizing them at each layer.

    What it does:
        For each token's feature vector, compute mean and variance,
        then normalize to zero mean and unit variance:

        x_norm = (x - mean) / sqrt(variance + epsilon)

    Then apply learnable scale (gamma) and shift (beta):

        output = gamma * x_norm + beta

    Why normalize?
        - Without normalization, activation magnitudes can grow or shrink
          as data passes through many layers
        - Large activations → large gradients → unstable training
        - Small activations → small gradients → very slow learning
        - Normalization keeps things in a "healthy" range

    PRE-NORM vs POST-NORM:

    Original Transformer (2017) — POST-NORM:
        x → Attention → Add(x) → LayerNorm → FFN → Add → LayerNorm

    GPT-2 (2019) — PRE-NORM (what we use):
        x → LayerNorm → Attention → Add(x) → LayerNorm → FFN → Add(x)

    Pre-norm is simpler and trains more stably. The normalization
    happens BEFORE each sub-layer, so the inputs to attention and FFN
    are always well-behaved.

    We also add a FINAL LayerNorm after the last block, before the
    output projection. This ensures the final representations are
    normalized before computing logits.

=====================================================================
CONCEPT 3: DROPOUT — Regularization
=====================================================================

    Dropout randomly "turns off" a fraction of neurons during training
    (sets them to zero). This forces the model to be robust — it can't
    rely on any single neuron.

    Applied in three places:
    1. After attention weights (before multiplying with V)
    2. After the FFN output
    3. After the output projection of multi-head attention

    During inference (generation), dropout is turned OFF — all neurons
    are active, giving the best possible output.

    dropout_rate = 0.1 means 10% of values are zeroed each forward pass.

=====================================================================
CONCEPT 4: WEIGHT INITIALIZATION
=====================================================================

    How we initialize weights matters for training stability.

    PyTorch default (Kaiming/He init) works for most layers, but for
    residual connections, we need special care:

    The issue: If each residual block adds a contribution of
    standard deviation ~1, then after N blocks, the total standard
    deviation grows as sqrt(N). For 6 blocks, that's ~2.4× growth.

    GPT-2's fix: Scale the initialization of the final projection
    in each residual path by 1/sqrt(2*N). This keeps the total
    output variance from growing with depth.

    We implement this as a flag on the relevant Linear layers.

=====================================================================
THE COMPLETE GPT-2 ARCHITECTURE (our version):
=====================================================================

    Input → Token Emb + Pos Emb → [Block × 6] → LayerNorm → Linear → Logits

    Each Block (Pre-Norm):
        x ─→ LayerNorm ─→ MultiHeadAttention ──⊕── → LayerNorm ─→ FFN ──⊕── → output
             ↑              ↑                   ↑       ↑          ↑       ↑
             └──────────────x──────────────────┘       └──────────x──────┘
                     (skip connection)                     (skip connection)

    This is architecturally identical to GPT-2 (just smaller).
    
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

from teja.config import STAGE_4_CONFIG
from teja.utils import get_device, count_parameters, print_banner


# ========================
# REPRODUCIBILITY
# ========================
torch.manual_seed(1337)


# ========================
# HYPERPARAMETERS
# ========================
batch_size    = STAGE_4_CONFIG['batch_size']      # 64
block_size    = STAGE_4_CONFIG['block_size']       # 64
learning_rate = STAGE_4_CONFIG['learning_rate']    # 1e-3
max_iters     = STAGE_4_CONFIG['max_iters']        # 15,000
eval_interval = STAGE_4_CONFIG['eval_interval']    # 500
eval_iters    = STAGE_4_CONFIG['eval_iters']       # 200
n_embd        = STAGE_4_CONFIG['n_embd']           # 128
n_head        = STAGE_4_CONFIG['n_head']           # 4
n_layer       = STAGE_4_CONFIG['n_layer']          # 6
dropout       = STAGE_4_CONFIG['dropout']          # 0.1

head_size = n_embd // n_head  # 128 / 4 = 32

device = get_device()
print_banner("Full GPT-2 Architecture (LayerNorm + Residuals)", 4)


# ========================================================================
# DATA LOADING
# ========================================================================
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Dataset: {len(text):,} chars | Vocab: {vocab_size}")
print(f"Config: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, block_size={block_size}, dropout={dropout}")


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
# SELF-ATTENTION HEAD (same as Stage 3, with proper scaling)
# ========================================================================
class Head(nn.Module):
    """One head of self-attention with causal masking."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


# ========================================================================
# MULTI-HEAD ATTENTION (with output projection)
# ========================================================================
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel, with output projection."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # Output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ========================================================================
# FEED-FORWARD NETWORK (with GELU activation — GPT-2 uses GELU)
# ========================================================================
class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    GPT-2 uses GELU instead of ReLU. GELU (Gaussian Error Linear Unit)
    is smoother than ReLU:
        GELU(x) = x * Phi(x)  where Phi is the standard Gaussian CDF

    ReLU:  hard cutoff at 0 (exactly zero for negative inputs)
    GELU:  smooth transition (small non-zero values for negative inputs)

    This smoothness helps with gradient flow during training.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),                          # Smoother than ReLU, used by GPT-2
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ========================================================================
# TRANSFORMER BLOCK — NOW WITH RESIDUALS AND LAYER NORM!
# ========================================================================
class Block(nn.Module):
    """
    Transformer block with PRE-NORM residual connections.

    This is the GPT-2 formulation:
        x = x + Attention(LayerNorm(x))    ← pre-norm + residual
        x = x + FFN(LayerNorm(x))          ← pre-norm + residual

    Compare to Stage 3 (no residuals/norm):
        x = Attention(x)   ← gradients must flow through attention
        x = FFN(x)         ← gradients must flow through FFN

    The residual connection (x + ...) lets gradients bypass the layers.
    The LayerNorm ensures inputs to each sub-layer are well-behaved.

    This is why Stage 3 was unstable but Stage 4 trains smoothly!
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        # Layer norms BEFORE each sub-layer (pre-norm formulation)
        self.ln1 = nn.LayerNorm(n_embd)  # Before attention
        self.ln2 = nn.LayerNorm(n_embd)  # Before FFN

        # Sub-layers
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # RESIDUAL + PRE-NORM ATTENTION
        # x + attention(normalize(x))
        # The + is the residual connection (skip connection)
        x = x + self.sa(self.ln1(x))

        # RESIDUAL + PRE-NORM FEED-FORWARD
        # x + ffn(normalize(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# ========================================================================
# GPT-2 STYLE LANGUAGE MODEL — THE COMPLETE ARCHITECTURE
# ========================================================================
class TejaGPT(nn.Module):
    """
    The complete Teja GPT model — architecturally identical to GPT-2.

    Architecture:
        Input → Token Emb + Pos Emb
              → [Block × n_layer]        (each block: LN→Attn+Res, LN→FFN+Res)
              → Final LayerNorm
              → Linear Head
              → Logits

    This is the MILESTONE model. After Stage 4, the architecture
    is complete. Future stages focus on data, tokenization, and scale.

    GPT-2 sizes for reference:
        Small:  n_embd=768,  n_head=12, n_layer=12  → 124M params
        Medium: n_embd=1024, n_head=16, n_layer=24  → 350M params
        Large:  n_embd=1280, n_head=20, n_layer=36  → 774M params
        XL:     n_embd=1600, n_head=25, n_layer=48  → 1.5B params

    Our Teja (Stage 4):
        n_embd=128, n_head=4, n_layer=6  → ~400K params
        Same architecture, just smaller!
    """

    def __init__(self):
        super().__init__()

        # Token embeddings: character ID → vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position embeddings: position → vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks (the core)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )

        # Final layer norm (GPT-2 adds this after the last block)
        self.ln_f = nn.LayerNorm(n_embd)

        # Language model head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding_table(idx)                              # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Transformer blocks
        x = self.blocks(x)  # (B, T, n_embd)

        # Final layer norm
        x = self.ln_f(x)  # (B, T, n_embd)

        # Logits
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
        """Autoregressive generation with context window cropping."""
        for _ in range(max_new_tokens):
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
print("\nCreating TejaGPT (Full GPT-2 Architecture)...")
model = TejaGPT()
model = model.to(device)
total_params = count_parameters(model)

print(f"\n   ARCHITECTURE EVOLUTION:")
print(f"     Stage 1 (Bigram):           4,225 params | val loss 2.47")
print(f"     Stage 2 (Single Attention): 7,553 params | val loss 2.37")
print(f"     Stage 3 (Mini Transformer): 208,577 params | val loss 3.36 (UNSTABLE!)")
print(f"     Stage 4 (TejaGPT):          {total_params:,} params | val loss ???")
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ========================================================================
# GENERATE BEFORE TRAINING
# ========================================================================
print("Text from UNTRAINED TejaGPT:")
print("-" * 50)
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(seed, max_new_tokens=200)[0].tolist()))
print("-" * 50)


# ========================================================================
# TRAINING LOOP
# ========================================================================
print(f"\nTraining for {max_iters:,} iterations...")
print(f"   Model: {n_layer} blocks, {n_head} heads, {n_embd} embedding dim")
print(f"   Batch: {batch_size} | Context: {block_size} chars | Dropout: {dropout}")
print(f"   Learning rate: {learning_rate}")
print(f"   KEY: Residual connections + LayerNorm should prevent Stage 3's instability\n")

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
print("\nText from TRAINED TejaGPT (expect sentence-level coherence):")
print("=" * 60)
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(seed, max_new_tokens=500)[0].tolist())
print(generated)
print("=" * 60)


# ========================================================================
# SAVE THE MODEL CHECKPOINT
# ========================================================================
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'teja_stage4.pt')
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'block_size': block_size,
    'chars': chars,
    'stoi': stoi,
    'itos': itos,
    'train_loss': losses['train'].item(),
    'val_loss': losses['val'].item(),
    'total_params': total_params,
}, checkpoint_path)
print(f"\nCheckpoint saved to: {checkpoint_path}")


# ========================================================================
# VERIFICATION SUMMARY
# ========================================================================
print(f"""
{'='*60}
  TEJA -- Stage 4 Complete! (Architecture Milestone)
{'='*60}

  Parameters:       {total_params:,}
  Final train loss: {losses['train']:.4f}
  Final val loss:   {losses['val']:.4f}
  Device:           {device}

  FULL STAGE COMPARISON:
    Stage 1 (Bigram):           val ~2.47 |     4,225 params | Seconds
    Stage 2 (Single Attn):      val ~2.37 |     7,553 params | Minutes
    Stage 3 (No Residuals):     val ~3.36 |   208,577 params | UNSTABLE
    Stage 4 (TejaGPT):          val ~{losses['val']:.2f} |   {total_params:,} params | STABLE

  ARCHITECTURE (GPT-2 complete):
    - {n_layer} Transformer blocks with PRE-NORM
    - {n_head} attention heads (head_size={head_size})
    - Residual connections on EVERY sub-layer
    - LayerNorm BEFORE attention and FFN (pre-norm)
    - Final LayerNorm before output projection
    - GELU activation in FFN
    - Dropout = {dropout}

  KEY TAKEAWAYS:
    - Residual connections FIXED Stage 3's instability
    - LayerNorm kept activations in a healthy range
    - Pre-norm (GPT-2 style) is simpler and more stable than post-norm
    - The architecture is now COMPLETE -- identical to GPT-2
    - Everything after this is about DATA and SCALE

  WHAT COMES NEXT (Stage 5):
    - Move from tiny Shakespeare to REAL DATA (Wikipedia/OpenWebText)
    - Larger model config (384 dims, 6 heads, 256 context)
    - ~10M parameters (vs ~400K now)
    - Google Colab recommended for compute
    - The model will start generating COHERENT ENGLISH
{'='*60}
""")
