"""
🔆 TEJA — Stage 8: Scale to GPT-2 Size with RoPE
==================================================
Built from zero. Trained to shine.
Created by Sabith, Nilambur, Kerala.

STAGE 8 GOAL:
    Jump from 30M → 117M parameters and 107M → 1B training tokens.
    This is the scale where outputs become genuinely coherent.

    Stage 7 Teja:  30M params, ~107M training tokens → broken answers
    Stage 8 Teja: 117M params, ~1B training tokens   → GPT-2 quality

=======================================================================
WHAT'S NEW IN STAGE 8
=======================================================================

1. BIGGER ARCHITECTURE
   Stage 5/7: n_embd=384, n_head=6,  n_layer=6,  block_size=256
   Stage 8:   n_embd=768, n_head=12, n_layer=12, block_size=512
   This matches GPT-2 small exactly.

2. RoPE (ROTARY POSITION EMBEDDINGS)
   Stage 5/7 used a LEARNED position table:
       position_embedding_table = nn.Embedding(block_size, n_embd)
       x = token_emb + pos_emb   ← just add a fixed learned vector

   Stage 8 uses ROPE — rotate Q and K vectors in attention:
       q_rotated = rotate(q, position)
       k_rotated = rotate(k, position)
       attention = q_rotated @ k_rotated.T

   WHY ROPE?
   - The attention score between token i and token j becomes a function
     of their RELATIVE distance (j - i), not absolute positions.
   - Generalizes better to sequences longer than training length.
   - Used in: LLaMA, Gemma, Mistral — every modern LLM.
   - No extra parameters (unlike learned pos embeddings).

   HOW IT WORKS:
   Each (query, key) head dimension pair gets rotated by angle:
       θ_d = position × (1 / 10000^(2d/head_dim))
   This is like a clock: dim 0 rotates fast, dim N rotates slow.
   When you compute q·k, the rotation cancels out into cos(θ_i - θ_j).
   So attention sees relative position, not absolute.

3. FLASH ATTENTION
   PyTorch 2.0+ has F.scaled_dot_product_attention which uses
   Flash Attention — a memory-efficient CUDA implementation.
   2-3x faster, 5-10x less VRAM for the attention step.
   Zero code change — one function call replaces the manual Q@K.T loop.

4. MORE DATA
   Stage 5: 100K docs → 107M tokens
   Stage 8: 1M docs  → ~1B tokens
   Same OpenWebText source, 10x more documents.
   Storage: ~3GB on disk (train.bin + val.bin).

=======================================================================
KAGGLE SETUP
=======================================================================

    !pip install -q tiktoken datasets tqdm

    !rm -rf teja
    !git clone https://github.com/sabithh/Teja.git teja
    %cd teja
    !mkdir -p checkpoints
    # No Stage 5 checkpoint needed — training from scratch at new scale
    !python stages/stage8_scale.py

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
from teja.config import STAGE_8_CONFIG
from teja.utils import get_device, count_parameters, print_banner

torch.manual_seed(1337)

# ========================================================================
# HYPERPARAMETERS
# ========================================================================

batch_size    = STAGE_8_CONFIG['batch_size']                    # 32
grad_accum    = STAGE_8_CONFIG['gradient_accumulation_steps']   # 4
block_size    = STAGE_8_CONFIG['block_size']                    # 512
learning_rate = STAGE_8_CONFIG['learning_rate']                 # 3e-4
max_iters     = STAGE_8_CONFIG['max_iters']                     # 50,000
eval_interval = STAGE_8_CONFIG['eval_interval']                 # 1,000
eval_iters    = STAGE_8_CONFIG['eval_iters']                    # 200
n_embd        = STAGE_8_CONFIG['n_embd']                        # 768
n_head        = STAGE_8_CONFIG['n_head']                        # 12
n_layer       = STAGE_8_CONFIG['n_layer']                       # 12
dropout       = STAGE_8_CONFIG['dropout']                       # 0.0
NUM_DOCS      = STAGE_8_CONFIG['num_docs']                      # 1,000,000

warmup_iters = 2_000
min_lr       = learning_rate / 10
grad_clip    = 1.0
vocab_size   = 50_257
eval_batch_size = 4

device = get_device()
print_banner("Scale to GPT-2 Size + RoPE", 8)

# ========================================================================
# DATA PREPARATION  (same as Stage 5, just 10x more docs)
# ========================================================================

def prepare_openwebtext(data_dir):
    train_bin = os.path.join(data_dir, 'train.bin')
    val_bin   = os.path.join(data_dir, 'val.bin')

    if os.path.exists(train_bin) and os.path.exists(val_bin):
        print(f"✓ Tokenized data already exists at {data_dir}")
        return

    print(f"\nPreparing OpenWebText ({NUM_DOCS:,} docs — this takes ~1 hour)...")

    try:
        import tiktoken
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("Run: pip install tiktoken datasets tqdm")
        sys.exit(1)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print("Streaming OpenWebText from HuggingFace...")
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    docs = []
    for doc in tqdm(dataset, total=NUM_DOCS, desc="Collecting docs"):
        docs.append(doc['text'])
        if len(docs) >= NUM_DOCS:
            break

    split_idx   = int(0.95 * len(docs))
    train_texts = docs[:split_idx]
    val_texts   = docs[split_idx:]

    def tokenize_and_save(texts, path, desc):
        from tqdm import tqdm
        all_tokens = []
        for text in tqdm(texts, desc=desc):
            tokens = enc.encode_ordinary(text)
            tokens.append(eot)
            all_tokens.extend(tokens)
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(path)
        print(f"  Saved {len(arr):,} tokens ({len(arr)/1e6:.1f}M) → {path}")
        return len(arr)

    os.makedirs(data_dir, exist_ok=True)
    n_train = tokenize_and_save(train_texts, train_bin, "Tokenizing train")
    n_val   = tokenize_and_save(val_texts,   val_bin,   "Tokenizing val")
    print(f"\nDone! Total tokens: {(n_train+n_val)/1e6:.1f}M")


data_dir   = os.path.join(os.path.dirname(__file__), '..', 'data', 'openwebtext8')
prepare_openwebtext(data_dir)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

print(f"\nDataset loaded:")
print(f"  Train: {len(train_data)/1e6:.1f}M tokens")
print(f"  Val:   {len(val_data)/1e6:.1f}M tokens")
print(f"  Context: {block_size} tokens (~{block_size*4} chars)\n")


def get_batch(split, bs=None):
    if bs is None:
        bs = batch_size
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (bs,))
    x = torch.stack([torch.from_numpy(data[i    :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1  :i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, bs=eval_batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ========================================================================
# ROPE — Rotary Position Embeddings
# ========================================================================

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute rotation frequencies for all positions and head dimensions.

    For each pair of dimensions (d, d+1) in a head, compute:
        freq_d = 1 / (theta ^ (2d / head_dim))

    Then for each position t:
        angle_d = t * freq_d

    Returns complex numbers e^(i * angle): shape (max_seq_len, head_dim/2)
    Multiplying a complex number by e^(i*angle) rotates it by that angle.
    """
    # freq for each dimension pair: (head_dim/2,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # position indices: (max_seq_len,)
    t = torch.arange(max_seq_len)
    # outer product → angles for every (position, dimension) pair
    angles = torch.outer(t, freqs)           # (max_seq_len, head_dim/2)
    # convert to complex: cos(angle) + i*sin(angle)
    return torch.polar(torch.ones_like(angles), angles)  # (max_seq_len, head_dim/2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.

    x:         (B, T, n_heads, head_dim)
    freqs_cis: (T, head_dim/2)   — precomputed rotation angles as complex numbers
    returns:   (B, T, n_heads, head_dim)  — rotated version of x
    """
    B, T, n_heads, head_dim = x.shape
    # Pair up consecutive dims and view as complex: (B, T, n_heads, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(B, T, n_heads, -1, 2))
    # Reshape freqs for broadcasting: (1, T, 1, head_dim/2)
    freqs = freqs_cis[:T].unsqueeze(0).unsqueeze(2)
    # Rotate: complex multiplication = rotation in 2D plane
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_rotated.type_as(x)


# ========================================================================
# MODEL — GPT-2 scale with RoPE + Flash Attention
# ========================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with:
    - RoPE applied to Q and K (no position_embedding_table needed)
    - Flash Attention via F.scaled_dot_product_attention (faster + less VRAM)
    - All heads computed together in one batched operation
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head

        self.q_proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.shape

        # Project and split into heads: (B, T, n_heads, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)

        # Apply RoPE to Q and K (NOT to V — only Q·K determines position)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Transpose to (B, n_heads, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention — fused CUDA kernel, causal mask handled automatically
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads and project: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.ffwd(self.ln2(x))
        return x


class TejaGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # NO position_embedding_table — RoPE handles positions inside attention
        self.blocks   = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f     = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying: input embeddings and output projection share weights
        self.token_embedding.weight = self.lm_head.weight

        # Precompute RoPE frequencies once — shared across all layers and heads
        freqs_cis = precompute_rope_freqs(n_embd // n_head, block_size)
        self.register_buffer('freqs_cis', freqs_cis)

        # Weight initialization (GPT-2 style)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)           # (B, T, n_embd)
        for block in self.blocks:
            x = block(x, self.freqs_cis)        # pass freqs_cis to each block
        x = self.ln_f(x)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ========================================================================
# BUILD MODEL
# ========================================================================

print(f"Creating TejaGPT (Stage 8 — GPT-2 scale)...")
model = TejaGPT(vocab_size=vocab_size).to(device)
total_params = count_parameters(model)
print(f"\n  ARCHITECTURE COMPARISON:")
print(f"    Stage 5/7 — n_embd=384, n_head=6,  n_layer=6,  block=256 → 30M params")
print(f"    Stage 8   — n_embd=768, n_head=12, n_layer=12, block=512 → {total_params/1e6:.0f}M params")
print(f"\n  NEW: RoPE positional embeddings (no position table)")
print(f"  NEW: Flash Attention (F.scaled_dot_product_attention)")


# ========================================================================
# CHECKPOINTS
# ========================================================================

checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

LOCAL_CKPT    = os.path.join(checkpoint_dir, 'teja_stage8_best.pt')
LOCAL_LATEST  = os.path.join(checkpoint_dir, 'teja_stage8_latest.pt')
KAGGLE_CKPT   = '/kaggle/working/teja_stage8_best.pt'
KAGGLE_LATEST = '/kaggle/working/teja_stage8_latest.pt'
DRIVE_CKPT    = '/content/drive/MyDrive/teja_checkpoints/teja_stage8_best.pt'
DRIVE_LATEST  = '/content/drive/MyDrive/teja_checkpoints/teja_stage8_latest.pt'


def save_checkpoint(path, step, val_loss, optimizer, is_latest=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        'step':                 step,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss':             val_loss,
        'vocab_size':           vocab_size,
        'n_embd':               n_embd,
        'n_head':               n_head,
        'n_layer':              n_layer,
        'block_size':           block_size,
        'total_params':         total_params,
    }
    torch.save(data, path)
    if os.path.exists('/kaggle/working'):
        torch.save(data, KAGGLE_LATEST if is_latest else KAGGLE_CKPT)
    if os.path.exists('/content/drive/MyDrive'):
        drive_path = DRIVE_LATEST if is_latest else DRIVE_CKPT
        os.makedirs(os.path.dirname(drive_path), exist_ok=True)
        torch.save(data, drive_path)


resume_path = (
    KAGGLE_LATEST if os.path.exists(KAGGLE_LATEST) else
    DRIVE_LATEST  if os.path.exists(DRIVE_LATEST)  else
    LOCAL_LATEST  if os.path.exists(LOCAL_LATEST)  else
    KAGGLE_CKPT   if os.path.exists(KAGGLE_CKPT)   else
    DRIVE_CKPT    if os.path.exists(DRIVE_CKPT)    else
    LOCAL_CKPT    if os.path.exists(LOCAL_CKPT)    else
    None
)

start_step    = 0
best_val_loss = float('inf')

if resume_path:
    print(f"\nResuming from: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    start_step    = ckpt['step']
    best_val_loss = ckpt['val_loss']
    # Preserve historical best if resuming from latest
    for p in (KAGGLE_CKPT, DRIVE_CKPT, LOCAL_CKPT):
        if os.path.exists(p) and p != resume_path:
            b = torch.load(p, map_location='cpu', weights_only=False)
            if b.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = b['val_loss']
            break
    print(f"  Step: {start_step:,} | Best val loss: {best_val_loss:.4f}\n")
else:
    print("\nNo checkpoint found — training from scratch.\n")


# ========================================================================
# OPTIMIZER + LR SCHEDULE
# ========================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True if device == 'cuda' else False,   # fused AdamW = faster on GPU
)

if resume_path:
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

effective_batch = batch_size * grad_accum
tokens_per_step = effective_batch * block_size

print(f"Training for {max_iters:,} steps...")
print(f"  Arch:         {n_layer} blocks × {n_head} heads × {n_embd} dims")
print(f"  Params:       {total_params/1e6:.0f}M")
print(f"  Context:      {block_size} tokens (~{block_size*4} chars)")
print(f"  Batch:        {batch_size} × {grad_accum} grad accum = {effective_batch} effective")
print(f"  Tokens/step:  {tokens_per_step:,}")
print(f"  LR:           {learning_rate} → {min_lr} (cosine, {warmup_iters} warmup)")
print(f"  Grad clip:    {grad_clip}\n")


def get_lr(step):
    if step < warmup_iters:
        return learning_rate * step / max(warmup_iters, 1)
    progress = (step - warmup_iters) / max(max_iters - warmup_iters, 1)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)


# ========================================================================
# TRAINING LOOP
# ========================================================================

t0 = time.time()

for step in range(start_step, max_iters + 1):

    # ------------------------------------------------------------------
    # Evaluation + checkpoint
    # ------------------------------------------------------------------
    if step % eval_interval == 0:
        losses  = estimate_loss()
        elapsed = time.time() - t0
        lr_now  = get_lr(step)
        tokens  = step * tokens_per_step

        print(f"   step {step:>6,} | train {losses['train']:.4f} | "
              f"val {losses['val']:.4f} | lr {lr_now:.2e} | "
              f"tokens {tokens/1e9:.2f}B | {elapsed:.0f}s")

        # Always save latest (safety net)
        save_checkpoint(LOCAL_LATEST, step, losses['val'].item(), optimizer, is_latest=True)
        print(f"   💾 latest saved (step {step:,})")

        # Save best if val improved
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val'].item()
            save_checkpoint(LOCAL_CKPT, step, best_val_loss, optimizer, is_latest=False)
            print(f"   ✓ best saved (step {step:,}, val {best_val_loss:.4f})")

    if step == max_iters:
        break

    # ------------------------------------------------------------------
    # LR update
    # ------------------------------------------------------------------
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # ------------------------------------------------------------------
    # Forward + backward with gradient accumulation
    # ------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum):
        xb, yb  = get_batch('train')
        _, loss = model(xb, yb)
        (loss / grad_accum).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


# ========================================================================
# SAMPLE TEXT
# ========================================================================

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    print("\nGenerating sample text:")
    print("=" * 60)
    model.eval()
    for prompt in ["The history of", "Scientists discovered", "In the future,"]:
        ids  = enc.encode(prompt)
        ctx  = torch.tensor([ids], dtype=torch.long, device=device)
        out  = model.generate(ctx, max_new_tokens=80, temperature=0.8, top_k=200)
        text = enc.decode(out[0].tolist())
        print(f"\nPrompt: '{prompt}'")
        print(text)
    print("=" * 60)
except ImportError:
    pass


# ========================================================================
# FINAL SAVE
# ========================================================================

losses = estimate_loss()
final_path = LOCAL_CKPT.replace('best', 'final')
save_checkpoint(final_path, max_iters, losses['val'].item(), optimizer, is_latest=False)

print(f"""
{'='*60}
  TEJA — Stage 8 Complete!
{'='*60}

  Parameters:    {total_params/1e6:.0f}M
  Best val loss: {best_val_loss:.4f}
  Final val loss:{losses['val']:.4f}
  Training data: ~1B tokens (OpenWebText, 1M docs)

  WHAT CHANGED FROM STAGE 5 → STAGE 8:
    Params:    30M → 117M  (4x bigger)
    Context:   256 → 512 tokens
    Data:      107M → ~1B tokens (10x more)
    Positions: learned table → RoPE
    Attention: manual loop → Flash Attention

  WHAT COMES NEXT:
    Stage 9: SFT on this bigger model (Alpaca / ShareGPT)
    Stage 10: RLHF
{'='*60}
""")
