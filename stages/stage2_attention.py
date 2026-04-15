"""
🔆 TEJA — Stage 2: Positional Encoding + Self-Attention (Single Head)
======================================================================
Built from zero. Trained to shine.
Created by Sabith, Nilambur, Kerala.

STAGE 2 GOAL:
    Make the model CONTEXT-AWARE. In Stage 1, the bigram model predicted
    the next character using ONLY the current character — it had zero
    memory of anything before it.

    Now we add two critical ingredients:
        1. Positional embeddings — so the model knows WHERE it is in the sequence
        2. Self-attention — so the model can LOOK BACK at all previous characters

    After this stage, instead of:
        P(next | current_char)                    ← Stage 1 (bigram)
    we'll have:
        P(next | all_previous_chars_in_context)   ← Stage 2 (attention)

=====================================================================
CONCEPT 1: WHY BIGRAM FAILS
=====================================================================

    Consider the sentence: "The cat sat on the ___"

    Bigram model sees only the 'e' in "the" and tries to predict next char.
    It has NO IDEA that "cat" or "sat" appeared earlier. Every prediction
    is made in isolation from just one character.

    What we need: a way for each position in the sequence to "look at"
    and gather information from all previous positions.

    That's exactly what self-attention does.

=====================================================================
CONCEPT 2: TOKEN EMBEDDINGS vs POSITIONAL EMBEDDINGS
=====================================================================

    In Stage 1, we had only a TOKEN embedding table:
        token_embedding_table[char_id] → vector

    This tells the model WHAT the character is ('a', 'b', 'c'...)
    but NOT WHERE it appears in the sequence.

    "cat" and "tac" would produce the same set of embeddings (in different
    order), but the model has no way to distinguish their order!

    Solution: Add a POSITION embedding table:
        position_embedding_table[position] → vector

    Now each character gets TWO pieces of information:
        x = token_embedding[char_id] + position_embedding[position]

    "I am at position 3, and I am the character 'a'"

    Both tables are learnable — the model discovers what position info
    is useful during training.

    KEY DETAIL: We now use a separate embedding dimension (n_embd = 32),
    not vocab_size. The token embedding maps:
        char_id → vector of size n_embd
    Then a final linear layer maps:
        n_embd → vocab_size (back to logits for prediction)

=====================================================================
CONCEPT 3: SELF-ATTENTION — The Core Innovation
=====================================================================

    Self-attention is the mechanism that allows each token to "look at"
    every previous token and decide which ones are relevant.

    The analogy: Imagine you're reading a sentence. At each word, you
    glance back at previous words to understand context. Some words
    are more relevant than others — self-attention learns which ones matter.

    HOW IT WORKS (step by step):

    For each token at position t, we compute THREE vectors:
        Q (Query)  = "What am I looking for?"
        K (Key)    = "What do I contain?"
        V (Value)  = "What information do I provide if chosen?"

    These are computed by multiplying the token's embedding by learned
    weight matrices:
        Q = x @ W_q     (x is the embedding, W_q is learnable)
        K = x @ W_k
        V = x @ W_v

    Then attention happens:
        1. For each pair (query_i, key_j): compute a "relevance score"
           score_{i,j} = Q_i · K_j  (dot product)

        2. Scale the scores: score / sqrt(head_size)
           (prevents scores from getting too large, which would make
            softmax output too "peaky" — nearly all weight on one token)

        3. Apply CAUSAL MASK: set score_{i,j} = -infinity for j > i
           (can't look into the future!)

        4. Apply softmax: convert scores to weights that sum to 1
           weights_{i,:} = softmax(scores_{i,:})

        5. Weighted sum of values:
           output_i = sum_j(weight_{i,j} * V_j)

    Result: each token's output is a weighted combination of information
    from all previous tokens, where the weights are learned!

    INTUITION:
        - If token 5 is 'o' and token 3 was 'h', the model might learn
          that 'h' at position 3 is very relevant when predicting what
          comes after 'o' (because "ho" → common in Shakespeare)
        - The attention weights encode these relevance patterns

=====================================================================
CONCEPT 4: SCALED DOT-PRODUCT ATTENTION (Mathematical View)
=====================================================================

    The full formula in matrix form:

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where:
        Q = queries matrix, shape (T, d_k)
        K = keys matrix, shape (T, d_k)
        V = values matrix, shape (T, d_k)
        d_k = head_size (dimension of each head)
        T = sequence length (block_size)

    Step by step:
        Q @ K^T → (T, T) matrix of attention scores
        / sqrt(d_k) → prevent softmax saturation
        softmax → each row sums to 1 (attention weights)
        @ V → weighted sum of values → (T, d_k) output

    WHY SCALE BY sqrt(d_k)?
        Without scaling, if d_k is large, the dot products Q·K can become
        very large in magnitude. Large values fed to softmax produce outputs
        very close to 0 or 1 (saturated). This means:
            - Gradients are nearly zero → training slows or stops
            - The model becomes "too confident" too early
        Dividing by sqrt(d_k) keeps the variance of the scores roughly 1,
        regardless of d_k. This is a simple but CRITICAL trick.

=====================================================================
CONCEPT 5: CAUSAL MASKING (The Triangular Mask)
=====================================================================

    In a language model, we must NEVER let position i attend to position j
    where j > i. That would be "looking into the future" — the model
    would see the answer before predicting it!

    We enforce this with a triangular mask:

        Position:  0  1  2  3  4
        Token 0:  [1, 0, 0, 0, 0]   ← can only see itself
        Token 1:  [1, 1, 0, 0, 0]   ← can see positions 0-1
        Token 2:  [1, 1, 1, 0, 0]   ← can see positions 0-2
        Token 3:  [1, 1, 1, 1, 0]   ← can see positions 0-3
        Token 4:  [1, 1, 1, 1, 1]   ← can see positions 0-4

    Before softmax, we set masked positions to -infinity:
        softmax(-inf) = 0 → zero attention weight → no information flows

    This is called a "causal" or "autoregressive" mask.
    
    Implementation: torch.tril (lower triangular) + masked_fill

=====================================================================
WHAT'S NEW vs STAGE 1:
=====================================================================

    Stage 1 (Bigram):
        char → token_embedding[char] → logits → prediction
        (One lookup table, 4,225 parameters)

    Stage 2 (Attention):
        char → token_emb + pos_emb → self-attention → linear → logits
        (~10K parameters, context-aware predictions)

    Key additions:
        ✦ Token embedding dimension: vocab_size → n_embd (32) → vocab_size
        ✦ Position embedding table: (block_size, n_embd)
        ✦ Self-attention head: Query, Key, Value projections + masking
        ✦ Final linear layer: n_embd → vocab_size

HARDWARE:
    Still trivially small. GTX 1660 is overkill. Minutes to train.

=====================================================================
Let's build it! 🔆
=====================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from teja.config import STAGE_2_CONFIG
from teja.utils import get_device, count_parameters, print_banner


# ========================
# REPRODUCIBILITY
# ========================
torch.manual_seed(1337)


# ========================
# HYPERPARAMETERS
# ========================
batch_size    = STAGE_2_CONFIG['batch_size']      # 32
block_size    = STAGE_2_CONFIG['block_size']       # 8
learning_rate = STAGE_2_CONFIG['learning_rate']    # 1e-3
max_iters     = STAGE_2_CONFIG['max_iters']        # 10,000
eval_interval = STAGE_2_CONFIG['eval_interval']    # 500
eval_iters    = STAGE_2_CONFIG['eval_iters']       # 200
n_embd        = STAGE_2_CONFIG['n_embd']           # 32 (embedding dimension)
head_size     = STAGE_2_CONFIG['head_size']         # 32 (attention head size)

# Device
device = get_device()

# Banner
print_banner("Positional Encoding + Self-Attention", 2)


# ========================================================================
# DATA LOADING (same as Stage 1 — we reuse the same infrastructure)
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

print(f"Dataset: {len(text):,} chars | Vocab: {vocab_size} | Train: {len(train_data):,} | Val: {len(val_data):,}")


def get_batch(split):
    """Get a random batch of input-target pairs."""
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i : i + block_size]     for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Evaluate model on train and val sets."""
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
# THE SELF-ATTENTION HEAD
# ========================================================================
# This is the NEW component in Stage 2.
# A single attention head that implements scaled dot-product attention
# with causal masking.

class Head(nn.Module):
    """
    One head of self-attention.

    This is the core building block of the Transformer.
    In Stage 3, we'll use MULTIPLE heads in parallel.
    For now, we use just one to understand the mechanism clearly.

    What it does:
        For each token in the sequence, compute attention weights
        over all previous tokens, then produce a weighted sum of
        their values.

    Architecture:
        Input (B, T, n_embd) → Q, K, V projections → Attention → Output (B, T, head_size)
    """

    def __init__(self, head_size):
        super().__init__()

        # ---- Query, Key, Value projections ----
        # These are simple linear layers (no bias) that project
        # each token's embedding into Q, K, V spaces.
        #
        # Why no bias? Convention from the original Transformer paper.
        # Adding bias doesn't significantly change performance.
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # ---- Causal mask ----
        # Pre-compute the lower-triangular mask.
        # register_buffer: stores it as part of the model but NOT as a
        # learnable parameter (no gradient computation needed).
        # Shape: (block_size, block_size)
        #
        #   [[1, 0, 0, ..., 0],
        #    [1, 1, 0, ..., 0],
        #    [1, 1, 1, ..., 0],
        #    ...
        #    [1, 1, 1, ..., 1]]
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Forward pass for one attention head.

        Args:
            x: (B, T, n_embd) — input embeddings

        Returns:
            out: (B, T, head_size) — attention-weighted output
        """
        B, T, C = x.shape  # B=batch, T=time (seq length), C=n_embd

        # Step 1: Compute Query, Key, Value for all tokens in parallel
        k = self.key(x)    # (B, T, head_size) — "what do I contain?"
        q = self.query(x)  # (B, T, head_size) — "what am I looking for?"
        v = self.value(x)  # (B, T, head_size) — "what info do I provide?"

        # Step 2: Compute attention scores (affinities)
        # Q @ K^T gives us a (T, T) matrix where entry (i, j) = how much
        # token i should attend to token j.
        # Scale by 1/sqrt(head_size) to control variance.
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)

        # Step 3: Apply causal mask
        # Set future positions to -infinity so softmax gives them weight 0.
        # We use tril[:T, :T] because T might be less than block_size during generation.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # Step 4: Softmax — convert scores to probabilities (weights sum to 1)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Step 5: Weighted aggregation of values
        # Each token's output = weighted sum of all values it attends to
        out = wei @ v  # (B, T, head_size)

        return out


# ========================================================================
# THE MODEL: BIGRAM + POSITION EMBEDDINGS + SELF-ATTENTION
# ========================================================================

class BigramWithAttention(nn.Module):
    """
    Stage 2 model: Bigram baseline enhanced with self-attention.

    Architecture:
        Input → Token Embedding + Position Embedding → Self-Attention Head → Linear → Logits

    What's new vs Stage 1:
        ✦ Token embeddings now map to n_embd (32), not vocab_size (65)
        ✦ Position embeddings add positional information
        ✦ Self-attention head lets tokens "communicate" with each other
        ✦ Final linear layer (lm_head) maps back to vocab_size for predictions

    Parameters breakdown:
        token_embedding:    vocab_size × n_embd  = 65 × 32 = 2,080
        position_embedding: block_size × n_embd  = 8  × 32 = 256
        attention Q:        n_embd × head_size   = 32 × 32 = 1,024
        attention K:        n_embd × head_size   = 32 × 32 = 1,024
        attention V:        n_embd × head_size   = 32 × 32 = 1,024
        lm_head:            head_size × vocab_size = 32 × 65 = 2,080 + 65 bias
        Total: ~7,553 parameters
    """

    def __init__(self):
        super().__init__()

        # Token embedding: character ID → vector of size n_embd
        # (Stage 1 used vocab_size → vocab_size; now we use a hidden dimension)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position embedding: position index → vector of size n_embd
        # This tells the model WHERE in the sequence each token sits.
        # "I am at position 0" vs "I am at position 7" → different vectors.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Self-attention head: the NEW component!
        # Takes in (B, T, n_embd) and outputs (B, T, head_size)
        self.sa_head = Head(head_size)

        # Language model head: project from head_size back to vocab_size
        # This converts the attention output into logits (prediction scores)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: (B, T) tensor of character indices
            targets: (B, T) tensor of target indices (None during generation)

        Returns:
            logits: (B, T, vocab_size) prediction scores
            loss: scalar cross-entropy loss (None if no targets)
        """
        B, T = idx.shape

        # Step 1: Token embeddings — WHAT each character is
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        # Step 2: Position embeddings — WHERE each character is
        # torch.arange(T) gives [0, 1, 2, ..., T-1]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)

        # Step 3: Combine — add token identity + position information
        # Broadcasting: (B, T, n_embd) + (T, n_embd) → (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Step 4: Self-attention — let tokens communicate!
        # This is where the magic happens. Each token can now "look at"
        # all previous tokens and gather relevant information.
        x = self.sa_head(x)  # (B, T, head_size)

        # Step 5: Project to vocabulary — get prediction logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressive text generation.

        Same approach as Stage 1, but now we must be careful about
        the context window: position embeddings only go up to block_size,
        so we must crop the input to the last block_size characters.

        In Stage 1 (bigram), this didn't matter because the model only
        looked at the last character anyway. Now it matters because the
        model uses positional information.
        """
        for _ in range(max_new_tokens):
            # Crop context to at most block_size tokens
            # (position embeddings only exist for indices 0..block_size-1)
            idx_cond = idx[:, -block_size:]  # (B, T) where T <= block_size

            # Forward pass
            logits, loss = self(idx_cond)

            # Get last token's predictions
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ========================================================================
# CREATE MODEL AND OPTIMIZER
# ========================================================================
print("\nCreating Stage 2 model (Bigram + Self-Attention)...")
model = BigramWithAttention()
model = model.to(device)
total_params = count_parameters(model)

# Compare to Stage 1
print(f"   Stage 1 had 4,225 params. Stage 2 has {total_params:,} params.")
print(f"   Increase: {total_params / 4225:.1f}x more parameters\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ========================================================================
# GENERATE TEXT BEFORE TRAINING (BASELINE)
# ========================================================================
print("Text from UNTRAINED Stage 2 model (expect randomness):")
print("-" * 50)
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(seed, max_new_tokens=200)[0].tolist()))
print("-" * 50)


# ========================================================================
# TRAINING LOOP
# ========================================================================
print(f"\nTraining for {max_iters:,} iterations...")
print(f"   Batch size: {batch_size} | Block size: {block_size} | Embedding dim: {n_embd}")
print(f"   Head size: {head_size} | Learning rate: {learning_rate}")
print(f"   Evaluating every {eval_interval} steps\n")

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"   Step {iter:>5,}: train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final evaluation
losses = estimate_loss()
print(f"   Step {max_iters:>5,}: train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f}")
print(f"\n   Training complete!")


# ========================================================================
# GENERATE TEXT AFTER TRAINING
# ========================================================================
print("\nText from TRAINED Stage 2 model (expect improvement over Stage 1):")
print("=" * 60)
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(seed, max_new_tokens=500)[0].tolist())
print(generated)
print("=" * 60)


# ========================================================================
# ATTENTION VISUALIZATION
# ========================================================================
# Let's peek inside the model to see what the attention head learned!
# We'll feed in a short sentence and visualize the attention weights.

print("\n--- Attention Weight Visualization ---")
print("(Shows which previous characters each position attends to)")

test_text = "First Ci"
test_encoded = torch.tensor([encode(test_text)], dtype=torch.long, device=device)

# Get attention weights by doing a manual forward pass through layers
with torch.no_grad():
    # Get embeddings
    tok_emb = model.token_embedding_table(test_encoded)
    pos_emb = model.position_embedding_table(torch.arange(len(test_text), device=device))
    x = tok_emb + pos_emb

    # Compute Q, K
    q = model.sa_head.query(x)
    k = model.sa_head.key(x)

    # Compute attention weights
    T = len(test_text)
    wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
    wei = wei.masked_fill(model.sa_head.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    # Print attention matrix
    print(f"\nInput: '{test_text}'")
    print(f"{'':>12}", end='')
    for ch in test_text:
        print(f"  {ch:>5}", end='')
    print()

    for i in range(T):
        print(f"  '{test_text[i]}' (pos {i})", end='')
        for j in range(T):
            w = wei[0, i, j].item()
            if w > 0.001:
                print(f"  {w:5.3f}", end='')
            else:
                print(f"      -", end='')
        print()

print("\nReading the matrix: Row = 'from' token, Column = 'to' token")
print("High values mean that token pays strong attention to that position.")


# ========================================================================
# VERIFICATION SUMMARY
# ========================================================================
print(f"""
{'='*60}
  TEJA -- Stage 2 Complete!
{'='*60}

  Parameters:       {total_params:,} (Stage 1 had 4,225)
  Final train loss: {losses['train']:.4f}
  Final val loss:   {losses['val']:.4f}
  Device:           {device}
  
  COMPARISON TO STAGE 1:
    Stage 1 (Bigram):    val loss ~ 2.47 | 4,225 params
    Stage 2 (Attention): val loss ~ {losses['val']:.2f} | {total_params:,} params

  WHAT WE LEARNED:
    - Position embeddings tell the model WHERE each character sits
    - Self-attention lets each token gather info from all previous tokens
    - Query/Key/Value projections are the heart of the mechanism
    - Causal masking prevents looking into the future
    - Even a single attention head improves over the context-free bigram

  WHAT COMES NEXT (Stage 3):
    - MULTI-HEAD attention: run several attention heads in PARALLEL
      (each head can learn to look for different patterns)
    - FEED-FORWARD NETWORK: add an MLP after attention
      (attention = "gather info", FFN = "process it")
    - Stack multiple TRANSFORMER BLOCKS for deeper reasoning
    - This is a real (mini) Transformer!
{'='*60}
""")
