"""
🔆 TEJA — Stage 1: Bigram Character-Level Language Model
=========================================================
Built from zero. Trained to shine.
Created by Sabith, Nilambur, Kerala.

STAGE 1 GOAL:
    Build the simplest possible language model — a bigram model.
    It predicts the next character based ONLY on the current character.
    This establishes our training loop, loss computation, and text
    generation pipeline that every future stage will build upon.

WHAT IS A LANGUAGE MODEL?
    A language model assigns probabilities to sequences of tokens.
    Given some text, it answers: "What comes next?"

    For example:
        Input:  "hell"
        Output: "o" (with high probability)

    Formally, a language model estimates:
        P(x_t | x_1, x_2, ..., x_{t-1})
    i.e., the probability of the next token given all previous tokens.

WHAT IS A BIGRAM MODEL?
    The simplest Markov assumption: the next character depends ONLY
    on the current character. We ignore all earlier context.

        P(x_t | x_{t-1})   ← that's it, just one character of history

    For example, if the current character is 'h':
        P('e' | 'h') = 0.35    ← "he" is common
        P('a' | 'h') = 0.20    ← "ha" is common
        P('z' | 'h') = 0.001   ← "hz" is rare

    This is obviously terrible — it has ZERO understanding of context.
    But it gives us a baseline to beat in later stages.

CHARACTER-LEVEL TOKENIZATION:
    In Stages 1-5, each unique character is one "token".
    Our vocabulary = all unique characters in the training data.

    Shakespeare has 65 unique characters:
        \n, ' ', '!', ... 'A', 'B', ... 'a', 'b', ... 'z'

    Each character gets an integer ID (index):
        'a' → 39, 'b' → 40, 'A' → 13, ' ' → 1, '\n' → 0, etc.

    This is the simplest possible tokenization. Stage 6 will replace
    this with BPE (Byte Pair Encoding) for much better performance.

THE EMBEDDING TABLE:
    The entire bigram model is just one matrix:
        E ∈ ℝ^(vocab_size × vocab_size)

    To predict what comes after character 'h' (ID=46):
        1. Look up row 46 of E → gives a vector of 65 numbers (logits)
        2. Apply softmax → probabilities over all 65 characters
        3. The highest probability character is the prediction

    That's the entire model. No hidden layers. No attention.
    The embedding lookup table IS the model.

CROSS-ENTROPY LOSS:
    How do we measure "how wrong" our predictions are?

    Cross-entropy loss = -log(P(correct_answer))

    If the model assigns probability 0.9 to the correct next character:
        loss = -log(0.9) = 0.105  ← low loss, good!

    If the model assigns probability 0.01 to the correct next character:
        loss = -log(0.01) = 4.605  ← high loss, bad!

    At initialization (random weights), every character is equally likely:
        P(any char) = 1/65 ≈ 0.0154
        loss = -log(1/65) = log(65) ≈ 4.17

    So our initial loss should be ~4.17. If it's not, something is wrong.
    After training, a bigram model on Shakespeare typically reaches ~2.45.

THE TRAINING LOOP:
    1. FORWARD PASS: Feed input through model → get predictions
    2. COMPUTE LOSS: Compare predictions to actual next characters
    3. BACKWARD PASS: Compute gradients (how to adjust each weight)
    4. UPDATE WEIGHTS: Move weights in the direction that reduces loss
    5. REPEAT for many iterations

TEXT GENERATION (INFERENCE):
    Once trained, we generate text autoregressively:
    1. Start with a "seed" character (or newline)
    2. Model predicts probability distribution over next character
    3. SAMPLE from that distribution (don't just pick the max — adds variety)
    4. Append the sampled character to input
    5. Repeat steps 2-4 for as many characters as desired

    Sampling vs. argmax:
        argmax → always picks most likely → repetitive, boring
        sampling → picks randomly weighted by probability → varied, interesting

WHAT TO EXPECT:
    The output will look like semi-random character soup. You'll see
    some recognizable short words (the, and, to, I) but no coherent
    sentences. This is expected — the bigram model is just a baseline.

    Stage 2 (self-attention) will dramatically improve output quality.

HARDWARE:
    This stage is trivially small. A GTX 1660 is massive overkill.
    Training will complete in seconds (< 1 minute).
    Even CPU would work fine, but we use GPU to verify CUDA setup.

=====================================================================
Let's build it! 🔆
=====================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Add project root to path so we can import from teja/
# ---------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from teja.config import STAGE_1_CONFIG
from teja.utils import get_device, count_parameters, print_banner


# ========================
# REPRODUCIBILITY
# ========================
# Setting a random seed ensures we get the same results every run.
# This is critical for debugging — if results change randomly,
# you can't tell if a code change helped or just got lucky.
torch.manual_seed(1337)


# ========================
# HYPERPARAMETERS
# ========================
# Load from config, but define here for clarity
batch_size    = STAGE_1_CONFIG['batch_size']      # 32 sequences per batch
block_size    = STAGE_1_CONFIG['block_size']       # 8 characters of context
learning_rate = STAGE_1_CONFIG['learning_rate']    # 1e-3
max_iters     = STAGE_1_CONFIG['max_iters']        # 10,000 training steps
eval_interval = STAGE_1_CONFIG['eval_interval']    # evaluate every 500 steps
eval_iters    = STAGE_1_CONFIG['eval_iters']       # average over 200 eval batches

# Detect GPU
device = get_device()

# Print the stage banner
print_banner("Bigram Character-Level Language Model", 1)


# ========================================================================
# STEP 1: LOAD AND PREPARE THE DATA
# ========================================================================
# Read the entire Shakespeare text file
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"📄 Dataset size: {len(text):,} characters")

# --- Build the vocabulary ---
# Get all unique characters in the text, sorted alphabetically.
# This is our "vocabulary" — the set of all possible tokens.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"📝 Vocabulary size: {vocab_size} unique characters")
print(f"   Characters: {''.join(chars)}")

# --- Create encoder/decoder mappings ---
# We need to convert between characters ↔ integers.
#   encode: 'hello' → [46, 43, 50, 50, 53]
#   decode: [46, 43, 50, 50, 53] → 'hello'
#
# stoi = "string to integer" (char → number)
# itos = "integer to string" (number → char)
stoi = {ch: i for i, ch in enumerate(chars)}  # {'a': 39, 'b': 40, ...}
itos = {i: ch for i, ch in enumerate(chars)}  # {39: 'a', 40: 'b', ...}

def encode(s):
    """Convert a string to a list of integers."""
    return [stoi[c] for c in s]

def decode(l):
    """Convert a list of integers back to a string."""
    return ''.join([itos[i] for i in l])

# --- Encode the entire dataset ---
# Convert all 1.1M characters into a single long tensor of integers.
data = torch.tensor(encode(text), dtype=torch.long)
print(f"🔢 Encoded data shape: {data.shape}")
print(f"   First 20 encoded values: {data[:20].tolist()}")
print(f"   Decoded back: '{decode(data[:20].tolist())}'")

# --- Train/Validation split ---
# Use 90% for training, 10% for validation.
# The validation set is data the model NEVER sees during training.
# It tells us if the model is actually learning general patterns
# or just memorizing the training data (overfitting).
n = int(0.9 * len(data))
train_data = data[:n]   # First 90%
val_data   = data[n:]   # Last 10%
print(f"📊 Train: {len(train_data):,} chars | Val: {len(val_data):,} chars")


# ========================================================================
# STEP 2: DATA LOADING (BATCHING)
# ========================================================================
# We can't feed the entire 1M-character sequence to the model at once.
# Instead, we randomly sample small chunks called "batches".
#
# Each batch contains `batch_size` independent sequences, each of
# length `block_size`. For every input position, we know the target
# (the next character).
#
# Example with block_size=4:
#   Input:  [18, 47, 56, 57]
#   Target: [47, 56, 57, 58]
#
# This gives us 4 training examples from one sequence:
#   When input is [18],             target is 47
#   When input is [18, 47],         target is 56
#   When input is [18, 47, 56],     target is 57
#   When input is [18, 47, 56, 57], target is 58

def get_batch(split):
    """
    Generate a small batch of input-target pairs.

    Args:
        split: 'train' or 'val' — which dataset to sample from

    Returns:
        x: (batch_size, block_size) tensor of input character indices
        y: (batch_size, block_size) tensor of target character indices
    """
    # Select the right dataset
    data_source = train_data if split == 'train' else val_data

    # Generate `batch_size` random starting positions.
    # We subtract block_size to ensure we don't go past the end.
    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    # Stack the sequences into a batch tensor
    x = torch.stack([data_source[i : i + block_size]     for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])

    # Move to GPU if available
    x, y = x.to(device), y.to(device)
    return x, y


# ========================================================================
# STEP 3: ESTIMATE LOSS (without affecting gradients)
# ========================================================================
# During training, we periodically want to check how well the model is
# doing on BOTH train and val sets. We average over many batches for
# a stable estimate (single batch loss is noisy).

@torch.no_grad()  # Don't compute gradients during evaluation (saves memory)
def estimate_loss():
    """
    Evaluate the model on train and val sets.

    Returns:
        dict with 'train' and 'val' average losses.
    """
    out = {}
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()  # Set back to training mode
    return out


# ========================================================================
# STEP 4: THE BIGRAM MODEL
# ========================================================================
# This is the heart of Stage 1. The entire model is just an embedding table.
#
# nn.Embedding(vocab_size, vocab_size) creates a matrix of shape (65, 65).
# Row i contains the logits for "given character i, predict next character".
#
# For the bigram model:
#   - Input: a character index (e.g., 46 for 'h')
#   - Output: 65 logits (unnormalized scores for each possible next char)
#   - Loss: cross-entropy between logits and actual next character

class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model.

    Given the current character, predict the next character.
    The model is just a single embedding table — no hidden layers,
    no attention mechanism, no complex architecture.

    This is our BASELINE. Every future stage must beat this.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # The token embedding table: (vocab_size, vocab_size)
        # Row i = "logits for predicting what comes after character i"
        # This single matrix IS the entire model.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass of the bigram model.

        Args:
            idx: (B, T) tensor of character indices
                 B = batch_size, T = block_size (sequence length)
            targets: (B, T) tensor of target indices (next characters)
                     None during generation (inference)

        Returns:
            logits: (B, T, vocab_size) prediction scores for each position
            loss:   scalar cross-entropy loss (None if no targets provided)
        """
        # Look up each character in the embedding table.
        # For each of the B*T input characters, we get vocab_size logits.
        # Shape: (B, T) → (B, T, C) where C = vocab_size
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            # Reshape for PyTorch's cross_entropy function.
            # cross_entropy expects: (N, C) logits and (N,) targets
            # where N = number of predictions, C = number of classes
            B, T, C = logits.shape
            logits = logits.view(B * T, C)    # (B*T, C)
            targets = targets.view(B * T)      # (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new text by repeatedly predicting the next character.

        This is AUTOREGRESSIVE generation:
            1. Model predicts probabilities for next character
            2. We sample from those probabilities
            3. Append the sampled character to the sequence
            4. Repeat

        Args:
            idx: (B, T) tensor — the initial "seed" sequence
            max_new_tokens: how many new characters to generate

        Returns:
            (B, T + max_new_tokens) tensor — the extended sequence
        """
        for _ in range(max_new_tokens):
            # Get predictions for the current sequence.
            # For bigram, only the LAST character matters,
            # but we pass the whole thing for consistency with later stages.
            logits, loss = self(idx)

            # We only care about the LAST time step's prediction
            logits = logits[:, -1, :]  # (B, C)

            # Convert logits to probabilities via softmax
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample the next character from the probability distribution
            # (not argmax — sampling gives more diverse, interesting output)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new character to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# ========================================================================
# STEP 5: CREATE THE MODEL AND OPTIMIZER
# ========================================================================
print("\n🏗️  Creating Bigram Language Model...")
model = BigramLanguageModel(vocab_size)
model = model.to(device)  # Move model to GPU
count_parameters(model)   # Should be 65 * 65 = 4,225 parameters

# The optimizer adjusts model weights to minimize the loss.
# Adam is the standard choice — it adapts learning rate per-parameter
# and generally works well without much tuning.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ========================================================================
# STEP 6: GENERATE TEXT BEFORE TRAINING (UNTRAINED BASELINE)
# ========================================================================
print("\n📝 Text from UNTRAINED model (expect pure randomness):")
print("-" * 50)
# Start with a single newline character (index 0) as the seed
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
untrained_text = decode(model.generate(seed, max_new_tokens=300)[0].tolist())
print(untrained_text)
print("-" * 50)


# ========================================================================
# STEP 7: TRAINING LOOP
# ========================================================================
print(f"\n🚀 Training for {max_iters:,} iterations...")
print(f"   Batch size: {batch_size} | Block size: {block_size}")
print(f"   Learning rate: {learning_rate}")
print(f"   Evaluating every {eval_interval} steps\n")

for iter in range(max_iters):

    # --- Evaluate periodically ---
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"   Step {iter:>5,}: train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f}")

    # --- Get a random batch of training data ---
    xb, yb = get_batch('train')

    # --- Forward pass ---
    logits, loss = model(xb, yb)

    # --- Backward pass ---
    optimizer.zero_grad(set_to_none=True)  # Clear old gradients
    loss.backward()                         # Compute new gradients

    # --- Update weights ---
    optimizer.step()

# --- Final evaluation ---
losses = estimate_loss()
print(f"   Step {max_iters:>5,}: train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f}")
print(f"\n✅ Training complete!")
print(f"   Loss improvement: {4.17:.2f} → {losses['val']:.4f} (random → trained)")


# ========================================================================
# STEP 8: GENERATE TEXT AFTER TRAINING
# ========================================================================
print("\n📝 Text from TRAINED model (expect character-soup with real words):")
print("=" * 60)
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
trained_text = decode(model.generate(seed, max_new_tokens=500)[0].tolist())
print(trained_text)
print("=" * 60)


# ========================================================================
# STEP 9: VERIFICATION SUMMARY
# ========================================================================
print(f"""
{'='*60}
  🔆 TEJA — Stage 1 Complete!
{'='*60}

  ✅ Initial loss:     ~4.17 (random, expected: -ln(1/{vocab_size}) = {-torch.log(torch.tensor(1.0/vocab_size)).item():.4f})
  ✅ Final train loss: {losses['train']:.4f}
  ✅ Final val loss:   {losses['val']:.4f}
  ✅ Parameters:       {sum(p.numel() for p in model.parameters()):,}
  ✅ Device:           {device}

  WHAT WE LEARNED:
    - Built a complete training pipeline (data → model → train → generate)
    - The bigram model is just an embedding table lookup
    - Loss decreased from ~4.17 to ~2.45 (model learned character-pair frequencies)
    - Generated text has recognizable short words but no coherent sentences

  WHAT COMES NEXT (Stage 2):
    - Add POSITIONAL EMBEDDINGS (so the model knows WHERE in the sequence it is)
    - Add SELF-ATTENTION (so the model can look at ALL previous characters)
    - This will be our first step toward a real Transformer

  🔮 KEY INSIGHT:
    The bigram model proves that even with zero context awareness,
    a model can learn basic character statistics. Self-attention
    (Stage 2) will give the model actual understanding of context.
{'='*60}
""")
