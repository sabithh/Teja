"""
Teja Model Configuration
========================
Hyperparameters for each stage of the Teja LLM.

Each config is a dictionary that can be unpacked into the model constructor.
As we progress through stages, configs grow to include more parameters.
"""


# =============================================================================
# Stage 1 — Bigram Character-Level Language Model
# =============================================================================
# The simplest possible config. The bigram model only needs:
#   - batch_size: how many independent sequences to process in parallel
#   - block_size: maximum context length (bigram only uses 1, but we set up infra)
#   - learning_rate: step size for the optimizer
#   - max_iters: total training iterations
#   - eval_interval: how often to evaluate on validation set
#   - eval_iters: how many batches to average for a stable eval loss

STAGE_1_CONFIG = {
    'batch_size': 32,       # Number of independent sequences per batch
    'block_size': 8,        # Maximum context length for predictions
    'learning_rate': 1e-3,  # Adam optimizer learning rate
    'max_iters': 10_000,    # Total training steps
    'eval_interval': 500,   # Evaluate every N steps
    'eval_iters': 200,      # Average loss over this many eval batches
}


# =============================================================================
# Stage 2 — Self-Attention (Single Head)
# =============================================================================
STAGE_2_CONFIG = {
    'batch_size': 32,
    'block_size': 8,
    'learning_rate': 1e-3,
    'max_iters': 10_000,
    'eval_interval': 500,
    'eval_iters': 200,
    'n_embd': 32,           # Embedding dimension
    'head_size': 32,        # Size of each attention head
}


# =============================================================================
# Stage 3 — Multi-Head Attention + Feed-Forward (Mini Transformer)
# =============================================================================
STAGE_3_CONFIG = {
    'batch_size': 32,
    'block_size': 32,       # Longer context now
    'learning_rate': 1e-3,
    'max_iters': 10_000,
    'eval_interval': 500,
    'eval_iters': 200,
    'n_embd': 64,           # Larger embeddings
    'n_head': 4,            # 4 attention heads (head_size = 64/4 = 16)
    'n_layer': 4,           # 4 transformer blocks
    'dropout': 0.0,         # No dropout for small model
}


# =============================================================================
# Stage 4 — Full GPT-2 Architecture (LayerNorm + Residuals)
# =============================================================================
STAGE_4_CONFIG = {
    'batch_size': 64,
    'block_size': 64,
    'learning_rate': 1e-3,
    'max_iters': 15_000,
    'eval_interval': 500,
    'eval_iters': 200,
    'n_embd': 128,          # GPT-2 small starts at 768, we use 128 for local
    'n_head': 4,
    'n_layer': 6,           # 6 transformer blocks
    'dropout': 0.1,         # Light regularization
}


# =============================================================================
# Stage 5+ — Scaled configs (for Colab)
# =============================================================================
STAGE_5_CONFIG = {
    'batch_size': 64,
    'block_size': 256,
    'learning_rate': 3e-4,  # Lower LR for larger model
    'max_iters': 50_000,
    'eval_interval': 1_000,
    'eval_iters': 200,
    'n_embd': 384,
    'n_head': 6,
    'n_layer': 6,
    'dropout': 0.2,
}
