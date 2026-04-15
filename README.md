# 🔆 Teja — Built from Zero. Trained to Shine.

<p align="center">
  <em>A from-scratch LLM, built incrementally with Python & PyTorch</em>
</p>

---

**Teja** (തേജ) means *"brightness / radiance"* in Malayalam.

Built by **Sabith**, from **Nilambur, Kerala**.

## What is this?

Teja is a large language model built **entirely from scratch** — no pre-trained weights, no HuggingFace shortcuts. Every piece of the architecture is written by hand, understood deeply, and tested at each stage.

The goal is not just to build an LLM, but to **understand** how every component works — from a simple bigram model all the way to a GPT-2 scale transformer.

## Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Bigram character-level language model (baseline) | 🔧 In Progress |
| 2 | Positional encoding + self-attention (single head) | ⏳ Upcoming |
| 3 | Multi-head attention + feed-forward (mini Transformer) | ⏳ Upcoming |
| 4 | Full GPT-2 style Transformer (LayerNorm + residuals) | ⏳ Upcoming |
| 5 | Train on real text dataset (Wikipedia / OpenWebText) | ⏳ Upcoming |
| 6 | Add BPE tokenizer (subword instead of character-level) | ⏳ Upcoming |
| 7 | Instruction fine-tuning (chat assistant) | ⏳ Upcoming |
| 8 | Scale up (more layers, more data, more params) | ⏳ Upcoming |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run a stage
python stages/stage1_bigram.py
```

## Hardware

- **Local**: GTX 1660 (6GB VRAM), Ryzen 5 5500 — Stages 1–4
- **Cloud**: Google Colab (T4/A100) — Stages 5+

## Philosophy

> *Start small. Understand deeply. Scale deliberately.*

Every stage is self-contained and runnable. Every concept is explained before being coded. Nothing is a black box.

---

<p align="center">
  <strong>🔆 Teja</strong> — <em>Built from zero. Trained to shine.</em>
</p>
