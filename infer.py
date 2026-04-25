import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# Fix Windows console encoding for non-ASCII tokens
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

device = 'cuda'
n_embd, n_head, n_layer, block_size, dropout = 384, 6, 6, 256, 0.2
TOTAL_VOCAB     = 50_261
SYSTEM_TOKEN    = 50_257
USER_TOKEN      = 50_258
ASSISTANT_TOKEN = 50_259
EOT_TOKEN       = 50_260

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(),
                                 nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa   = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
    def forward(self, x):
        return x + self.ffwd(self.ln2(x + self.sa(self.ln1(x))))

class TejaGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == EOT_TOKEN:
                break
        return idx

enc   = tiktoken.get_encoding('gpt2')
model = TejaGPT(vocab_size=TOTAL_VOCAB).to(device)
ckpt  = torch.load('teja_stage7_best.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded! val loss: {ckpt['val_loss']:.4f}, step: {ckpt['step']}\n")

def chat(prompt):
    ids = ([SYSTEM_TOKEN] + enc.encode("You are Teja, a helpful assistant.") + [EOT_TOKEN] +
           [USER_TOKEN]   + enc.encode(prompt) + [EOT_TOKEN] +
           [ASSISTANT_TOKEN])
    x = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
    out = model.generate(x)[0].tolist()
    return enc.decode([t for t in out[len(ids):] if t < 50_257])

questions = [
    "What is the capital of France?",
    "Write a Python function to reverse a string.",
    "What is 2 + 2?",
    "Who created you?",
]

for q in questions:
    print(f"User: {q}")
    print(f"Teja: {chat(q)}")
    print()
