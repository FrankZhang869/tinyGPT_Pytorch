import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import embed_dim, num_heads, num_layers, dropout, block_size, device
from data import vocab_size

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        T, B, C = x.shape

        mask = torch.tril(torch.ones(T, T, device=device)) == 0
        attn_out, _ = self.sa(x, x, x, attn_mask=mask)

        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=device)
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb
        x = x.transpose(0, 1)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        x = x.transpose(0, 1)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
