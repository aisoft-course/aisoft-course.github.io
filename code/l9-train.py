import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
import time

# ============================================================
# 0. 下载数据
# ============================================================
if not os.path.exists("shakespeare.txt"):
    print("下载 Shakespeare 数据...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    with open("shakespeare.txt", "w", encoding="utf-8") as f:
        f.write(text)

# ============================================================
# 1. 读取数据
# ============================================================
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()[:10000]

print("文本长度:", len(text))

# ============================================================
# 2. tokenizer（字符级）
# ============================================================
chars = sorted(list(set(text)))
vocab_size = len(chars)

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

# ============================================================
# 3. Dataset
# ============================================================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, i):
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+self.block_size+1]
        return x, y


block_size = 128
batch_size = 64

dataset = TextDataset(data, block_size)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ============================================================
# 4. 模型
# ============================================================

# ---------------- Self-Attention ----------------
class SelfAttention(nn.Module):
    def __init__(self, d_model=256, d_k=256, d_v=256):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_v)

        self.out_proj = nn.Linear(d_v, d_model)

    def forward(self, x):
        B, T, _ = x.shape

        Q = self.q_proj(x)  # (B, T, d_k)
        K = self.k_proj(x)  # (B, T, d_k)
        V = self.v_proj(x)  # (B, T, d_v)

        # Attention score
        scores = Q @ K.transpose(-2, -1)  # (B, T, T)
        scores = scores / (Q.shape[-1] ** 0.5)

        # 因果 mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)

        out = attn @ V  # (B, T, d_v)

        return self.out_proj(out)  # (B, T, d_model)


# ---------------- Feed Forward ----------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Decoder Block ----------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model=256, d_k=256, d_v=256, d_ff=1024):
        super().__init__()

        self.attn = SelfAttention(d_model, d_k, d_v)
        self.ffn = FeedForward(d_model, d_ff)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


# ---------------- Decoder-only Transformer ----------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_q=256, d_v=256, d_ff=1024, num_layers=6, max_len=512):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, d_q, d_v, d_ff) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)

        x = self.token_emb(idx) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits


# ============================================================
# 5. 训练
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DecoderOnlyTransformer(vocab_size=vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

epochs = 5
log_interval = 100

global_step = 0

for ep in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if global_step % log_interval == 0:
            elapsed = time.time() - start_time
            tokens = log_interval * batch_size * block_size
            speed = tokens / elapsed

            print(f"[Epoch {ep+1} | Step {global_step}] "
                  f"loss={total_loss/log_interval:.4f}, "
                  f"{speed:.0f} tokens/s")

            total_loss = 0
            start_time = time.time()

    print(f"Epoch {ep+1} 完成")

# ============================================================
# 6. 保存模型
# ============================================================
torch.save({
    "model": model.state_dict(),
    "vocab_size": vocab_size,
    "char2idx": char2idx,
    "idx2char": idx2char
}, "decoder_only.pt")

print("模型已保存")
