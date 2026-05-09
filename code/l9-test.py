import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. 模型定义（必须和训练时完全一致）
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

        self.max_len = max_len

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
# 2. 加载模型
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("decoder_only.pt", map_location=device)

model = DecoderOnlyTransformer(ckpt["vocab_size"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

char2idx = ckpt["char2idx"]
idx2char = ckpt["idx2char"]

block_size = model.max_len
eos_token = "\n"
print("模型加载完成！")


# ============================================================
# 3. 文本生成函数（支持 temperature）
# ============================================================
def generate(model, prompt, max_new_tokens=200, temperature=1.0):
    model.eval()

    idx = torch.tensor(
        [char2idx[c] for c in prompt],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            # ✅ ===== 加入 EOS 停止条件 =====
            next_char = idx2char[next_idx.item()]
            if next_char == eos_token:
                break

            idx = torch.cat([idx, next_idx], dim=1)

    return "".join([idx2char[i] for i in idx[0].tolist()])

# ============================================================
# 4. 交互模式
# ============================================================
print("\n进入交互模式（输入 exit 退出）")

while True:
    prompt = input("\n你: ")

    if prompt.strip().lower() == "exit":
        break

    try:
        output = generate(
            model,
            prompt,
            max_new_tokens=300,
            temperature=0.8
        )
        print("\n模型:\n", output)

    except KeyError:
        print("⚠️ 输入包含未见字符（训练语料中没有）")
