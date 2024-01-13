import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embed = 384
n_layer = 6
n_head = 6
dropout = 0.2
# end of hyperparameters

# set seed for reproducibility
torch.manual_seed(1337)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data)) # 90% train, 10% val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        # q, k, v are all (B, T, head_size)
        # compute scaled dot-product attention
        wei = q @ k.transpose(-1, -2) * q.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B, T, n_embed)
        out = [h(x) for h in self.heads] # list of (B, T, head_size)
        out = torch.cat(out, dim=-1) # (B, T, n_heads * head_size)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # projection layer going back to the residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block with communication followed by computation"""
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_head, n_embed//n_head)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # apply self-attention (B, T, n_embed)
        x = x + self.ff(self.ln2(x)) # apply feed-forward (B, T, n_embed)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embed)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T, n_embed)
        x = tok_emb + pos_emb # (B, T, n_embed)
        x = self.blocks(x) # (B, T, n_embed)
        x = self.ln_final(x) # (B, T, n_embed)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B * T)
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop to block_size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond) # B, T, C
            # focus only on the las time stamp
            logits = logits[:, -1, :] # Becomes B, C
            # apply softmax
            probs = F.softmax(logits, dim=-1) # B, C
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat((idx, idx_next), dim=1) # B, T + 1
        return idx
    
@torch.no_grad()
def estimate_loss():
  # implementation of the estimate_loss function goes here
  out = {}
  model.eval()
  for split in ['train', 'val']:
    total_loss = 0
    for _ in range(eval_iters):
      x, y = get_batch(split)
      _, loss = model(x, y)
      total_loss += loss.item()
    out[split] = total_loss / eval_iters
  model.train()
  return out


model = BigramLanguageModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  
  # evaluate loss on train and val dataset

  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"iter {iter:5d} | train loss {losses['train']:5.4f} | val loss {losses['val']:5.4f}")

  # train the model
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

model.eval()

# generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))