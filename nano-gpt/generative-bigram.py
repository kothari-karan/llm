import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

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
            logits, _ = self(idx) # B, T, C
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


model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  
  # evaluate loss on train and val dataset

  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"iter {iter:5d} | train loss {losses['train']:5.2f} | val loss {losses['val']:5.2f}")

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
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))