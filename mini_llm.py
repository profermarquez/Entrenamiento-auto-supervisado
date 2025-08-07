import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= Cargar y preparar el texto =======
with open("el_principito.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# ======= Preparar batches =======
block_size = 64
batch_size = 32
# Acá es donde se produce el aprendizaje auto-supervisado.
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # entrada
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # salida esperada (objetivo)
    return x, y

# ======= Definir el modelo =======
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, emb_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

# ======= Entrenamiento =======
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Entrenando modelo...")

for step in range(12000):
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)
# Acá se compara lo que el modelo predice (logits) con lo que debería haber predicho (yb), y se ajustan los pesos del modelo para que la próxima vez falle menos.
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# ======= Generación de texto =======
def sample(model, start_text="Había", length=200):
    model.eval()
    context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    output = encode(start_text)  # empezamos con lista de enteros

    for _ in range(length):
        logits = model(context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_char = torch.multinomial(probs, num_samples=1)
        next_char_int = next_char.item()
        output.append(next_char_int)
        context = torch.cat([context, next_char], dim=1)

    return decode(output)

print("\n=== Texto generado ===\n")
print(sample(model, start_text="Había una vez "))

# Guardar el modelo
torch.save(model.state_dict(), "mini_llm.pt")
