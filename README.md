[main.md](https://github.com/user-attachments/files/24590725/main.md)
``` python
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
```
``` python
with open("input.txt", "r", encoding="utf-8") as f:
    file = f.read()
letters = list(file)
print(len(letters))
```
``` python
chars = sorted(set(letters))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}

print(stoi)

vocab_size = len(stoi)
d_model = 512
token_embedding = nn.Embedding(vocab_size, d_model)

indices = [stoi[ch] for ch in letters]
C = token_embedding(torch.tensor(indices, dtype=torch.long))
print(C.shape)
```

::: {.output .stream .stdout}
    {'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}
    torch.Size([1115394, 512])
:::
::::

``` python
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```
``` python
block_size = 128
batch_size = 32

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x, y

data = torch.tensor(indices, dtype=torch.long)
x, y = get_batch(data, block_size=128, batch_size=32) # 32 x 128
x_emb = token_embedding(x) # 32 x 128 x 512
pos_enc = positional_encoding(128, 512)
x_emb = x_emb + pos_enc
```
``` python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None): # x.shape is (32 x 128 x 512)
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k) # (32 x 128 x 8 x 64)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        Q = Q.transpose(1, 2) # (32 x 8 x 128 x 64)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k) # (32 x 8 x 128 x 64) @ (32 x 8 x 64 x 128) -> (32 x 8 x 128 x 128)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V # (32 x 8 x 128 x 128) @ (32 x 8 x 128 x 64) -> (32 x 8 x 128 x 64)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(out)
```
``` python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x
```
``` python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.masked_attn(x, mask=mask)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x
```
``` python
def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    print(mask)
    mask = mask == 0
    return mask

def decode(indices):
    return ''.join([itos[int(i)] for i in indices])
```
``` python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, block_size):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.register_buffer(
            'causal_mask',
            create_causal_mask(block_size).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x) # (B, T, d_model)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb # (B, T, d_model)

        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx
```
``` python
model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    block_size=128
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
eval_interval = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    for step in range(1000):
        x, y = get_batch(data, block_size=128, batch_size=32)
        x, y = x.to(device), y.to(device)

        logits = model(x) # (32 x 128 x 65)
        logits = logits.view(-1, vocab_size)
        y = y.view(-1)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    print("\n=== Generation ====")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200)

    print(decode(generated[0]))
    print("=" * 50 + "\n")
```

::: {.output .stream .stdout}
    tensor([[0., 1., 1.,  ..., 1., 1., 1.],
            [0., 0., 1.,  ..., 1., 1., 1.],
            [0., 0., 0.,  ..., 1., 1., 1.],
            ...,
            [0., 0., 0.,  ..., 0., 1., 1.],
            [0., 0., 0.,  ..., 0., 0., 1.],
            [0., 0., 0.,  ..., 0., 0., 0.]])
    Epoch 0, Step 0, Loss: 4.3087
    Epoch 0, Step 500, Loss: 1.7103

    === Generation ====


    BALTHASTIO:
    My lords, I did them not no mark you give
    The shine of heard to hear heards and these was
    I sexcurtuted him it deper as one seving
    Will my follour'd pleacent wiced beas look.

    Bless Of Gl
    ==================================================

    Epoch 1, Step 0, Loss: 1.4198
    Epoch 1, Step 500, Loss: 1.3780

    === Generation ====

    Would up he time were thou wilt profare
    Your cabins their partiest. Then, may well, are he--

    BRUTUS:
    And all rest thou like wanteding diver
    To Eithor they have at the poor mother?

    BENVOLIO:
    'Tis lea
    ==================================================

    Epoch 2, Step 0, Loss: 1.3015
    Epoch 2, Step 500, Loss: 1.1628

    === Generation ====

    YORK:
    Therein ralm'd from effect not him to thee hence!

    GLOUCESTER:

    GLOUCESTER:
    If, read for me.

    KING EDWARD IV:
    Why, Camillo,--

    LUCIO:
    My name rest is will lady and 'gainst,'
    Hath lookest nock my
    ==================================================

    Epoch 3, Step 0, Loss: 1.1508
    Epoch 3, Step 500, Loss: 1.0776

    === Generation ====

    QUEN LADY MONTAGUE:
    So they she's do; and so reports no thousand dear
    When best thou bring the like to Richard's ground:
    Tell his men hardly venom'd,
    Nor one that thou hast a French'd with too,
    Which 
    ==================================================

    Epoch 4, Step 0, Loss: 1.0580
    Epoch 4, Step 500, Loss: 0.9555

    === Generation ====

    Second sad a healter way to France,
    And I the midnight to sore my joy
    Rider hand together: he shall burn the buyer of his eyes,
    Accase his sworn be warrain 'gaged cures:
    And the fault got that ways fa
    ==================================================

    Epoch 5, Step 0, Loss: 0.8348
    Epoch 5, Step 500, Loss: 0.7566

    === Generation ====


    WARWICK:
    Salunes him his charity, with all gross spirit.

    KING LEWIS RICHARD III:
    Faith, for he first it.

    QUEEN MARGARET:
    But thou to joy; lie thee enemy to hito.

    WARWICK:
    Sweet rest his suit will 
    ==================================================

    Epoch 6, Step 0, Loss: 0.6961
    Epoch 6, Step 500, Loss: 0.6624

    === Generation ====

    DUKE VINCENTIO:
    Nor never till death, my lord,
    Live, in my brain, a penitent trade:
    My state, in brief, a very sin,
    And spit fit then by the storm of state,
    Begpara's 'em; and I do think it out of my 
    ==================================================

    Epoch 7, Step 0, Loss: 0.5466
    Epoch 7, Step 500, Loss: 0.5226

    === Generation ====

    MIRANDA:
    Am I like king? I do not long again.

    DERBY:
    Well, well, bring thee to a slave, so evident,
    I will may be well and be returned:
    But wherefore I am lute, all commit me do
    I not know, your fath
    ==================================================

    Epoch 8, Step 0, Loss: 0.4457
    Epoch 8, Step 500, Loss: 0.3864

    === Generation ====

    And give him greatness! mild, ah, a drum, pair:
    And if you were as good as breath, that strike
    Which now the terms of Rome is excell'd
    The mortal, corruption of a devil
    Beguile of his country's bjesic
    ==================================================

    Epoch 9, Step 0, Loss: 0.4395
    Epoch 9, Step 500, Loss: 0.4014

    === Generation ====

    Me, smear'd; she was come with thee
    And with safety odds extecholness,
    To whose flatterers hath clear'd out in some fewlyvorn
    And as when so discover a few blasting foul minds
    traitors? as they are ch
    ==================================================
``` python
loss.item()
```
    0.3808194696903229
``` python
# trying to make smth with user-input. Of course model is not good enough but we have what we have :)
```
``` python
prompt = "tell me about the lord, how kind is he?"
context_indices = torch.tensor([stoi[c] for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(context_indices, max_new_tokens=300)
output_text = decode(generated[0])
print(output_text)
```

::: {.output .stream .stdout}
    tell me about the lord, how kind is he?

    DUKE OF AUNT:
    It may be your countrymen, you shall stiff dear this:
    Then, Petruchio is friending in Marshal:
    Then all believe me, of Juliet.
    The valiant heart is not white here;
    If she plays else you all happiness after byOr this, which, by something approach
    Hath made to if the jewel. Then he had
:::
