#!/usr/bin/env python3
"""microvlm-run1.py

Generated from the Python code blocks in microvlm.md.
Runs the location-only toy training demo and greedy generation examples.
"""

# ---- note code block 1 ----
COLORS = ['red', 'green', 'blue', 'yellow']
RGB = {'red': (1.0, 0.0, 0.0), 'green': (0.0, 1.0, 0.0),
       'blue': (0.0, 0.0, 1.0), 'yellow': (1.0, 1.0, 0.0)}
VROWS, HCOLS = ['top', 'bottom'], ['left', 'right']

VOCAB = ['<pad>', '<bos>', '<eos>', '<image>',
         'red', 'green', 'blue', 'yellow', 'top', 'bottom', 'left', 'right']
stoi = {w: i for i, w in enumerate(VOCAB)}
itos = {i: w for w, i in stoi.items()}
PAD, BOS, EOS, IMG = stoi['<pad>'], stoi['<bos>'], stoi['<eos>'], stoi['<image>']

def num_image_tokens(cfg):
    """#image tokens the projector emits = #<image> slots to reserve."""
    n = (cfg['H'] // cfg['P']) * (cfg['W'] // cfg['P'])     # patch count
    return n // (cfg.get('shuffle_r', 1) ** 2)             # pixel shuffle shrinks by r²

def make_image(color, vrow, hcol, cfg, noise=0.0):
    """8×8 RGB: one quadrant filled with `color`, the rest black."""
    C, H, W = cfg['C'], cfg['H'], cfg['W']
    qh, qw = H // 2, W // 2
    img = [[[0.0] * W for _ in range(H)] for _ in range(C)]
    gy = 0 if vrow == 'top' else qh
    gx = 0 if hcol == 'left' else qw
    for c in range(C):
        for y in range(gy, gy + qh):
            for x in range(gx, gx + qw):
                img[c][y][x] = RGB[color][c]
    if noise:
        for c in range(C):
            for y in range(H):
                for x in range(W):
                    img[c][y][x] = min(1.0, max(0.0, img[c][y][x] + random.uniform(-noise, noise)))
    return img

def make_example(cfg, noise=0.0, color=None, vrow=None, hcol=None):
    color = color or random.choice(COLORS)
    vrow  = vrow  or random.choice(VROWS)
    hcol  = hcol  or random.choice(HCOLS)
    img   = make_image(color, vrow, hcol, cfg, noise)
    k = num_image_tokens(cfg)
    answer = [stoi[color], stoi[vrow], stoi[hcol]]
    input_ids   = [BOS] + [IMG] * k + answer + [EOS]
    image_slots = list(range(1, 1 + k))
    targets   = [PAD] * len(input_ids)                     # next-token targets
    loss_mask = [0]   * len(input_ids)                     # 1 where supervised
    for i in range(k, len(input_ids) - 1):                 # from last <image> onward
        targets[i], loss_mask[i] = input_ids[i + 1], 1
    return dict(img=img, input_ids=input_ids, image_slots=image_slots,
                targets=targets, loss_mask=loss_mask, label=f"{color} {vrow} {hcol}")

def make_batch(n, cfg, noise=0.0):
    return [make_example(cfg, noise) for _ in range(n)]

def all_combinations(cfg, noise=0.0):                       # the 16 fixed combos
    return [make_example(cfg, noise, c, v, h)
            for c in COLORS for v in VROWS for h in HCOLS]

def decode(ids):                                           # ids -> caption string
    return ' '.join(itos[i] for i in ids if i not in (PAD, BOS, EOS, IMG))

# ---- note code block 2 ----
def patchify(img, cfg):
    """img[c][y][x] -> list of flat patch vectors, each len C*P*P."""
    C, H, W, P = cfg['C'], cfg['H'], cfg['W'], cfg['P']
    patches = []
    for gy in range(0, H, P):
        for gx in range(0, W, P):
            vec = [img[c][y][x]
                   for c in range(C)
                   for y in range(gy, gy + P)
                   for x in range(gx, gx + P)]
            patches.append(vec)
    return patches

# ---- note code block 3 ----
import math, random

class Value:
    """Minimal scalar autograd — identical idea to micrograd/microgpt."""
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,))
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        out = Value(math.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    def log(self):
        out = Value(math.log(self.data), (self,))
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    def __neg__(self):        return self * -1
    def __sub__(self, o):     return self + (-o)
    def __rsub__(self, o):    return o + (-self)
    def __radd__(self, o):    return self + o
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * o ** -1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

def linear(x, w):
    """w: out_dim rows of in_dim weights; x: in_dim vector -> out_dim vector."""
    return [sum((wij * xj for wij, xj in zip(row, x)), Value(0.0)) for row in w]

def rmsnorm(x, g):
    n = len(x)
    ms = sum((xi * xi for xi in x), Value(0.0)) * (1.0 / n)
    inv = (ms + 1e-5) ** -0.5
    return [xi * inv * gi for xi, gi in zip(x, g)]

def softmax(logits):
    m = max(l.data for l in logits)               # for numerical stability
    exps = [(l - m).exp() for l in logits]
    s = sum(exps, Value(0.0))
    return [e / s for e in exps]

def randn(*shape, scale=1.0):
    if len(shape) == 1:
        return [Value(random.gauss(0, 1) * scale) for _ in range(shape[0])]
    return [randn(*shape[1:], scale=scale) for _ in range(shape[0])]

def mlp(x, p):
    h = [v.relu() for v in linear(x, p['fc'])]   # D -> 4D, ReLU
    return linear(h, p['proj'])                  # 4D -> D

# ---- note code block 4 ----
def causal_attention(tokens, p, n_head, cache=None):
    """CAUSAL (masked) self-attention: position i attends only to keys 0..i.
    Pass a KV `cache` to decode incrementally — then `tokens` is just the NEW
    token(s) and past keys/values are reused from the cache, not recomputed.
    With cache=None it processes the whole sequence (training) unchanged."""
    hd = len(tokens[0]) // n_head
    scale = 1.0 / math.sqrt(hd)
    Kc = cache['K'] if cache else []                      # keys cached from earlier tokens
    Vc = cache['V'] if cache else []                      # values cached from earlier tokens
    base = len(Kc)                                        # how many tokens are already cached
    K = Kc + [linear(t, p['Wk']) for t in tokens]        # past (reused) + new keys
    V = Vc + [linear(t, p['Wv']) for t in tokens]        # past (reused) + new values
    if cache is not None:
        cache['K'], cache['V'] = K, V                    # remember for the next step
    out = []
    for i, t in enumerate(tokens):
        qi_full = linear(t, p['Wq'])
        pos = base + i                                   # absolute position of this query
        heads = []
        for h in range(n_head):
            sl = slice(h * hd, (h + 1) * hd)
            qi = qi_full[sl]
            scores = [sum((a * b for a, b in zip(qi, K[j][sl])), Value(0.0)) * scale
                      for j in range(pos + 1)]            # attend to keys 0..pos
            attn = softmax(scores)
            ctx = [Value(0.0) for _ in range(hd)]
            for j in range(pos + 1):
                ctx = [c + attn[j] * v for c, v in zip(ctx, V[j][sl])]
            heads.extend(ctx)
        out.append(linear(heads, p['Wo']))
    return out

def decoder_block(tokens, p, n_head, cache=None):
    normed   = [rmsnorm(t, p['ln1']) for t in tokens]
    attended = causal_attention(normed, p, n_head, cache)      # cache threaded through for decoding
    tokens   = [[a + b for a, b in zip(t, at)] for t, at in zip(tokens, attended)]
    normed   = [rmsnorm(t, p['ln2']) for t in tokens]
    mlped    = [mlp(t, p) for t in normed]
    return [[a + b for a, b in zip(t, m)] for t, m in zip(tokens, mlped)]

def gpt_forward(input_ids, lparams, cfg):
    """Text-only GPT: token ids -> per-position next-token logits (no cache)."""
    stream = [[v for v in lparams['tok_emb'][tid]] for tid in input_ids]   # ids -> D_lm
    stream = [[e + pe for e, pe in zip(tok, lparams['pos_emb'][i])]
              for i, tok in enumerate(stream)]                             # + positions
    for layer in lparams['blocks']:
        stream = decoder_block(stream, layer, cfg['lm_n_head'])
    stream = [rmsnorm(t, lparams['lnf']) for t in stream]
    return [linear(t, lparams['lm_head']) for t in stream]                 # per-position logits

def gpt_generate(prompt_ids, lparams, cfg, max_new=20, eos=None):
    """Greedy generation with a KV cache: prefill the prompt once, then emit one token
    at a time — each step runs ONLY the new token through the decoder, reusing every
    earlier token's cached keys/values (O(seq) work per token instead of O(seq²))."""
    caches = [{'K': [], 'V': []} for _ in lparams['blocks']]   # one KV cache per layer

    def step(chunk_ids, start):                                # run ids at positions start, start+1, ...
        stream = [[v for v in lparams['tok_emb'][tid]] for tid in chunk_ids]
        stream = [[e + pe for e, pe in zip(tok, lparams['pos_emb'][start + i])]
                  for i, tok in enumerate(stream)]
        for layer, cache in zip(lparams['blocks'], caches):
            stream = decoder_block(stream, layer, cfg['lm_n_head'], cache)
        return linear(rmsnorm(stream[-1], lparams['lnf']), lparams['lm_head'])   # logits, last pos

    ids = list(prompt_ids)
    logits = step(ids, 0)                                      # PREFILL: whole prompt, fills the caches
    for _ in range(max_new):
        nxt = max(range(len(logits)), key=lambda j: logits[j].data)
        ids.append(nxt)
        if nxt == eos:
            break
        logits = step([nxt], len(ids) - 1)                    # DECODE: only the new token
    return ids

def init_lm(cfg):
    D_lm, vocab, max_seq, nl = cfg['D_lm'], cfg['vocab'], cfg['max_seq'], cfg['lm_n_layer']
    mlp_hidden, s = 4 * D_lm, 0.02
    params = {
        'tok_emb':  randn(vocab, D_lm, scale=s),    # token embedding table
        'pos_emb':  randn(max_seq, D_lm, scale=s),  # decoder positions
        'lnf':      [Value(1.0) for _ in range(D_lm)],
        'lm_head':  randn(vocab, D_lm, scale=s),    # hidden -> vocab logits
        'blocks':   [],
    }
    for _ in range(nl):
        params['blocks'].append({
            'ln1': [Value(1.0) for _ in range(D_lm)], 'ln2': [Value(1.0) for _ in range(D_lm)],
            'Wq': randn(D_lm, D_lm, scale=s), 'Wk': randn(D_lm, D_lm, scale=s),
            'Wv': randn(D_lm, D_lm, scale=s), 'Wo': randn(D_lm, D_lm, scale=s),
            'fc': randn(mlp_hidden, D_lm, scale=s), 'proj': randn(D_lm, mlp_hidden, scale=s),
        })
    return params

# ---- note code block 5 ----
def patchify(img, cfg):
    """img[c][y][x] -> list of flat patch vectors, each len C*P*P."""
    C, H, W, P = cfg['C'], cfg['H'], cfg['W'], cfg['P']
    patches = []
    for gy in range(0, H, P):
        for gx in range(0, W, P):
            vec = [img[c][y][x]
                   for c in range(C)
                   for y in range(gy, gy + P)
                   for x in range(gx, gx + P)]
            patches.append(vec)
    return patches

def attention(tokens, p, n_head):
    """Full BIDIRECTIONAL self-attention — the only change vs a GPT block."""
    N, D = len(tokens), len(tokens[0])
    hd, scale = D // n_head, 1.0 / math.sqrt(D // n_head)
    Q = [linear(t, p['Wq']) for t in tokens]
    K = [linear(t, p['Wk']) for t in tokens]
    V = [linear(t, p['Wv']) for t in tokens]
    out = []
    for i in range(N):
        heads = []
        for h in range(n_head):
            sl = slice(h * hd, (h + 1) * hd)
            qi = Q[i][sl]
            scores = [sum((a*b for a, b in zip(qi, K[j][sl])), Value(0.0)) * scale
                      for j in range(N)]          # attend to EVERY patch (no mask)
            attn = softmax(scores)
            ctx = [Value(0.0) for _ in range(hd)]
            for j in range(N):
                ctx = [c + attn[j] * v for c, v in zip(ctx, V[j][sl])]
            heads.extend(ctx)
        out.append(linear(heads, p['Wo']))
    return out

def block(tokens, p, n_head):
    normed   = [rmsnorm(t, p['ln1']) for t in tokens]
    attended = attention(normed, p, n_head)
    tokens   = [[a + b for a, b in zip(t, at)] for t, at in zip(tokens, attended)]
    normed   = [rmsnorm(t, p['ln2']) for t in tokens]
    mlped    = [mlp(t, p) for t in normed]
    return [[a + b for a, b in zip(t, m)] for t, m in zip(tokens, mlped)]

def vit_forward(img, params, cfg):
    tokens = []
    for i, patch in enumerate(patchify(img, cfg)):
        emb = linear([Value(x) for x in patch], params['patch_emb'])  # patch -> D
        emb = [e + pe for e, pe in zip(emb, params['pos_emb'][i])]    # + position
        tokens.append(emb)
    for layer in params['blocks']:
        tokens = block(tokens, layer, cfg['n_head'])
    return [rmsnorm(t, params['lnf']) for t in tokens]                # patch tokens

def init_vit(cfg):
    C, H, W, P = cfg['C'], cfg['H'], cfg['W'], cfg['P']
    D, nl = cfg['D'], cfg['n_layer']
    n_patches, patch_in, mlp_hidden = (H // P) * (W // P), C * P * P, 4 * D
    s = 0.02
    params = {
        'patch_emb': randn(D, patch_in, scale=s),   # C*P*P -> D
        'pos_emb':   randn(n_patches, D, scale=s),   # one vector per patch slot
        'lnf':       [Value(1.0) for _ in range(D)],
        'blocks':    [],
    }
    for _ in range(nl):
        params['blocks'].append({
            'ln1': [Value(1.0) for _ in range(D)], 'ln2': [Value(1.0) for _ in range(D)],
            'Wq': randn(D, D, scale=s), 'Wk': randn(D, D, scale=s),
            'Wv': randn(D, D, scale=s), 'Wo': randn(D, D, scale=s),
            'fc': randn(mlp_hidden, D, scale=s), 'proj': randn(D, mlp_hidden, scale=s),
        })
    return params

# ---- note code block 6 ----
def pixel_shuffle(tokens, grid_w, r):
    """Merge each r×r block of neighboring patch tokens into one wider token.
    tokens: N vectors (dim D), row-major on a (grid_h × grid_w) grid.
    Returns N/r² tokens, each of dim D·r²  (token count shrinks, width grows)."""
    N, D = len(tokens), len(tokens[0])
    grid_h = N // grid_w
    out = []
    for by in range(0, grid_h, r):
        for bx in range(0, grid_w, r):
            merged = []
            for dy in range(r):
                for dx in range(r):
                    merged.extend(tokens[(by + dy) * grid_w + (bx + dx)])  # concat channels
            out.append(merged)                                             # length D·r²
    return out

def project(tokens, params):
    """Each vision token (dim D or D·r²) -> one LM embedding (dim D_lm).
    A 2-layer MLP (LLaVA-style)."""
    out = []
    for t in tokens:
        h = [v.relu() for v in linear(t, params['fc1'])]   # D·r² -> hidden
        out.append(linear(h, params['fc2']))               # hidden -> D_lm
    return out

def vision_to_language(patch_tokens, params, cfg):
    """ViT patch tokens -> tokens ready to splice into the LM stream."""
    toks = patch_tokens
    if cfg.get('shuffle_r', 1) > 1:
        toks = pixel_shuffle(toks, cfg['grid_w'], cfg['shuffle_r'])
    return project(toks, params)

def init_projector(cfg):
    r = cfg.get('shuffle_r', 1)
    in_dim = cfg['D'] * r * r          # width after pixel shuffle
    hidden, D_lm = cfg['proj_hidden'], cfg['D_lm']
    s = 0.02
    return {
        'fc1': randn(hidden, in_dim, scale=s),
        'fc2': randn(D_lm, hidden, scale=s),
    }

# ---- note code block 7 ----
def splice(text_embeds, image_tokens, image_slots):
    """Overwrite placeholder positions in the text stream with image tokens.
    text_embeds: list of D_lm vectors (the embedded prompt)
    image_slots: indices of the <image> placeholders, in order."""
    out = list(text_embeds)
    for slot, img in zip(image_slots, image_tokens):
        out[slot] = img
    return out

def vlm_forward(img, input_ids, image_slots, vparams, pparams, lparams, cfg):
    text_embeds  = [[v for v in lparams['tok_emb'][tid]] for tid in input_ids]  # 1. ids -> D_lm
    patch_tokens = vit_forward(img, vparams, cfg)                               # 2. image -> patches
    image_tokens = vision_to_language(patch_tokens, pparams, cfg)               # 3. -> D_lm
    stream = splice(text_embeds, image_tokens, image_slots)                     # 4. fill <image> slots
    stream = [[e + pe for e, pe in zip(tok, lparams['pos_emb'][i])]
              for i, tok in enumerate(stream)]                                  # 5. + positions
    for layer in lparams['blocks']:
        stream = decoder_block(stream, layer, cfg['lm_n_head'])                 # 6. causal decoder
    stream = [rmsnorm(t, lparams['lnf']) for t in stream]
    return [linear(t, lparams['lm_head']) for t in stream]                      # 7. per-position logits

# ---- note code block 8 ----
cfg = dict(C=3, H=8, W=8, P=4, D=16, n_head=4, n_layer=2,        # vision
           grid_w=2, shuffle_r=2, proj_hidden=24,                # projector
           D_lm=32, lm_n_head=4, lm_n_layer=2, vocab=50, max_seq=16)  # language
vparams, pparams, lparams = init_vit(cfg), init_projector(cfg), init_lm(cfg)
img = [[[random.random() for _ in range(cfg['W'])] for _ in range(cfg['H'])]
       for _ in range(cfg['C'])]
input_ids   = [BOS, IMG, EOS]   # one <image> placeholder
image_slots = [1]               # shuffle_r=2 on a 2x2 grid -> 1 image token -> 1 slot
logits = vlm_forward(img, input_ids, image_slots, vparams, pparams, lparams, cfg)
# logits: 3 positions x 50 vocab; loss.backward() reaches patch_emb, fc1, tok_emb, lm_head

# ---- note code block 9 ----
def cross_entropy(logits, target):
    return -softmax(logits)[target].log()                # -log P(correct token)

# ---- note code block 10 ----
class Adam:
    """Minimal Adam over a flat list of Value parameters."""
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [0.0] * len(params); self.v = [0.0] * len(params); self.t = 0
    def zero_grad(self):
        for p in self.params: p.grad = 0.0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (vhat ** 0.5 + self.eps)

def collect_params(*trees):
    """Gather every Value leaf from the encoder / projector / LM parameter trees."""
    out = []
    def walk(p):
        if isinstance(p, Value):  out.append(p)
        elif isinstance(p, list): [walk(x) for x in p]
        elif isinstance(p, dict): [walk(x) for x in p.values()]
    for t in trees: walk(t)
    return out

# ---- note code block 11 ----
def make_location_example(cfg, color, vrow, hcol):
    """One example whose target is the quadrant only — no color word."""
    img = make_image(color, vrow, hcol, cfg)
    k = num_image_tokens(cfg)
    answer = [stoi[vrow], stoi[hcol]]                          # location only
    ids = [BOS] + [IMG] * k + answer + [EOS]
    targets, mask = [PAD] * len(ids), [0] * len(ids)
    for i in range(k, len(ids) - 1):                           # supervise the answer + EOS
        targets[i], mask[i] = ids[i + 1], 1
    return dict(img=img, input_ids=ids, image_slots=list(range(1, 1 + k)),
                targets=targets, loss_mask=mask, label=f"{vrow} {hcol}")

def argmax(xs):
    return max(range(len(xs)), key=lambda j: xs[j].data)

random.seed(0)
cfg = dict(C=3, H=8, W=8, P=4, D=8, n_head=2, n_layer=1,
           grid_w=2, shuffle_r=2, proj_hidden=16,
           D_lm=16, lm_n_head=2, lm_n_layer=1, vocab=12, max_seq=16)
dataset = [make_location_example(cfg, c, v, h) for c in COLORS for v in VROWS for h in HCOLS]
vp, pp, lp = init_vit(cfg), init_projector(cfg), init_lm(cfg)
opt = Adam(collect_params(vp, pp, lp))

def accuracy():                                                # teacher-forced exact match on all 16
    hit = 0
    for ex in dataset:
        lg = vlm_forward(ex['img'], ex['input_ids'], ex['image_slots'], vp, pp, lp, cfg)
        hit += all(argmax(lg[i]) == ex['targets'][i]
                   for i, keep in enumerate(ex['loss_mask']) if keep)
    return hit / len(dataset)

EPOCHS = 40
for epoch in range(1, EPOCHS + 1):
    opt.lr = 0.002 + 0.5 * (0.02 - 0.002) * (1 + math.cos(math.pi * epoch / EPOCHS))  # cosine 0.02->0.002
    random.shuffle(dataset)
    total = 0.0
    for ex in dataset:
        logits = vlm_forward(ex['img'], ex['input_ids'], ex['image_slots'], vp, pp, lp, cfg)
        loss, n = Value(0.0), 0
        for i, keep in enumerate(ex['loss_mask']):             # masked cross-entropy
            if keep:
                loss = loss + cross_entropy(logits[i], ex['targets'][i]); n += 1
        loss = loss * (1.0 / n)                                # mean over the answer tokens
        opt.zero_grad(); loss.backward(); opt.step()           # one backward -> all three modules
        total += loss.data
    if epoch % 5 == 0:
        acc = accuracy()
        print(f"epoch {epoch:3d}  loss {total/len(dataset):.4f}  acc {acc:.2f}")
        if acc == 1.0:
            break

# ---- note code block 12 ----
def generate(img, vparams, pparams, lparams, cfg, max_new=8):
    """Greedy decode with a KV cache: encode the image ONCE, prefill [BOS, <image>×k],
    then emit one token at a time, reusing the decoder's cached keys/values."""
    k = num_image_tokens(cfg)
    image_tokens = vision_to_language(vit_forward(img, vparams, cfg), pparams, cfg)  # encode once
    caches = [{'K': [], 'V': []} for _ in lparams['blocks']]               # KV cache per layer

    def step(vectors, start):                            # decode `vectors` at positions start, start+1...
        s = [[e + pe for e, pe in zip(v, lparams['pos_emb'][start + i])]
             for i, v in enumerate(vectors)]
        for layer, cache in zip(lparams['blocks'], caches):
            s = decoder_block(s, layer, cfg['lm_n_head'], cache)
        return linear(rmsnorm(s[-1], lparams['lnf']), lparams['lm_head'])  # logits, last position

    bos = [v for v in lparams['tok_emb'][BOS]]
    logits = step([bos] + image_tokens, 0)               # PREFILL: BOS + image tokens (positions 0..k)
    ids = [BOS] + [IMG] * k
    for _ in range(max_new):
        nxt = max(range(len(logits)), key=lambda j: logits[j].data)
        if nxt == EOS:
            break
        ids.append(nxt)
        logits = step([[v for v in lparams['tok_emb'][nxt]]], len(ids) - 1)  # DECODE: only the new token
    return decode(ids)                                    # drop special tokens -> string

# ---- note code block 13 ----
for color, vrow, hcol in [('red', 'top', 'left'), ('blue', 'bottom', 'right'), ('yellow', 'top', 'right')]:
    answer = generate(make_image(color, vrow, hcol, cfg), vp, pp, lp, cfg)
    print(f"input: {color} block, {vrow} {hcol}  ->  model: {answer!r}")

# ---- note code block 14 ----
def cross_entropy(logits, target):
    m = max(l.data for l in logits)                  # shift by the max for stability
    s = sum(((l - m).exp() for l in logits), Value(0.0))
    return (s.log() + m) - logits[target]            # logsumexp(logits) - logit[target]
