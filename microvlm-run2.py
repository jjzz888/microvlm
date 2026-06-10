"""
microvlm-run2.py

A single-file, dependency-free micro Vision-Language Model:

  Value autograd -> GPT decoder -> ViT encoder -> projector -> VLM splice
  -> toy data -> masked cross-entropy -> Adam -> location-only training.

Default run:
    python3 microvlm-run2.py

The default location-only run uses shuffle_r=2, which merges the four image
patches into one image token and usually converges in minutes on this scalar
autograd implementation. Use --shuffle-r 1 for the clearer four-token spatial
version, but expect a longer run.
"""

import argparse
import json
import math
import random
import time


# ============================================================================
# Scalar autograd + shared helpers
# ============================================================================
class Value:
    """Minimal scalar autograd, micrograd/microgpt style."""

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

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
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
    m = max(l.data for l in logits)
    exps = [(l - m).exp() for l in logits]
    s = sum(exps, Value(0.0))
    return [e / s for e in exps]


def randn(*shape, scale=1.0):
    if len(shape) == 1:
        return [Value(random.gauss(0, 1) * scale) for _ in range(shape[0])]
    return [randn(*shape[1:], scale=scale) for _ in range(shape[0])]


def mlp(x, p):
    h = [v.relu() for v in linear(x, p["fc"])]
    return linear(h, p["proj"])


# ============================================================================
# GPT decoder
# ============================================================================
def causal_attention(tokens, p, n_head, cache=None):
    """Causal self-attention. With cache, only new tokens are processed."""
    hd = len(tokens[0]) // n_head
    scale = 1.0 / math.sqrt(hd)
    Kc = cache["K"] if cache else []
    Vc = cache["V"] if cache else []
    base = len(Kc)
    K = Kc + [linear(t, p["Wk"]) for t in tokens]
    V = Vc + [linear(t, p["Wv"]) for t in tokens]
    if cache is not None:
        cache["K"], cache["V"] = K, V

    out = []
    for i, t in enumerate(tokens):
        qi_full = linear(t, p["Wq"])
        pos = base + i
        heads = []
        for h in range(n_head):
            sl = slice(h * hd, (h + 1) * hd)
            qi = qi_full[sl]
            scores = [
                sum((a * b for a, b in zip(qi, K[j][sl])), Value(0.0)) * scale
                for j in range(pos + 1)
            ]
            attn = softmax(scores)
            ctx = [Value(0.0) for _ in range(hd)]
            for j in range(pos + 1):
                ctx = [c + attn[j] * v for c, v in zip(ctx, V[j][sl])]
            heads.extend(ctx)
        out.append(linear(heads, p["Wo"]))
    return out


def decoder_block(tokens, p, n_head, cache=None):
    normed = [rmsnorm(t, p["ln1"]) for t in tokens]
    attended = causal_attention(normed, p, n_head, cache)
    tokens = [[a + b for a, b in zip(t, at)] for t, at in zip(tokens, attended)]
    normed = [rmsnorm(t, p["ln2"]) for t in tokens]
    mlped = [mlp(t, p) for t in normed]
    return [[a + b for a, b in zip(t, m)] for t, m in zip(tokens, mlped)]


def init_lm(cfg):
    D_lm = cfg["D_lm"]
    vocab = cfg["vocab"]
    max_seq = cfg["max_seq"]
    nl = cfg["lm_n_layer"]
    mlp_hidden, s = 4 * D_lm, 0.02
    params = {
        "tok_emb": randn(vocab, D_lm, scale=s),
        "pos_emb": randn(max_seq, D_lm, scale=s),
        "lnf": [Value(1.0) for _ in range(D_lm)],
        "lm_head": randn(vocab, D_lm, scale=s),
        "blocks": [],
    }
    for _ in range(nl):
        params["blocks"].append(
            {
                "ln1": [Value(1.0) for _ in range(D_lm)],
                "ln2": [Value(1.0) for _ in range(D_lm)],
                "Wq": randn(D_lm, D_lm, scale=s),
                "Wk": randn(D_lm, D_lm, scale=s),
                "Wv": randn(D_lm, D_lm, scale=s),
                "Wo": randn(D_lm, D_lm, scale=s),
                "fc": randn(mlp_hidden, D_lm, scale=s),
                "proj": randn(D_lm, mlp_hidden, scale=s),
            }
        )
    return params


# ============================================================================
# ViT encoder: GPT block without the causal mask
# ============================================================================
def patchify(img, cfg):
    """img[c][y][x] -> flat patch vectors, each length C*P*P."""
    C, H, W, P = cfg["C"], cfg["H"], cfg["W"], cfg["P"]
    patches = []
    for gy in range(0, H, P):
        for gx in range(0, W, P):
            vec = []
            for c in range(C):
                for y in range(gy, gy + P):
                    for x in range(gx, gx + P):
                        vec.append(img[c][y][x])
            patches.append(vec)
    return patches


def attention(tokens, p, n_head):
    """Bidirectional self-attention: every patch attends to every patch."""
    N, D = len(tokens), len(tokens[0])
    hd = D // n_head
    scale = 1.0 / math.sqrt(hd)
    Q = [linear(t, p["Wq"]) for t in tokens]
    K = [linear(t, p["Wk"]) for t in tokens]
    V = [linear(t, p["Wv"]) for t in tokens]
    out = []
    for i in range(N):
        heads = []
        for h in range(n_head):
            sl = slice(h * hd, (h + 1) * hd)
            qi = Q[i][sl]
            scores = [
                sum((a * b for a, b in zip(qi, K[j][sl])), Value(0.0)) * scale
                for j in range(N)
            ]
            attn = softmax(scores)
            ctx = [Value(0.0) for _ in range(hd)]
            for j in range(N):
                ctx = [c + attn[j] * v for c, v in zip(ctx, V[j][sl])]
            heads.extend(ctx)
        out.append(linear(heads, p["Wo"]))
    return out


def vit_block(tokens, p, n_head):
    normed = [rmsnorm(t, p["ln1"]) for t in tokens]
    attended = attention(normed, p, n_head)
    tokens = [[a + b for a, b in zip(t, at)] for t, at in zip(tokens, attended)]
    normed = [rmsnorm(t, p["ln2"]) for t in tokens]
    mlped = [mlp(t, p) for t in normed]
    return [[a + b for a, b in zip(t, m)] for t, m in zip(tokens, mlped)]


def vit_forward(img, params, cfg):
    tokens = []
    for i, patch in enumerate(patchify(img, cfg)):
        emb = linear([Value(x) for x in patch], params["patch_emb"])
        emb = [e + pe for e, pe in zip(emb, params["pos_emb"][i])]
        tokens.append(emb)
    for layer in params["blocks"]:
        tokens = vit_block(tokens, layer, cfg["n_head"])
    return [rmsnorm(t, params["lnf"]) for t in tokens]


def init_vit(cfg):
    C, H, W, P = cfg["C"], cfg["H"], cfg["W"], cfg["P"]
    D, nl = cfg["D"], cfg["n_layer"]
    n_patches = (H // P) * (W // P)
    patch_in = C * P * P
    mlp_hidden = 4 * D
    s = 0.02
    params = {
        "patch_emb": randn(D, patch_in, scale=s),
        "pos_emb": randn(n_patches, D, scale=s),
        "lnf": [Value(1.0) for _ in range(D)],
        "blocks": [],
    }
    for _ in range(nl):
        params["blocks"].append(
            {
                "ln1": [Value(1.0) for _ in range(D)],
                "ln2": [Value(1.0) for _ in range(D)],
                "Wq": randn(D, D, scale=s),
                "Wk": randn(D, D, scale=s),
                "Wv": randn(D, D, scale=s),
                "Wo": randn(D, D, scale=s),
                "fc": randn(mlp_hidden, D, scale=s),
                "proj": randn(D, mlp_hidden, scale=s),
            }
        )
    return params


# ============================================================================
# Projector: optional pixel shuffle -> MLP into LM embedding space
# ============================================================================
def pixel_shuffle(tokens, grid_w, r):
    """Merge each r*r block of neighboring patch tokens into one wider token."""
    N, D = len(tokens), len(tokens[0])
    grid_h = N // grid_w
    out = []
    for by in range(0, grid_h, r):
        for bx in range(0, grid_w, r):
            merged = []
            for dy in range(r):
                for dx in range(r):
                    merged.extend(tokens[(by + dy) * grid_w + (bx + dx)])
            out.append(merged)
    return out


def project(tokens, params):
    out = []
    for t in tokens:
        h = [v.relu() for v in linear(t, params["fc1"])]
        out.append(linear(h, params["fc2"]))
    return out


def vision_to_language(patch_tokens, params, cfg):
    toks = patch_tokens
    if cfg.get("shuffle_r", 1) > 1:
        toks = pixel_shuffle(toks, cfg["grid_w"], cfg["shuffle_r"])
    return project(toks, params)


def init_projector(cfg):
    r = cfg.get("shuffle_r", 1)
    in_dim = cfg["D"] * r * r
    hidden, D_lm = cfg["proj_hidden"], cfg["D_lm"]
    s = 0.02
    return {
        "fc1": randn(hidden, in_dim, scale=s),
        "fc2": randn(D_lm, hidden, scale=s),
    }


# ============================================================================
# Full VLM forward pass
# ============================================================================
def splice(text_embeds, image_tokens, image_slots):
    out = list(text_embeds)
    for slot, img in zip(image_slots, image_tokens):
        out[slot] = img
    return out


def vlm_forward(img, input_ids, image_slots, vparams, pparams, lparams, cfg):
    text_embeds = [[v for v in lparams["tok_emb"][tid]] for tid in input_ids]
    patch_tokens = vit_forward(img, vparams, cfg)
    image_tokens = vision_to_language(patch_tokens, pparams, cfg)
    stream = splice(text_embeds, image_tokens, image_slots)
    stream = [
        [e + pe for e, pe in zip(tok, lparams["pos_emb"][i])]
        for i, tok in enumerate(stream)
    ]
    for layer in lparams["blocks"]:
        stream = decoder_block(stream, layer, cfg["lm_n_head"])
    stream = [rmsnorm(t, lparams["lnf"]) for t in stream]
    return [linear(t, lparams["lm_head"]) for t in stream]


# ============================================================================
# Toy data
# ============================================================================
COLORS = ["red", "green", "blue", "yellow"]
RGB = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}
VROWS = ["top", "bottom"]
HCOLS = ["left", "right"]

VOCAB = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<image>",
    "red",
    "green",
    "blue",
    "yellow",
    "top",
    "bottom",
    "left",
    "right",
]
stoi = {w: i for i, w in enumerate(VOCAB)}
itos = {i: w for w, i in stoi.items()}
PAD, BOS, EOS, IMG = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"], stoi["<image>"]


def num_image_tokens(cfg):
    n = (cfg["H"] // cfg["P"]) * (cfg["W"] // cfg["P"])
    r = cfg.get("shuffle_r", 1)
    return n // (r * r)


def make_image(color, vrow, hcol, cfg, noise=0.0):
    C, H, W = cfg["C"], cfg["H"], cfg["W"]
    qh, qw = H // 2, W // 2
    img = [[[0.0] * W for _ in range(H)] for _ in range(C)]
    gy = 0 if vrow == "top" else qh
    gx = 0 if hcol == "left" else qw
    rgb = RGB[color]
    for c in range(C):
        for y in range(gy, gy + qh):
            for x in range(gx, gx + qw):
                img[c][y][x] = rgb[c]
    if noise:
        for c in range(C):
            for y in range(H):
                for x in range(W):
                    img[c][y][x] = min(
                        1.0,
                        max(0.0, img[c][y][x] + random.uniform(-noise, noise)),
                    )
    return img


def make_example(cfg, noise=0.0, color=None, vrow=None, hcol=None):
    color = color or random.choice(COLORS)
    vrow = vrow or random.choice(VROWS)
    hcol = hcol or random.choice(HCOLS)
    img = make_image(color, vrow, hcol, cfg, noise)
    k = num_image_tokens(cfg)
    answer = [stoi[color], stoi[vrow], stoi[hcol]]
    input_ids = [BOS] + [IMG] * k + answer + [EOS]
    image_slots = list(range(1, 1 + k))
    answer_start = 1 + k
    targets = [PAD] * len(input_ids)
    loss_mask = [0] * len(input_ids)
    for i in range(answer_start - 1, len(input_ids) - 1):
        targets[i] = input_ids[i + 1]
        loss_mask[i] = 1
    return {
        "img": img,
        "input_ids": input_ids,
        "image_slots": image_slots,
        "targets": targets,
        "loss_mask": loss_mask,
        "label": f"{color} {vrow} {hcol}",
    }


def all_combinations(cfg, noise=0.0):
    return [
        make_example(cfg, noise, color, vrow, hcol)
        for color in COLORS
        for vrow in VROWS
        for hcol in HCOLS
    ]


def decode(ids):
    return " ".join(itos[i] for i in ids if i not in (PAD, BOS, EOS, IMG))


def make_location_example(cfg, color, vrow, hcol):
    """Location-only target: color varies but answer is just row/column."""
    img = make_image(color, vrow, hcol, cfg)
    k = num_image_tokens(cfg)
    answer = [stoi[vrow], stoi[hcol]]
    ids = [BOS] + [IMG] * k + answer + [EOS]
    targets = [PAD] * len(ids)
    loss_mask = [0] * len(ids)
    for i in range(k, len(ids) - 1):
        targets[i] = ids[i + 1]
        loss_mask[i] = 1
    return {
        "img": img,
        "input_ids": ids,
        "image_slots": list(range(1, 1 + k)),
        "targets": targets,
        "loss_mask": loss_mask,
        "label": f"{vrow} {hcol}",
        "color": color,
    }


def location_dataset(cfg):
    return [
        make_location_example(cfg, color, vrow, hcol)
        for color in COLORS
        for vrow in VROWS
        for hcol in HCOLS
    ]


# ============================================================================
# Training helpers
# ============================================================================
def cross_entropy(logits, target):
    m = max(l.data for l in logits)
    s = sum(((l - m).exp() for l in logits), Value(0.0))
    return (s.log() + m) - logits[target]


def collect_params(*trees):
    out = []

    def walk(p):
        if isinstance(p, Value):
            out.append(p)
        elif isinstance(p, list):
            for x in p:
                walk(x)
        elif isinstance(p, dict):
            for x in p.values():
                walk(x)

    for tree in trees:
        walk(tree)
    return out


def argmax(logits):
    return max(range(len(logits)), key=lambda i: logits[i].data)


class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (vhat ** 0.5 + self.eps)


def teacher_forced_acc(dataset, vp, pp, lp, cfg):
    correct = 0
    for ex in dataset:
        logits = vlm_forward(ex["img"], ex["input_ids"], ex["image_slots"], vp, pp, lp, cfg)
        ok = all(
            argmax(logits[i]) == ex["targets"][i]
            for i, keep in enumerate(ex["loss_mask"])
            if keep
        )
        correct += int(ok)
    return correct / len(dataset)


def greedy_location(ex, vp, pp, lp, cfg, steps=2):
    k = num_image_tokens(cfg)
    ids = [BOS] + [IMG] * k
    for _ in range(steps):
        logits = vlm_forward(ex["img"], ids, list(range(1, 1 + k)), vp, pp, lp, cfg)
        ids.append(argmax(logits[-1]))
    return decode(ids)


def train_location(args):
    random.seed(args.seed)
    cfg = {
        "C": 3,
        "H": 8,
        "W": 8,
        "P": 4,
        "D": args.d,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "grid_w": 2,
        "shuffle_r": args.shuffle_r,
        "proj_hidden": args.proj_hidden,
        "D_lm": args.d_lm,
        "lm_n_head": args.lm_n_head,
        "lm_n_layer": args.lm_n_layer,
        "vocab": len(VOCAB),
        "max_seq": 16,
    }
    dataset = location_dataset(cfg)
    k = num_image_tokens(cfg)
    vp, pp, lp = init_vit(cfg), init_projector(cfg), init_lm(cfg)
    params = collect_params(vp, pp, lp)
    opt = Adam(params)

    print(
        f"params={len(params)} examples={len(dataset)} k={k} "
        f"seq_len={len(dataset[0]['input_ids'])} shuffle_r={args.shuffle_r}",
        flush=True,
    )

    hist_loss, hist_acc = [], []
    t0 = time.time()
    converged = False
    for epoch in range(1, args.epochs + 1):
        opt.lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)
        )
        random.shuffle(dataset)
        total = 0.0
        for ex in dataset:
            logits = vlm_forward(ex["img"], ex["input_ids"], ex["image_slots"], vp, pp, lp, cfg)
            loss, n = Value(0.0), 0
            for i, keep in enumerate(ex["loss_mask"]):
                if keep:
                    loss = loss + cross_entropy(logits[i], ex["targets"][i])
                    n += 1
            loss = loss * (1.0 / n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.data
        avg_loss = total / len(dataset)
        hist_loss.append(avg_loss)

        do_eval = epoch <= 3 or epoch % args.eval_every == 0
        if do_eval:
            acc = teacher_forced_acc(dataset, vp, pp, lp, cfg)
            hist_acc.append([epoch, acc])
            print(
                f"epoch {epoch:3d} lr={opt.lr:.4f} loss={avg_loss:.4f} "
                f"acc={acc:.2f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            if acc == 1.0:
                print("CONVERGED.", flush=True)
                converged = True
                break
        elif args.print_every_epoch:
            print(
                f"epoch {epoch:3d} lr={opt.lr:.4f} loss={avg_loss:.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    print("\ngreedy predictions (real autoregressive decode):", flush=True)
    preds = []
    for ex in sorted(dataset, key=lambda e: (e["label"], e["color"])):
        pred = greedy_location(ex, vp, pp, lp, cfg)
        ok = pred == ex["label"]
        preds.append({"color": ex["color"], "label": ex["label"], "pred": pred, "ok": ok})
        suffix = "" if ok else "   WRONG"
        print(f"  {ex['color']:<7} {ex['label']:<14} -> {pred}{suffix}", flush=True)

    final_acc = teacher_forced_acc(dataset, vp, pp, lp, cfg)
    history = {
        "config": cfg,
        "loss": hist_loss,
        "acc": hist_acc,
        "preds": preds,
        "seconds": round(time.time() - t0, 1),
        "final_acc": final_acc,
        "converged": converged,
    }
    if args.history:
        with open(args.history, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nsaved history to {args.history}", flush=True)
    print(f"final_acc={final_acc:.2f}; total={time.time() - t0:.0f}s", flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description="Train the all-in-one microvlm toy model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--print-every-epoch", action="store_true")
    parser.add_argument("--shuffle-r", type=int, default=2, choices=[1, 2])
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--d-lm", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--lm-n-head", type=int, default=2)
    parser.add_argument("--lm-n-layer", type=int, default=1)
    parser.add_argument("--proj-hidden", type=int, default=16)
    parser.add_argument("--lr-max", type=float, default=0.02)
    parser.add_argument("--lr-min", type=float, default=0.002)
    parser.add_argument("--history", default="loc_history_all_in_one.json")
    args = parser.parse_args()
    train_location(args)


if __name__ == "__main__":
    main()
