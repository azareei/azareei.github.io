---
layout: post
title: Vision Transformers
date: 2026-02-24 10:00:00-0400
description: A comprehensive deep dive into ViT, DeiT, BEiT, MAE, DINO, Swin, DETR, SAM, CLIP, DiT, and beyond — with PyTorch code, training tips, and practical intuitions.
toc:
  beginning: true
---

The transformer took NLP from a fragmented landscape of task-specific architectures to a single unified paradigm. Vision followed the same path — later, and with more friction. This post covers everything an AI researcher needs to know about vision transformers: the core architecture, training paradigms, key variants, working PyTorch code, and the open questions that remain.

---

## 1. Why Transformers for Vision?

For a decade, convolutional neural networks were the uncontested backbone of computer vision. They work because images have exploitable structure: nearby pixels are correlated, and that structure is approximately shift-invariant. CNNs bake both assumptions into the architecture:

- **Locality**: each filter operates on a small spatial neighborhood
- **Weight sharing**: the same filter is applied across all positions (translation equivariance)
- **Hierarchy**: stacked layers build up receptive field and abstraction progressively

These inductive biases are a double-edged sword. They make CNNs extremely data-efficient — a ResNet-50 trains well on 1M images. But they also constrain the model. Long-range spatial dependencies require many stacked layers. The architecture assumes the statistics of natural images in ways that may not transfer to other domains (satellite imagery, medical scans, point clouds, video).

The transformer has none of these biases. Every token attends to every other token from the very first layer. That is a liability when data is scarce — the model must learn spatial structure entirely from data. But when data is abundant, it is a strength: the model can discover whatever spatial relationships actually exist, including non-local ones that CNNs would need dozens of layers to capture.

The other driver is **unification**. By 2020, language models were pretrained as giant transformers and fine-tuned everywhere. If vision used the same architecture, the same pretraining infrastructure, the same scaling playbook — the ecosystem simplification would be enormous. That bet has paid off.

---

## 2. ViT: The Original Vision Transformer

**Paper:** *An Image is Worth 16×16 Words* (Dosovitskiy et al., 2020)

The key insight is disarmingly simple: slice an image into fixed-size patches and treat each patch as a token, exactly as a word is treated in BERT. Then run a standard transformer encoder.

### 2.1 Patch Embedding

Given an image $$x \in \mathbb{R}^{H \times W \times C}$$, split it into $$N$$ non-overlapping patches of size $$P \times P$$:

$$N = \frac{H \times W}{P^2}$$

Each patch $$x_i \in \mathbb{R}^{P^2 C}$$ is linearly projected to a $$D$$-dimensional embedding:

$$z_i = x_i \mathbf{E}, \quad \mathbf{E} \in \mathbb{R}^{P^2 C \times D}$$

This linear projection is equivalent to a `Conv2d` with `kernel_size=P` and `stride=P`, which is the standard implementation.

### 2.2 [CLS] Token and Positional Encoding

A learnable classification token $$\mathbf{x}_\text{cls}$$ is prepended to the sequence. Learnable 1D positional embeddings $$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$ are added:

$$\mathbf{z}_0 = [\mathbf{x}_\text{cls};\, z_1\mathbf{E};\, z_2\mathbf{E};\, \ldots;\, z_N\mathbf{E}] + \mathbf{E}_{pos}$$

The [CLS] token aggregates global image information through attention and is used for classification. There is no spatial pooling — the token acts as the global summary by design.

### 2.3 Transformer Encoder

The sequence passes through $$L$$ identical transformer blocks. Each block uses **pre-norm** (LayerNorm before the sublayer, not after):

$$\mathbf{z}'_\ell = \text{MHSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$
$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

The MLP has an expansion ratio of 4×: $$D \to 4D \to D$$ with GELU activation. The residual connections are critical — they allow gradients to flow and let early layers be bypassed when not needed.

### 2.4 Multi-Head Self-Attention (MHSA)

For each of $$H$$ heads, project the input into queries, keys, and values:

$$Q_h = \mathbf{z} W_h^Q,\quad K_h = \mathbf{z} W_h^K,\quad V_h = \mathbf{z} W_h^V$$

Compute attention and aggregate:

$$A_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right), \quad \text{head}_h = A_h V_h$$

$$\text{MHSA}(\mathbf{z}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W^O$$

The $$\sqrt{d_k}$$ scaling prevents the dot products from growing large and pushing the softmax into a saturated region with near-zero gradients.

### 2.5 Classification Head

After $$L$$ blocks, the [CLS] token is normalized and classified:

$$\hat{y} = \text{LN}(\mathbf{z}_L^{[\text{CLS}]})\, \mathbf{W}_\text{head}$$

### 2.6 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel=stride=patch_size is identical to a linear patch projection
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2) # (B, N, D)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)      # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))   # classify via [CLS]


# Standard sizes
def vit_base():  return ViT(embed_dim=768,  depth=12, num_heads=12)  # 86M
def vit_large(): return ViT(embed_dim=1024, depth=24, num_heads=16)  # 307M
def vit_huge():  return ViT(embed_dim=1280, depth=32, num_heads=16)  # 632M
```

### 2.7 Key Practical Points

**Patch size is a major hyperparameter.** ViT-B/16 uses 16×16 patches → 196 tokens for a 224×224 image. ViT-B/32 → 49 tokens. Smaller patches give finer spatial resolution but quadratic cost. For most tasks, patch size 16 is the right trade-off.

**ViT needs a lot of pretraining data.** On ImageNet-1k alone, ViT-B underperforms a ResNet-50. The model has no spatial inductive bias, so it must learn patch locality from scratch. On JFT-300M (300M images), it dominates. The lesson: raw architecture quality matters less than the pretraining recipe.

**Pre-norm vs post-norm.** ViT uses pre-norm (LN before each sublayer). This makes training more stable than post-norm (the original transformer), especially at larger scale. Use pre-norm.

---

## 3. Positional Encodings — A Full Design Space

Positional encodings are non-trivial for vision. Unlike 1D text, images are 2D — and at inference you often need to handle resolutions different from training.

### 3.1 Learned 1D Absolute (ViT default)

One learned embedding per position in $\{0, \ldots, N\}$. Simple and effective at fixed resolution. Fails to generalize to new resolutions without interpolation.

### 3.2 Fixed 2D Sinusoidal

Separate sinusoidal encodings for row and column indices, concatenated or added. Better spatial awareness than 1D, still resolution-fixed.

### 3.3 Relative Positional Encodings

Rather than encoding absolute positions, encode the *relative offset* between token pairs. Added directly to attention logits:

$$A_{ij} = \frac{(q_i)(k_j + r_{i-j})^\top}{\sqrt{d_k}}$$

where $$r_{i-j}$$ is a learned embedding for the relative displacement. Used in Swin and many dense-prediction models. More flexible across resolutions.

### 3.4 RoPE (Rotary Position Embedding)

Now standard in LLMs (LLaMA, GPT-NeoX) and migrating to vision. Instead of adding position to embeddings, rotate query and key vectors by an angle proportional to position:

$$q_m \to q_m\, e^{im\theta},\quad k_n \to k_n\, e^{in\theta}$$

The dot product $$\langle q_m, k_n \rangle$$ then depends only on relative position $$m - n$$. Extends naturally to 2D by using independent rotation frequencies for the row and column axes. Excellent generalization to unseen resolutions.

### 3.5 No Positional Encoding

In fully windowed models (Swin), position is encoded implicitly by the window structure. Some models drop global positional encoding entirely and add only a local relative position bias within each window.

**Rule of thumb:** learned 1D absolute for fixed-resolution classification. RoPE or relative for anything requiring resolution flexibility (dense prediction, variable-resolution inference).

---

## 4. DeiT: Training ViT Without Giant Datasets

**Paper:** *Training Data-efficient Image Transformers* (Touvron et al., 2021)

The problem: ViT requires JFT-300M. Most researchers do not have that. DeiT makes ViT work on ImageNet-1k alone through a strong training recipe and knowledge distillation.

### 4.1 Training Recipe

DeiT shows that aggressive regularization and augmentation can substitute for data volume:

| Technique | Effect |
|-----------|--------|
| RandAugment | Strong spatial and color augmentation |
| Mixup | Blends two images and linearly interpolates labels |
| CutMix | Cuts a patch from one image into another |
| Label smoothing | Prevents overconfident predictions |
| Repeated augmentation | Same image augmented multiple ways per batch |
| Stochastic depth | Randomly drops entire transformer blocks |
| Cosine LR schedule | Smooth decay, better final convergence |

This alone brings DeiT-B to 81.8% top-1 on ImageNet — competitive with EfficientNet.

### 4.2 Distillation Token

DeiT adds a second special token alongside [CLS]: the **distillation token** $$\mathbf{x}_\text{dist}$$. It participates in attention like any other token. At the output, it is passed to a separate classification head trained against the teacher's prediction (argmax from a RegNet CNN):

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{CE}(\hat{y}_\text{cls},\, y) + \lambda\,\mathcal{L}_{CE}(\hat{y}_\text{dist},\, y_\text{teacher})$$

Hard distillation (argmax of teacher) works slightly better than soft distillation. At inference, average both head predictions.

```python
class DeiT(nn.Module):
    def __init__(self, base_vit: ViT, num_classes=1000):
        super().__init__()
        self.vit = base_vit
        D = base_vit.head.in_features
        n = base_vit.patch_embed.n_patches
        self.dist_token = nn.Parameter(torch.zeros(1, 1, D))
        # Expand pos_embed to cover [CLS] + [DIST] + N patches
        self.vit.pos_embed = nn.Parameter(torch.zeros(1, n + 2, D))
        self.dist_head = nn.Linear(D, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x   = self.vit.patch_embed(x)
        cls  = self.vit.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls, dist, x], dim=1) + self.vit.pos_embed
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)
        cls_out  = self.vit.head(x[:, 0])
        dist_out = self.dist_head(x[:, 1])
        if self.training:
            return cls_out, dist_out
        return (cls_out + dist_out) / 2   # average at inference
```

**Key lesson from DeiT III (2022):** With an even stronger recipe (3-Augment, LAMB optimizer, longer schedule), the distillation token becomes unnecessary. **Training recipe is at least as important as architecture.**

---

## 5. Self-Supervised Pretraining: BEiT and MAE

Supervised ViT needs labeled data. What if you want to pretrain on unlabeled images? The masked pretraining paradigm — a direct copy of BERT's success in NLP — answers this.

### 5.1 BEiT: BERT Pre-Training of Image Transformers

**Paper:** Bao et al., 2022

**The problem with naively applying BERT to images:** BERT predicts masked word tokens from a finite vocabulary. Images have no such vocabulary — predicting masked raw pixels teaches the model blurry spatial interpolation, not semantics.

**BEiT's solution:** create a discrete visual vocabulary with a **dVAE** (discrete VAE from DALL-E), then predict masked *visual tokens* rather than pixels.

**Stage 1 — Learn a visual tokenizer:** Train a dVAE to encode image patches into a codebook of 8192 discrete visual tokens. The tokenizer is then frozen.

**Stage 2 — Masked Image Modeling:** Mask ~40% of patches, replace with a learned `[MASK]` embedding, and train the ViT to predict the dVAE code for each masked patch:

$$\mathcal{L}_\text{BEiT} = -\sum_{i \in \mathcal{M}} \log p(z_i \mid \tilde{x})$$

Masked patches attend to visible ones through self-attention, forcing contextual understanding of image content.

### 5.2 MAE: Masked Autoencoders

**Paper:** He et al., 2022 — the simplest masked pretraining recipe, and one of the most effective.

Three departures from BEiT:

**1. High masking ratio (75%).** Far more aggressive than BEiT's 40%. At 75%, you cannot recover the image by interpolating neighbors — genuine semantic understanding is required.

**2. Asymmetric encoder-decoder.** The encoder processes *only visible patches* (25%). A small, shallow decoder reconstructs the full image. This makes pretraining ~4× faster because the encoder processes far fewer tokens.

**3. Predict raw pixels, not visual tokens.** No dVAE needed. Targets are per-patch mean-normalized pixel values. Simpler and comparably effective.

```python
class MAE(nn.Module):
    def __init__(self, encoder: ViT, decoder_dim=512, decoder_depth=8,
                 mask_ratio=0.75, patch_size=16):
        super().__init__()
        self.encoder    = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        D = encoder.pos_embed.shape[-1]
        N = encoder.patch_embed.n_patches

        self.decoder_embed    = nn.Linear(D, decoder_dim)
        self.mask_token       = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, N + 1, decoder_dim))
        self.decoder_blocks   = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads=16) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * 3)

    def random_masking(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep    = ids_shuffle[:, :keep]
        x_masked    = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0)   # 0 = visible, 1 = masked
        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # --- Encoder: process only visible patches ---
        x = self.encoder.patch_embed(imgs)
        x = x + self.encoder.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x)
        cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x   = torch.cat([cls + self.encoder.pos_embed[:, :1, :], x], dim=1)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)

        # --- Decoder: fill in mask tokens, reconstruct ---
        x = self.decoder_embed(x)
        B, _, Dd = x.shape
        n_masked = ids_restore.shape[1] - (x.shape[1] - 1)
        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = x_.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, Dd))
        x  = torch.cat([x[:, :1, :], x_], dim=1)
        x  = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        pred = self.decoder_pred(self.decoder_norm(x)[:, 1:, :])
        return pred, mask

    def loss(self, imgs, pred, mask):
        p = self.patch_size
        # Patchify target: (B, N, patch_size^2 * 3)
        target = imgs.unfold(2, p, p).unfold(3, p, p)
        target = target.permute(0, 2, 3, 1, 4, 5).reshape(imgs.shape[0], -1, p*p*3)
        # Normalize per patch
        mean = target.mean(-1, keepdim=True)
        var  = target.var(-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        loss = ((pred - target) ** 2).mean(-1)
        return (loss * mask).sum() / mask.sum()   # loss only on masked patches
```

The decoder is discarded after pretraining. Only the encoder is kept and fine-tuned downstream.

| | BEiT | MAE |
|--|------|-----|
| Prediction target | Discrete visual tokens | Raw pixels |
| Masking ratio | ~40% | 75% |
| Extra component | dVAE tokenizer | None |
| Pretraining cost | Moderate | Low (encoder sees 25% of tokens) |
| Decoder complexity | Transformer | Small transformer |

In practice, MAE is the default choice: simpler, cheaper, and broadly competitive.

---

## 6. DINO and DINOv2: Emergent Segmentation from Self-Distillation

**Papers:** Caron et al., 2021 (DINO); Oquab et al., 2023 (DINOv2)

DINO revealed something unexpected: a ViT trained purely with self-supervised self-distillation develops attention heads that **spontaneously segment objects** — without any segmentation supervision. This emergent property pointed to fundamentally richer representations than contrastive methods on CNNs had produced.

### 6.1 Self-Distillation Setup

DINO uses two networks with identical architecture: a student and a teacher. The teacher's weights are an **exponential moving average (EMA)** of the student's — it is never trained by gradient descent.

Both networks receive differently augmented views of the same image:
- **Student** sees local crops (~50% of image area) and global crops
- **Teacher** sees only global crops (~100% of image area)

The student is trained to match the teacher's output distribution:

$$\mathcal{L}_\text{DINO} = -\sum_{\text{global}\, v'} \sum_{\substack{v \neq v'}} p_t(v') \log p_s(v)$$

where $$p_s, p_t$$ are softmax outputs over $$K$$ prototype dimensions. The loss is asymmetric: the teacher generates targets; the student is optimized.

### 6.2 Collapse Prevention

Two mechanisms prevent trivial solutions:

**Centering:** subtract a running mean from teacher logits before softmax. This prevents the teacher from collapsing to outputting the same distribution for all images.

**Sharpening:** use a low temperature on the teacher softmax to produce peaked, confident distributions. The student uses a higher temperature.

```python
# EMA teacher update after each optimizer step
momentum = 0.996  # annealed toward 1.0 over training
for ps, pt in zip(student.parameters(), teacher.parameters()):
    pt.data = momentum * pt.data + (1 - momentum) * ps.data
# No gradients flow through the teacher
```

### 6.3 Why Do Attention Maps Segment Objects?

The local-to-global matching objective forces the model to understand *which image regions are semantically informative*. To predict global content from a small local crop, the [CLS] token must attend to the image parts that carry meaning — and that turns out to be the foreground objects. This is purely emergent: no segmentation loss is used at any point.

```python
# Visualize DINO attention maps
def get_dino_attention(model, img):
    model.eval()
    attentions = []
    def hook(m, inp, out):
        attentions.append(inp[0].detach())
    h = model.blocks[-1].attn.attn_drop.register_forward_hook(hook)
    with torch.no_grad():
        model(img.unsqueeze(0))
    h.remove()
    # [CLS] attending to patch tokens, across all heads
    return attentions[0][:, :, 0, 1:]  # (1, num_heads, N)
```

### 6.4 DINOv2

DINOv2 scales DINO significantly and produces features that work off-the-shelf — no encoder fine-tuning needed:

- **Data:** 142M curated images (LVD-142M) filtered from the web for quality and diversity
- **Training objectives:** DINO self-distillation + iBOT (token-level masked prediction) + KoLeo regularizer (encourages uniform coverage of embedding space)
- **Patch size:** 14×14 (instead of 16×16) for finer spatial resolution
- **Models:** ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14

DINOv2-L and DINOv2-g are now the de facto starting point for any task requiring a strong visual feature extractor.

```python
# DINOv2 off-the-shelf features (no fine-tuning of encoder)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval()

with torch.no_grad():
    features = model.forward_features(img.unsqueeze(0))

patch_features = features['x_norm_patchtokens']  # (1, N, 1024) — spatial
cls_feature    = features['x_norm_clstoken']     # (1, 1024)    — global
```

---

## 7. Swin Transformer: Hierarchical ViTs for Dense Prediction

**Paper:** *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al., 2021)

The original ViT has two properties that make it awkward for dense prediction (segmentation, detection):

1. **Single resolution:** all layers output features at the same spatial scale. CNNs produce a feature pyramid (stride 2 at each stage) that detection and segmentation heads depend on.
2. **Quadratic attention cost:** self-attention over $$N$$ tokens is $$O(N^2)$$. At 800px, $$N \approx 2500$$. Full attention becomes prohibitively expensive.

Swin solves both.

### 7.1 Hierarchical Feature Maps

Swin processes images through 4 stages, each halving spatial resolution:

| Stage | Resolution | Channels | Tokens per window |
|-------|-----------|----------|-------------------|
| 1 | H/4 × W/4 | 96 | 7×7 = 49 |
| 2 | H/8 × W/8 | 192 | 7×7 = 49 |
| 3 | H/16 × W/16 | 384 | 7×7 = 49 |
| 4 | H/32 × W/32 | 768 | 7×7 = 49 |

Between stages, **patch merging** concatenates 2×2 neighboring tokens and projects to 2× channels — equivalent to stride-2 downsampling. The resulting multi-scale feature map is directly compatible with FPN-based detection and segmentation heads.

### 7.2 Window Self-Attention (W-MSA)

Instead of global attention, each token attends only within its local $$M \times M$$ window. Cost per layer:

$$O\!\left(\frac{H'W'}{M^2} \cdot M^4\right) = O(M^2 H' W')$$

Linear in image size, quadratic only in the fixed window size $$M=7$$. The trade-off: no cross-window communication within a single layer.

### 7.3 Shifted Window Self-Attention (SW-MSA)

Alternate layers shift the window grid by $$\lfloor M/2 \rfloor$$ pixels. Tokens near window boundaries in layer $$\ell$$ find themselves in the same window in layer $$\ell+1$$. Two layers together provide connectivity across the entire feature map.

Partial windows at the borders are handled via **cyclic shifting**: roll the feature map, compute attention with an additive attention mask ($$-\infty$$ for cross-window pairs), then roll back. This avoids padding overhead.

```python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # Learned relative position bias table: (2M-1)^2 entries per head
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

    def forward(self, x, mask=None):
        # x: (num_windows * B, M*M, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0) * -100.0
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

    def _get_rel_pos_bias(self):
        # Index into table using precomputed relative position indices
        # (implementation detail: indices precomputed in __init__)
        return self.rel_pos_bias[self.rel_pos_index].permute(2, 0, 1).unsqueeze(0)
```

### 7.4 Swin vs ViT — When to Use Which

| | ViT | Swin |
|--|-----|------|
| Feature maps | Single scale | Multi-scale pyramid |
| Attention scope | Global | Local windows |
| Detection / segmentation | Needs adapter (ViTDet) | Native (FPN-ready) |
| Attention complexity | $$O(N^2)$$ | $$O(N)$$ |
| Positional encoding | Absolute | Relative (within window) |
| Best use | Large-scale pretraining, multimodal | Dense prediction pipelines |

---

## 8. DETR: Object Detection as Set Prediction

**Paper:** *End-to-End Object Detection with Transformers* (Carion et al., 2020)

Traditional detection pipelines are complex: anchor design, region proposals, non-maximum suppression. DETR replaces all of this with a single, end-to-end trainable model that produces detections directly as a set.

### 8.1 Architecture

**Backbone + Encoder:** A CNN (or ViT) backbone extracts a feature map. Flatten the spatial dimensions, add 2D positional encodings, and pass through a transformer encoder.

**Decoder:** $$N$$ learned **object queries** ($$N=100$$, far more than objects per image) are passed to the transformer decoder. Queries attend to each other (self-attention) and to the encoder output (cross-attention). Each query slot specializes in detecting one object.

**Prediction heads:** Two parallel FFN heads per query:
- Class prediction: softmax over $$C + 1$$ classes (the extra class is "no object")
- Box prediction: sigmoid over $$(c_x, c_y, w, h)$$ in normalized coordinates

### 8.2 Bipartite Matching Loss

The central contribution: given $$N$$ predictions and $$M$$ ground-truth objects ($$M \ll N$$), find the optimal one-to-one assignment via the **Hungarian algorithm**:

$$\hat{\sigma} = \underset{\sigma \in \mathfrak{S}_N}{\arg\min} \sum_{i} \mathcal{L}_\text{match}(y_i,\, \hat{y}_{\sigma(i)})$$

The matching cost balances classification confidence and box quality (L1 + GIoU). Once matched, unmatched predictions are assigned to the "no object" class. The loss is computed only on matched pairs.

```python
from scipy.optimize import linear_sum_assignment

def hungarian_match(pred_logits, pred_boxes, targets):
    """
    pred_logits: (N, C+1)
    pred_boxes:  (N, 4) — normalized (cx, cy, w, h)
    targets: list of {'labels': (M,), 'boxes': (M, 4)}
    """
    out_prob  = pred_logits.softmax(-1)
    tgt_ids   = torch.cat([t['labels'] for t in targets])
    tgt_boxes = torch.cat([t['boxes']  for t in targets])

    cost_class = -out_prob[:, tgt_ids]
    cost_l1    = torch.cdist(pred_boxes, tgt_boxes, p=1)
    cost_giou  = -generalized_box_iou(pred_boxes, tgt_boxes)

    C = cost_class + 5 * cost_l1 + 2 * cost_giou  # (N, M)
    rows, cols = linear_sum_assignment(C.detach().cpu().numpy())
    return torch.as_tensor(rows), torch.as_tensor(cols)
```

No anchors. No NMS. Fully differentiable. The model either predicts an object at a position or it doesn't — there is no duplicate suppression step.

### 8.3 Limitations and Follow-ups

DETR converges slowly (~500 epochs vs 36 for Faster R-CNN) and struggles with small objects because the encoder operates at a single, low-resolution feature map.

**Deformable DETR** (Zhu et al., 2020): replaces full cross-attention with deformable attention — each query attends to a small set of learned reference points on a multi-scale feature map. Dramatically faster convergence (50 epochs) and better small-object performance.

**DINO-DETR** (not to be confused with the self-supervised DINO): adds contrastive denoising training and improved query initialization, pushing DETR-family methods to state-of-the-art on COCO.

---

## 9. SAM: Segment Anything

**Paper:** Kirillov et al., 2023 (Meta AI)

SAM demonstrates the **promptable foundation model** paradigm: given any prompt (click, box, mask, text), return a valid segmentation mask in real time. It is trained on 1.1 billion masks across 11 million images — SA-1B, the largest segmentation dataset by two orders of magnitude.

### 9.1 Architecture

Three components with very different computational budgets:

**Image encoder (heavy, run once):** A MAE-pretrained ViT-H (632M params) encodes a 1024×1024 image into a $$64 \times 64 \times 256$$ feature map. This is the expensive step — but it runs once per image regardless of how many prompts follow.

**Prompt encoder (lightweight):**
- Points → positional encoding + learned foreground/background token
- Boxes → two corner point encodings
- Masks → spatial convolution + element-wise addition to image features
- Text → CLIP text encoder (available but experimental)

**Mask decoder (very lightweight — 2 transformer layers):** Prompt tokens attend to image features via cross-attention. Self-attention among prompt tokens. Two output tokens produce three candidate masks per query (handling ambiguity: a click on a person may mean the person, torso, or shirt) plus a confidence score per mask.

```python
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").cuda()
predictor = SamPredictor(sam)

predictor.set_image(image)          # expensive encoder step — run once

# Prompt: foreground point click
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),     # 1 = foreground, 0 = background
    multimask_output=True,          # returns 3 candidate masks
)
best_mask = masks[scores.argmax()]
```

**SAM 2 (2024)** extends this to video with a Hiera backbone (hierarchical MAE) and a memory bank that accumulates per-frame features, enabling real-time promptable video segmentation from a single frame annotation.

---

## 10. CLIP: Vision-Language Pretraining

**Paper:** *Learning Transferable Visual Models from Natural Language Supervision* (Radford et al., OpenAI, 2021)

CLIP jointly trains a ViT image encoder and a text transformer on 400M image-text pairs scraped from the web using **contrastive learning**. The result is visual features that can be aligned to arbitrary text descriptions — enabling zero-shot transfer to any classification task.

### 10.1 Contrastive Objective

For a batch of $$N$$ (image, text) pairs, compute an $$N \times N$$ similarity matrix. Diagonal entries are matched pairs; off-diagonal are negatives. The loss pulls matched pairs together and pushes mismatches apart:

$$\mathcal{L}_\text{CLIP} = \frac{1}{2N}\sum_{i=1}^N \left[ -\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} - \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}} \right]$$

where $$s_{ij} = f_I(x_i)^\top f_T(t_j)$$ (after L2 normalization) and $$\tau$$ is a learnable temperature.

```python
def clip_loss(image_features, text_features, temperature):
    image_features = F.normalize(image_features, dim=-1)
    text_features  = F.normalize(text_features,  dim=-1)

    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_i2t = F.cross_entropy(logits,   labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

### 10.2 Zero-Shot Classification

CLIP's killer capability: classify images into arbitrary categories without any fine-tuning, using text prompts as classifiers.

```python
class_names = ["cat", "dog", "airplane", ...]
templates   = ["a photo of a {}.", "a picture of a {}.", "{}"]

# Encode all class text prompts
with torch.no_grad():
    text_tokens   = [clip.tokenize(t.format(c)) for t in templates for c in class_names]
    text_features = model.encode_text(torch.cat(text_tokens).cuda())
    text_features = text_features.reshape(len(templates), len(class_names), -1).mean(0)
    text_features = F.normalize(text_features, dim=-1)

# Classify new image
image_features = F.normalize(model.encode_image(image), dim=-1)
probs = (image_features @ text_features.T).softmax(dim=-1)
```

**Prompt engineering is not cosmetic.** Ensembling over 80 text templates gives 3–4% better ImageNet accuracy than a single template. The right prompt can be the difference between a useful classifier and a poor one.

### 10.3 CLIP as a Universal Backbone

CLIP's image encoder is the visual backbone underlying most modern multimodal systems:

- **DALL-E 2:** generates images conditioned on CLIP image embeddings
- **Stable Diffusion:** text guidance runs through CLIP text encoder
- **LLaVA, InstructBLIP, GPT-4V:** CLIP ViT-L/14 as the visual tokenizer feeding into an LLM
- **Zero-shot retrieval:** embed a query image or text, find nearest neighbors in the other modality

---

## 11. DiT: Transformers for Diffusion Models

**Paper:** *Scalable Diffusion Models with Transformers* (Peebles & Xie, 2022)

DiT replaces the U-Net backbone in latent diffusion models with a transformer operating on latent patches. It is now the architecture behind **Stable Diffusion 3**, **FLUX**, and **Sora**.

### 11.1 Architecture

An image is first encoded to a latent space via a VAE (e.g., $$256 \times 256 \to 32 \times 32 \times 4$$). The latent is then patchified (typically 2×2 patches) into a sequence of tokens. A noise timestep $$t$$ and class label $$c$$ are encoded as conditioning signals.

**adaLN-Zero:** The key conditioning mechanism. Standard LayerNorm parameters $$(\gamma, \beta)$$ are predicted from the conditioning signal $$(t, c)$$ by a small MLP. The MLP output is initialized to zero so conditioning starts as a no-op, making early training stable:

$$\text{adaLN}(h, t, c) = \gamma(t,c) \cdot \text{LN}(h) + \beta(t,c)$$

An additional gating scalar $$\alpha(t,c)$$ is applied before each residual connection. All three — $$\gamma, \beta, \alpha$$ — are predicted jointly.

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn  = MultiHeadSelfAttention(hidden_size, num_heads)
        hidden_mlp = int(hidden_size * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_size, hidden_mlp),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_mlp, hidden_size),
        )
        # Predict 6 scalars: (shift, scale, gate) × (attn, mlp)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-init so conditioning starts as identity
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.adaLN(c).chunk(6, dim=-1)

        # Modulate LayerNorm parameters from conditioning
        def modulate(h, shift, scale):
            return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp( modulate(self.norm2(x), shift_m, scale_m))
        return x
```

### 11.2 Scaling Behavior

DiT follows smooth power-law scaling: DiT-XL/2 (675M params, patch size 2) achieves state-of-the-art FID on class-conditional ImageNet-256 generation. Larger models trained longer always improve — the compute-optimal frontier follows Chinchilla-like scaling.

### 11.3 From DiT to Sora and FLUX

**Sora** (OpenAI, 2024) extends DiT to video by replacing 2D spatial patches with **spatiotemporal tubes** (patches across both space and time). Variable resolution and duration are handled by treating all tokens as a single flat sequence regardless of shape.

**FLUX** (Black Forest Labs) introduces **Multimodal Diffusion Transformer (MMDiT)**: image tokens and text tokens are processed in separate streams with weights for each modality, but attend to each other in every layer. This gives both modalities equal capacity in the attention computation.

---

## 12. Video and 3D Transformers

### 12.1 Video

Extending ViT to video requires handling the temporal axis. The core question is how to factorize spatiotemporal attention.

**TimeSformer** (Bertasius et al., 2021): within each block, run temporal attention first (each patch attends to same-position patches across all $$T$$ frames) then spatial attention (each patch attends to all patches within its frame). Cost is $$O(T^2 + H'W'^2)$$ per layer — far cheaper than full 3D attention $$O((T \cdot H'W')^2)$$.

**Video Swin** (Liu et al., 2022): extends Swin's shifted window attention to 3D spatiotemporal windows. Highly efficient and achieves strong results on Kinetics-400/600.

**Practical tube embedding:**

```python
class TubeletEmbedding(nn.Module):
    """Embeds a video clip into spatiotemporal patch tokens."""
    def __init__(self, embed_dim, tubelet_size=2, patch_size=16, in_channels=3):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)                    # (B, D, T', H', W')
        return x.flatten(2).transpose(1, 2) # (B, T'*H'*W', D)
```

### 12.2 Point Clouds

Point clouds are unordered sets of $$(x, y, z)$$ coordinates. Self-attention is permutation-equivariant — a natural fit.

**Point Transformer** (Zhao et al., 2021): uses **vector self-attention** — attention weights are vectors, not scalars, so different feature dimensions can attend with different strengths. k-NN grouping defines local neighborhoods within which attention is computed.

**PCT** (Guo et al., 2021): full self-attention over all points after farthest-point sampling. Offset-attention (attention on residuals rather than values) improves geometric sensitivity.

The key challenge is position encoding: unlike image patches on a grid, 3D points are irregular. Input coordinate features and local neighborhood encodings substitute for standard positional embeddings.

---

## 13. ViT vs CNN: Properties and Robustness

Understanding the inductive biases of ViTs versus CNNs matters for choosing the right backbone and anticipating failure modes.

### 13.1 Texture Bias vs Shape Bias

CNNs are **texture-biased**: a cat image rendered with elephant texture is classified as elephant. Local filters naturally pick up texture statistics, and ImageNet training reinforces this.

ViTs — especially those trained with DINO — develop more **shape-biased** representations, better aligned with human perception. Global self-attention integrates object boundary information across the full image, regardless of local texture. This has practical consequences:

- ViTs generalize better to style-shifted and texture-shifted domains
- DINO/DINOv2 attention maps track object shapes, not texture patches
- CNNs can partially close the gap with texture-augmentation training (AugMix, Stylized-ImageNet fine-tuning)

### 13.2 Out-of-Distribution Robustness

ViTs consistently outperform CNNs on OOD benchmarks (ImageNet-C corruption robustness, ImageNet-R renditions, ObjectNet):

| Benchmark | ResNet-50 | ViT-B/16 | DINOv2-L |
|-----------|-----------|----------|----------|
| ImageNet-C (mean corruption error ↓) | ~76% | ~55% | ~35% |
| ImageNet-R top-1 ↑ | ~36% | ~51% | ~75% |
| ObjectNet top-1 ↑ | ~25% | ~43% | ~68% |

The combination of global attention (less texture reliance) and large pretraining (better prior over image statistics) drives this gap.

### 13.3 Adversarial Robustness

Here ViTs are *not* inherently better. Without adversarial training, ViTs and CNNs are similarly vulnerable to $$\ell_\infty$$-bounded perturbations. The Lipschitz properties of softmax attention are no better than those of convolution. Adversarial training (PGD-AT, TRADES) improves both architectures to comparable levels.

### 13.4 Feature Structure Across Depth

CNN representations become increasingly local and discriminative with depth. ViT representations are more **uniform across layers** — global attention is present from layer 1, so earlier and later layers differ less dramatically in spatial resolution. This changes how you fine-tune: CNNs benefit from freezing early layers; ViTs often benefit from fine-tuning all layers with layer-wise LR decay.

### 13.5 When CNNs Still Win

- **Small datasets** (< 50K images): inductive biases pay off; CNNs generalize better
- **Latency-constrained inference**: ConvNeXt, EfficientNet, and MobileNet remain faster at equal accuracy for edge devices
- **Fine-grained local texture tasks**: material classification, medical texture grading
- **Strong augmentation closes the gap** on medium datasets — with the DeiT recipe, the advantage of CNNs at ImageNet scale disappears

---

## 14. Scaling Laws for ViTs

One of the strongest practical arguments for ViTs: they scale predictably. This is why industry has moved heavily to ViT-based architectures.

### 14.1 Standard Model Sizes

| Model | Params | Layers | Dim | Heads | Patch |
|-------|--------|--------|-----|-------|-------|
| ViT-Ti | 5.7M | 12 | 192 | 3 | 16 |
| ViT-S | 22M | 12 | 384 | 6 | 16 |
| ViT-B | 86M | 12 | 768 | 12 | 16 |
| ViT-L | 307M | 24 | 1024 | 16 | 16 |
| ViT-H | 632M | 32 | 1280 | 16 | 14 |
| ViT-G | 1.8B | 40 | 1664 | 16 | 14 |
| ViT-22B | 22B | 48 | 6144 | 48 | 14 |

### 14.2 Scaling Behavior

- **Power-law scaling:** loss decreases as a power law in both model size and training tokens
- **No observed saturation** up to ViT-22B on JFT-3B
- **Compute-optimal allocation:** as with Chinchilla in language, the optimal compute allocation shifts more toward data (tokens) at larger scale

Larger ViTs with more pretraining data consistently outperform smaller ViTs trained longer. If you have a compute budget, make the model bigger rather than training the same model for more epochs.

### 14.3 Resolution Adaptation

Pretrain at lower resolution, fine-tune at higher resolution. Interpolate positional embeddings to the new grid:

```python
def interpolate_pos_embed(model, new_img_size):
    """Resize pos_embed when fine-tuning at a different resolution than pretraining."""
    pos_embed = model.pos_embed                           # (1, N+1, D)
    cls_embed = pos_embed[:, :1, :]                       # (1, 1, D) — keep as-is
    patch_embed = pos_embed[:, 1:, :]                     # (1, N, D)

    old_n = int(patch_embed.shape[1] ** 0.5)             # old grid size
    new_n = new_img_size // model.patch_size              # new grid size

    patch_embed = patch_embed.reshape(1, old_n, old_n, -1).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(patch_embed, size=(new_n, new_n),
                                mode='bicubic', align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_n**2, -1)

    model.pos_embed = nn.Parameter(torch.cat([cls_embed, patch_embed], dim=1))
```

---

## 15. Hybrid Architectures

The line between CNNs and transformers has blurred. The best practical models often borrow from both.

### 15.1 CvT and CoAtNet

**CvT (Convolutional Vision Transformer):** replaces linear patch projection with convolutional projection at each stage. Convolutions carry local inductive bias cheaply in early layers; attention handles global reasoning in later layers. Fewer parameters needed because the network does not have to learn patch locality from scratch.

**CoAtNet:** interleaves convolution blocks (depthwise conv for local processing) and full attention blocks (for global reasoning). Early stages use convolution; later stages use attention. Achieves top ImageNet performance at competitive inference speed. The key insight: convolutions and attention are complementary, not competing.

### 15.2 ConvNeXt: Modernizing CNNs with ViT Design Choices

ConvNeXt (Liu et al., 2022) takes a ResNet and applies ViT-inspired design decisions one at a time:

| Change | From | To |
|--------|------|-----|
| Kernel size | 3×3 | 7×7 depthwise conv |
| Activation | ReLU | GELU |
| Normalization | BatchNorm | LayerNorm |
| Architecture | Bottleneck (1×1, 3×3, 1×1) | Inverted bottleneck (1×1 expand, 7×7 dw, 1×1 project) |
| Fewer activations/norms | One per block | One per block (but positioned like transformers) |

The result: a pure CNN that matches Swin Transformer on ImageNet and downstream tasks. ConvNeXt is a reminder that many of ViT's gains come from training improvements and design philosophy, not from attention per se.

### 15.3 Practical Backbone Selection Guide

| Scenario | Recommended Architecture |
|----------|--------------------------|
| Large-scale pretraining from scratch | ViT-L/H with MAE or DINO |
| Detection / instance segmentation | Swin-L or ViT-L with Mask2Former |
| Semantic segmentation | Swin + UPerNet or DINOv2 + linear head |
| Latency-constrained deployment | ConvNeXt-T or Swin-T |
| Video understanding | Video Swin or TimeSformer |
| Foundation model (no fine-tuning) | DINOv2-L or DINOv2-g |
| Generative modeling (diffusion) | DiT-XL/2 or FLUX architecture |
| Multimodal / zero-shot | CLIP ViT-L/14 or ViT-G/14 |
| Point cloud / 3D | Point Transformer v2 or PCT |

---

## 16. Training Tips and Test-Time Considerations

### 16.1 Optimizer

AdamW with weight decay is the standard. Apply weight decay **only to weight matrices** — not to biases, LayerNorm parameters, or positional embeddings:

```python
def get_param_groups(model, weight_decay=0.05):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

optimizer = torch.optim.AdamW(get_param_groups(model), lr=1e-3, betas=(0.9, 0.999))
```

For very large batches (> 4096), LAMB scales better than AdamW.

### 16.2 Learning Rate Schedule

Linear warmup + cosine decay is universal:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
```

Scale peak LR linearly with batch size: `lr = base_lr * batch_size / 256`. For ViT-B, `base_lr = 1e-3` with AdamW. Use 5–20 epochs of warmup.

### 16.3 Layer-Wise LR Decay (llrd)

Fine-tuned ViTs benefit from applying a smaller LR to earlier layers. Typical decay factor: 0.65–0.9:

```python
def build_llrd_param_groups(model, base_lr, decay=0.75):
    num_layers = len(model.blocks)
    groups = []
    for i, block in enumerate(model.blocks):
        lr = base_lr * (decay ** (num_layers - i))
        groups.append({'params': list(block.parameters()), 'lr': lr})
    groups.append({'params': list(model.head.parameters()), 'lr': base_lr})
    return groups
```

### 16.4 Stochastic Depth

Randomly drop entire transformer blocks during training. Later blocks are dropped more often (linear schedule from 0 to `drop_path_rate`):

```python
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * (random_tensor < keep_prob).float()
```

Typical `drop_path_rate`: 0.1 for ViT-B, 0.2 for ViT-L, 0.3+ for ViT-H.

### 16.5 Flash Attention

Standard attention materializes the full $$N \times N$$ matrix in GPU HBM — slow and memory-intensive for long sequences. FlashAttention computes attention in SRAM tiles without writing the full matrix to HBM. Same numerical output, 2–4× faster, significantly reduced memory.

```python
# PyTorch 2.0+ dispatches to FlashAttention automatically
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
```

Use `F.scaled_dot_product_attention` instead of manually computing `softmax(QK^T/√d)V`. It handles FlashAttention, memory-efficient attention, and math backends transparently.

### 16.6 Mixed Precision and Gradient Clipping

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(x)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

Gradient clipping at `max_norm=1.0` is important early in training when gradients can spike.

### 16.7 EMA of Model Weights

Maintain an EMA of model weights for evaluation. Typically adds 0.1–0.5% accuracy at no training cost:

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.9999))
# After each optimizer step:
ema.update_parameters(model)
# Evaluate using ema, not model
```

### 16.8 Test-Time Considerations

**Multi-crop evaluation:** run at multiple scales and crops, average logits. Adds ~5–10% compute for ~0.2–0.5% accuracy gain on ImageNet.

**Attention map visualization:**
```python
def extract_attention(model, x, layer_idx=-1):
    """Return (B, num_heads, N, N) attention weights from specified block."""
    attentions = []
    def hook(m, inp, out):
        attentions.append(inp[0].detach())
    h = model.blocks[layer_idx].attn.attn_drop.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    h.remove()
    return attentions[0]  # pre-dropout attention weights
```

**Register tokens:** some models (DINOv2) add extra `register` tokens that absorb high-norm artifacts from background patches, producing cleaner attention maps and patch features for dense tasks.

---

## 17. Where the Field Is Going

### Open Problems

**Efficiency at scale.** FlashAttention helps, but $$O(N^2)$$ attention remains a bottleneck for video, high-resolution images, and 3D volumetric data. **Vision Mamba** and **VMamba** adapt state-space models to vision for $$O(N)$$ complexity — early results are promising but not yet at ViT quality. The irregular spatial structure of images does not map as naturally to 1D SSM scanning as sequential text does.

**Unified multimodal architectures.** Models like **4M**, **UnifiedIO**, and **Gemini** treat vision as one channel in a broader multi-modal transformer. Vision encoding, language, audio, and actions share a single architecture. The question is shifting from "which visual architecture?" to "how do modalities share representations?"

**Label efficiency.** With DINOv2-g frozen features, a linear probe achieves 86%+ top-1 on ImageNet — rivaling full supervised fine-tuning from 5 years ago. The research question is no longer "how do we train with less data" but "how do we adapt powerful frozen features to novel tasks with minimal additional supervision?"

**3D and spatial intelligence.** Transformers are being applied to Neural Radiance Fields, 3D Gaussian splatting, and point cloud sequences for autonomous driving. Consistent 3D representations across frames remain an open challenge.

**Scaling beyond ViT-22B.** Google's ViT-22B showed continued scaling gains but at enormous training cost. Whether scaling continues to improve quality in a cost-effective way — or whether qualitatively new architectures are needed at billion-parameter scale — remains an active debate.

---

## Summary Reference Table

| Model | Year | Primary Task | Core Idea |
|-------|------|-------------|-----------|
| ViT | 2020 | Classification | Images as patch token sequences |
| DeiT | 2021 | Classification | CNN distillation + strong training recipe |
| DINO | 2021 | Self-supervised | Self-distillation; emergent object segmentation |
| DETR | 2020 | Detection | Set prediction + Hungarian matching |
| Swin | 2021 | All (hierarchical) | Shifted window attention + feature pyramid |
| BEiT | 2022 | Pretraining | Masked visual token prediction |
| CLIP | 2021 | Vision-Language | Contrastive image-text pretraining |
| MAE | 2022 | Pretraining | 75% masking + pixel reconstruction |
| SAM | 2023 | Segmentation | Promptable segmentation foundation model |
| DiT | 2022 | Generation | Transformer backbone for latent diffusion |
| DINOv2 | 2023 | Universal backbone | Curated data + combined objectives |
| Video Swin | 2022 | Video | 3D shifted windows for video understanding |
| SAM 2 | 2024 | Video segmentation | Memory-augmented promptable segmentation |

The unifying story: transformers impose no domain-specific constraints, scale predictably with data and compute, and compose naturally with other modalities and tasks. The field has not found a ceiling yet.
