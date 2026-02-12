import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import time
from fvcore.nn import FlopCountAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class SSTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, num_patches, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.k_layers = nn.ModuleList([])
        self.channels_to_embedding = nn.Linear(num_patches, b_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, b_dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        for _ in range(b_depth):
            self.k_layers.append(nn.ModuleList([
                Residual(PreNorm(b_dim, Attention(dim=b_dim, heads=b_heads, dim_head=b_dim_head, dropout = dropout))),
                Residual(PreNorm(b_dim, FeedForward(b_dim, b_mlp_head, dropout = dropout)))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.channels_to_embedding(x)
        b, d, n = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        for attn, ff in self.k_layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class SSTTransformerEncoder(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, num_patches, cross_attn_depth=3, cross_attn_heads=8, dropout = 0):
        super().__init__()

        self.transformer = SSTransformer(dim, depth, heads, dim_head, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, num_patches, dropout)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(PreNorm(b_dim, CrossAttention(b_dim, heads = cross_attn_heads, dim_head=dim_head, dropout=0)))

    def forward(self, x1, x2):
        x1 = self.transformer(x1)
        x2 = self.transformer(x2)

        for cross_attn in self.cross_attn_layers:
            x1_class = x1[:, 0]
            x1 = x1[:, 1:]
            x2_class = x2[:, 0]
            x2 = x2[:, 1:]

            # Cross Attn
            cat1_q = x1_class.unsqueeze(1)
            cat1_qkv = torch.cat((cat1_q, x2), dim=1)
            cat1_out = cat1_q+cross_attn(cat1_qkv)
            x1 = torch.cat((cat1_out, x1), dim=1)
            cat2_q = x2_class.unsqueeze(1)
            cat2_qkv = torch.cat((cat2_q, x1), dim=1)
            cat2_out = cat2_q+cross_attn(cat2_qkv)
            x2 = torch.cat((cat2_out, x2), dim=1)

        return cat1_out, cat2_out

class SSTFormer(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, dim_head = 16, dropout=0., emb_dropout=0., multi_scale_enc_depth=1):
        super().__init__()

        patch_dim = image_size ** 2 * near_band
        self.num_patches = num_patches+1
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token_t1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_t2 = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(SSTTransformerEncoder(dim, depth, heads, dim_head, mlp_dim,b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, self.num_patches,
                                                                                    dropout = 0.))

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(b_dim),
            nn.Linear(b_dim, num_classes)
        )
    def forward(self, x1, x2):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')
        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x1 = self.patch_to_embedding(torch.transpose(x1, 1, 2)) #[b,n,dim]
        x2 = self.patch_to_embedding(torch.transpose(x2, 1, 2))
        b, n, _ = x1.shape
        # add position embedding
        cls_tokens_t1 = repeat(self.cls_token_t1, '() n d -> b n d', b = b) #[b,1,dim]
        cls_tokens_t2 = repeat(self.cls_token_t2, '() n d -> b n d', b = b)

        x1 = torch.cat((cls_tokens_t1, x1), dim = 1) #[b,n+1,dim]
        x1 += self.pos_embedding[:, :(n + 1)]
        x1 = self.dropout(x1)
        x2 = torch.cat((cls_tokens_t2, x2), dim = 1) #[b,n+1,dim]
        x2 += self.pos_embedding[:, :(n + 1)]
        x2 = self.dropout(x2)
        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        for multi_scale_transformer in self.multi_scale_transformers:
            out1, out2 = multi_scale_transformer(x1, x2)
        # classification: using cls_token output
        out1 = self.to_latent(out1[:,0])
        out2 = self.to_latent(out2[:,0])
        out = out1+out2
        # MLP classification layer
        return self.mlp_head(out)

if __name__ == "__main__":
    # 创建模型实例，定义输入参数

    patches = 5
    band_patches = 3
    num_classes = 3
    band = 166

    model = SSTFormer(
        image_size = patches,
        near_band = band_patches,
        num_patches = band,
        num_classes = num_classes,
        dim = 64,
        depth = 2,
        heads = 4,
        dim_head = 16,
        mlp_dim = 8,
        b_dim = 512,
        b_depth = 3,
        b_heads = 8,
        b_dim_head= 32,
        b_mlp_head = 8,
        dropout = 0.2,
        emb_dropout = 0.1,
    )

    # 测试推理时间
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    input1 = torch.randn(3, patches**2*band_patches, band).to(device)
    input2 = torch.randn(3, patches**2*band_patches, band).to(device)
    print(input1.shape)
    output = model(input1,  input2)
    print(output.shape)

    # GPU 计时
    if device == 'cuda':
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            _ = model(input1, input2)
        ender.record()
        torch.cuda.synchronize()
        print(f"Inference time: {starter.elapsed_time(ender)} ms")

    # CPU 计时
    else:
        start_time = time.time()
        with torch.no_grad():
            _ = model(input1, input2)
        end_time = time.time()
        print(f"Inference time: {(end_time - start_time) * 1000} ms")

    # 计算并打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6}M")
    print(f"Trainable parameters: {trainable_params/1e6}M")

    flops = FlopCountAnalysis(model, (input1, input2))
    print(f"FLOPs: {flops.total() / 1e6} MFLOPs")  # 将FLOPs转换为GFLOPs

