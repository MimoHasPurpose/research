import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

# --- Parallel Spiking Neuron (PPSN) ---
class TriangularSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha=1.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (1 / ctx.alpha) * (1 / ctx.alpha) * ((ctx.alpha - input.abs()).clamp(min=0))
        return grad_input * temp, None

surrogate_fn = TriangularSurrogate.apply

class pa_lif_n(nn.Module):
    def __init__(self, T=4, tau=0.25, threshold=0.5):
        super().__init__()
        self.T = T
        self.tau = tau
        self.threshold = nn.Parameter(torch.as_tensor(threshold))
        self.surrogate_function = surrogate_fn
        
        self.register_buffer('indices', torch.arange(self.T).unsqueeze(1))
        self.powers = (self.indices - self.indices.T).clamp(min=0)
        self.register_buffer('W', torch.tril(self.tau * ((1 - self.tau) ** self.powers)))
        
        indices_1 = torch.arange(self.T)
        row, col = torch.meshgrid(indices_1, indices_1, indexing='ij')
        self.register_buffer('coefficients', (self.tau ** (row - col)).tril().unsqueeze(-1))
        
        self.i_grid, self.j_grid = torch.meshgrid(indices_1, indices_1, indexing='ij')
        self.mask = ((self.i_grid > self.j_grid) & (self.j_grid <= self.i_grid - 2)) | (self.i_grid == self.j_grid + 1)
        self.register_buffer('mask2', self.mask.unsqueeze(-1).float())
        self.register_buffer('zeros', torch.zeros(1))

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x_flat = x.view(self.T, -1)
        p1 = torch.sigmoid(self.threshold - 0.5 * (self.tau * x_flat + torch.matmul(self.W, x_flat)))
        log_cumsum = torch.cat([self.zeros.expand(1, x_flat.shape[1]), torch.cumsum(torch.log(torch.clamp(p1, min=1e-12)), dim=0)])
        reset1 = torch.exp((log_cumsum[self.i_grid] - log_cumsum[self.j_grid]) * self.mask2)
        eff_input = torch.einsum('ijc,jc->ic', self.tau * reset1 * self.coefficients.expand(-1,-1,x_flat.shape[1]), x_flat)
        return self.surrogate_function(eff_input - self.threshold).reshape(original_shape)

# --- Attention & Transformer ---
class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.q_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = pa_lif_n(tau=0.5)
        self.k_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = pa_lif_n(tau=0.5)
        self.attn_lif = pa_lif_n(tau=0.5, threshold=0.5)
        self.proj_conv = nn.Conv1d(dim, dim, 1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = pa_lif_n(tau=0.5)
        self.mem_alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_in = self.proj_lif(x).flatten(3).flatten(0, 1) # (TB, C, N)
        T_dim, _, N = x.shape[0], C, H*W

        q = self.q_lif(self.q_bn(self.q_conv(x_in)).reshape(T, B, C, N)).unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        k = self.k_lif(self.k_bn(self.k_conv(x_in)).reshape(T, B, C, N)).unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        indices = torch.arange(T, device=x.device).unsqueeze(1)
        Wei = torch.tril(self.mem_alpha * ((1 - self.mem_alpha) ** (indices - indices.T).clamp(min=0)))
        mems = torch.matmul(Wei, q.view(T, -1)).view(T, B, self.num_heads, C // self.num_heads, N)
        
        attn = self.attn_lif(torch.sum(torch.cat([mems, q], 3), dim=3, keepdim=True))
        x_out = torch.mul(attn, k).flatten(2, 3)
        return self.proj_bn(self.proj_conv(x_out.flatten(0, 1))).reshape(T, B, C, H, W)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden, 1); self.bn1 = nn.BatchNorm2d(hidden); self.lif1 = pa_lif_n()
        self.fc2 = nn.Conv2d(hidden, in_features, 1); self.bn2 = nn.BatchNorm2d(in_features); self.lif2 = pa_lif_n()
        self.hidden = hidden
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn1(self.fc1(self.lif1(x).flatten(0, 1))).reshape(T, B, self.hidden, H, W)
        return self.bn2(self.fc2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)

class QKFormer(nn.Module):
    def __init__(self, in_channels=155, num_classes=2, embed_dims=64, num_heads=4, mlp_ratios=4, depths=2, T=4):
        super().__init__()
        self.T = T
        self.proj = nn.Conv2d(in_channels, embed_dims, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(embed_dims); self.lif = pa_lif_n()
        self.layers = nn.ModuleList([nn.Sequential(Token_QK_Attention(embed_dims, num_heads), MLP(embed_dims, int(embed_dims*mlp_ratios))) for _ in range(depths)])
        self.diff_conv = nn.Conv2d(embed_dims, 32, 3, 1, 1, bias=False); self.diff_bn = nn.BatchNorm2d(32); self.diff_lif = pa_lif_n()
        self.head = nn.Linear(32, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02); 
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # x1, x2: (B, L, C) -> (B, C, H, W)
        B, L, C = x1.shape; H = int(L**0.5)
        x1, x2 = x1.transpose(1, 2).reshape(B, C, H, H), x2.transpose(1, 2).reshape(B, C, H, H)
        x = torch.cat([x1.unsqueeze(0).repeat(self.T//2, 1, 1, 1, 1), x2.unsqueeze(0).repeat(self.T//2, 1, 1, 1, 1)], dim=0)
        
        # Embed
        x = self.bn(self.proj(self.lif(x).flatten(0, 1))).reshape(self.T, B, -1, H, H)
        # Transform (Simple Additive loop for demo)
        for blk in self.layers: x = x + blk[1](x + blk[0](x))
        # SDM
        x = self.diff_bn(self.diff_conv(self.diff_lif(x - torch.roll(x, shifts=1, dims=0)).flatten(0, 1))).reshape(self.T, B, -1, H, H)
        return self.head(x.flatten(3).mean(3).mean(0))
