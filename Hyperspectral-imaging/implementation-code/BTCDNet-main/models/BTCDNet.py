import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time
from fvcore.nn import FlopCountAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
            
    def forward(self, x):
        return x + self.conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class TileAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        if refine:
            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)                      
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh,ww)           
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
        
        if self.refine:
            stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


class TileAttentionLayer(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.pos_embed = ResDWC(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        self.attn = TileAttention(dim, stoken_size=stoken_size, 
                                    n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self, x):
        x = self.pos_embed(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))        
        return x
    
class BasicLayer(nn.Module):        
    def __init__(self, num_layers, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5, downsample=False):
        super().__init__()        
                
        self.blocks = nn.ModuleList([TileAttentionLayer(
                                           dim=dim[0],  n_iter=n_iter, stoken_size=stoken_size,                                          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,act_layer=act_layer, layerscale=layerscale, init_values=init_values) for i in range(num_layers)])

        if downsample:            
            self.downsample = PatchMerging(dim[0], dim[1])
        else:
            self.downsample = None
         
    def forward(self, x):
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
       
class PatchEmbed(nn.Module):        
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()

        if downsample:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.GELU(),
                nn.BatchNorm2d(out_channels),
                
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.GELU(),
                nn.BatchNorm2d(out_channels),      
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.GELU(),
                nn.BatchNorm2d(out_channels),
                
                nn.Conv2d(out_channels, out_channels, 1, 1, 0),
                nn.GELU(),
                nn.BatchNorm2d(out_channels),      
            )

    def forward(self, x):
        x = self.proj(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    
class BTCDNet(nn.Module):   
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 n_iter=[1, 1, 1, 1], stoken_size=[8, 4, 2, 1],                
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 projection=None, freeze_bn=False,
                 layerscale=[False, False, False, False], init_values=1e-6, temperature=0.5, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim        
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.freeze_bn = freeze_bn
        self.temperature = temperature

        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.f_prob = nn.ModuleList()
        self.f_prob.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim[0], num_classes - 1), 
            nn.Softmax(dim=1)                       
        ))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
                
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer], embed_dim[i_layer+1] if i_layer<self.num_layers-1 else None],                              
                               n_iter=n_iter[i_layer],
                               stoken_size=to_2tuple(stoken_size[i_layer]),                                                       
                               num_heads=num_heads[i_layer], 
                               mlp_ratio=self.mlp_ratio, 
                               qkv_bias=qkv_bias, qk_scale=qk_scale, 
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=i_layer < self.num_layers - 1,                           
                               layerscale=layerscale[i_layer],
                               init_values=init_values)
            self.layers.append(layer)
            self.f_prob.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim[i_layer] * 2 if i_layer < self.num_layers - 1 else embed_dim[i_layer], num_classes - 1), 
            nn.Softmax(dim=1)                       
        ))
    
        self.proj = nn.Conv2d(self.num_features, projection, 1) if projection else None
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish() 
        self.f_prob.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(projection, num_classes - 1), 
            nn.Softmax(dim=1)                       
        ))       
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.f_prob.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(projection or self.num_features, num_classes - 1), 
            nn.Softmax(dim=1)                       
        ))

        self.head = nn.Linear(projection or self.num_features, num_classes) if num_classes > 0 else nn.Identity()                              

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
       

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def get_center_values(self, batch_tensor):
        # 获取 p 的维度
        _, _, p, _ = batch_tensor.shape
        # 计算中心点的索引
        center_idx = p // 2
        # 取出每个 batch 中中心点的值
        center_values = batch_tensor[:, :, center_idx, center_idx]
        return center_values

    def compute_normalized_spectral_angles(self, tensor1, tensor2):
        # 计算每个向量的范数
        norms1 = torch.norm(tensor1, dim=1, keepdim=True)
        norms2 = torch.norm(tensor2, dim=1, keepdim=True)
        
        # 避免除零错误：确保不为零
        normalized_tensor1 = tensor1 / (norms1 + 1e-8)
        normalized_tensor2 = tensor2 / (norms2 + 1e-8)
        
        # 计算每一对向量的余弦相似度
        cosine_similarity = torch.sum(normalized_tensor1 * normalized_tensor2, dim=1, keepdim=True)
        
        # 计算光谱角（弧度），并使用 acos 将余弦相似度转换为角度
        spectral_angles = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
        
        # 将光谱角归一化为差异概率（0到1的范围）
        difference_probability = spectral_angles / torch.pi
        
        # 计算相似概率为 1 - 差异概率
        similarity_probability = 1 - difference_probability
        
        # 合并差异概率和相似概率，得到 [batch, 2]
        probabilities = torch.cat([difference_probability, similarity_probability], dim=1)
        
        return probabilities

    def compute_posterior(self, prior_probs, likelihood_probs):
        # 计算未标准化的后验概率
        unnormalized_posterior = prior_probs * likelihood_probs
        
        # 计算分母，即所有类别的概率之和
        evidence = unnormalized_posterior.sum(dim=1, keepdim=True)
        
        # 计算每个类别的后验概率
        posterior_probs = unnormalized_posterior / evidence
        return posterior_probs

    def forward(self, x1, x2):
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        bayesian_prob_step = 0
        f_prob_likelyhood_list = []
        f_prob_posterior_list = []
        f = []
        # print("origin:", x1.shape)

        x1_center = self.get_center_values(x1)
        x2_center = self.get_center_values(x2)
        # print("center:", x1_center.shape)

        sam_prob = self.compute_normalized_spectral_angles(x1_center, x2_center)
        f_prob_likelyhood = sam_prob
        f_prob_likelyhood_list.append(f_prob_likelyhood)
        f_prob_posterior = sam_prob
        f_prob_posterior_list.append(f_prob_posterior)
        # print("sam_prob:", sam_prob.shape)
        # print(sam_prob)

        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        # print("patch embedding:", x1.shape)

        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        # print("patch dropout:", x1.shape)

        f.append(x1+x2)
        f_prob_likelyhood = self.f_prob[bayesian_prob_step](x1+x2)
        f_prob_likelyhood_list.append(f_prob_likelyhood)
        f_prob_posterior = self.compute_posterior(f_prob_posterior, f_prob_likelyhood)
        f_prob_posterior_list.append(f_prob_posterior)
        bayesian_prob_step = bayesian_prob_step + 1

        for layer in self.layers:            
            x1 = layer(x1)
            x2 = layer(x2)
            f.append(x1+x2)
            # print("layer:",x1.shape)
            f_prob_likelyhood = self.f_prob[bayesian_prob_step](x1+x2)
            f_prob_likelyhood_list.append(f_prob_likelyhood)
            f_prob_posterior = self.compute_posterior(f_prob_posterior, f_prob_likelyhood)
            f_prob_posterior_list.append(f_prob_posterior)
            bayesian_prob_step = bayesian_prob_step + 1
        
        x1 = self.proj(x1)
        x2 = self.proj(x2)
        # print("project:", x1.shape)

        x1 = self.norm(x1)
        x2 = self.norm(x2)
        # print("layernorm:", x1.shape)

        x1 = self.swish(x1)
        x2 = self.swish(x2)
        # print("swish:", x1.shape)

        f.append(x1+x2)
        f_prob_likelyhood = self.f_prob[bayesian_prob_step](x1+x2)
        f_prob_likelyhood_list.append(f_prob_likelyhood)
        f_prob_posterior = self.compute_posterior(f_prob_posterior, f_prob_likelyhood)
        f_prob_posterior_list.append(f_prob_posterior)
        bayesian_prob_step = bayesian_prob_step + 1
        
        x1 = self.avgpool(x1).flatten(1)  # B C 1  
        x2 = self.avgpool(x2).flatten(1)  # B C 1   
        # print("avgpool:", x1.shape)  

        f.append(x1+x2)
        f_prob_likelyhood = self.f_prob[bayesian_prob_step](x1+x2)
        f_prob_likelyhood_list.append(f_prob_likelyhood)
        f_prob_posterior = self.compute_posterior(f_prob_posterior, f_prob_likelyhood)
        f_prob_posterior_list.append(f_prob_posterior)

        # print("likelyhood:", f_prob_likelyhood_list)
        # print("posterior:", f_prob_posterior_list)

        x = self.head(x1+x2)

        x_sub = x[:, 1:]
        x_posterior = x_sub * f_prob_posterior / self.temperature
        x = torch.cat((x[:, :1], x_posterior), dim=1)

        # print("output:", x.shape)  

        return x
    
if __name__ == "__main__":
    # 创建模型实例，定义输入参数

    patches = 5
    band_patches = 3
    num_classes = 3
    band = 166

    # 测试推理时间
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BTCDNet(in_chans=166, num_classes=3, embed_dim=[96, 192], depths=[2, 2], num_heads=[3, 6], n_iter=[1, 1], stoken_size=[2, 1], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, projection=512, freeze_bn=False, layerscale=[False, False, False, False], init_values=1e-6).to(device)

    # 打印模型结构（可选）
    print(model)

    # 随机生成一个输入，假设输入尺寸为 (B, C, H, W) = (1, 3, 224, 224)
    input1 = torch.randn(2, 5, 5, 166).to(device)
    input2 = torch.randn(2, 5, 5, 166).to(device)

    # 运行模型并打印输出形状
    output = model(input1, input2)
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