# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from escnn import group
from escnn import gspaces
import escnn

from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model

from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time



def drop_path(x, drop_prob: float = 0., training: bool = False): # form DINO ViT
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module): # form DINO ViT
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N**.5)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        distances = indd**.5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N
        
        if return_map:
            return dist, attn_map
        else:
            return dist

            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gateSO2cnn=True, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gateSO2cnn = use_gateSO2cnn
        self.num_heads = num_heads
        if self.use_gateSO2cnn:
            self.gating_param = nn.Parameter(torch.ones(1)) # 创建门控参数
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
            self.SO2En_Decoder = SO2En_Decoder_layer()
            
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        
    def forward(self, x, x_SO2_rep):
#         print("x.shape = ", x.shape)
#         print("self.norm1(x).shape", self.norm1(x).shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_gateSO2cnn: # 如果使用SO2等变，则再跑一次前向
            x_SO2tensor, x_SO2_rep = self.SO2En_Decoder(x_SO2_rep)
            # 融合ViT与cnn特征
#             print("x_SO2tensor.shape", x_SO2tensor.shape)
            gating = self.gating_param.view(-1,1,1)
            x[:,1:,:] *= (1.-torch.sigmoid(gating)) # * x[:,1:,:] 
            x[:,1:,:] += torch.sigmoid(gating) * x_SO2tensor.transpose(1,2) # x中排除cls token
        return x, x_SO2_rep
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
class SO2PatchEmbed(torch.nn.Module):
    def __init__(self):
        super(SO2PatchEmbed, self).__init__()
        # the model is equivariant under all planar rotations
        self.r2_act = gspaces.rot2dOnR2(N=-1)
        # the group SO(2)
        self.G: SO2 = self.r2_act.fibergroup
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = escnn.nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        activation1 = escnn.nn.FourierELU(self.r2_act, 15, irreps=self.G.bl_irreps(6), N=16, inplace=True) # 前一个份数乘以后面的（irreps(3)*2+1）
        out_type = activation1.in_type
        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=16, stride=16, padding=0, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation1,
        )

    def forward(self, input: torch.Tensor):
        x = self.input_type(input)
        x = self.block1(x)
        return x

class SO2En_Decoder_layer(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(SO2En_Decoder_layer, self).__init__()
        # the model is equivariant under all planar rotations
        self.r2_act = gspaces.rot2dOnR2(N=-1)
        # the group SO(2)
        self.G: SO2 = self.r2_act.fibergroup
        # the input image is a scalar field, corresponding to the trivial representation
        act_temp = escnn.nn.FourierELU(self.r2_act, 15, irreps=self.G.bl_irreps(6), N=16, inplace=True) # 前一个份数乘以后面的（irreps(3)*2+1）。用于调整输入尺寸，需要与SO2Embed的输出一致
        in_type = act_temp.in_type
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        # transpose Conv
        activation_Trans1 = escnn.nn.FourierELU(self.r2_act, 8, irreps=self.G.bl_irreps(6), N=16, inplace=True)
        out_type = activation_Trans1.in_type
        self.Trans_block1 = escnn.nn.SequentialModule(
            escnn.nn.R2ConvTransposed(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation_Trans1
        )
        in_type = self.Trans_block1.out_type
        activation1 = escnn.nn.FourierELU(self.r2_act, 15, irreps=self.G.bl_irreps(6), N=16, inplace=True)
        out_type = activation1.in_type
        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=3, stride=2, padding=1, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation1,)
        # number of output invariant chaescnn.nnels
        c = 195  # 153664 # 64
        # last 1x1 convolution layer, which maps the regular fields to c=195 invariant scalar fields
        # this is essential to provide *invariant* features in the final classification layer
        output_invariant_type = escnn.nn.FieldType(self.r2_act, c*[self.r2_act.trivial_repr])
        self.invariant_map = escnn.nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

    def forward(self, input: escnn.nn.geometric_tensor.GeometricTensor):
        x = self.Trans_block1(input)
        x = self.block1(x)
        x = x + input     # 同尺度残差连接
#         x = self.pool1(x)
        x_neck = self.invariant_map(x)
        x_neck = x_neck.tensor
        x_neck = x_neck.flatten(start_dim=2,end_dim=3) 
        return x_neck, x # 返回第一个为旋转群等变的向量，x为保持群等变性质的向量
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.apply(self._init_weights)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
class Rotation_Eq_Transformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=0, embed_dim=195, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=None,
                 local_up_to_layer=3, locality_strength=1., use_pos_embed=True):
        super().__init__()
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

# #         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(
#                 hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
# #         else:
#         self.patch_embed = PatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.SO2patch_embed = SO2PatchEmbed()
        
        num_patches = 196 # self.patch_embed.num_patches
        self.num_patches = num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gateSO2cnn=True,)
#                 locality_strength=locality_strength
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gateSO2cnn=False)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.SO2patch_embed(x) # 此处的x为SO2表达
        x_SO2 = x # .clone() # 拷贝一份
        x = x.tensor.clone() # 转为普通tensor
        x = x.flatten(start_dim=2,end_dim=3).transpose(1,2) # 打平
#         print("x.shape = ", x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
#         print("x.shape = ", x.shape)
#         print("x_SO2.shape = ", x_SO2.shape)
        for u, blk in enumerate(self.blocks):
#             print("Ran Block no:",u)
            x, x_SO2 = blk(x, x_SO2)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, _):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def GRET_NO1( **kwargs):
    model = Rotation_Eq_Transformer()
    return model

class Rotation_Eq_Transformer_NO2(Rotation_Eq_Transformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=0, embed_dim=195, depth=12,
             num_heads=3, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
             drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
             local_up_to_layer=3, locality_strength=1., use_pos_embed=True): # 【注意】这些参数不会被super().__init__()调用，仅供在init函数内使用
        super().__init__()
#         del self.
        self.vanilla_patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
    def forward_features(self, x):
        B = x.shape[0]
        x_SO2 = self.SO2patch_embed(x) # 此处的x为SO2表达
#         x_SO2 = x # .clone() # 拷贝一份
#         x = x.tensor.clone() # 转为普通tensor
        x = self.vanilla_patch_embed(x)
#         x = x.flatten(start_dim=2,end_dim=3).transpose(1,2) # 打平
#         print("x.shape = ", x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
#         print("x.shape = ", x.shape)
#         print("x_SO2.shape = ", x_SO2.shape)
        for u, blk in enumerate(self.blocks):
#             print("Ran Block no:",u)
            x, x_SO2 = blk(x, x_SO2)

        x = self.norm(x)
        return x[:, 0] 

def GRET_NO2_NormalPatchEm( **kwargs):
    model = Rotation_Eq_Transformer_NO2(local_up_to_layer=0)
    return model

















