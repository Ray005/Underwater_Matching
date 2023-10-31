# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_

from escnn import group
from escnn import gspaces
import escnn

from bisect import bisect_left, bisect

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

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
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gateSO2cnn=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gateSO2cnn = use_gateSO2cnn
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.use_gateSO2cnn:
            self.gating_param = nn.Parameter(torch.ones(1)) # 创建门控参数
            self.SO2En_Decoder = SO2En_Decoder_layer()
        
    def forward(self, x, x_SO2_rep, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_gateSO2cnn: # 如果使用SO2等变，则再跑一次前向
            x_SO2tensor, x_SO2_rep = self.SO2En_Decoder(x_SO2_rep)
            # 融合ViT与cnn特征
#             print("x_SO2tensor.shape", x_SO2tensor.shape)
            gating = self.gating_param.view(-1,1,1)
            x[:,1:,:] *= (1.-torch.sigmoid(gating)) # * x[:,1:,:] 
            x[:,1:,:] += torch.sigmoid(gating) * x_SO2tensor.transpose(1,2) # x中排除cls token
        
        return x, x_SO2_rep

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
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Rotation_Eq_Transformer_DinoBase(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=1, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 local_up_to_layer=3, drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
#         print("local_up_to_layer = ", local_up_to_layer)
#         assert 0
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.local_up_to_layer = local_up_to_layer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if local_up_to_layer != 0:
            self.SO2patch_embed = SO2PatchEmbed()
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
        self.head = nn.Identity()
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        if self.local_up_to_layer != 0:
            x_SO2 = self.SO2patch_embed(x) # 此处的x为SO2表达
        else:
            x_SO2 = None
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x), x_SO2

    def forward(self, x, _):
        x, x_SO2 = self.prepare_tokens(x)
        for blk in self.blocks:
            x, x_SO2 = blk(x, x_SO2)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def GRET_DB_NO1(patch_size=16, **kwargs):
    model = Rotation_Eq_Transformer_DinoBase(
        patch_size=patch_size, embed_dim=195, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# 【脚本】测试脚本
# model = GRET_DB_NO1(local_up_to_layer=0)
# data = torch.randn([2,1,224,224])
# out = model(data)
# print("out.shape", out.shape)

# 【GRET】【ELE】加上高程编码
class Rotation_Eq_TFM_EleEmbed_DinoBase(nn.Module):
    """ Transformer """
    def __init__(self, num_ele_slice, per_li, num_ele_slice_array=10, img_size=[224], patch_size=16, in_chans=1, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 local_up_to_layer=3, drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
#         print("local_up_to_layer = ", local_up_to_layer)
#         assert 0
        self.per_li = per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.local_up_to_layer = local_up_to_layer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.num_ele_slice = num_ele_slice
        self.num_ele_slice_ori = num_ele_slice_array # 表示内部的高程嵌入矩阵尺寸
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice_array, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        if local_up_to_layer != 0:
            self.SO2patch_embed = SO2PatchEmbed()
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
        self.head = nn.Identity()
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.eleva_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_ele_encoding_inter(self, x, x_g_norm):
        # 【EEnT】计算高程编码（list）。放入list
        eleva_embed_batch = torch.zeros([x_g_norm.shape[0], x.shape[1], x.shape[2]], device=torch.device('cuda:0')) # .cuda() 【B，197，192】还有一个cls token
        # 【EEnT】
        Ele_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】计算高程编码（list）。放入list
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
            Ele_idx_li.append(idx)
            
        # 【插值】插值为 10000份
        ele_embed_inter = torch.nn.functional.interpolate(
            self.eleva_embed.permute(1,0).unsqueeze(0),
            size=self.num_ele_slice,
            mode='linear',
        )
        ele_embed_inter = ele_embed_inter.permute(0,2,1).squeeze()
        
        # 针对每个真实batch进行操作
        for batch_num in range(batch_numTrue): 
            eleva_embed_batch[batch_num, :, :] += ele_embed_inter[Ele_idx_li[batch_num],:].repeat(x.shape[1],1) # repaet的含义是，所有token加上同一个编码值
        # 重复到每个crops
        eleva_embed_batch = eleva_embed_batch.repeat(x.shape[0]//x_g_norm.shape[0], 1, 1)
        return eleva_embed_batch
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        if self.local_up_to_layer != 0:
            x_SO2 = self.SO2patch_embed(x) # 此处的x为SO2表达
        else:
            x_SO2 = None
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】add ELevation encoding to each token
        x = x + self.get_ele_encoding_inter(x, x_g_norm)

        return self.pos_drop(x), x_SO2

    def forward(self, x, x_g_norm):
        x, x_SO2 = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x, x_SO2 = blk(x, x_SO2)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_map_idx(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率。
        return idx

def GRET_ELE_DB_NO2(patch_size=16, **kwargs):
    model = Rotation_Eq_TFM_EleEmbed_DinoBase(
        patch_size=patch_size, embed_dim=195, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

