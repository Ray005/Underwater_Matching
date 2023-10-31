from torch.utils.data import Dataset
import os
import numpy as np
import math

import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_
from vision_transformer import DropPath, drop_path, Mlp, Attention, Block, PatchEmbed, DINOHead
import sys
import utils

from bisect import bisect_left, bisect

from SelfAttention import ScaledDotProductAttention

class TerrainDataset_for_per_li(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = os.listdir(self.data_folder)
#         self.transform = transform
#         self.global_max = global_max
#         self.global_min = global_min

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.file_list[index].endswith('.npy'):
             # 使用NumPy库读取网格化地形数据
            data = np.load(os.path.join(self.data_folder, self.file_list[index]))
#             return np.min(data), np.max(data)
            return np.min(data), np.max(data), data.mean()
        else:
            return None

def get_dataset_per(dataset_path, num_slice):
    print("获取数据集分位数……")
    min_global = math.inf
    max_global = -math.inf
    dataset_for_min_max = TerrainDataset_for_per_li(dataset_path)
    mean_li = []
    for i in range(len(dataset_for_min_max)):
        min_temp, max_temp, mean_temp = dataset_for_min_max.__getitem__(i)
        mean_li.append(mean_temp)
        if min_temp < min_global:
            min_global = min_temp
        if max_temp > max_global:
            max_global = max_temp
    per_li = []
    # for i in range(args.num_ele_slice):
    #     per_test = np.percentile(z, 100 / args.num_ele_slice * (i + 1))
    #     per_test = (per_test - global_min) / (global_max - global_min)
    #     per_li.append(per_test)
    per_li = np.percentile(mean_li, np.linspace(0,100, num_slice))
    per_li = (per_li - min_global) / (max_global - min_global)
    per_li[-1] = 1
#     print("已读取分位数列表：per_li， 份数" + str(args.num_ele_slice) )
    print("数据集最大最小值为" + str((min_global, max_global)))
    return min_global, max_global, per_li


class ElevationTransformer_NO2(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】
        Ele_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】计算高程编码（list）。放入list
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
            Ele_idx_li.append(idx)
#         print("x.shape[0] = " + str(x.shape[0]))
#         print("idx = " + str(Ele_idx_li))
#         print("batch_numTrue = " + str(batch_numTrue))
#         assert 0 
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                x[(crops_num*batch_numTrue+batch_num), :, :] += self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1) # repaet的含义是，所有token加上同一个编码值
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率。
#         for i, per in enumerate(self.per_li):
#             if ele_mean < per:
#                 return i
#         else:
#             return None
        return idx


def EEnT_NO2_tiny(patch_size=16, **kwargs):
    model = ElevationTransformer_NO2(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def load_Pre_EEnT_NO2_tiny(args, per_li):
    if "EEnT_tiny_NO2" in args.arch:
        model = EEnT_NO2_tiny(num_ele_slice = args.num_ele_slice, per_li = per_li, in_chans=1)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    return model

# 加入自注意力机制加权的高程嵌入（无sift）
class ElevationTransformer_NO3_AT(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.SAT_embed = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】
        Ele_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】计算高程编码（list）。放入list
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
            Ele_idx_li.append(idx)
#         print("x.shape[0] = " + str(x.shape[0]))
#         print("idx = " + str(Ele_idx_li))
#         print("batch_numTrue = " + str(batch_numTrue))
#         assert 0 
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
                token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                # 提取这个batch的高程编码，并repeat为196*1*192
                eleEm_in_a_batch = self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                at_in = torch.cat((token_in_a_batch, eleEm_in_a_batch), dim=1) # 拼接为196*2*192，方便做注意力
#                 print(at_in.shape) # torch.Size([197, 2, 192])
                # 进行自注意力操作
                at_out = self.SAT_embed(at_in, at_in, at_in) # 注意力层前向
#                 print(at_out.shape) # torch.Size([197, 2, 192])
                token_in_a_batch, eleEm_in_a_batch = torch.chunk(at_out, 2, dim=1) # 将at_out拆分，再直接相加合成。将其拆分为token和eleEM
#                 print(x[(crops_num*batch_numTrue+batch_num), :, :].shape) # torch.Size([197, 192])
                x[(crops_num*batch_numTrue+batch_num), :, :] = token_in_a_batch.squeeze(1) + eleEm_in_a_batch.squeeze(1)
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率
#         for i, per in enumerate(self.per_li):
#             if ele_mean < per:
#                 return i
#         else:
#             return None
        return idx

def EEnT_NO3_AT(patch_size=16, **kwargs):
    model = ElevationTransformer_NO3_AT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# 加入自注意力机制加权的高程嵌入（无sift）【方差嵌入】【Abort】
class ElevationTransformer_NO4_Var_AT(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, per_li_var, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.per_li_var = per_li_var
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.eleva_var_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.SAT_embed = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.eleva_embed, std=.02)
        trunc_normal_(self.eleva_var_embed, std=.02)
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

    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】
        Ele_idx_li = []
        Ele_var_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】计算高程编码（list）。放入list
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
            Ele_idx_li.append(idx)
        for batch_num in range(batch_numTrue):
            idx = self.get_map_var_idx(x_g_norm[batch_num, :, :, :])
            Ele_var_idx_li.append(idx)
        
#         print("x.shape[0] = " + str(x.shape[0]))
#         print("idx = " + str(Ele_idx_li))
#         print("batch_numTrue = " + str(batch_numTrue))
#         assert 0 
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
                token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                # 提取这个batch的高程编码，并repeat为196*1*192
                eleEm_in_a_batch = self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                elevarEm_in_a_batch = self.eleva_var_embed[Ele_var_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                
                at_in = torch.cat((token_in_a_batch, eleEm_in_a_batch, elevarEm_in_a_batch), dim=1) # 拼接为196*2*192，方便做注意力
#                 print(at_in.shape) # torch.Size([197, 2, 192])
                # 进行自注意力操作
                at_out = self.SAT_embed(at_in, at_in, at_in) # 注意力层前向
#                 print(at_out.shape) # torch.Size([197, 2, 192])
                token_in_a_batch, eleEm_in_a_batch,  elevarEm_in_a_batch= torch.chunk(at_out, 3, dim=1) # 将at_out拆分，再直接相加合成。将其拆分为token和eleEM
#                 print(x[(crops_num*batch_numTrue+batch_num), :, :].shape) # torch.Size([197, 192])
                x[(crops_num*batch_numTrue+batch_num), :, :] = token_in_a_batch.squeeze(1) + eleEm_in_a_batch.squeeze(1) + elevarEm_in_a_batch.squeeze(1)
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率
#         for i, per in enumerate(self.per_li):
#             if ele_mean < per:
#                 return i
#         else:
#             return None
        return idx
    def get_map_var_idx(self, x_g_norm):
        ele_var = x_g_norm.var()
        idx = bisect_left(self.per_li_var, ele_var) # 二分查找提高效率
#         for i, per in enumerate(self.per_li):
#             if ele_mean < per:
#                 return i
#         else:
#             return None
        return idx

def EEnT_NO4_Var_AT(patch_size=16, **kwargs):
    model = ElevationTransformer_NO4_Var_AT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# 【EEnT】【高程嵌入】【方差嵌入】【注意力机制】无sift【batch内高程嵌入】
class ElevationTransformer_NO5_inBatchPer_AT(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li # 这一per_li是允许在外部改变的，在匹配中，这很重要。在训练时将使用in batch 分位数
        self.use_in_batch_per = True # 默认在训练中使用in batch 分位数。如果为false，则会使用传入的per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.SAT_embed = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】
        Ele_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】计算高程编码（list）。放入list
        if self.use_in_batch_per:
            per_li = self.get_per_li_in_batch(x_g_norm)  # 计算当前batch的per_li
            for batch_num in range(batch_numTrue):
                # 查找与append
                idx = bisect_left(per_li, x_g_norm[batch_num, :, :, :].mean())
                Ele_idx_li.append(idx)
        else: # 不使用in batch分位数时，使用外部给的per_li，注意，在将use_in_batch_per参数置为False时，一定要同时传入per_li
            for batch_num in range(batch_numTrue):
                idx = self.get_map_idx(self.per_li, x_g_norm[batch_num, :, :, :])
                Ele_idx_li.append(idx)
#         print("x.shape[0] = " + str(x.shape[0]))
#         print("idx = " + str(Ele_idx_li))
#         print("batch_numTrue = " + str(batch_numTrue))
#         assert 0 
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
                token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                # 提取这个batch的高程编码，并repeat为196*1*192
                eleEm_in_a_batch = self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                at_in = torch.cat((token_in_a_batch, eleEm_in_a_batch), dim=1) # 拼接为196*2*192，方便做注意力
#                 print(at_in.shape) # torch.Size([197, 2, 192])
                # 进行自注意力操作
                at_out = self.SAT_embed(at_in, at_in, at_in) # 注意力层前向
#                 print(at_out.shape) # torch.Size([197, 2, 192])
                token_in_a_batch, eleEm_in_a_batch = torch.chunk(at_out, 2, dim=1) # 将at_out拆分，再直接相加合成。将其拆分为token和eleEM
#                 print(x[(crops_num*batch_numTrue+batch_num), :, :].shape) # torch.Size([197, 192])
                x[(crops_num*batch_numTrue+batch_num), :, :] = token_in_a_batch.squeeze(1) + eleEm_in_a_batch.squeeze(1)
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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

    def get_map_idx(self, per_li, x_g_norm): # 加入pooling计算均值，提高效率
        batch_numTrue = x_g_norm.shape[0]
        # 【TODO】加入pooling计算均值，提高计算效率
        ele_mean = x_g_norm.mean()
        idx = bisect_left(per_li, ele_mean) # 二分查找提高效率
        return idx
    
    def get_per_li_in_batch(self, x_g_norm):
        mean_li=[]
        batch_numTrue = x_g_norm.shape[0]
        for batch_num in range(batch_numTrue): # 池化后求平均似乎没那么快
            mean_li.append(x_g_norm[batch_num, :, :, :].mean().detach().cpu())
        per_li = np.percentile(mean_li, np.linspace(0,99,99)) # 提高分位数获取效率
        return per_li
    
def EEnT_NO5_iBP_AT(patch_size=16, **kwargs):
    model = ElevationTransformer_NO5_inBatchPer_AT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# 【EEnT】【无极高程嵌入】加入自注意力机制加权的高程嵌入（无sift）
class ElevationTransformer_NO6_contiEm_AT(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, g_max, g_min, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.g_max = g_max
        self.g_min = g_min
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.eleva_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.SAT_embed = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】计算高程编码（list）。放入list
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        Ele_idx_li = torch.zeros([batch_numTrue])
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx_inter_norm(x_g_norm[batch_num, :, :, :])
            Ele_idx_li[batch_num] = idx
#         Ele_idx_li = Ele_idx_li / self.num_ele_slice # 将编号归一化
#         print("x.shape[0] = " + str(x.shape[0]))
#         print("idx = " + str(Ele_idx_li))
#         print("batch_numTrue = " + str(batch_numTrue))
#         assert 0 
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
                token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                # 提取这个batch的高程编码，并repeat为196*1*192
#                 eleEm_in_a_batch = self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                ele_weight = Ele_idx_li[batch_num]
#                 print("ele_weight", ele_weight)
                eleEm_in_a_batch = (self.eleva_embed * ele_weight).repeat(x.shape[1],1).unsqueeze(1) # 【无极嵌入】
                at_in = torch.cat((token_in_a_batch, eleEm_in_a_batch), dim=1) # 拼接为196*2*192，方便做注意力
#                 print(at_in.shape) # torch.Size([197, 2, 192])
                # 进行自注意力操作
                at_out = self.SAT_embed(at_in, at_in, at_in) # 注意力层前向
#                 print(at_out.shape) # torch.Size([197, 2, 192])
                token_in_a_batch, eleEm_in_a_batch = torch.chunk(at_out, 2, dim=1) # 将at_out拆分，再直接相加合成。将其拆分为token和eleEM
#                 print(x[(crops_num*batch_numTrue+batch_num), :, :].shape) # torch.Size([197, 192])
                x[(crops_num*batch_numTrue+batch_num), :, :] = token_in_a_batch.squeeze(1) + eleEm_in_a_batch.squeeze(1)
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx_inter_norm(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率
        if idx == self.num_ele_slice-1: # 如果是头上的话就不用插值
            return idx / self.num_ele_slice 
        elif idx == 0:
            idx_inter = idx - 1/(self.per_li[idx] - 0)*(self.per_li[idx]-x_g_norm.mean())
            return idx_inter / self.num_ele_slice 
        else: # 执行插值操作
            idx_inter = idx - 1/(self.per_li[idx] - self.per_li[idx-1])*(self.per_li[idx]-x_g_norm.mean()) # 计算区间长度，即为idx为1时的增量
            return idx_inter / self.num_ele_slice 
        if idx >= self.num_ele_slice:
            assert 0, "高程编码索引计算错误"
        return None        # 如果非上述条件，则返回None

def EEnT_NO6_contiEm_AT(patch_size=16, **kwargs):
    model = ElevationTransformer_NO6_contiEm_AT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# ==========================================
# 【EEnT】【无极高程嵌入V2】加入自注意力机制加权的高程嵌入（无sift）
# 使用非线性层输入到高程编码中
# ==========================================
class ElevationTransformer_NO7_contiEmV2_AT(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, g_max, g_min, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.g_max = g_max
        self.g_min = g_min
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
#         self.eleva_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.SAT_embed = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
#         trunc_normal_(self.eleva_embed, std=.02)
        self.apply(self._init_weights)
        
        # 引入一个非线性层用于高程嵌入
        self.ele_porj = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 192),
            nn.LeakyReLU(inplace=True),
        )

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

    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】计算高程编码（list）。放入list
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        Ele_idx_li = torch.zeros([batch_numTrue])
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx_inter_norm(x_g_norm[batch_num, :, :, :])
            Ele_idx_li[batch_num] = idx
#         Ele_idx_li = Ele_idx_li / self.num_ele_slice # 将编号归一化
#         print("x.shape[0] = " + str(x.shape[0]))
#         print("idx = " + str(Ele_idx_li))
#         print("batch_numTrue = " + str(batch_numTrue))
#         assert 0 
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
                token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                # 提取这个batch的高程编码，并repeat为196*1*192
#                 eleEm_in_a_batch = self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                ele_weight = Ele_idx_li[batch_num]
#                 print("ele_weight", ele_weight)
#                 eleEm_in_a_batch = (self.eleva_embed * ele_weight).repeat(x.shape[1],1).unsqueeze(1) # 【无极嵌入】
                # 【无极嵌入与非线性层】
#                 print(torch.tensor(ele_weight).unsqueeze(0).unsqueeze(0).shape)
                eleEm_in_a_batch = self.ele_porj(ele_weight.unsqueeze(0).unsqueeze(0).cuda()).repeat(x.shape[1],1).unsqueeze(1)
#                 print(eleEm_in_a_batch.shape)
                at_in = torch.cat((token_in_a_batch, eleEm_in_a_batch), dim=1) # 拼接为196*2*192，方便做注意力
#                 print(at_in.shape) # torch.Size([197, 2, 192])
                # 进行自注意力操作
                at_out = self.SAT_embed(at_in, at_in, at_in) # 注意力层前向
#                 print(at_out.shape) # torch.Size([197, 2, 192])
                token_in_a_batch, eleEm_in_a_batch = torch.chunk(at_out, 2, dim=1) # 将at_out拆分，再直接相加合成。将其拆分为token和eleEM
#                 print(x[(crops_num*batch_numTrue+batch_num), :, :].shape) # torch.Size([197, 192])
                x[(crops_num*batch_numTrue+batch_num), :, :] = token_in_a_batch.squeeze(1) + eleEm_in_a_batch.squeeze(1)
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx_inter_norm(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率
        if idx == self.num_ele_slice-1: # 如果是头上的话就不用插值
            return idx / self.num_ele_slice 
        elif idx == 0:
            idx_inter = idx - 1/(self.per_li[idx] - 0)*(self.per_li[idx]-x_g_norm.mean())
            return idx_inter / self.num_ele_slice 
        else: # 执行插值操作
            idx_inter = idx - 1/(self.per_li[idx] - self.per_li[idx-1])*(self.per_li[idx]-x_g_norm.mean()) # 计算区间长度，即为idx为1时的增量
            return idx_inter / self.num_ele_slice 
        if idx >= self.num_ele_slice:
            assert 0, "高程编码索引计算错误"
        return None        # 如果非上述条件，则返回None

def EEnT_NO7_contiEmV2_AT(patch_size=16, **kwargs):
    model = ElevationTransformer_NO7_contiEmV2_AT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ==========================================
# 【EEnT】【无极高程嵌入V3】加入自注意力机制加权的高程嵌入（无sift）
# 使用非线性层输入到高程编码中
# 【token】将高程作为一个token嵌入
# ==========================================
class ElevationTransformer_NO8_contiEmV3_AT(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, g_max, g_min, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.g_max = g_max
        self.g_min = g_min
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
#         self.eleva_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
#         self.SAT_embed = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
#         trunc_normal_(self.eleva_embed, std=.02)
        self.apply(self._init_weights)
        
        # 引入一个非线性层用于高程嵌入
        self.ele_porj = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 192),
            nn.LeakyReLU(inplace=True),
        )

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

    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】计算高程编码（list）。放入list
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        Ele_idx_li = torch.zeros([batch_numTrue])
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx_inter_norm(x_g_norm[batch_num, :, :, :])
            Ele_idx_li[batch_num] = idx
#         Ele_idx_li = Ele_idx_li / self.num_ele_slice # 将编号归一化
#         print("x.shape[0] = " + str(x.shape[0]))
#         print("idx = " + str(Ele_idx_li))
#         print("batch_numTrue = " + str(batch_numTrue))
#         assert 0 
        # 【EEnT】进行高程嵌入
        ele_tokens = torch.zeros(B, 1, self.embed_dim).cuda()
        x = torch.cat((x, ele_tokens), dim=1)
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
#                 token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                ele_weight = Ele_idx_li[batch_num]
                # 【无极嵌入与非线性层】
#                 print(torch.tensor(ele_weight).unsqueeze(0).unsqueeze(0).shape)
                
                x[(crops_num*batch_numTrue+batch_num), -1, :] = self.ele_porj(ele_weight.unsqueeze(0).unsqueeze(0).cuda()) # .repeat(x.shape[1],1).unsqueeze(1)
        
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx_inter_norm(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率
        if idx == self.num_ele_slice-1: # 如果是头上的话就不用插值
            return idx / self.num_ele_slice 
        elif idx == 0:
            idx_inter = idx - 1/(self.per_li[idx] - 0)*(self.per_li[idx]-x_g_norm.mean())
            return idx_inter / self.num_ele_slice 
        else: # 执行插值操作
            idx_inter = idx - 1/(self.per_li[idx] - self.per_li[idx-1])*(self.per_li[idx]-x_g_norm.mean()) # 计算区间长度，即为idx为1时的增量
            return idx_inter / self.num_ele_slice 
        if idx >= self.num_ele_slice:
            assert 0, "高程编码索引计算错误"
        return None        # 如果非上述条件，则返回None

def EEnT_NO8_contiEmV3_AT(patch_size=16, **kwargs):
    model = ElevationTransformer_NO8_contiEmV3_AT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# ==========================================
# 【EEnT】【Patch加高程嵌入】返璞归真。回到原来的基本思路，通过微调结构：
# 三个思路：对小patch单独计算高程嵌入、双线性插值、分位数加权嵌入
# 首先进行每个patch的高程嵌入
# ==========================================
class ElevationTransformer_NO9_EleOnPatch(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.ele_avgpooling = torch.nn.AvgPool2d( kernel_size=patch_size , stride=patch_size , padding=0 )
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
    
    def get_ele_encoding(self, x, x_g_norm):
        # 【EEnT】计算高程编码（list）。放入list
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】进行高程嵌入
        ele_em_avg = self.ele_avgpooling(x_g_norm)
        ele_em_avg = ele_em_avg.flatten(2).transpose(1, 2) # dim=【B，196，1】
        
        eleva_embed = torch.zeros([batch_numTrue, x.shape[1], x.shape[2]], device=torch.device('cuda:0')) # .cuda() 【B，197，192】还有一个cls token
        # 索引为【B, 196, 192】
        for i in range(ele_em_avg.shape[0]): # 遍历每个batch
            for j in range(ele_em_avg.shape[1]): # 遍历每个patch
                idx = bisect_left(self.per_li, ele_em_avg[i, j, 0])
                eleva_embed[i, j+1, :] = self.eleva_embed[idx, :] # 加1为了避开第0个位置上的cls token
        # 索引完后，在Batch维度重复就可以了
        eleva_embed = eleva_embed.repeat(x.shape[0] // x_g_norm.shape[0], 1 , 1) # 在第0个维度上，也就是batch上数量一致
        # Cat上第一个cls token的嵌入为0
        return eleva_embed
        
    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】add ELevation encoding to each token
        x = x + self.get_ele_encoding(x, x_g_norm)
        
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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

def EEnT_NO9_EleOnPatch(patch_size=16, **kwargs):
    model = ElevationTransformer_NO9_EleOnPatch(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# ==========================================
# 【EEnT】【Patch加高程嵌入】返璞归真。回到原来的基本思路，通过微调结构：
# 三个思路：对小patch单独计算高程嵌入、双线性插值、分位数加权嵌入
# 首先进行每个patch的高程嵌入【高程嵌入中pooling以提高速度】
# ==========================================
class ElevationTransformer_NO10_EleOnPatch_poolingEm(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.eleva_embed, std=.02)
        self.apply(self._init_weights)
        
        self.em_patchSideLen = 7
        self.patch_pooling_li = []
        for i in range(self.em_patchSideLen + 1):
            self.patch_pooling_li.append(img_size[0]//patch_size//self.em_patchSideLen*i) # img_size[0]//patch_size=14
        print("self.patch_pooling_li = ", self.patch_pooling_li)
        
        self.ele_avgpooling = torch.nn.AvgPool2d( kernel_size=224//self.em_patchSideLen , stride=224//self.em_patchSideLen , padding=0 )


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
    
    def get_ele_encoding(self, x, x_g_norm):
        # 【EEnT】计算高程编码（list）。放入list
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】进行高程嵌入
        ele_em_avg = self.ele_avgpooling(x_g_norm) # dim=[B, 1,  2, 2]or[B, 1, 7, 7]
#         print("ele_em_avg", ele_em_avg[0])
        eleva_embed = torch.zeros([batch_numTrue, x.shape[1], x.shape[2]], device=torch.device('cuda:0')) # .cuda() 【B，197，192】还有一个cls token
        
        # 索引为【B, 196, 192】
        idx_array = [[0]*self.em_patchSideLen]*self.em_patchSideLen # 2*2或 7*7，减少查找次数
        for i in range(batch_numTrue): # 遍历每个batch (50、60)
            # 对于每个batch，提前计算好pooling的idx
            for m in range(self.em_patchSideLen):
                for n in range(self.em_patchSideLen):
                    idx_array[m][n] = bisect_left(self.per_li, ele_em_avg[i, 0, m, n])
#             print("idx_array", idx_array)
            for j in range(eleva_embed.shape[1] - 1): # 遍历每个patch 196(197)
                idx_x = bisect(self.patch_pooling_li, j//14) - 1
                idx_y = bisect(self.patch_pooling_li, j% 14) - 1 
                idx = idx_array[idx_x][idx_y]
                eleva_embed[i, j+1, :] = self.eleva_embed[idx, :] # 加1为了避开第0个位置上的cls token
        # 索引完后，在Batch维度重复就可以了
        eleva_embed = eleva_embed.repeat(x.shape[0] // x_g_norm.shape[0], 1 , 1) # 在第0个维度上，也就是batch上数量一致
        # Cat上第一个cls token的嵌入为0
        return eleva_embed
        
    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】add ELevation encoding to each token
        x = x + self.get_ele_encoding(x, x_g_norm)
        
        return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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

def EEnT_NO10_EleOnPatch_poolingEm(patch_size=16, **kwargs):
    model = ElevationTransformer_NO10_EleOnPatch_poolingEm(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# ==========================================
# 【EEnT】【Patch加高程嵌入】返璞归真。回到原来的基本思路，通过微调结构：
# 三个思路：对小patch单独计算高程嵌入、线性插值、分位数加权嵌入
# 【第二】对高度编码线性插值
# 初步尝试加载100的模型，插值到10000进行测试
# ==========================================
class ElevationTransformer_NO11_interEleEncode(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, num_ele_slice_array=10, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.num_ele_slice_ori = num_ele_slice_array # 表示内部的高程嵌入矩阵尺寸
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice_array, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        
    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # 【EEnT】add ELevation encoding to each token
        x = x + self.get_ele_encoding_inter(x, x_g_norm)
        
        return self.pos_drop(x)

    
#     def prepare_tokens(self, x, x_g_norm):
#         B, nc, w, h = x.shape
#         x = self.patch_embed(x)  # patch linear embedding

#         # add the [CLS] token to the embed patch tokens
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # add positional encoding to each token
#         x = x + self.interpolate_pos_encoding(x, w, h)
        
#         # 【EEnT】
#         Ele_idx_li = []
#         batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
#         # 【EEnT】计算高程编码（list）。放入list
#         for batch_num in range(batch_numTrue):
#             idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
#             Ele_idx_li.append(idx)
# #         print("x.shape[0] = " + str(x.shape[0]))
# #         print("idx = " + str(Ele_idx_li))
# #         print("batch_numTrue = " + str(batch_numTrue))
# #         assert 0 
#         # 【EEnT】进行高程嵌入
#         for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
#             for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
#                 x[(crops_num*batch_numTrue+batch_num), :, :] += self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1) # repaet的含义是，所有token加上同一个编码值
#         return self.pos_drop(x)

    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率。
#         for i, per in enumerate(self.per_li):
#             if ele_mean < per:
#                 return i
#         else:
#             return None
        return idx


def EEnT_NO11_interEleEncode(patch_size=16, **kwargs):
    model = ElevationTransformer_NO11_interEleEncode(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ==========================================
# 【EEnT】【Patch加高程嵌入】返璞归真。回到原来的基本思路，通过微调结构：
# 三个思路：对小patch单独计算高程嵌入、线性插值、分位数加权嵌入
# 【第二】对高度编码线性插值
# 初步尝试加载100的模型，插值到10000进行测试
# ==========================================
class ElevationTransformer_NO12_interEle_AsToken(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, num_ele_slice_array=10, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.per_li = per_li
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.num_ele_slice = num_ele_slice
        self.num_ele_slice_ori = num_ele_slice_array # 表示内部的高程嵌入矩阵尺寸
        self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice_array, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        ele_pos_embed = self.pos_embed[:, 1]
        patch_pos_embed = self.pos_embed[:, 2:]
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
        return torch.cat((class_pos_embed.unsqueeze(0), ele_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    
    def get_ele_encoding_inter(self, x, x_g_norm):
        # 【EEnT】计算高程编码（list）。放入list
        eleva_embed_batch = torch.zeros([x_g_norm.shape[0], 1, x.shape[2]], device=torch.device('cuda:0')) # .cuda() 【B，197，192】还有一个cls token
        # 【EEnT】
        Ele_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】计算高程编码（list）。放入list
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
            Ele_idx_li.append(idx)
            
        # 【插值】插值为 10000份（指定的份数self.num_ele_slice）
        ele_embed_inter = torch.nn.functional.interpolate(
            self.eleva_embed.permute(1,0).unsqueeze(0),
            size=self.num_ele_slice,
            mode='linear',
        )
        ele_embed_inter = ele_embed_inter.permute(0,2,1).squeeze()
        
        # 针对每个真实batch进行操作
        for batch_num in range(batch_numTrue): 
            # repaet的含义是，所有token加上同一个编码值（不repeat，直接为1，准备cat上去）
            eleva_embed_batch[batch_num, :, :] += ele_embed_inter[Ele_idx_li[batch_num],:] # .repeat(x.shape[1],1) 
        # 重复到每个crops
        eleva_embed_batch = eleva_embed_batch.repeat(x.shape[0]//x_g_norm.shape[0], 1, 1)
        return eleva_embed_batch
        
    def prepare_tokens(self, x, x_g_norm):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        
        # 【EEnT】add ELevation encoding to each token
        ele_tokens = self.get_ele_encoding_inter(x, x_g_norm)
        x = torch.cat((ele_tokens, x), dim=1)
        
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return self.pos_drop(x)


    def forward(self, x, x_g_norm):
        x = self.prepare_tokens(x, x_g_norm)
        for blk in self.blocks:
            x = blk(x)
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


def EEnT_NO12_interEle_AsToken(patch_size=16, **kwargs):
    model = ElevationTransformer_NO12_interEle_AsToken(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
