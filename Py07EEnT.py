import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_
from vision_transformer import DropPath, drop_path, Mlp, Attention, Block, PatchEmbed, DINOHead
import sys
import utils

class ElevationTransformer(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

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
            idx = self.get_encode_vec_idx(x_g_norm[batch_num, :, :, :])
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
    
    def get_encode_vec_idx(self, global_norm_data):
        # minmax norm
#         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
        # 计算idx
        idx = int(global_norm_data.mean() * self.num_ele_slice)
        return idx

def EEnT_tiny(patch_size=16, **kwargs):
    model = ElevationTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def load_Pre_EEnT_tiny_No1(args):
    if "EEnT_tiny_NO1" in args.arch:
        model = EEnT_tiny(num_ele_slice = args.num_ele_slice, in_chans=1)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    return model

