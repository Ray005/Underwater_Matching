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

from bisect import bisect_left

from SelfAttention import ScaledDotProductAttention

import os, cv2, time, numpy, argparse
import logging as log
# from multiprocessing import Pool
from sklearn.svm import LinearSVC
# from sklearn.externals import joblib
import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

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
            return np.nanmin(data), np.nanmax(data), np.nanmean(data), data
        else:
            return None

def get_dataset_per(dataset_path, num_slice):
    print("获取数据集分位数……")
    min_global = math.inf
    max_global = -math.inf
    dataset_for_min_max = TerrainDataset_for_per_li(dataset_path)
    mean_li = []
    for i in range(len(dataset_for_min_max)):
        min_temp, max_temp, mean_temp, _ = dataset_for_min_max.__getitem__(i)
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

def get_dataset_var_per(dataset_path, num_slice):
    print("获取数据集分位数……(VAR)")
    min_global = math.inf
    max_global = -math.inf
    dataset_for_min_max = TerrainDataset_for_per_li(dataset_path)
    var_li = []
    for i in range(len(dataset_for_min_max)):
        min_temp, max_temp, _, data = dataset_for_min_max.__getitem__(i)
        var_temp = data.var()
        var_li.append(var_temp)
        if min_temp < min_global:
            min_global = min_temp
        if max_temp > max_global:
            max_global = max_temp
    per_li_var = []
#     for i in range(num_slice):
#         per_test = np.percentile(var_li, 100 / num_slice * (i + 1))
#         per_test = (per_test - min_global) / (max_global - min_global)
#         per_li_var.append(per_test)
    per_li_var = np.percentile(var_li, np.linspace(0,100, num_slice))
    per_li_var = (per_li_var - min_global) / (max_global - min_global)
    per_li_var[-1] = 1
    per_li_var[-1] = 1
    print("数据集最大最小值为" + str((min_global, max_global)))
    return min_global, max_global, per_li_var

class TerrainDataset_for_bof_voc(Dataset):
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
            return torch.tensor(data, dtype=torch.float32), index
        else:
            return None

def get_bof_voc(dataset_path, using_num, voc_k = 100):
    # Detecting points and extracting features
    dataset_full = TerrainDataset_for_bof_voc(dataset_path)
    if len(dataset_full) >= using_num:
        use_size = using_num
        other_size = len(dataset_full) - use_size
        dataset, _ = torch.utils.data.random_split(dataset_full, [use_size, other_size])
    else:
        dataset = dataset_full
    set_size = len(dataset)
        
    print("Detecting points and extracting features")
    features = detectAndCompute_set(dataset)

    # Stack all the descriptors vertically in a numpy array using Pool
    print("Stack all descriptors vertically in a numpy array")
    descriptors = stack_descriptors(features)
    descriptors_result = descriptors

    # 获取类心
    # Perform k-means clustering
    print("Perform k-means clustering, ", "descriptors_result.shape = ", descriptors_result.shape, "\n")

    voc, _ = kmeans(descriptors_result, voc_k, 100) # voc就是类心的矩阵
#     torch.save(torch.tensor(voc, dtype=torch.float32), "./voc/voc.pth")
#     voc = torch.load("./voc/voc.pth")
    print("k-means Done")
    print("类心矩阵尺寸：voc.shape = ", voc.shape)
    
    # 获取stdSlr
    print("Creating codebook")
    im_features = numpy.zeros((set_size, voc_k), "float32")
    i = 0
    for i, feature in enumerate(features):
        if feature[1] is not None:
            for descriptor in feature[1]:
                descriptor = numpy.expand_dims(descriptor, axis = 0)
        #             print("descriptor" + str(descriptor))
        #             print("voc" + str(voc))
                words, _ = vq(descriptor, voc)
                for w in words:
                    im_features[i][w] +=1
            i += 1
    # Scaling the words
    print("Scaling words")
    stdSlr = StandardScaler().fit(im_features)
    return voc, stdSlr

# 获取bof，输入参数为【数据集】
def get_bof_Pacific_voc(dataset, using_num, voc_k = 100):
    # Detecting points and extracting features
    dataset_full = dataset
    if len(dataset_full) >= using_num:
        use_size = using_num
        other_size = len(dataset_full) - use_size
        dataset, _ = torch.utils.data.random_split(dataset_full, [use_size, other_size])
    else:
        dataset = dataset_full
    set_size = len(dataset)
        
    print("Detecting points and extracting features")
    features = detectAndCompute_set_Pacific(dataset)

    # Stack all the descriptors vertically in a numpy array using Pool
    print("Stack all descriptors vertically in a numpy array")
    descriptors = stack_descriptors(features)
    descriptors_result = descriptors

    # 获取类心
    # Perform k-means clustering
    print("Perform k-means clustering, ", "descriptors_result.shape = ", descriptors_result.shape, "\n")

    voc, _ = kmeans(descriptors_result, voc_k, 100) # voc就是类心的矩阵
#     torch.save(torch.tensor(voc, dtype=torch.float32), "./voc/voc.pth")
#     voc = torch.load("./voc/voc.pth")
    print("k-means Done")
    print("类心矩阵尺寸：voc.shape = ", voc.shape)
    
    # 获取stdSlr
    print("Creating codebook")
    im_features = numpy.zeros((set_size, voc_k), "float32")
    i = 0
    for i, feature in enumerate(features):
        if feature[1] is not None:
            for descriptor in feature[1]:
                descriptor = numpy.expand_dims(descriptor, axis = 0)
        #             print("descriptor" + str(descriptor))
        #             print("voc" + str(voc))
                words, _ = vq(descriptor, voc)
                for w in words:
                    im_features[i][w] +=1
            i += 1
    # Scaling the words
    print("Scaling words")
    stdSlr = StandardScaler().fit(im_features)
    return voc, stdSlr
    
# 【对一个数据集返回所有SiFT特征】
def detectAndCompute_set(data_loader):
    # Detect, compute and return all features found on images
    # 实例化一个记录者
    metric_logger = utils.MetricLogger(delimiter="  ")
    descriptions = []
    keypoint_detector = cv2.SIFT_create()
    keypoint_descriptor = cv2.SIFT_create()
    for samples, index in metric_logger.log_every(data_loader, 100):
        image=cv2.normalize(samples.squeeze().cpu().numpy(),None,0,255,cv2.NORM_MINMAX).astype('uint8')
        keypoints = keypoint_detector.detect(image, None)
        (keypoints, description) = keypoint_descriptor.compute(image, keypoints)
        descriptions.append((index,description))
    return descriptions

# 【对一个数据集返回所有SiFT特征】输入为数据集时
def detectAndCompute_set_Pacific(data_loader):
    # Detect, compute and return all features found on images
    # 实例化一个记录者
    metric_logger = utils.MetricLogger(delimiter="  ")
    descriptions = []
    keypoint_detector = cv2.SIFT_create()
    keypoint_descriptor = cv2.SIFT_create()
    for samples, _, index, _ in metric_logger.log_every(data_loader, 100):
        image=cv2.normalize(samples.squeeze().cpu().numpy(),None,0,255,cv2.NORM_MINMAX).astype('uint8')
        keypoints = keypoint_detector.detect(image, None)
        (keypoints, description) = keypoint_descriptor.compute(image, keypoints)
        descriptions.append((index,description))
    return descriptions

def stack_descriptors(features):
    # Stack all the descriptors vertically in a numpy array
    descriptors = None
    while(descriptors is None):
        descriptors = features.pop(0)[1]
    i = 0
    for _, descriptor in features:
        if descriptor is not None:
#             print(descriptors.shape)
#             print(descriptor.shape)
            descriptors = numpy.concatenate((descriptors, descriptor), axis=0)
#             i = i + 1
#             print(i)
    return descriptors

# ======================================================================
# 以下为网络
# ======================================================================

# 加入自注意力机制加权的高程嵌入，加上Sift
# ===========================================
# 【SA-FETM】【高程】【SiFT】【注意力机制】
# ===========================================
class SelfAttention_Feature_Embedding_Encoder(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, bof_voc, bof_stdSlr, voc_k, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
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
        
        self.SAT = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # BoF相关
        self.bof_voc = bof_voc
        self.bof_stdSlr = bof_stdSlr
        self.voc_k = voc_k # self.voc_k应与sift的切分数量一致
        self.sift_embed = nn.Parameter(torch.zeros(self.voc_k, embed_dim))  # 此处第一个维度应为BoF的kmeans聚类维度，即voc第一个维度
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.eleva_embed, std=.02)
        trunc_normal_(self.sift_embed, std=.02)
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
        
        # 【SiFT】计算 sift特征（bof）
        sift_li = torch.zeros([B, self.voc_k, 1], dtype=torch.float32) # self.voc_k应与sift的切分数量一致
#         print("B = ", B)
        for batch_num in range(B):
            sift_bof_fea = self.get_bof_fea(x[batch_num, :, :, :])
#             print("sift_bof_fea dim = ", sift_bof_fea.shape)
            sift_li[batch_num, :, 0] = sift_bof_fea  / sift_bof_fea.sum() # 归一化到0到1
        sift_li = torch.where(torch.isnan(sift_li), torch.full_like(sift_li, 0), sift_li)
#         print(sift_li[0])
#         print("self.sift_embed.shape" , self.sift_embed.shape)
#         print("sift_li.shape", sift_li.shape)
        # 利用BoF加权
        sift_em_batch = (self.sift_embed * sift_li.cuda()) # .sum(dim = 1) # 直接相乘（100*192）*（B*100*1），变为B*100*192。sum为B*192
#         print("sift_em_batch.shape = ", sift_em_batch.shape)
#         print(sift_em_batch[0]) # 应该有很多0。是的，这就对了
        sift_em_batch = sift_em_batch.sum(dim = 1)
#         assert 0 
        
        x = self.patch_embed(x)  # patch linear embedding
        
        # 【EEnT】
        Ele_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
        # 【EEnT】计算高程编码（list）。放入list
        for batch_num in range(batch_numTrue):
            idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
            Ele_idx_li.append(idx)
#         print(Ele_idx_li)
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
                token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                # 【高程】提取这个batch的高程编码，并repeat为196*1*192
                eleEm_in_a_batch = self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                # 【sift】计算sift嵌入
#                 print("sift_em_batch[batch_num]  =", sift_em_batch[batch_num].shape)
                siftEm_in_a_batch = sift_em_batch[batch_num].unsqueeze(0).repeat(x.shape[1],1).unsqueeze(1)
                
                at_in = torch.cat((token_in_a_batch, eleEm_in_a_batch, siftEm_in_a_batch), dim=1) # 拼接为196*2*192，方便做注意力
#                 print(at_in.shape) # torch.Size([197, 2, 192])
                # 进行自注意力操作
                at_out = self.SAT(at_in, at_in, at_in) # 注意力层前向
#                 print(at_out.shape) # torch.Size([197, 2, 192])
                token_in_a_batch, eleEm_in_a_batch, siftEm_in_a_batch= torch.chunk(at_out, 3, dim=1) # 将at_out拆分，再直接相加合成。将其拆分为token和eleEM
#                 print(x[(crops_num*batch_numTrue+batch_num), :, :].shape) # torch.Size([197, 192])
                x[(crops_num*batch_numTrue+batch_num), :, :] = token_in_a_batch.squeeze(1) + eleEm_in_a_batch.squeeze(1) + siftEm_in_a_batch.squeeze(1)
    
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率
        if idx >= self.num_ele_slice:
            assert 0, "高程编码索引计算错误"
#         for i, per in enumerate(self.per_li):
#             if ele_mean < per:
#                 return i
#         else:
#             return None
        return idx
    def get_bof_fea(self, data):
        output_vec = self.siftAndBoF_onePic(data)
#         output_vec = self.bof_stdSlr.transform(output_vec)
        output_vec = torch.tensor(output_vec, dtype=torch.float32)
        return output_vec
    
    def siftAndBoF_onePic(self, sample): # 注：需要voc, stdSlr为全局变量
        SiFT_feature = self.detectAndCompute_onePic(sample)
        im_feature = numpy.zeros((1, self.bof_voc.shape[0]), "float32")
        if SiFT_feature is not None:
            words, _ = vq(SiFT_feature, self.bof_voc)
            for w in words:
                im_feature[0][w] += 1
    #     im_feature = self.bof_stdSlr.transform(im_feature)
    #     im_feature = torch.tensor(im_feature, dtype=torch.float32)
        return im_feature
    
    def detectAndCompute_onePic(self, sample):
        # Detect, compute and return all features found on images
        descriptions = []
        keypoint_detector = cv2.SIFT_create()
        keypoint_descriptor = cv2.SIFT_create()
    #     for image_path in image_paths[0]:
    #     print(image_path)
    #     image = cv2.imread(image_path)
        image=cv2.normalize(sample.squeeze().cpu().numpy(),None,0,255,cv2.NORM_MINMAX).astype('uint8')
        keypoints = keypoint_detector.detect(image, None)
        (keypoints, description) = keypoint_descriptor.compute(image, keypoints)
    #     descriptions.append(description)
        return description

def SA_FETM_bi(patch_size=16, **kwargs):
    model = SelfAttention_Feature_Embedding_Encoder(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ===========================================
# 【SA-FETM】【无高程】【SiFT】【注意力机制】不使用高程，只使用BoF+ViT尝试
# ===========================================
class SelfAttention_Feature_Embedding_BoFOnly(nn.Module):
    """ Elevation Transformer """
    def __init__(self, num_ele_slice, per_li, bof_voc, bof_stdSlr, voc_k, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
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
#         self.eleva_embed = nn.Parameter(torch.zeros(num_ele_slice, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.SAT = ScaledDotProductAttention(d_model=192, d_k=32, d_v=32, h=8) # 实例化自注意力层
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # BoF相关
        self.bof_voc = bof_voc
        self.bof_stdSlr = bof_stdSlr
        self.voc_k = voc_k # self.voc_k应与sift的切分数量一致
        self.sift_embed = nn.Parameter(torch.zeros(self.voc_k, embed_dim))  # 此处第一个维度应为BoF的kmeans聚类维度，即voc第一个维度
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
#         trunc_normal_(self.eleva_embed, std=.02)
        trunc_normal_(self.sift_embed, std=.02)
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
        
        # 【SiFT】计算 sift特征（bof）
        sift_li = torch.zeros([B, self.voc_k, 1], dtype=torch.float32) # self.voc_k应与sift的切分数量一致
#         print("B = ", B)
        for batch_num in range(B):
            sift_bof_fea = self.get_bof_fea(x[batch_num, :, :, :])
#             print("sift_bof_fea dim = ", sift_bof_fea.shape)
            sift_li[batch_num, :, 0] = sift_bof_fea  / sift_bof_fea.sum() # 归一化到0到1
        sift_li = torch.where(torch.isnan(sift_li), torch.full_like(sift_li, 0), sift_li)
#         print(sift_li[0])
#         print("self.sift_embed.shape" , self.sift_embed.shape)
#         print("sift_li.shape", sift_li.shape)
        # 利用BoF加权
        sift_em_batch = (self.sift_embed * sift_li.cuda()) # .sum(dim = 1) # 直接相乘（100*192）*（B*100*1），变为B*100*192。sum为B*192
#         print("sift_em_batch.shape = ", sift_em_batch.shape)
#         print(sift_em_batch[0]) # 应该有很多0。是的，这就对了
        sift_em_batch = sift_em_batch.sum(dim = 1)
#         assert 0 
        
        x = self.patch_embed(x)  # patch linear embedding
        
        # 【EEnT】
#         Ele_idx_li = []
        batch_numTrue = x_g_norm.shape[0] # 全局归一化的batchsize（真实的batchsize）。局部归一化后的增强，会将多增强出来的数据，叠加到batchsize上，故不是真实的
#         # 【EEnT】计算高程编码（list）。放入list
#         for batch_num in range(batch_numTrue):
#             idx = self.get_map_idx(x_g_norm[batch_num, :, :, :])
#             Ele_idx_li.append(idx)
#         print(Ele_idx_li)
        # 【EEnT】进行高程嵌入
        for crops_num in range(x.shape[0] // x_g_norm.shape[0]): # 对每个不同的crops进行高程编码。(range的解释：叠加后的维度，除以真实维度，得到的是crops个数，也就是叠加维度)
            for batch_num in range(batch_numTrue): # 针对每个真实batch进行操作
                # 在每个batch中操作自注意力
                # 将其变为196*1*192，将其196作为自注意力的batch、1为token个数，192为编码长度。（原来的196是ViT中token的个数）
                token_in_a_batch = x[(crops_num*batch_numTrue+batch_num), :, :] .unsqueeze(1)
                # 【高程】提取这个batch的高程编码，并repeat为196*1*192
#                 eleEm_in_a_batch = self.eleva_embed[Ele_idx_li[batch_num],:].repeat(x.shape[1],1).unsqueeze(1)
                # 【sift】计算sift嵌入
#                 print("sift_em_batch[batch_num]  =", sift_em_batch[batch_num].shape)
                siftEm_in_a_batch = sift_em_batch[batch_num].unsqueeze(0).repeat(x.shape[1],1).unsqueeze(1)
                
                at_in = torch.cat((token_in_a_batch, siftEm_in_a_batch), dim=1) # 拼接为196*2*192，方便做注意力
#                 print(at_in.shape) # torch.Size([197, 2, 192])
                # 进行自注意力操作
                at_out = self.SAT(at_in, at_in, at_in) # 注意力层前向
#                 print(at_out.shape) # torch.Size([197, 2, 192])
                token_in_a_batch, siftEm_in_a_batch= torch.chunk(at_out, 2, dim=1) # 将at_out拆分，再直接相加合成。将其拆分为token和eleEM
#                 print(x[(crops_num*batch_numTrue+batch_num), :, :].shape) # torch.Size([197, 192])
                x[(crops_num*batch_numTrue+batch_num), :, :] = token_in_a_batch.squeeze(1) + siftEm_in_a_batch.squeeze(1)
    
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
#     def get_encode_vec_idx(self, global_norm_data):
#         # minmax norm
# #         global_norm_data = (global_norm_data - global_norm_data.min()) / global_norm_data.max() 
#         # 计算idx
#         idx = int(global_norm_data.mean() * self.num_ele_slice)
#         return idx
    def get_map_idx(self, x_g_norm):
        ele_mean = x_g_norm.mean()
        idx = bisect_left(self.per_li, ele_mean) # 二分查找提高效率
        if idx >= self.num_ele_slice:
            assert 0, "高程编码索引计算错误"
#         for i, per in enumerate(self.per_li):
#             if ele_mean < per:
#                 return i
#         else:
#             return None
        return idx
    def get_bof_fea(self, data):
        output_vec = self.siftAndBoF_onePic(data)
#         output_vec = self.bof_stdSlr.transform(output_vec)
        output_vec = torch.tensor(output_vec, dtype=torch.float32)
        return output_vec
    
    def siftAndBoF_onePic(self, sample): # 注：需要voc, stdSlr为全局变量
        SiFT_feature = self.detectAndCompute_onePic(sample)
        im_feature = numpy.zeros((1, self.bof_voc.shape[0]), "float32")
        if SiFT_feature is not None:
            words, _ = vq(SiFT_feature, self.bof_voc)
            for w in words:
                im_feature[0][w] += 1
    #     im_feature = self.bof_stdSlr.transform(im_feature)
    #     im_feature = torch.tensor(im_feature, dtype=torch.float32)
        return im_feature
    
    def detectAndCompute_onePic(self, sample):
        # Detect, compute and return all features found on images
        descriptions = []
        keypoint_detector = cv2.SIFT_create()
        keypoint_descriptor = cv2.SIFT_create()
    #     for image_path in image_paths[0]:
    #     print(image_path)
    #     image = cv2.imread(image_path)
        image=cv2.normalize(sample.squeeze().cpu().numpy(),None,0,255,cv2.NORM_MINMAX).astype('uint8')
        keypoints = keypoint_detector.detect(image, None)
        (keypoints, description) = keypoint_descriptor.compute(image, keypoints)
    #     descriptions.append(description)
        return description

def SA_FETM_BoFOnly_bi(patch_size=16, **kwargs):
    model = SelfAttention_Feature_Embedding_BoFOnly(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

