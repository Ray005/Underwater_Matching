import timm
import torch
import torch.nn as nn

from Py01shared_code import Proj_layer
import vision_transformer as vits

import utils
from utils import trunc_normal_

from torch.utils.data import Dataset
import math
import os
import numpy as np
from PIL import Image
import torch.distributed as dist
import os
import sys
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from functools import partial

from Py05_Matching_fun import feature_normalize, minmaxscaler
from Py07EEnT import EEnT_tiny
from Py08EEnT_NO2 import get_dataset_per, ElevationTransformer_NO2, EEnT_NO2_tiny
import Py10Bi_model
from Py13_SA_FETM import get_dataset_var_per, get_bof_voc


def get_convnext_model(fea_dim = 16):
    # 使用nano模型
    model = timm.create_model('convnext_nano', pretrained=False)
    # 改变第一层的输入通道数为1
    conv0 = nn.Conv2d(1, 80, kernel_size=(4, 4), stride=(4, 4))
    model.stem[0] = conv0
    # 改变最后一层全连接
    fc = nn.Sequential(
        nn.Linear(640, 256),
        nn.Linear(256, fea_dim)
    #     fc4 = nn.Linear(64, 16)
                        )
    model.head[4] = fc
    return model

def vit_tiny(patch_size=16, embed_dim=192, **kwargs):
    model = vits.VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def get_Vit_model_local():
    model = vit_tiny(patch_size=16, num_classes=0, embed_dim=192)
    
    # 维度更改
    PatchEmbed_1ch = Proj_layer()
    model.patch_embed = PatchEmbed_1ch
    return model
# def get_Vit_model(embed_dim=192):
#     model = vits.__dict__["vit_tiny"](patch_size=16, num_classes=0)
    
#     # 维度更改
#     PatchEmbed_1ch = Proj_layer()
#     model.patch_embed = PatchEmbed_1ch
#     return model
    
class TBiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = get_convnext_model(fea_dim = 16)
        self.model_vit = get_Vit_model()
        self.embed_dim = 192 + 16
 
    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
        output = torch.cat((fea_conv, fea_vit), dim = -1)
        return output

def get_resnet18_model(fea_dim = 16):
    # 使用模型
    model = timm.create_model('resnet18', in_chans=1, pretrained=False)
    # 改变最后一层全连接
    fc_fea = nn.Sequential(
        nn.Linear(512, 256),
        nn.Linear(256, fea_dim)
    #     fc4 = nn.Linear(64, 16)
        )
    model.global_pool = nn.Sequential(model.global_pool , fc_fea)
    model.fc = nn.Identity()
    return model

def get_resnet18_model_ori(fea_dim = 16):
    # 使用模型
    model = timm.create_model('resnet18', in_chans=1, pretrained=False)
    # 改变最后一层全连接
    fc_fea = nn.Sequential(
        nn.Linear(512, 256),
        nn.Linear(256, fea_dim)
    #     fc4 = nn.Linear(64, 16)
        )
#     model.global_pool = nn.Sequential(model.global_pool , fc_fea)
    model.fc = fc_fea
    return model

class TBiNet_Res(nn.Module):
    def __init__(self, conv_em_dim = 16):
        super().__init__()
        self.model_conv = get_resnet18_model_ori(fea_dim = conv_em_dim)
        self.model_vit = get_Vit_model_local()
        self.embed_dim = 192 + conv_em_dim

    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output

class MultiCropWrapper_Bi(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper_Bi, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x, x_global_norm):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]), x_global_norm)
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)
    
class DINOHead_Bi(nn.Module): # 其实和原始Head一样
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
#         print(x.shape)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class TerrainDataset_BiNorm(Dataset):
    def __init__(self, data_folder, transform, global_min, global_max ):
        self.data_folder = data_folder
        self.file_list = os.listdir(self.data_folder)
        self.transform = transform
        self.global_max = global_max
        self.global_min = global_min
        self.global_aug = pth_transforms.Compose([
        # pth_transforms.Resize(256, interpolation=3),
        # pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
#         pth_transforms.Normalize(mean=[0.485], std=[0.229]),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.file_list[index].endswith('.npy'):
             # 使用NumPy库读取网格化地形数据
            data = np.load(os.path.join(self.data_folder, self.file_list[index]))
            # 全局归一化
            data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
#             data_global_norm = torch.tensor(data_global_norm, dtype = torch.float32).unsqueeze(0)
            # 对数据进行局部归一化处理
            min_loc = np.min(data)
            max_loc = np.max(data)
            data = (data - min_loc) / (max_loc - min_loc) * 255
            # 转为图像
            data = Image.fromarray(data.astype(np.uint8), mode='L')
            data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
            # 将图像裁剪为多个大小不同的块
            crops = self.transform(data)
            data_global_norm = self.global_aug(data_global_norm)
            # 构建由多个块组成的元组，作为模型输入
            inputs = []
            for crop in crops:
                inputs.append(crop)
            if len(inputs) == 1: # 数据增强后，仅为一个时，直接返回tensor
                inputs = inputs[0].unsqueeze(0) # 取第一个为输入，而不是变为list
                data_global_norm = data_global_norm[0].unsqueeze(0)
                
            return (inputs, data_global_norm, index)
        else:
            return None

# 20230718【全局归一化加噪声】为在全局归一化数据上加入少量噪声而设计
class TerrainDataset_BiNorm_Noise_on_g(Dataset):
    def __init__(self, data_folder, transform, transform_g, global_min, global_max ):
        self.data_folder = data_folder
        self.file_list = os.listdir(self.data_folder)
        self.transform = transform
        self.global_max = global_max
        self.global_min = global_min
        self.global_aug = transform_g
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.file_list[index].endswith('.npy'):
             # 使用NumPy库读取网格化地形数据
            data = np.load(os.path.join(self.data_folder, self.file_list[index]))
            # 全局归一化
            data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
#             data_global_norm = torch.tensor(data_global_norm, dtype = torch.float32).unsqueeze(0)
            # 对数据进行局部归一化处理
            min_loc = np.min(data)
            max_loc = np.max(data)
            data = (data - min_loc) / (max_loc - min_loc) * 255
            # 转为图像
            data = Image.fromarray(data.astype(np.uint8), mode='L')
            data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
            # 将图像裁剪为多个大小不同的块
            crops = self.transform(data)
            data_global_norm = self.global_aug(data_global_norm)
            # 构建由多个块组成的元组，作为模型输入
            inputs = []
            for crop in crops:
                inputs.append(crop)
            if len(inputs) == 1: # 数据增强后，仅为一个时，直接返回tensor
                inputs = inputs[0].unsqueeze(0) # 取第一个为输入，而不是变为list
                data_global_norm = data_global_norm[0].unsqueeze(0)
                
            return (inputs, data_global_norm, index)
        else:
            return None

class TerrainDataset_for_minmax(Dataset):
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
            return np.min(data), np.max(data)
        else:
            return None
def get_dataset_minmax(dataset_path):
    print("获取数据集最大最小值……")
    min_global = math.inf
    max_global = -math.inf
    dataset_for_min_max = TerrainDataset_for_minmax(dataset_path)
    for i in range(len(dataset_for_min_max)):
        min_temp, max_temp = dataset_for_min_max.__getitem__(i)
        if min_temp < min_global:
            min_global = min_temp
        if max_temp > max_global:
            max_global = max_temp
    print("数据集最大最小值为" + str((min_global, max_global)))
    return min_global, max_global

def get_dataset_ele_li(dataset):
    li = []
    for i in range(len(dataset)):
        _,data_g_n,_ = dataset.__getitem__(i)
        li.append(data_g_n.mean())
    return li

def get_dataset_ele_var_li(dataset):
    li = []
    for i in range(len(dataset)):
        _,data_g_n,_ = dataset.__getitem__(i)
        li.append(data_g_n.var())
    return li
    

@torch.no_grad()
def extract_features_Bi(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, sam_global_norm , index in metric_logger.log_every(data_loader, 10):
#         print("samples")
#         print(samples)
#         print("sam_global_norm")
#         print(sam_global_norm)
#         assert 0 
        samples = samples.cuda(non_blocking=True)
        sam_global_norm = sam_global_norm.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples, sam_global_norm).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

def extract_feature_pipeline_Bi(args):
    load_bof_voc = True
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
#         pth_transforms.Resize(256, interpolation=3),
#         pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean=[0.485], std=[0.229]),
#         pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
#     transform = DataAugmentationDINO(
#     args.global_crops_scale,
#     args.local_crops_scale,
#     args.local_crops_number,
#     )
    # train集数据
    data_train_path = os.path.join(args.data_path, "train")
    global_min, global_max = get_dataset_minmax(data_train_path)
    dataset_train = TerrainDataset_BiNorm(data_train_path, transform, global_min, global_max)
    # test集数据
    data_val_path = os.path.join(args.data_path, "test")
#     global_min, global_max = get_dataset_minmax(data_val_path)
    dataset_val   = TerrainDataset_BiNorm(data_val_path, transform, global_min, global_max)
    
    # 获取bof特征
    if load_bof_voc:
        bof_voc, bof_stdSlr = torch.load("./bof_voc/voc.pth")
        voc_k = bof_voc.shape[0]
        print("voc_k = ", voc_k, " (代表sift_bof特征维度为此值，后续的嵌入会沿用这一切分维度)")
        print("成功加载bof_voc")
    else:
        voc_k = 100
        bof_voc, bof_stdSlr = get_bof_voc(dataset_path, using_num = 2000, voc_k = voc_k)
        torch.save((bof_voc, bof_stdSlr), "./bof_voc/voc.pth")
    
    global_min, global_max, per_li_train = get_dataset_per(data_train_path, args.num_ele_slice)
    print("获取数据集data_train高程分位数完成")
    _, _, per_li_val = get_dataset_per(data_val_path, args.num_ele_slice)
    print("获取数据集data_val高程分位数完成\r\n")
    
    _, _, per_li_var_train = get_dataset_var_per(data_train_path, args.num_ele_slice)
    print("获取数据集data_train(var)高程方差分位数完成")
    _, _, per_li_var_val = get_dataset_var_per(data_val_path, args.num_ele_slice)
    print("获取数据集data_val(var)高程方差分位数完成\r\n")
    
    data_train_ele_li = get_dataset_ele_li(dataset_train)
    print("获取数据集data_train高度均值列表完成（用于上色）")
    data_val_ele_li = get_dataset_ele_li(dataset_val)
    print("获取数据集data_val高度均值列表完成（用于上色）")
    
#     dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
#     dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "test"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    elif args.arch == "BiNet_NO1":
        model = TBiNet()
        embed_dim = model.embed_dim
    elif args.arch == "EEnT_tiny_NO1":
        model = EEnT_tiny(num_ele_slice = args.num_ele_slice, in_chans=1)
        embed_dim = model.embed_dim
    elif args.arch == "EEnT_tiny_NO2":
        model = EEnT_NO2_tiny(num_ele_slice = args.num_ele_slice, per_li = per_li_train, in_chans=1)
        embed_dim = model.embed_dim
    elif args.arch == "BiNet_Res_NO1":
        model = TBiNet_Res(conv_em_dim = 16)
        embed_dim = model.embed_dim
    elif args.arch in Py10Bi_model.__dict__.keys():
        try:
            model = Py10Bi_model.__dict__[args.arch]() # 调用函数
            embed_dim = model.embed_dim
        except: # 兼容需要slice和分位数列表的网络
            model = Py10Bi_model.__dict__[args.arch](num_ele_slice=args.num_ele_slice, per_li=per_li_train, per_li_var=per_li_var_train,
                                                       bof_voc=bof_voc, bof_stdSlr=bof_stdSlr, in_chans=1, voc_k = voc_k,
                                                      g_min = global_min, g_max = global_max)
            embed_dim = model.embed_dim
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
        
    # 维度更改
#     PatchEmbed_1ch = Proj_layer()
#     model.patch_embed = PatchEmbed_1ch
    
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features_Bi(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    if args.arch == "EEnT_tiny_NO2": # EEnT NO2中，模型分位数与数据集有关，故需要覆盖
        model.per_li = per_li_val
    try:
        model.per_li = per_li_val
        model.per_li_var = per_li_var_val
#         print("per_li_val", per_li_val)
    except:
        pass
    test_features = extract_features_Bi(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

#     train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
#     test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()

#     train_labels = torch.zeros([dataset_train.__len__()]).long()
#     test_labels = torch.zeros([dataset_val.__len__()]).long()

    print("Reading Labels from train set...")
    train_labels = torch.tensor([dataset_train.__getitem__(i)[-1] for i in range(0, len(dataset_train))]).long()
    print("Reading Labels from train set...")
    test_labels = torch.tensor([dataset_val.__getitem__(i)[-1] for i in range(0, len(dataset_val))]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels, data_train_ele_li, data_val_ele_li


# class ReturnIndexDataset(TerrainDataset):
#     def __getitem__(self, idx):
#         img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
#         return img, idx

def load_Pre_BiNet_No1(args):
    if "BiNet_NO1" in args.arch:
        model = TBiNet()
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    return model

def get_global_norm(data, global_min, global_max):
    data_global_norm = (data - global_min) / (global_max - global_min)
    # 给左上角与右下角赋值，便于观察最大最小（实际输入网络不需要这一步）
    data_global_norm[0] = 0
    data_global_norm[-1]  = 1
    return data_global_norm

def get_local_aug(data, transform):
    #  转换为图片后增强
    # 均值方差归一化与最大最小值归一化
#     data = feature_normalize(data)
    data = minmaxscaler(data) * 255 # 归一化到0-255
    # print(data)
    # 转换为图片并量化
    data = Image.fromarray(data.squeeze().numpy().astype(np.uint8), mode='L')
    # 数据增强
    data = transform(data)
    return data

def load_Pre_BiNet_Res_NO1(args):
    if "BiNet_Res_NO1" in args.arch:
        model = TBiNet_Res(conv_em_dim = 16)
#         print(model)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    return model

@torch.no_grad()
def extract_features_geoTXT_Bi(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index, box in metric_logger.log_every(data_loader, 100):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()
        
        if index[0] % 10000 == 0:
            print("Processed:" + str(index[0]))
        
#         print(index)
        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features