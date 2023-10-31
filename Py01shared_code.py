from torch.utils.data import Dataset
import os
from PIL import Image
from torch import nn
import numpy as np
from einops import rearrange
from torchvision import datasets, transforms
import utils
import torch
import argparse
import random

import re
import datetime
import socket

class TerrainDataset(Dataset):
    def __init__(self, data_folder, transform):
        self.data_folder = data_folder
        self.file_list = os.listdir(self.data_folder)
        self.transform = transform
#         self.transform = DataAugmentationDINO(
#             args.global_crops_scale,
#             args.local_crops_scale,
#             args.local_crops_number,
#         )
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.file_list[index].endswith('.npy'):
             # 使用NumPy库读取网格化地形数据
            data = np.load(os.path.join(self.data_folder, self.file_list[index]))
            # 对数据进行归一化处理
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            # 使用OpenCV库将数据转换为灰度图
            # 设置图像的高度和宽度
            image_shape = np.load(os.path.join(self.data_folder, self.file_list[0])).shape
            height = image_shape[1]
            width = image_shape[0]
            
            data = Image.fromarray(data.astype(np.uint8), mode='L')
            # 将图像裁剪为多个大小不同的块  
#             print(type(data.filter))
#             try:
#                 print("to_PIL")
#                 print(type(data.filter))
#                 data = ToPILImage()(data) #tensor转为PIL Image
#             except:
#                 pass
#             print("data type = " + str(type(data)))
            crops = self.transform(data)
            
            # 构建由多个块组成的元组，作为模型输入            
            inputs = []
            for crop in crops:
                inputs.append(crop)
            if len(inputs) == 1: # 数据增强后，仅为一个时，直接返回tensor
                inputs = inputs[0].unsqueeze(0) # 取第一个为输入，而不是变为list
            return (inputs, 0)
        else:
            return None
        
class TerrainDataset_withXY(Dataset):
    def __init__(self, data_folder, box_label_path, transform, len_pre, len_suf):
        self.data_folder = data_folder
        self.file_list = os.listdir(self.data_folder)
        self.key_fun = lambda x: int(x[len_pre:-len_suf])
        self.file_list.sort(key = self.key_fun) ##文件名按数字排序 key是个lambda函数
        self.transform = transform
#         self.label_path = label_path
        # 检验连续性
        for i in range(0, len(self.file_list)-1):
            if self.key_fun(self.file_list[i+1]) - self.key_fun(self.file_list[i]) != 1:
                assert 0, "【han】Number not Sequential"
        # 读取框的标签值
        self.box_label = np.loadtxt(open(box_label_path,"rb"),delimiter=",",skiprows=0,usecols=[0,1,2]) 
        assert len(self.box_label)==len(self.file_list), "【han】label and data number not match"
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.file_list[index].endswith('.npy'):
             # 使用NumPy库读取网格化地形数据
            data = np.load(os.path.join(self.data_folder, self.file_list[index]))
            # 对数据进行归一化处理
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            # 使用OpenCV库将数据转换为灰度图
            # 设置图像的高度和宽度
            image_shape = np.load(os.path.join(self.data_folder, self.file_list[0])).shape
            height = image_shape[1]
            width = image_shape[0]
            
            data = Image.fromarray(data.astype(np.uint8), mode='L')
            # 将图像裁剪为多个大小不同的块  
#             print(type(data.filter))
#             try:
#                 print("to_PIL")
#                 print(type(data.filter))
#                 data = ToPILImage()(data) #tensor转为PIL Image
#             except:
#                 pass
#             print("data type = " + str(type(data)))
            crops = self.transform(data)
            
            # 构建由多个块组成的元组，作为模型输入            
            inputs = []
            for crop in crops:
                inputs.append(crop)
            if len(inputs) == 1: # 数据增强后，仅为一个时，直接返回tensor
                inputs = inputs[0].unsqueeze(0) # 取第一个为输入，而不是变为list
            # 检验标签是否正确
            if int(self.box_label[index][0]) == self.key_fun(self.file_list[index]):
                box = self.box_label[index][1:3] # 只取中心的x、y坐标，第一列为序号，不需要
            else:
                assert 0, "【han】label and data not match"
            return (inputs, index, box)
        else:
            return None
        
class Proj_layer(nn.Module):
    def __init__(self):
        super(Proj_layer, self).__init__()
        self.patch_size = 224
    #         self.params=nn.Parameter(torch.randn(4, 1))
        self.conv_2d = nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(16, 16))
    def forward(self, x):
        x = self.conv_2d(x)
        x = x.transpose(1,3)
        x = x.squeeze(2)
        x = rearrange(x, 'b c h w -> b (c h) w')
        return x
    
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.RandomApply([transforms.Lambda(lambda x: x + np.random.normal(0, 1, x.shape))]),
            
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(0.485, 0.229)
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
#             transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
#             transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []

        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

# class DataAugmentationDINO_ADJ(object):
#     def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
#         flip_and_color_jitter = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#         ])
#         normalize = transforms.Compose([
#             transforms.ToTensor(),
# #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             transforms.Normalize(mean=[0.485], std=[0.229]),
#         ])

#         # first global crop
#         self.global_transfo1 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(1.0),
#             normalize,
#         ])
#         # second global crop
#         self.global_transfo2 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(0.1),
#             utils.Solarization(0.2),
#             normalize,
#         ])
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#             transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(p=0.5),
#             normalize,
#         ])

#     def __call__(self, image):
#         crops = []
#         crops.append(self.global_transfo1(image))
#         crops.append(self.global_transfo2(image))
#         for _ in range(self.local_crops_number):
#             crops.append(self.local_transfo(image))
#         return crops
    
class DataAugmentationDINO_ADJ(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, USE_Noise_Aug=True, Use_HFlip_aug=True, Use_VFlip_aug=True):
        all_aug_li = [
            
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8),
        ]
        if USE_Noise_Aug:
#             all_aug_li.append(AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45)))
            all_aug_li.append(AddGaussianNoise(mean=0, variance=1, amplitude=20))
            all_aug_li.append(AddPepperNoise(snr = 0.99, p = 1.0))
        if Use_HFlip_aug:
            all_aug_li.insert(0, transforms.RandomHorizontalFlip(p=0.5))
        if Use_VFlip_aug:
            all_aug_li.insert(0, transforms.RandomVerticalFlip(p=0.5))
        if True:
            all_aug_li.append(transforms.RandomGrayscale(p=0.2))
        flip_and_color_jitter = transforms.Compose(all_aug_li)
        normalize = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
class DataAugmentationDINO_ADJ_V2(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, USE_Noise_Aug=True, Use_HFlip_aug=True, Use_VFlip_aug=True, Use_Rotate_aug=True, Rotate_angle = 1):
        all_aug_li = [
            
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8),
        ]
        if USE_Noise_Aug:
            all_aug_li.append(AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45)))
            all_aug_li.append(AddPepperNoise(snr = 0.99, p = 1.0))
        if Use_HFlip_aug:
            all_aug_li.insert(0, transforms.RandomHorizontalFlip(p=0.5))
        if Use_VFlip_aug:
            all_aug_li.insert(0, transforms.RandomVerticalFlip(p=0.5))
        if Use_Rotate_aug:
            t_temp = transforms.RandomRotation((-Rotate_angle, Rotate_angle))
            all_aug_li.append(t_temp)
        if True:
            all_aug_li.append(transforms.RandomGrayscale(p=0.2))
        flip_and_color_jitter = transforms.Compose(all_aug_li)
        normalize = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
class DataAugmentationDINO_ADJ_V3(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, USE_Noise_Aug=True, Use_HFlip_aug=True, Use_VFlip_aug=True, Use_Rotate_aug=True, Rotate_angle = 5):
        all_aug_li = [
            
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8),
        ]
        if USE_Noise_Aug:
#             all_aug_li.append(AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45)))
            all_aug_li.append(AddGaussianNoise(mean=0, variance=1, amplitude=20))
            all_aug_li.append(AddPepperNoise(snr = 0.99, p = 1.0))
        if Use_HFlip_aug:
            all_aug_li.insert(0, transforms.RandomHorizontalFlip(p=0.5))
        if Use_VFlip_aug:
            all_aug_li.insert(0, transforms.RandomVerticalFlip(p=0.5))
        if Use_Rotate_aug:
            t_temp = transforms.RandomRotation((-Rotate_angle, Rotate_angle))
            all_aug_li.append(t_temp)
        if True:
            all_aug_li.append(transforms.RandomGrayscale(p=0.2))
        flip_and_color_jitter = transforms.Compose(all_aug_li)
        normalize = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p
    
    def __str__(self):
        return "AddPepperNoise(snr={:f}, p={:.2f})".format(self.snr, self.p)

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            if len(img_.shape) == 2:
                img_ = np.expand_dims(img_, axis=2)
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
#             print(mask)
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            img_ = np.squeeze(img_)
            return Image.fromarray(img_.astype('uint8'))# .convert('RGB') # 转化为PIL的形式
        else:
            return img
            
#添加高斯噪声
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p
    
    def __str__(self):
        return "AddGaussianNoise(mean={:.2f}, variance={:.2f}, amplitude={:.2f}, p={:.2f})".format(self.mean, self.variance, self.amplitude, self.p)

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img[img < 0] = 0                       # 避免有值超过255而反转
#             print(img)
            img = np.squeeze(img)
            img = Image.fromarray(img.astype('uint8')) # .convert('RGB')
            return img
        else:
            return img

#添加高斯噪声
class AddGaussianNoise_Pre(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p
    
    def __str__(self):
        return "AddGaussianNoise_Pre(mean={:.2f}, variance={:.2f}, amplitude={:.2f}, p={:.2f})".format(self.mean, self.variance, self.amplitude, self.p)

    def __call__(self, data):
#         if self.amplitude == 0:
#             return data
        if random.uniform(0, 1) < self.p:
            data = np.array(data)
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)
            c, h, w = data.shape
            min_data = np.nanmin(data)
            max_data = np.nanmax(data)
            N = (self.amplitude) * np.random.normal(loc=self.mean, scale=self.variance, size=(1, h, w))
            N = np.repeat(N, c, axis=0)
            data = N + data
            data[data > max_data] = max_data                       # 避免有值超过255而反转
            data[data < min_data] = min_data                       # 避免有值超过255而反转
#             print(data)
            data = np.squeeze(data)
            # data = Image.fromarray(data.astype('uint8')) # .convert('RGB')
            return data
        else:
            return data
        
#添加高斯噪声
class AddGaussianNoise_Pre_Snr(object):

    def __init__(self, mean=0.0, variance=1.0, snr=1.0, p=1):

        self.mean = mean
        self.variance = variance
        self.snr = snr
        self.p=p
    
    def __str__(self):
        return "AddGaussianNoise_Pre_Snr(mean={:.2f}, variance={:.2f}, snr={:.2f}, p={:.2f})".format(self.mean, self.variance, self.snr, self.p)

    def __call__(self, data):
        if self.snr == 1.0:
            return data
        if random.uniform(0, 1) < self.p:
            data = np.array(data)
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)
            c, h, w = data.shape
            min_data = np.nanmin(data)
            max_data = np.nanmax(data)
            N = (max_data - min_data) * (1-self.snr) * np.random.normal(loc=self.mean, scale=self.variance, size=(1, h, w))
            N = np.repeat(N, c, axis=0)
            data = N + data
            data[data > max_data] = max_data                       # 避免有值超过255而反转
            data[data < min_data] = min_data                       # 避免有值超过255而反转
#             print(data)
            data = np.squeeze(data)
            # data = Image.fromarray(data.astype('uint8')) # .convert('RGB')
            return data
        else:
            return data
        
#添加高斯噪声
class AddGaussianNoise_snr(object):

    def __init__(self, mean=0.0, variance=1.0, snr=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = (1 - snr) * 255
        self.p=p
    
    def __str__(self):
        return "AddGaussianNoise(mean={:.2f}, variance={:.2f}, amplitude={:.2f}, p={:.2f})".format(self.mean, self.variance, self.amplitude, self.p)

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img[img < 0] = 0                       # 避免有值超过255而反转
#             print(img)
            img = np.squeeze(img)
            img = Image.fromarray(img.astype('uint8')) # .convert('RGB')
            return img
        else:
            return img


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters模型参数

    parser.add_argument('--patch_size', default=16, type=int,
                        help="""Size in pixels of input square patches - default 16 (for 16x16 patches). Using smaller values leads to better performance but requires more memory. Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int,
                        help="""Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.Not normalizing leads to better performance but can make the training unstable.In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                        help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool_flag, help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters温度教师参数
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help="""Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int, help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters训练/优化器参数
    parser.add_argument('--use_fp16', type=bool_flag, default=True,
                        help="""Whether or not to use half precision for training. Improves training time and memory requirements, but can provoke instability and slight decay of performance. We recommend disabling mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float,
                        help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters多裁剪参数
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8,
                        help="""Number of small local views to generate. Set this parameter to 0 to disable multi-crop training. When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    # 额外
    parser.add_argument('--arch', default="vit_tiny", type=str, metavar='M', help='arch')
    parser.add_argument('--num_ele_slice', default=100, type=int, help='num_ele_slice, 切分高度的层数')
    return parser

def bool_flag(s):
    """
    Parse boolean arguments from the command line.从命令行解析布尔参数。
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
        
        

def get_netpath_unique(netname): # 从列表出发来匹配
    now_time = datetime.datetime.now().strftime('%Y%m%d')
    li = os.listdir("./【checkpoint存档】/")
    this_arch_li = []
    for s in li:
        result = re.search('BiNet_ViT96_Convnext96_s', s)
        if result is not None:
            this_arch_li.append(result.string)
    num_max = 0
    for s in this_arch_li:
        result = re.search('\d+$', s[0:-4]) # 匹配除去.pth后缀的末尾数字
        num = int(result.group())
        if num > num_max:
            num_max = num
    num_added = "{:02d}".format(int(num_max) + 1) # 至少两位
    netname = now_time + netname + get_host_name_DINO_X() + "_" + num_added
    save_net_name = "./【checkpoint存档】/" + netname + ".pth"
    return save_net_name, netname

def get_host_name_DINO_X():
    # 获取当前系统主机名
    host_name = socket.gethostname()
    # print('主机名   --> %s' % host_name)
    host_id = host_name[-8:-1] + host_name[-1]
    if host_id == "49fd416f":
        host_friendly_name = "DINO_3"
    elif host_id == "e2024fcf": 
        host_friendly_name = "DINO_4"
    elif host_id == "93ec44ce":
        host_friendly_name = "DINO_5"
    elif host_id == "51387963":
        host_friendly_name = "DINO_6"
    elif host_id == "0ae706a4":
        host_friendly_name = "DINO_7"
    elif host_id == "65a230e0":
        host_friendly_name = "DINO_8"
    elif host_id == "0bec62e0":
        host_friendly_name = "DINO_9"
    elif host_id == "05b18165":
        host_friendly_name = "DINO_10"
    elif host_id == "2571cfc4":
        host_friendly_name = "DINO_11"
    else:
        host_friendly_name = "Unknow_Host"
    return host_friendly_name