# from Py01shared_code import AddPepperNoise, AddGaussianNoise
# from Py05_Matching_fun import idx2xy, xy2idx, idx_ex_range , dataset_from_geoTXT_aug_BiNorm, feature_normalize, minmaxscaler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms as pth_transforms
import torch

class MuskApply(object):

    def __init__(self, mask_size = 80, target_size = 224, mode = "constant"):

        self.mask_size = mask_size
        self.target_size = target_size
        self.mode = mode
    
    def __str__(self):
        return "MuskApply(mask_size={:d},target_size={:d},mode={:s} )".format(int(self.mask_size), self.target_size, self.mode)
    
    def get_mask(self):
        center_1_mask = np.ones([self.mask_size, self.mask_size])
        pad_size = (self.target_size - self.mask_size) // 2
        mask_pad = np.pad(center_1_mask, pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
            
        mask_pad = np.expand_dims(mask_pad, axis=2)
        return mask_pad
    
    def __call__(self, img):
        img = np.array(img)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        h, w, c = img.shape
        mask = self.get_mask()
#         plt.imshow(mask)
        mask = np.repeat(mask, c, axis=2)
        img = mask * img # 应用mask
        # 【TODO】计算中心部分的均值
        if self.mode == "mean": # 如果是均值填充而非常数填充，则加上一个mask
            mean = img.sum() / (self.mask_size * self.mask_size)
            mask_pad_add = np.full_like(mask, mean)
#             print(mean)
            mask_pad_add = mask_pad_add * (~mask.astype(np.bool)).astype(np.int32)
#             print(mask_pad_add)
            img = mask_pad_add + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img[img < 0] = 0                       # 避免有值超过255而反转
        
        img = np.squeeze(img)
        img = Image.fromarray(img.astype('uint8')) # .convert('RGB')
        return img
    
class MuskApply_tensor(object):

    def __init__(self, mask_size = 80, target_size = 224, mode = "constant"):

        self.mask_size = mask_size
        self.target_size = target_size
        self.mode = mode
    
    def __str__(self):
        return "MuskApply_tensor(mask_size={:d},target_size={:d},mode={:s} )".format(int(self.mask_size), self.target_size, self.mode)
    
    def get_mask(self):
        center_1_mask = np.ones([self.mask_size, self.mask_size])
        pad_size = (self.target_size - self.mask_size) // 2
        mask_pad = np.pad(center_1_mask, pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
            
        mask_pad = np.expand_dims(mask_pad, axis=0)
        return mask_pad
    
    def __call__(self, img):
        img = np.array(img)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        c, h, w = img.shape
        mask = self.get_mask()
#         plt.imshow(mask)
        mask = np.repeat(mask, c, axis=0)
        img = mask * img # 应用mask
        # 【TODO】计算中心部分的均值
        if self.mode == "mean": # 如果是均值填充而非常数填充，则加上一个mask
            mean = img.sum() / (self.mask_size * self.mask_size)
            mask_pad_add = np.full_like(mask, mean)
#             print(mean)
            mask_pad_add = mask_pad_add * (~mask.astype(np.bool)).astype(np.int32)
#             print(mask_pad_add)
            img = mask_pad_add + img
#         img[img > 255] = 255                       # 避免有值超过255而反转
#         img[img < 0] = 0                       # 避免有值超过255而反转
        
        img = np.squeeze(img)
#         img = Image.fromarray(img.astype('uint8')) # .convert('RGB')
        
        return torch.tensor(img, dtype = torch.float32).unsqueeze(0)
# aug = MuskApply()
# data = np.random.rand(224,224)*255
# data_aug = aug(data)
# plt.imshow(data_aug)
# print(data.shape)

