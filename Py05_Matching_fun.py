import torch
from PIL import Image
from torchvision import transforms as pth_transforms
import numpy as np

## 标签→中心点坐标函数
def idx2xy(idx, n_rows, n_cols, shape = (224, 224), stride = 1):
    # 计算裁剪区域的左上角坐标
    if idx >= n_rows * n_cols:
        assert 0, "访问越界"
    i = idx // n_cols
    j = idx %  n_cols
    start_i = i * stride
    start_j = j * stride
    # 对中心点进行编号 
    center_i = start_i + shape[0] // 2
    center_j = start_j + shape[1] // 2
    return center_i, center_j
# idx2xy(6370575,  n_rows, n_cols)


## 坐标到标签函数
# 不考虑在边缘的情况
# x,y 为中心点坐标
def xy2idx(x, y, n_rows, n_cols, shape = (224, 224), stride = 1):
    assert x < n_rows + shape[0] // 2 , "超出最大范围" + str(x)
    assert y < n_cols + shape[1] // 2 , "超出最大范围" + str(y)
    
    assert x >= shape[0] // 2 , "小于最小范围" + str(x)
    assert y >= shape[1] // 2 , "小于最小范围" + str(y)
    
    # 还原到开始点
    start_i = x - shape[0] // 2
    start_j = y - shape[1] // 2
    
    i = start_i / stride
    j = start_j / stride
    
    return int(i * n_cols + j)
# xy2idx(2635, 2635, n_rows, n_cols)

## 范围先验函数
def idx_ex_range(idx, ex_range, n_rows, n_cols, shape = (224, 224), stride = 1):
    x, y = idx2xy(idx, n_rows, n_cols, shape = shape)
#     assert 0
    idx_li = []
    # 将范围限制在左上角112，112.右下角2635,2635（限制的是索引值）
    if x - ex_range < shape[0] // 2: # 小于上最小索引
        x_ex_min = shape[0] // 2
        x_ex_max = shape[0] // 2 + 2 * ex_range
    elif x + ex_range > n_rows + shape[0] // 2 - 1: # 加上ex后，大于下最大索引
        x_ex_min = n_rows + shape[0] // 2 - 1 - 2 * ex_range + 1
        x_ex_max = n_rows + shape[0] // 2 - 1 + 1
    else:
        x_ex_min = x - ex_range
        x_ex_max = x + ex_range 

    if y - ex_range < shape[1] // 2: # 小于左上角最小
        y_ex_min = shape[1] // 2
        y_ex_max = shape[1] // 2 + 2 * ex_range 
    elif y + ex_range > n_cols + shape[1] // 2 - 1:
        y_ex_min = n_cols + shape[1] // 2 - 1 - 2 * ex_range + 1
        y_ex_max = n_cols + shape[1] // 2 - 1 + 1
    else:
        y_ex_min = y - ex_range
        y_ex_max = y + ex_range
    
    for i in range(x_ex_min, x_ex_max):
        for j in range(y_ex_min, y_ex_max):
            idx_li.append(xy2idx(i, j, n_rows, n_cols, shape = shape))
    if idx not in idx_li:
        assert 0, "截取错误，范围内不包含原索引"
    assert len(idx_li) == ex_range * 2 * ex_range * 2, "长度错误"
    return idx_li


# test_idx = xy2idx(2524 + 111, 2524 + 111 , n_rows, n_cols)
# print("test_idx = " + str(test_idx))
# idx_li = idx_ex_range(test_idx, 10, n_rows, n_cols)
# print("Len of extended idx list = " + str(len(idx_li)))

# 测试范围先验函数
# for i in range(n_matrices):
#     idx_li = idx_ex_range(i, 2, n_rows, n_cols)
#     if i % 10000==0:
#         print("Processing [" + str(i) + "/" + str(n_matrices) + "]")


class dataset_from_geoTXT_aug():
    def __init__(self, map_raw, shape, stride, transform):
        self.map_raw = map_raw
        self.shape = shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.transform = transform
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
        data = feature_normalize(data)
        data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
        data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        return (data,
        torch.tensor(idx, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))
    
class dataset_from_geoTXT_aug_BiNorm():
    def __init__(self, map_raw, shape, stride, transform, global_min, global_max):
        self.map_raw = map_raw
        self.shape = shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = pth_transforms.Compose([
        # pth_transforms.Resize(256, interpolation=3),
        # pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
#         pth_transforms.Normalize(mean=[0.485], std=[0.229]),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
        data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        return (data, data_global_norm, 
        torch.tensor(idx, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

class dataset_from_geoTXT_aug_BiNorm_BiAug():
    def __init__(self, map_raw, shape, stride, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.shape = shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global

#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
        data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        return (data, data_global_norm, 
        torch.tensor(idx, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

# 【大连海图】大连海图排除NaN数据集，经典版
class dataset_from_geoTXT_aug_BiNorm_BiAug_smallNaN_map():
    def __init__(self, map_raw, shape, out_shape, stride, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.shape = shape
        self.out_shape = out_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global
        self.available_idx_li = []
        
        self.pad_size = (self.out_shape[0] - self.shape[0]) // 2
        # TODO: 得出能够用的list
        print("计算可用区域……")
        for i in range(self.n_rows):
            for j in range(self.n_cols):
            # 切片裁剪
                start_i = i * self.stride
                start_j = j * self.stride
                data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
                contain_nan = (True in np.isnan(data))
                if contain_nan:
                    continue
                else:
                    # 对中心点进行编号
                    center_i = start_i + self.shape[0] // 2
                    center_j = start_j + self.shape[1] // 2
                    idx = xy2idx(center_i, center_j, self.n_rows, self.n_cols, self.shape, stride = 1)
                    self.available_idx_li.append(idx)
        print("原总数为：" + str(self.n_matrices) + "   可用数量为：" + str(self.__len__()))
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return len(self.available_idx_li)

    def __getitem__(self, idx_set):
        if idx_set >= self.__len__():
            assert 0, "超出图像大小"
        idx = self.available_idx_li[idx_set]        # 覆盖idx

        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
        data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        # 扩充为224
        data = np.pad(data.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
        data_global_norm = np.pad(data_global_norm.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0))
        return (torch.tensor(data).unsqueeze(0), 
                torch.tensor(data_global_norm).unsqueeze(0), 
        torch.tensor(idx_set, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))
    
# 【大连海图】【noIMG版本】大连海图排除NaN数据集，不转图像版本
class dataset_from_geoTXT_aug_BiNorm_BiAug_smallNaN_map_noIMG():
    def __init__(self, map_raw, shape, out_shape, stride, pre_aug, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.shape = shape
        self.out_shape = out_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global
        self.pre_aug = pre_aug
        self.available_idx_li = []
        
        self.pad_size = (self.out_shape[0] - self.shape[0]) // 2
        print("计算可用区域……")
        for i in range(self.n_rows):
            for j in range(self.n_cols):
            # 切片裁剪
                start_i = i * self.stride
                start_j = j * self.stride
                data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
                contain_nan = (True in np.isnan(data))
                if contain_nan:
                    continue
                else:
                    # 对中心点进行编号
                    center_i = start_i + self.shape[0] // 2
                    center_j = start_j + self.shape[1] // 2
                    idx = xy2idx(center_i, center_j, self.n_rows, self.n_cols, self.shape, stride = 1)
                    self.available_idx_li.append(idx)
        print("原总数为：" + str(self.n_matrices) + "   可用数量为：" + str(self.__len__()))
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return len(self.available_idx_li)

    def __getitem__(self, idx_set):
        if idx_set >= self.__len__():
            assert 0, "超出图像大小"
        idx = self.available_idx_li[idx_set]        # 覆盖idx

        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
        
        # 预增强（加入高斯噪声（匹配时）或什么都不做（编码时））
        data = self.pre_aug(data)
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) # * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) #  * 255 # 归一化到0-255
        # 转换为图片并量化
#         data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
#         data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        # 扩充为224
        data = np.pad(data.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
        data_global_norm = np.pad(data_global_norm.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0))
        return (torch.tensor(data, dtype = torch.float32).unsqueeze(0), 
                torch.tensor(data_global_norm, dtype = torch.float32).unsqueeze(0), 
        torch.tensor(idx_set, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

# 【大连海图】【noIMG】【计算动态范围用】
class dataset_from_geoTXT_aug_BiNorm_BiAug_smallNaN_map_noIMG_for_DYNA():
    def __init__(self, map_raw, shape, out_shape, stride, global_min, global_max):
        self.map_raw = map_raw
        self.shape = shape
        self.out_shape = out_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.available_idx_li = []
        
        self.pad_size = (self.out_shape[0] - self.shape[0]) // 2
        print("计算可用区域……")
        for i in range(self.n_rows):
            for j in range(self.n_cols):
            # 切片裁剪
                start_i = i * self.stride
                start_j = j * self.stride
                data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
                contain_nan = (True in np.isnan(data))
                if contain_nan:
                    continue
                else:
                    # 对中心点进行编号
                    center_i = start_i + self.shape[0] // 2
                    center_j = start_j + self.shape[1] // 2
                    idx = xy2idx(center_i, center_j, self.n_rows, self.n_cols, self.shape, stride = 1)
                    self.available_idx_li.append(idx)
        print("原总数为：" + str(self.n_matrices) + "   可用数量为：" + str(self.__len__()))
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return len(self.available_idx_li)

    def __getitem__(self, idx_set):
        if idx_set >= self.__len__():
            assert 0, "超出图像大小"
        idx = self.available_idx_li[idx_set]        # 覆盖idx

        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i + self.shape[0], start_j:start_j + self.shape[1]]
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        return (torch.tensor(data, dtype = torch.float32).unsqueeze(0), 
#                 torch.tensor(data_global_norm, dtype = torch.float32).unsqueeze(0), 
        torch.tensor(idx_set, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

# 【实验1】：先裁剪再做归一化、padding
class dataset_from_geoTXT_aug_BiNorm_BiAug_crop_and_padding():
    def __init__(self, map_raw, crop_shape, in_shape, stride, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.crop_shape = crop_shape
        self.in_shape = in_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - in_shape[0]) // stride + 1 # 计算这个仍以原来的序号坐标转换规则来
        self.n_cols = (map_raw.shape[1] - in_shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global
        
        self.pad_size = (self.in_shape[0] - self.crop_shape[0]) // 2
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.crop_shape[0], start_j:start_j+self.crop_shape[1]]
        
        # 裁剪
#         data = data[self.in_shape]
        
        # 对中心点进行编号
        center_i = start_i + self.in_shape[0] // 2
        center_j = start_j + self.in_shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
        data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        
        # 填充padding
        data = np.pad(data.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
        data_global_norm = np.pad(data_global_norm.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0))
        return (torch.tensor(data).unsqueeze(0), 
                torch.tensor(data_global_norm).unsqueeze(0), 
                torch.tensor(idx, dtype = torch.long),
                torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

# 【实验2】：先归一化，再mask(效果不好)
class dataset_from_geoTXT_aug_BiNorm_BiAug_Norm_first_then_mask():
    def __init__(self, map_raw, crop_shape, in_shape, stride, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.crop_shape = crop_shape
        self.in_shape = in_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - in_shape[0]) // stride + 1 # 计算这个仍以原来的序号坐标转换规则来
        self.n_cols = (map_raw.shape[1] - in_shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global
        
        self.pad_size = (self.in_shape[0] - self.crop_shape[0]) // 2
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
#         data = self.map_raw[start_i:start_i+self.crop_shape[0], start_j:start_j+self.crop_shape[1]]
        data = self.map_raw[start_i:start_i+self.in_shape[0], start_j:start_j+self.in_shape[1]]
        
        # 裁剪
#         data = data[self.in_shape]
        
        # 对中心点进行编号
        center_i = start_i + self.in_shape[0] // 2
        center_j = start_j + self.in_shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
        data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        
#         # 填充padding
#         data = np.pad(data.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
#         data_global_norm = np.pad(data_global_norm.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0))
        return (data, 
                data_global_norm, 
                torch.tensor(idx, dtype = torch.long),
                torch.tensor(np.array((center_i, center_j)), dtype = torch.long))
    
# transform_for_match = pth_transforms.Compose([
#     MuskApply(mask_size = mask_size, mode=pad_mode),  # “constant”为0填充、‘mean’——表示均值填充、‘median’——表示中位数
# #     AddGaussianNoise(mean=0, variance=1, amplitude=10),
#     pth_transforms.ToTensor(),
#     pth_transforms.Normalize(mean=[0.485], std=[0.229]),
# ])

# transform_for_global_match = pth_transforms.Compose([
#     MuskApply(mask_size = mask_size, mode=pad_mode),  # “constant”为0填充、‘mean’——表示均值填充、‘median’——表示中位数
# #     AddGaussianNoise(mean=0, variance=1, amplitude=10),
#     pth_transforms.ToTensor(),
# ])
# print(stride)
# 无mask编码
# dataset_geo = dataset_from_geoTXT_aug_BiNorm_BiAug_Norm_first_then_mask(z, (20,20), (224,224), stride, transform_for_match, transform_for_global_match, global_min, global_max)


# 【测试5】在转为图像之前，就加入高斯噪声。（仍为图像方案，可能会损失精度，实验目的只是测试统一信噪比增强，对小尺寸与大尺寸上，会不会有理想的结果）【结果是会的】
class dataset_from_geoTXT_aug_BiNorm_BiAug_Pre_aug():
    def __init__(self, map_raw, crop_shape, in_shape, stride, pre_aug, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.crop_shape = crop_shape
        self.in_shape = in_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - in_shape[0]) // stride + 1 # 计算这个仍以原来的序号坐标转换规则来
        self.n_cols = (map_raw.shape[1] - in_shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global
        self.pre_aug = pre_aug
        
        self.pad_size = (self.in_shape[0] - self.crop_shape[0]) // 2
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.crop_shape[0], start_j:start_j+self.crop_shape[1]]
        
        # 裁剪
#         data = data[self.in_shape]
        data = self.pre_aug(data)
        # 对中心点进行编号
        center_i = start_i + self.in_shape[0] // 2
        center_j = start_j + self.in_shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
        data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        
        # 填充padding
        data = np.pad(data.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
        data_global_norm = np.pad(data_global_norm.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0))
        return (torch.tensor(data).unsqueeze(0), 
                torch.tensor(data_global_norm).unsqueeze(0), 
                torch.tensor(idx, dtype = torch.long),
                torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

# 【测试6】彻底不使用图像化增强
class dataset_from_geoTXT_aug_BiNorm_BiAug_Pre_aug_noIMG():
    def __init__(self, map_raw, crop_shape, in_shape, stride, pre_aug, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.crop_shape = crop_shape
        self.in_shape = in_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - in_shape[0]) // stride + 1 # 计算这个仍以原来的序号坐标转换规则来
        self.n_cols = (map_raw.shape[1] - in_shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global
        self.pre_aug = pre_aug
        
        self.pad_size = (self.in_shape[0] - self.crop_shape[0]) // 2
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.crop_shape[0], start_j:start_j+self.crop_shape[1]]
        
        # 裁剪
#         data = data[self.in_shape]
        data = self.pre_aug(data)
        # 对中心点进行编号
        center_i = start_i + self.in_shape[0] // 2
        center_j = start_j + self.in_shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) # * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) # * 255 # 归一化到0-255
        # 转换为图片并量化
#         data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
#         data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        
        # 填充padding
        data = np.pad(data.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
        data_global_norm = np.pad(data_global_norm.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0))
        return (torch.tensor(data, dtype = torch.float32).unsqueeze(0), 
                torch.tensor(data_global_norm, dtype = torch.float32).unsqueeze(0), 
                torch.tensor(idx, dtype = torch.long),
                torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

# 【测试7】不使用图像与resize
class dataset_from_geoTXT_aug_BiNorm_BiAug_Pre_aug_noIMG_Resize():
    def __init__(self, map_raw, crop_shape, in_shape, stride, pre_aug, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.crop_shape = crop_shape
        self.in_shape = in_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - in_shape[0]) // stride + 1 # 计算这个仍以原来的序号坐标转换规则来
        self.n_cols = (map_raw.shape[1] - in_shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global
        self.pre_aug = pre_aug
        
        self.pad_size = (self.in_shape[0] - self.crop_shape[0]) // 2
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.crop_shape[0], start_j:start_j+self.crop_shape[1]]
        
        # 裁剪
#         data = data[self.in_shape]
        data = self.pre_aug(data)
        # 对中心点进行编号
        center_i = start_i + self.in_shape[0] // 2
        center_j = start_j + self.in_shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) # * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
        data = minmaxscaler(data) # * 255 # 归一化到0-255
        # 转换为图片并量化
#         data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
#         data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
        data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        
        # 填充padding
#         print(data.squeeze().unsqueeze(0).shape)
        data = torch.nn.functional.interpolate(data.squeeze().unsqueeze(0).unsqueeze(0), size=self.in_shape, mode='bilinear').squeeze() #  nearest
#         data = np.resize(data.squeeze().numpy(), (self.in_shape))
        data_global_norm = torch.nn.functional.interpolate(data_global_norm.squeeze().unsqueeze(0).unsqueeze(0), size=self.in_shape, mode='bilinear').squeeze() #  nearest
#         data_global_norm = np.resize(data_global_norm.squeeze().numpy(), self.in_shape)
#         data = np.pad(data.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0)) # constant为0填充、‘mean’——表示均值填充、‘median’——表示中位数填充、‘minimum’——表示最小值填充
#         data_global_norm = np.pad(data_global_norm.squeeze(), self.pad_size, mode = "constant", constant_values = (0,0))
        return (torch.tensor(data, dtype = torch.float32).unsqueeze(0), 
                torch.tensor(data_global_norm, dtype = torch.float32).unsqueeze(0), 
                torch.tensor(idx, dtype = torch.long),
                torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

# 【太平洋大图】【常规数据集】计算动态范围用
class dataset_from_geoTXT_aug_BiNorm_BiAug_Pre_aug_noIMG_for_DYNA():
    def __init__(self, map_raw, crop_shape, in_shape, stride, global_min, global_max):
        self.map_raw = map_raw
        self.crop_shape = crop_shape
        self.in_shape = in_shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - in_shape[0]) // stride + 1 # 计算这个仍以原来的序号坐标转换规则来
        self.n_cols = (map_raw.shape[1] - in_shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        
        self.pad_size = (self.in_shape[0] - self.crop_shape[0]) // 2
#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.crop_shape[0], start_j:start_j+self.crop_shape[1]]
        
        # 对中心点进行编号
        center_i = start_i + self.in_shape[0] // 2
        center_j = start_j + self.in_shape[1] // 2
        return (torch.tensor(data, dtype = torch.float32).unsqueeze(0), 
#                 torch.tensor(data_global_norm, dtype = torch.float32).unsqueeze(0), 
                torch.tensor(idx, dtype = torch.long),
                torch.tensor(np.array((center_i, center_j)), dtype = torch.long))

def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std
def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data - min)/(max-min)

# feature_normalize
# data = np.array(torch.rand([10,10])*1000)
# data = feature_normalize(data)
# data = minmaxscaler(data)*255
# print(data)

class dataset_from_geoTXT_aug_No_norm():
    def __init__(self, map_raw, shape, stride, crop_size, transform, transform_global, global_min, global_max):
        self.map_raw = map_raw
        self.shape = shape
        self.stride = stride
        self.crop_size = crop_size
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
        self.global_max = global_max
        self.global_min = global_min
        self.transform = transform
        self.global_aug = transform_global

#         assert 0, "切分后行、列、总数：" + str((self.n_rows, self.n_cols, self.n_matrices)) + "。 确认无误后注释此行。 " 
    def __len__(self):
        return self.n_matrices

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        if idx >= self.n_matrices:
            assert 0, "超出图像大小"
        # 计算裁剪区域的左上角坐标
        i = idx // self.n_cols
        j = idx %  self.n_cols
        start_i = i * self.stride
        start_j = j * self.stride
        
        # 切片裁剪
        data = self.map_raw[start_i:start_i+self.crop_size, start_j:start_j+self.crop_size]
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        
        # 全局归一化
        data_global_norm = (data - self.global_min) / (self.global_max - self.global_min) * 255
        ##  转换为图片后增强
        # 均值方差归一化与最大最小值归一化
#         data = feature_normalize(data)
#         data = minmaxscaler(data) * 255 # 归一化到0-255
        # 转换为图片并量化
#         data = Image.fromarray(data.squeeze().astype(np.uint8), mode='L')
        data_global_norm = Image.fromarray(data_global_norm.astype(np.uint8), mode='L')
        # 数据增强
#         data = self.transform(data)
        data_global_norm = self.global_aug(data_global_norm)
        return (data, data_global_norm, 
        torch.tensor(idx, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))