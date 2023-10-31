# input：样本（1，1，224，224）
# return：1*192维向量
import torch
import cv2
import numpy as np
from PIL import Image


# =========================================================
# 【phash】感知哈希
# =========================================================
class phash_encode():
    def __init__(self):
#         self.dist = torch.nn.PairwiseDistance(p=2)
        self.dist = self.hm_dist
        self.distName = "phash中的汉明距离hm_dist"
        self.dim_encode = 2
        self.store_dtype = torch.int64
        
    def encode(self, sample):
        hash_val = self.phash(sample.squeeze().numpy())
        # 存为64位的整型会有bug
        hash_int64_high = int(hash_val[0:32], 2)
        hash_int64_low =  int(hash_val[32:64], 2)
        
        hash_tensor = torch.zeros([1, 2], dtype=torch.int64) # torch没有无符号32位，将32位的转为整型后，第一位是1的话，没办法使用，会溢出
        hash_tensor[0,0] = hash_int64_high
        hash_tensor[0,1] = hash_int64_low
        return hash_tensor
    
    #定义感知哈希
    def phash(self, img):
        #step1：调整大小32x32
#         img=cv2.resize(img,(224,224))
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=img.astype(np.float32)

        #step2:离散余弦变换
        img=cv2.dct(img)
        img=img[0:8,0:8]
        sum=0.
        hash_str=''

        #step3:计算均值
        # avg = np.sum(img) / 64.0
        for i in range(8):
            for j in range(8):
                sum+=img[i,j]
        avg=sum/64

        #step4:获得哈希
        for i in range(8):
            for j in range(8):
                if img[i,j]>avg:
                    hash_str=hash_str+'1'
                else:
                    hash_str=hash_str+'0'
        return hash_str
    #计算汉明距离
    def hmdistance(self, hash1,hash2):
        num=0
        assert len(hash1)==len(hash2), print("【错误】请检查尺寸，两哈希值长度为：",len(hash1), len(hash2),'\n')
        for i in range(len(hash1)):
            if hash1[i]!=hash2[i]:
                num+=1
        return num
    
    def hm_dist(self, hash1_tensor,hash2_tensor):
        hash_int64_high = int(hash1_tensor[0,0])
        hash_int64_low = int(hash1_tensor[0,1])
        hash1 = format(hash_int64_high, '032b') + format(hash_int64_low, '032b')
        
        hash_int64_high = int(hash2_tensor[0,0])
        hash_int64_low = int(hash2_tensor[0,1])
        hash2 = format(hash_int64_high, '032b') + format(hash_int64_low, '032b')

        hash1 = hash1[0:64]
        hash2 = hash2[0:64]
        return self.hmdistance(hash1,hash2)

# =========================================================
# 【dhash】差异值哈希
# =========================================================
class dhash_encode():
    def __init__(self):
        self.dist = self.hm_dist
        self.distName = "dhash中的汉明距离hm_dist"
        self.dim_encode = 8
        self.store_dtype = torch.uint8
        
    def encode(self, sample):
        hash_string = self.dhash(sample.squeeze().numpy())
        # 16进制字符串转为uint8存储
        hash_uint8_tensor = torch.zeros([1, len(hash_string)//2],dtype=torch.uint8)
        for i in range(0, len(hash_string)//2):
            hash_uint8_tensor[0,i] = int(hash_string[2*i:2*i+2], 16)
#         print(hash_uint8_tensor)
        return hash_uint8_tensor
    
    #定义d哈希
    def dhash(self, img):
        image = Image.fromarray(img)
        # #缩放图片
        resize_width = 9
        resize_height = 8
        # # 1. resize to (9,8)
        image = image.resize((resize_width, resize_height))
        
        # 3. 比较相邻像素
        pixels = list(image.getdata())
        difference = []
        for row in range(resize_height):    
            row_start_index = row * resize_width    
            for col in range(resize_width - 1):        
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
        #4. 转换为hash值
        # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):    
            if value:  # value为0, 不用计算, 程序优化        
                decimal_value += value * (2 ** (index % 8))   
            if index % 8 == 7:  # 每8位的结束        
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f        
                decimal_value = 0
        return hash_string
        
        
    #计算汉明距离
    def hmdistance(self, hash1,hash2):
        assert len(hash1)==len(hash2), print("【错误】请检查尺寸，两哈希值长度为：",len(hash1), len(hash2),'\n')
        difference = (int(hash1, 16)) ^ (int(hash2, 16))
        return bin(difference).count("1")
    
    def hm_dist(self, hash1_tensor,hash2_tensor):
        # 将uint8的tensor转回字符串
        hash1 = ''
        for i, item in enumerate(hash1_tensor.squeeze()):
            hash1 = hash1 + format(int(item.squeeze()), '02x')

        hash2 = ''
        for i, item in enumerate(hash2_tensor.squeeze()):
            hash2 = hash2 + format(int(item.squeeze()), '02x')
        return self.hmdistance(hash1,hash2)


# =========================================================
# 【ncc】归一化互相关方法（本质上是皮尔逊相关系数）
# =========================================================
class ncc_class():
    def __init__(self):
        pass
        
    def Compute_Similarity(self, sample1, sample2):
        img1 = sample1.clone().detach().numpy().copy()
        img2 = sample2.clone().detach().numpy().copy()
        img1.resize([10,10])
        img2.resize([10,10])
        ncc_val = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
        return (1 - ncc_val) # 皮尔逊相关系数不会大于一，由于源程序以距离衡量，所以这里取负号，再加一使其一般不小于0，作为“皮尔逊距离”
    
# =========================================================
# 【SSD】Sum-of-Squared-Differences （与欧氏距离类似）
# =========================================================
class ssd_class():
    def __init__(self):
        pass
        
    def Compute_Similarity(self, sample1, sample2):
        img1 = sample1.clone().detach().numpy().copy()
        img2 = sample2.clone().detach().numpy().copy()
#         img1 = torch.nn.functional.interpolate(img1, size=[10], scale_factor=None, mode='nearest', align_corners=None)
#         img2 = torch.nn.functional.interpolate(img2, size=[10], scale_factor=None, mode='nearest', align_corners=None)
        img1.resize([10,10])
        img2.resize([10,10])
        diff = img1-img2
        ssd_dis = diff * diff
        return ssd_dis.sum()

# =========================================================
# 【TERCOM】（）
# =========================================================
class TERCOM_class():
    def __init__(self):
        pass
        
    def Compute_Similarity(self, sample1, sample2):
        img1 = sample1.clone().detach().numpy().copy()
        img2 = sample2.clone().detach().numpy().copy()
#         img1 = torch.nn.functional.interpolate(img1, size=[10], scale_factor=None, mode='nearest', align_corners=None)
#         img2 = torch.nn.functional.interpolate(img2, size=[10], scale_factor=None, mode='nearest', align_corners=None)
        img1.resize([10,10])
        img2.resize([10,10])
        diff = img1-img2
        ssd_dis = diff * diff
        return np.sqrt(ssd_dis.sum())
