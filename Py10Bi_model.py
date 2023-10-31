# 本文件中为定义双输入结构网络的代码，包含创建模型与加载模型
# 模型均以 "_bi" 结尾表示双结构
# 加载预训练函数，均以 load_pre_ 开头加模型名称
# from Py06BiNet import get_Vit_model, get_convnext_model, get_resnet18_model

from vision_transformer import VisionTransformer
import torch
import torch.nn as nn
import timm
from functools import partial
import utils

import Py08EEnT_NO2
import Py13_SA_FETM
import Py14_SO2
import Py15_SO2_DinoBase

import ViT_test

def get_vit_any(embed_dim=192, patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def load_Pre_type1(model, args):
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    return model

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

def get_convnext_any(fea_dim = 16, netname = "convnext_atto_ols"):
    # 支持模型：timm库中的所有convnext模型
    # 更改输入图片通道数与输出维度
    # 如果报错未知模型，请尝试更新timm库!pip install --upgrade timm
    model = timm.create_model(netname, pretrained=False)
    # 改变第一层的输入通道数为1
    out_ch = model.stem[0].out_channels
    conv0 = nn.Conv2d(1, out_ch, kernel_size=(4, 4), stride=(4, 4))
    model.stem[0] = conv0
    # 改变最后一层全连接
    in_fea = model.head.fc.in_features
    fc = nn.Linear(in_fea, fea_dim)
    model.head.fc = fc
    setattr(model, 'embed_dim', fea_dim)
    setattr(model, 'DONOT_CHANGE_HeadFC', True)
    return model

# ViT嵌入96+Resnet96组合
class TBiNet_Res_NO2_bi(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = get_resnet18_model(fea_dim = 96)
        self.model_vit = get_vit_any(embed_dim = 96, in_chans=1)
        self.embed_dim = 96 + 96
        self.vit_em_dim = 96
        self.conv_em_dim = 96

    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
        output = torch.cat((fea_conv, fea_vit), dim = -1)
        return output

# ViT嵌入96+ convnext 96组合
class BiNet_ViT96_Convnext96_bi(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = get_convnext_model(fea_dim = 96)
        self.model_vit = get_vit_any(embed_dim = 96, in_chans=1)
        self.embed_dim = 96 + 96
        self.vit_em_dim = 96
        self.conv_em_dim = 96
 
    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
#         fea_vit = fea_vit.repeat(fea_conv.shape[0] // fea_vit.shape[0],1)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output
    
# ViT嵌入129+ convnext 32组合
# 为什么ViT的嵌入维度必须是3的倍数？
class BiNet_ViT128_Convnext32_bi(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = get_convnext_model(fea_dim = 32)
        self.model_vit = get_vit_any(embed_dim = 129, in_chans=1)
        self.embed_dim = 129 + 32
        self.vit_em_dim = 129
        self.conv_em_dim = 32
 
    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
#         fea_vit = fea_vit.repeat(fea_conv.shape[0] // fea_vit.shape[0],1)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output

# ViT嵌入192+ convnext 32组合
# 为什么ViT的嵌入维度必须是3的倍数？
class BiNet_ViT192_Convnext32_bi(nn.Module):
    def __init__(self):
        super().__init__()
        vit_dim = 192
        conv_dim = 32
        self.model_conv = get_convnext_model(fea_dim = conv_dim)
        self.model_vit = get_vit_any(embed_dim = vit_dim, in_chans=1)
        self.embed_dim = vit_dim + conv_dim
        self.vit_em_dim = vit_dim
        self.conv_em_dim = conv_dim
 
    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
#         fea_vit = fea_vit.repeat(fea_conv.shape[0] // fea_vit.shape[0],1)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output
    
# ViT嵌入192+ convnext 16组合
class BiNet_ViT192_Convnext16_bi(nn.Module):
    def __init__(self):
        super().__init__()
        vit_dim = 192
        conv_dim = 16
        self.model_conv = get_convnext_model(fea_dim = conv_dim) # only for nano
        self.model_vit = get_vit_any(embed_dim = vit_dim, in_chans=1)
        self.embed_dim = vit_dim + conv_dim
        self.vit_em_dim = vit_dim
        self.conv_em_dim = conv_dim
 
    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
#         fea_vit = fea_vit.repeat(fea_conv.shape[0] // fea_vit.shape[0],1)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output
    
# ViT嵌入192+ convnext 16组合
class BiNet_ViT192_Convnext_atto_ols_16_bi(nn.Module):
    def __init__(self):
        super().__init__()
        vit_dim = 192
        conv_dim = 16
        self.model_conv = get_convnext_any(fea_dim = conv_dim, netname = "convnext_atto_ols")
        self.model_vit = get_vit_any(embed_dim = vit_dim, in_chans=1)
        self.embed_dim = vit_dim + conv_dim
        self.vit_em_dim = vit_dim
        self.conv_em_dim = conv_dim

    def forward(self, data, data_global_norm):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data_global_norm)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
        fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
#         fea_vit = fea_vit.repeat(fea_conv.shape[0] // fea_vit.shape[0],1)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output

# 【测试】测试原有的EEnT NO2能否正常工作，这是将高程映射到均匀分布的做法
def EEnT_per_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO2_tiny(num_ele_slice = num_ele_slice, per_li = per_li, in_chans=1)
    
# 【EEnT】【高程嵌入】【注意力机制】无sift
def EEnT_NO3_AT_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO3_AT(num_ele_slice = num_ele_slice, per_li = per_li, in_chans=1)

# 【EEnT】【高程嵌入】【方差嵌入】【注意力机制】无sift
def EEnT_NO4_Var_AT_bi(num_ele_slice, per_li, per_li_var, **kwargs):
    return Py08EEnT_NO2.EEnT_NO4_Var_AT(num_ele_slice = num_ele_slice, per_li = per_li, per_li_var=per_li_var, **kwargs)

# 【SA-FETM】【高程】【SiFT】【注意力机制】
def SA_FETM_bi(num_ele_slice, per_li, bof_voc, bof_stdSlr, **kwargs):
    return Py13_SA_FETM.SA_FETM_bi(num_ele_slice=num_ele_slice, per_li=per_li, bof_voc=bof_voc, bof_stdSlr=bof_stdSlr, **kwargs)

# 【EEnT】【高程嵌入】【方差嵌入】【注意力机制】无sift【batch内高程嵌入】
def EEnT_NO5_iBP_AT_bi(num_ele_slice, per_li, per_li_var, **kwargs):
    return Py08EEnT_NO2.EEnT_NO5_iBP_AT(num_ele_slice = num_ele_slice, per_li = per_li, per_li_var=per_li_var, **kwargs)

# # 【SA-FETM】【高程】【SiFT】【注意力机制】【batch内高程嵌入】
# def SA_FETM_NO2_iBP_bi(num_ele_slice, per_li, bof_voc, bof_stdSlr, **kwargs):
#     return Py13_SA_FETM.SA_FETM_bi(num_ele_slice=num_ele_slice, per_li=per_li, bof_voc=bof_voc, bof_stdSlr=bof_stdSlr, **kwargs)

# 【EEnT】【无极嵌入】【高程嵌入】【方差嵌入】【注意力机制】无sift【batch内高程嵌入】
def EEnT_NO6_contiEm_AT_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO6_contiEm_AT(num_ele_slice = num_ele_slice, per_li = per_li, **kwargs)

# 【EEnT】【无极嵌入V2 with porj】【高程嵌入】【方差嵌入】【注意力机制】无sift【batch内高程嵌入】
def EEnT_NO7_contiEmV2_AT_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO7_contiEmV2_AT(num_ele_slice = num_ele_slice, per_li = per_li, **kwargs)

# 【EEnT】【无极嵌入V3 ele as token】【高程嵌入】【方差嵌入】【注意力机制】无sift【batch内高程嵌入】
def EEnT_NO8_contiEmV3_AT_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO8_contiEmV3_AT(num_ele_slice = num_ele_slice, per_li = per_li, **kwargs)

# 【SA-FETM】【无高程】【SiFT】【注意力机制】【batch内高程嵌入】
def SA_FETM_NO2_BoFViT_bi(num_ele_slice, per_li, bof_voc, bof_stdSlr, **kwargs):
    return Py13_SA_FETM.SA_FETM_BoFOnly_bi(num_ele_slice=num_ele_slice, per_li=per_li, bof_voc=bof_voc, bof_stdSlr=bof_stdSlr, **kwargs)

# 【EEnT】【返璞归真】【在每一个Patch上做高程编码】
def EEnT_NO9_EleOnPatch_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO9_EleOnPatch(num_ele_slice = num_ele_slice, per_li = per_li, in_chans=1)

# 【EEnT】【返璞归真】【在每一个Patch上做高程编码】【嵌入pooling】
def EEnT_NO10_EleOnPatch_poolingEm_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO10_EleOnPatch_poolingEm(num_ele_slice = num_ele_slice, per_li = per_li, in_chans=1)

# 【EEnT】【返璞归真】【插值】
def EEnT_NO11_interEleEncode_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO11_interEleEncode(num_ele_slice = num_ele_slice, per_li = per_li, in_chans=1)

# 【EEnT】【插值与as Token】
def EEnT_NO12_interEle_AsToken_bi(num_ele_slice, per_li, **kwargs):
    return Py08EEnT_NO2.EEnT_NO12_interEle_AsToken(num_ele_slice = num_ele_slice, per_li = per_li, in_chans=1)

# 【GRET】
def GRET_NO1_bi(num_ele_slice, per_li, **kwargs):
    return Py14_SO2.GRET_NO1(in_chans=1)

# 【GRET】
def GRET_NO2_NormalPatchEm_bi(num_ele_slice, per_li, **kwargs):
    return Py14_SO2.GRET_NO2_NormalPatchEm(in_chans=1)

# 【ViT】纯ViT测试
def ViT_Tiny_bi(num_ele_slice, per_li, **kwargs):
    return ViT_test.vit_tiny(in_chans=1, local_up_to_layer=0)

# 【ViT】纯ViT测试V2，使用BiNet中一样的ViT：替换层
def ViT_Tiny_repalce_bi(num_ele_slice, per_li, **kwargs):
    return ViT_test.get_Vit_model_local()

# 【GRET】
def GRET_DB_NO1_bi(num_ele_slice, per_li, **kwargs):
    return Py15_SO2_DinoBase.GRET_DB_NO1(in_chans=1, local_up_to_layer=3)

# 【GRET】
def GRET_ELE_DB_NO2_bi(num_ele_slice, per_li, **kwargs):
    return Py15_SO2_DinoBase.GRET_ELE_DB_NO2(num_ele_slice = num_ele_slice, per_li = per_li, in_chans=1, local_up_to_layer=9)




