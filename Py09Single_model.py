# 本文件中为定义单输入结构网络的代码，包含创建模型与加载模型
# 模型均以 "_s" 结尾表示单结构
# 加载预训练函数，均以 load_pre_ 开头加模型名称
import timm
import torch
import torch.nn as nn
import utils
from functools import partial
from vision_transformer import VisionTransformer

from Py01shared_code import Proj_layer
# from functools import partial

def load_Pre_s_type1(model, args):
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
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
    setattr(model, 'embed_dim', fea_dim)
    setattr(model, 'DONOT_CHANGE_HeadFC', True)
    return model


# def get_Vit_model_local():
#     model = vit_tiny(patch_size=16, num_classes=0, embed_dim=192)
    
#     # 维度更改
#     PatchEmbed_1ch = Proj_layer()
#     model.patch_embed = PatchEmbed_1ch
#     return model

def get_convnext_any(fea_dim = 16, netname = "convnext_atto_ols"):
    # 支持模型：timm库中的所有convnext模型
    # 更改输入图片通道数与输出维度
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

def get_vit_any(embed_dim=192, patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def resnet18_s(fea_dim = 192):
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
    setattr(model, 'embed_dim', fea_dim)
    return model


# ViT嵌入96+ convnext 96组合
class BiNet_ViT96_Convnext96_s(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = get_convnext_model(fea_dim = 96)
        self.model_vit = get_vit_any(embed_dim = 96, in_chans=1)
        self.embed_dim = 96 + 96
        self.vit_em_dim = 96
        self.conv_em_dim = 96
 
    def forward(self, data):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
#         fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
        output = torch.cat((fea_conv, fea_vit), dim = -1)
        return output

# 很占显存
def Convnext208_s():
    model = get_convnext_model(fea_dim = 208)
    return model
    
# 尝试使用小的atto 全224归一化时，batchsize降到50勉强能训
def Convnextv2_atto_208_s():
    model = get_convnext_any(fea_dim = 208, netname = "convnext_atto_ols")
    return model

# ViT嵌入96+ convnext 96组合
class BiNet_ViT96_Convnextv2_atto_96_s(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = get_convnext_any(fea_dim = 96, netname = "convnext_atto_ols")
        self.model_vit = get_vit_any(embed_dim = 96, in_chans=1)
        self.embed_dim = 96 + 96
        self.vit_em_dim = 96
        self.conv_em_dim = 96
 
    def forward(self, data):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data)
        # 维度复制
#         print(fea_vit.shape)
#         print(fea_conv.shape)
#         fea_conv = fea_conv.repeat(fea_vit.shape[0] // fea_conv.shape[0],1)
        output = torch.cat((fea_conv, fea_vit), dim = -1)
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
 
    def forward(self, data):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output
    
def vit_tiny_192_s():
    model = get_vit_any(embed_dim = 192, in_chans=1)
    return model

def resnet18_PulsBN_dim16_s(fea_dim = 16):
    # 使用模型
    model = resnet18_s(fea_dim = fea_dim)
    # 首层加上BN
    model.conv1 = nn.Sequential(
        nn.BatchNorm2d(1),
        model.conv1
    )
    return model


def resnet18_dim16_s(fea_dim = 16):
    # 使用模型
    model = resnet18_s(fea_dim = fea_dim)
    return model

# ViT嵌入192+ resnet18_PulsBN 16组合
class vit192_res_PlusBN16_s(nn.Module):
    def __init__(self):
        super().__init__()
        vit_dim = 192
        conv_dim = 16
        self.model_conv = resnet18_PulsBN_dim16_s(fea_dim = conv_dim)
        self.model_vit = get_vit_any(embed_dim = vit_dim, in_chans=1)
        self.embed_dim = vit_dim + conv_dim
        self.vit_em_dim = vit_dim
        self.conv_em_dim = conv_dim
 
    def forward(self, data):
        fea_vit  = self.model_vit(data)
        fea_conv = self.model_conv(data)
        output = torch.cat((fea_vit, fea_conv), dim = -1)
        return output
    
def vit_tiny_replace_Proj_192_s():
    model = get_vit_any(embed_dim = 192)
    
    # 维度更改
    PatchEmbed_1ch = Proj_layer()
    model.patch_embed = PatchEmbed_1ch
    return model