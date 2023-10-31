
import argparse
import utils

import utils
import vision_transformer as vits
import numpy as np
from PIL import Image
from einops import rearrange

from Py01shared_code import Proj_layer

import torch
import torch.distributed as dist

@torch.no_grad()
def extract_features_geoTXT(model, data_loader, use_cuda=True, multiscale=False):
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

class dataset_from_geoTXT():
    def __init__(self, map_raw, shape, stride):
        self.map_raw = map_raw
        self.shape = shape
        self.stride = stride
        self.n_rows = (map_raw.shape[0] - shape[0]) // stride + 1
        self.n_cols = (map_raw.shape[1] - shape[1]) // stride + 1
        self.n_matrices = self.n_rows * self.n_cols
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
        cropped_data = self.map_raw[start_i:start_i+self.shape[0], start_j:start_j+self.shape[1]]
        
        # 对中心点进行编号
        center_i = start_i + self.shape[0] // 2
        center_j = start_j + self.shape[1] // 2
        
        return (torch.tensor(cropped_data, dtype = torch.float32).unsqueeze(0),
        torch.tensor(idx, dtype = torch.long),
        torch.tensor(np.array((center_i, center_j)), dtype = torch.long))


def load_Pre_ViT_No1(args):
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    
    # 维度更改
    PatchEmbed_1ch = Proj_layer()
    model.patch_embed = PatchEmbed_1ch
    
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    return model

# def extract_features_geoTXT(model, data_loader, use_cuda=True, multiscale=False):
#     pass
    
def get_args(args_self):
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='./checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_tiny', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default='./dump_features',
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='./Data_geo/深度学习地形数据', type=str)
    parser.add_argument('--box_label_path', default="./path_to【Label】.csv", type=str)
    parser.add_argument('--num_ele_slice', default=100, type=int, help='num_ele_slice, 切分高度的层数')
    args = parser.parse_args(args=args_self)
    return args