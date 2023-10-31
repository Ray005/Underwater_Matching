import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
import numpy as np
from PIL import Image
from einops import rearrange

from Py01shared_code import TerrainDataset, Proj_layer, TerrainDataset_withXY

def extract_feature_pipeline_forTerrainDataset_withXY(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
#         pth_transforms.Resize(256, interpolation=3),
#         pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean=[0.485], std=[0.229]),
#         pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_train = TerrainDataset_withXY(os.path.join(args.data_path), args.box_label_path, transform=transform, len_pre=7, len_suf=4)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
        
    # 维度更改
    PatchEmbed_1ch = Proj_layer()
    model.patch_embed = PatchEmbed_1ch
    
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features_forTerrainDataset_withXY(model, data_loader_train, args.use_cuda)
#     print("Extracting features for val set...")
#     test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
#         test_features = nn.functional.normalize(test_features, dim=1, p=2)

#     train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
#     test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()

#     train_labels = torch.zeros([dataset_train.__len__()]).long()
    print("Reading Labels from train set...")
    train_labels = torch.tensor([dataset_train.__getitem__(i)[-2] for i in range(0, len(dataset_train))]).long()
    print("Reading Bounding Box Info from train set...")
    train_box    = torch.tensor([dataset_train.__getitem__(i)[-1] for i in range(0, len(dataset_train))])
#     test_labels = torch.zeros([dataset_val.__len__()]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
#         torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(train_box.cpu(),    os.path.join(args.dump_features, "train_box.pth"))
#         torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, train_labels, train_box

@torch.no_grad()
def extract_features_forTerrainDataset_withXY(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index, box in metric_logger.log_every(data_loader, 10):
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

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features
    
# class ReturnIndexDataset_withXY(TerrainDataset_withXY):
#     def __getitem__(self, idx):
#         img, index, box = super(ReturnIndexDataset, self).__getitem__(idx)
#         return img, index, box
    
class Dataset_fea_XY():
    def __init__(self, train_fea_in, train_labels, train_boxes):
#         self.images = torch.permute(torch.from_numpy(train_images),(0,3,1,2)).float()
        self.fea_in = train_fea_in
        self.labels = train_labels.type(torch.LongTensor)
        self.boxes = train_boxes.float()

    def __len__(self):
        return len(self.labels)

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        return (self.fea_in[idx],
              self.labels[idx],
              self.boxes[idx])

# Inheriting from Dataset class
class ValDataset_fea_XY(Dataset_fea_XY):

    def __init__(self, val_fea_in, val_labels, val_boxes):

#         self.images = torch.permute(torch.from_numpy(val_fes),(0,3,1,2)).float()
        self.fea_in = val_fea_in
        self.labels = val_labels.type(torch.LongTensor)
        self.boxes = torch.from_numpy(val_boxes).float()