import json
import logging
import os
import time
import traceback
import sys
import warnings
from typing import Any, Mapping

import torch
import pickle
import pandas as pd
from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler

from datasets.dataset import DistilledDataset
from utils.torch_utils import dict_to_tensor, collate_fn_identity, collate_fn_first, collate_padding_fn, collate_padding_speed_fn
from utils.distributed import DIST_WRAPPER

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


# 3. 生成每个进程的本地样本索引（基于DistributedSampler的逻辑）
def get_local_indices(dataset, rank, world_size, shuffle):
    """模拟DistributedSampler的索引生成逻辑，获取当前进程的本地样本索引"""
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    # sampler.set_epoch(epoch)  # 确保每个epoch的shuffle一致（分布式必须）
    # 获取全局索引列表（所有进程的样本索引，长度=len(dataset)）
    global_indices = list(range(len(dataset)))
    # if shuffle:
        # 模拟DistributedSampler的打乱逻辑（基于epoch的随机种子）
        # g = torch.Generator()
        # g.manual_seed(epoch)
        # global_indices = torch.randperm(len(dataset)).tolist()
    # 划分当前进程的本地索引（与DistributedSampler一致：按rank间隔取）
    local_indices = global_indices[rank::world_size]
    return local_indices

# 4. 每个进程创建带权重的采样器
def create_weighted_distributed_sampler(dataset, weights, rank, world_size, shuffle):
    # 获取当前进程的本地样本索引（全局视角）
    local_indices = get_local_indices(dataset, rank, world_size, shuffle)
    # 提取本地索引对应的权重（本地权重列表）
    local_weights = [weights[idx] for idx in local_indices]
    # 创建本地加权采样器（基于本地索引和权重）
    # num_samples：每个epoch从本地分片中采样的数量（通常等于本地分片大小）
    # replacement=True：允许重复采样（权重低的样本可能不被采到，根据需求调整）
    g = torch.Generator()
    g.manual_seed(rank)
    weighted_sampler = WeightedRandomSampler(
        weights=weights,#local_weights,
        num_samples=len(local_indices),
        replacement=True,
        generator=g
    )
    return weighted_sampler, local_indices

def get_dataloader(configs: Any, batchsize=1, shuffle=False, input_json_path="", mode="test",weights=None) -> DataLoader:
    if mode == 'train':
        crop_size = configs.train_crop_size
        drop_last = True
    else:
        crop_size = -1
        drop_last = False
    distilled_dataset = DistilledDataset(
        input_json_path=input_json_path,
        configs=configs,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
        crop_size=crop_size
    )
    if mode == 'train':
        sampler, local_indices = create_weighted_distributed_sampler(
        dataset=distilled_dataset,
        weights=weights,
        rank=DIST_WRAPPER.rank,
        world_size=DIST_WRAPPER.world_size,
        shuffle=shuffle,
        )
    else:
        sampler = DistributedSampler(
            dataset=distilled_dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=shuffle,
        )
    dataloader = DataLoader(
        dataset=distilled_dataset,
        batch_size=batchsize,
        sampler=sampler,
        # collate_fn=collate_fn_custom,
        collate_fn=collate_padding_speed_fn,
        num_workers=configs.num_workers,
        prefetch_factor=2,
        drop_last=drop_last,
    )
    return dataloader
