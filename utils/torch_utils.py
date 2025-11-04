# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from typing import Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def batch_avg_with_mask(
    value: torch.Tensor,
    mask: torch.Tensor,
    avg_dim: Union[int, tuple[int]] = None,
    batch_reduction: str = "mean",
    eps: float = 1e-12,
):
    """Average values with mask.
    Args:
        value: tensor of shape [BS, ...]
        mask: tensor with same shape and type of value, 1 means valid, 0 means maksed
        avg_dim: dimensions to apply average, if None, all dims excluding BS dim will be averaged
        batch_reduction: mean/sum/none, reduction operation applied on BS dim
    """
    if avg_dim is None:
        avg_dim = tuple(range(1, len(value.shape)))
    avg = (value * mask).sum(dim=avg_dim) / (mask.sum(dim=avg_dim) + eps)
    if batch_reduction == "mean":
        return avg.mean()
    elif batch_reduction == "sum":
        return avg.sum()
    elif batch_reduction == "none":
        return avg
    else:
        raise Exception(f"Invalid batch_reduction: {batch_reduction}")


def dict_to_tensor(feature_dict):
    for k, v in feature_dict.items():
        if not isinstance(v, torch.Tensor):
            dtype = feature_dict[k].dtype
            feature_dict[k] = torch.tensor(v)

            if dtype in [np.int64, np.int32]:
                feature_dict[k] = feature_dict[k].to(torch.int64)
            elif dtype in [np.float32, np.float64]:
                feature_dict[k] = feature_dict[k].to(torch.float32)

    return feature_dict


def to_device(obj, device):
    """Move tensor or dict of tensors to device"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                to_device(v, device)
            elif isinstance(v, torch.Tensor):
                obj[k] = obj[k].to(device)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_device(item, device) for item in obj]
    else:
        raise Exception(f"type {type(obj)} not supported")
    return obj


def autocasting_disable_decorator(disable_casting):
    def func_wrapper(func):
        def new_func(*args, **kwargs):
            _amp_context = (
                torch.autocast(device_type="cuda", enabled=False)
                if disable_casting
                else nullcontext()
            )

            # Helper function to conditionally cast tensors
            def conditioned_cast(tensor):
                if (
                    disable_casting
                    and isinstance(tensor, torch.Tensor)
                    and torch.is_floating_point(tensor)
                ):
                    return tensor.to(dtype=torch.float32)
                return tensor

            with _amp_context:
                return func(
                    *(conditioned_cast(v) for v in args),
                    **{k: conditioned_cast(v) for k, v in kwargs.items()},
                )

        return new_func

    return func_wrapper


def collate_fn_identity(x):
    return x

def collate_fn_first(x):
    return x[0]

def recursive_collate(values):
    if not values: return values
    
    ele = values[0]
    if torch.is_tensor(ele):
        try:
            if ele.dim() == 0: return torch.stack([v.unsqueeze(0) for v in values], dim=0).squeeze(1)
            else:    return torch.stack(values, dim=0)
        except Exception as e:
            print(e)
    elif isinstance(ele, str):
        return values
    elif isinstance(ele, dict):
        result_dict = {}
        keys = ele.keys()
        if not all(k in v.keys() for v in values[1:] for k in keys):
            raise ValueError("All dictionaries must have the same keys")
        for key in keys:
            result_dict[key] = recursive_collate([v[key] for v in values])
        return result_dict
    else:
        return values

def collate_fn_custom(batch):
    if not batch: return batch
    return recursive_collate(batch)


def calculate_max_lengths(lengths):
    # Step 1: Initialize max_token_len and max_atom_len
    max_token_len = max(token_len for token_len, _ in lengths)
    max_atom_len = max(atom_len for _, atom_len in lengths)

    # Step 2: Adjust max_atom_len to satisfy the second condition
    for token_len, atom_len in lengths:
        required_padding_tokens = max_token_len - token_len
        available_padding_atoms = max_atom_len - atom_len

        if required_padding_tokens > available_padding_atoms:
            # Increase max_atom_len to ensure max_atom_len - atom_len >= max_token_len - token_len
            max_atom_len += required_padding_tokens - available_padding_atoms
    
    # Step 3: Extra check for the case where max_token_len - token_len == 0
    for token_len, atom_len in lengths:
        if token_len == max_token_len and atom_len < max_atom_len:
            # # If a sample has max_token_len but atom_len is smaller
            ## 1 more token for leave atoms
            max_atom_len = max_atom_len + 1
            max_token_len = max_token_len + 1

    # Step 4: Return the final max_token_len and max_atom_len
    return max_token_len, max_atom_len


## TODO: implement the padding for batchsize > 1
def collate_padding_fn(batch):
    ## 1. get the max length of token and atom
    max_token_len = max(item['N_token'] for item in batch)
    max_atom_len = max(item['N_atom'] for item in batch)
    
    ## 2. some samples maybe have fewer tokens, but more atoms, so we will update the max_token_len and max_atom_len
    lengths = []
    for item in batch:
        token_len = item['input_feature_dict']['residue_index'].shape[0]
        atom_len = item['input_feature_dict']['atom_to_token_idx'].shape[0]
        
        token_padding_size = max_token_len - token_len
        atom_padding_size = max_atom_len - atom_len
        lengths.append((token_len, atom_len))

    max_token_len, max_atom_len = calculate_max_lengths(lengths)
    
    # padding
    features_padded = []
    for idx in range(len(batch)):
        features_tmp = batch[idx]['input_feature_dict']
        
        token_len = features_tmp['residue_index'].shape[0]
        atom_len = features_tmp['atom_to_token_idx'].shape[0]

        token_padding_size = max_token_len - token_len
        atom_padding_size = max_atom_len - atom_len
        
        ## TODO: add seq mask
        seq_mask = torch.ones(
            features_tmp["restype"].shape[0], dtype=torch.float32
        )
        features_tmp['seq_mask'] = np.pad(seq_mask, (0, token_padding_size), 'constant', constant_values=(0, 0))

        ## TODO: add atom mask
        all_atom_mask = torch.ones(
            features_tmp["atom_to_token_idx"].shape[0], dtype=torch.float32
        )
        features_tmp['all_atom_mask'] = np.pad(all_atom_mask, (0, atom_padding_size), 'constant', constant_values=(0, 0))
        
        # DEBUG: move, after add seq_mask, atom_mask
        if max_token_len == token_len and max_atom_len == atom_len:
            features_padded.append(features_tmp)
            continue

        token2atom_nums = atom_padding_size // token_padding_size
        token2atom_nums_list = [token2atom_nums] * token_padding_size
        token2atom_nums_list[-1] = atom_padding_size - token2atom_nums * (token_padding_size-1)  ## the last one

        ## padding tokens
        token_level_padding_0_keys = {'residue_index'}
        for key in token_level_padding_0_keys:
            features_tmp[key] = np.pad(features_tmp[key], (0, token_padding_size), 'constant', constant_values=(0, 0))        

        features_tmp['restype'] = np.pad(features_tmp['restype'], ((0, token_padding_size), (0, 0)), 'constant', constant_values=(0, 0))
        features_tmp['asym_id'] = np.pad(features_tmp['asym_id'], (0, token_padding_size), 'constant', constant_values=(max(features_tmp['asym_id'])+1, max(features_tmp['asym_id'])+1))  ## a new chain for padding
        features_tmp['sym_id'] = np.pad(features_tmp['sym_id'], (0, token_padding_size), 'constant', constant_values=(1, 1))  
        features_tmp['entity_id'] = np.pad(features_tmp['entity_id'], (0, token_padding_size), 'constant', constant_values=(max(features_tmp['entity_id'])+1, max(features_tmp['entity_id'])+1))  ## a new entity for padding
        features_tmp['token_index'] = torch.cat([features_tmp['token_index'], torch.arange(token_padding_size)+max(features_tmp['token_index'])+1], axis=0)

        features_tmp['has_frame'] = np.pad(features_tmp['has_frame'], (0, token_padding_size), 'constant', constant_values=(0, 0)) 
        features_tmp['frame_atom_index'] = np.pad(features_tmp['frame_atom_index'], ((0, token_padding_size), (0, 0)), 'constant', constant_values=0) 

        features_tmp['token_bonds'] = np.pad(features_tmp['token_bonds'], ((0, token_padding_size), (0, token_padding_size)), mode='constant', constant_values=0)

        features_tmp['esm_token_embedding'] = np.pad(features_tmp['esm_token_embedding'], ((0, token_padding_size), (0, 0)), 'constant', constant_values=(0, 0))

        ## padding atoms
        atom_level_padding_0_keys = {'is_protein', 'is_dna', 'is_rna', 'is_ligand'}
        for key in atom_level_padding_0_keys:
            features_tmp[key] = np.pad(features_tmp[key], (0, atom_padding_size), 'constant', constant_values=(0, 0))

        features_tmp['atom_to_token_idx'] = np.pad(features_tmp['atom_to_token_idx'], (0, atom_padding_size), 'constant', constant_values=max(features_tmp['atom_to_token_idx'])+1)
        features_tmp['atom_to_tokatom_idx'] = np.pad(features_tmp['atom_to_tokatom_idx'], (0, atom_padding_size), 'constant', constant_values=(0, 0))

        features_tmp['mol_id'] = np.pad(features_tmp['mol_id'], (0, atom_padding_size), 'constant', constant_values=(max(features_tmp['mol_id'])+1, max(features_tmp['mol_id'])+1)) 
        features_tmp['mol_atom_index'] = np.pad(features_tmp['mol_atom_index'], (0, atom_padding_size), 'constant', constant_values=(0, 0))
        features_tmp['entity_mol_id'] = np.pad(features_tmp['entity_mol_id'], (0, atom_padding_size), 'constant', constant_values=(max(features_tmp['entity_mol_id'])+1, max(features_tmp['mol_id'])+1)) 

        ## ref
        features_tmp['ref_pos'] = np.pad(features_tmp['ref_pos'], ((0, atom_padding_size), (0, 0)), 'constant', constant_values=(0.0, 0.0))
        features_tmp['ref_mask'] = np.pad(features_tmp['ref_mask'], (0, atom_padding_size), 'constant', constant_values=(0, 0))
        features_tmp['ref_element'] = np.pad(features_tmp['ref_element'], ((0, atom_padding_size), (0, 0)), 'constant', constant_values=(0, 0))   ## use C for padding
        features_tmp['ref_charge'] = np.pad(features_tmp['ref_charge'], (0, atom_padding_size), 'constant', constant_values=(0, 0))
        features_tmp['ref_atom_name_chars'] = torch.cat([features_tmp['ref_atom_name_chars'], torch.zeros(atom_padding_size, 4, 64)], axis=0)

        ref_space_uid_padding = []
        for i, atom_nums in enumerate(token2atom_nums_list):
            ref_space_uid_padding.extend([i+1] * atom_nums)
        ref_space_uid_padding = [r + features_tmp['ref_space_uid'][-1].item() for r in ref_space_uid_padding]
        ref_space_uid_padding = torch.tensor(ref_space_uid_padding)

        features_tmp['ref_space_uid'] = torch.cat([features_tmp['ref_space_uid'], ref_space_uid_padding], axis=0)

        # mask padding
        features_tmp['pae_rep_atom_mask'] = np.pad(features_tmp['pae_rep_atom_mask'], ((0, atom_padding_size)), 'constant', constant_values=0.0)
        features_tmp['plddt_m_rep_atom_mask'] = np.pad(features_tmp['plddt_m_rep_atom_mask'], ((0, atom_padding_size)), 'constant', constant_values=0.0)
        features_tmp['distogram_rep_atom_mask'] = np.pad(features_tmp['distogram_rep_atom_mask'], ((0, atom_padding_size)), 'constant', constant_values=0.0)
        features_tmp['modified_res_mask'] = np.pad(features_tmp['modified_res_mask'], ((0, atom_padding_size)), 'constant', constant_values=0.0)
        features_tmp['bond_mask'] = np.pad(features_tmp['bond_mask'], ((0, atom_padding_size), (0, atom_padding_size)), mode='constant', constant_values=0)


        features_tmp['msa'] = np.pad(features_tmp['msa'], ((0, 0), (0, token_padding_size)), 'constant', constant_values=(21, 21))
        features_tmp['profile'] = np.pad(features_tmp['profile'], ((0, token_padding_size), (0, 0)), 'constant', constant_values=(0, 0))
        features_tmp['has_deletion'] = np.pad(features_tmp['has_deletion'], ((0, 0), (0, token_padding_size)), 'constant', constant_values=(0, 0))
        features_tmp['deletion_mean'] = np.pad(features_tmp['deletion_mean'], (0, token_padding_size), 'constant', constant_values=(0, 0))
        features_tmp['deletion_value'] = np.pad(features_tmp['deletion_value'], ((0, 0), (0, token_padding_size)), 'constant', constant_values=(0, 0))

        features_padded.append(features_tmp)
        
    ## conconate
    all_features = {}
    for key in features_padded[0].keys():
        try:
            if key not in ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids']:
                # print(key, [item[key].shape for item in features_padded])
                all_features[key] = np.stack([item[key] for item in features_padded], axis=0)
        except:
            # print([item[key].shape for item in features_padded])
            # print(f"collate_padding_fn, {key} dim is inconsistent, skip")
            continue


    features = {}
    for k, v in all_features.items():
        try:
            if k in ['ref_element', 'ref_atom_name_chars', 'residue_index', 'restype', 'asym_id', 'sym_id', 'entity_id', 'token_index', 'msa', 'num_alignments']:
                features[k] = torch.tensor(v, dtype=torch.long)
            else:
                features[k] = torch.tensor(v)
        except:
            features[k] = v

    # features['ref_element'] = F.one_hot(features['ref_element'], num_classes=128)
    # features['ref_atom_name_chars'] = F.one_hot(features['ref_atom_name_chars'], num_classes=64)
    batch_data = {'input_feature_dict': features}
    for key in batch[0].keys():
        if key not in ['input_feature_dict', 'sample_label']:
            batch_data[key] = [example[key] for example in batch]
        elif key in ['sample_label']:
            batch_data[key] = torch.tensor([[example[key]] for example in batch])
    return batch_data


def calculate_max_lengths_speed(lengths):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.long)
    
    # Step 1 & 2: 单次遍历计算初始 max_token_len 和 max_atom_len
    max_token_len, max_atom_len = lengths.max(dim=0).values

    # Step 2: 计算所需调整量
    token_lens, atom_lens = lengths[:, 0], lengths[:, 1]
    required_padding = max_token_len - token_lens
    available_padding = max_atom_len - atom_lens
    deficit = (required_padding > available_padding).long() * (required_padding - available_padding)
    max_atom_len += deficit.max()

    # Step 3: 处理 max_token_len 但 atom_len 不足的情况
    is_max_token = (token_lens == max_token_len)
    needs_extra = (is_max_token & (atom_lens < max_atom_len)).any()
    if needs_extra:
        max_token_len += 1
        max_atom_len += 1

    return max_token_len.item(), max_atom_len.item()

def collate_padding_speed_fn(batch):
    token_lens = [item['input_feature_dict']['residue_index'].shape[0] for item in batch]
    atom_lens = [item['input_feature_dict']['atom_to_token_idx'].shape[0] for item in batch]

    max_token_len, max_atom_len = calculate_max_lengths_speed(list(zip(token_lens, atom_lens)))

    token_padding_sizes = [max_token_len - tl for tl in token_lens]
    atom_padding_sizes = [max_atom_len - al for al in atom_lens]

    # padding
    features_padded = []
    for idx, item in enumerate(batch):
        features_tmp = item['input_feature_dict']

        token_padding_size = token_padding_sizes[idx]
        atom_padding_size = atom_padding_sizes[idx]
        
        if token_padding_size == 0 and atom_padding_size == 0:
            features_tmp['seq_mask'] = torch.ones(token_lens[idx], dtype=torch.float32)
            features_tmp['all_atom_mask'] = torch.ones(atom_lens[idx], dtype=torch.float32)
            features_padded.append(features_tmp)
            continue
        
        features_tmp = pad_features_batch(features_tmp, token_padding_size, atom_padding_size, token_lens[idx], atom_lens[idx])
        features_padded.append(features_tmp)

    return stack_features(features_padded, batch)

def pad_features_batch(features, token_pad, atom_pad, token_len, atom_len):
    token_pad_zeros_1d = torch.zeros(token_pad, dtype=torch.float32)
    atom_pad_zeros_1d = torch.zeros(atom_pad, dtype=torch.float32)
    ## add seq/atom masks
    features['seq_mask'] = torch.cat([torch.ones(token_len, dtype=torch.float32), token_pad_zeros_1d], dim=0)
    features['all_atom_mask'] = torch.cat([torch.ones(atom_len, dtype=torch.float32), atom_pad_zeros_1d], dim=0)

    token_1d_features = {
        'residue_index': token_pad_zeros_1d,
        'asym_id': torch.full((token_pad,), features['asym_id'].max() + 1, dtype=features['asym_id'].dtype),
        'sym_id': torch.full((token_pad,), 1, dtype=features['sym_id'].dtype),
        'entity_id': torch.full((token_pad,), features['entity_id'].max() + 1, dtype=features['entity_id'].dtype),
        'has_frame': token_pad_zeros_1d
    }
    for key, pad_tensor in token_1d_features.items():
        features[key] = pad_1d_tensor(features[key], pad_tensor)
    features['token_index'] = pad_1d_tensor(features['token_index'], torch.arange(token_pad) + features['token_index'].max() + 1)

    token_2d_features = ['restype', 'frame_atom_index', 'esm_token_embedding']
    for key in token_2d_features:
        features[key] = pad_2d_tensor(features[key], token_pad)
    # 处理token_bonds
    new_size = features['token_bonds'].size(0) + token_pad
    features['token_bonds'] = torch.cat([torch.cat([features['token_bonds'], torch.zeros((features['token_bonds'].size(0), token_pad), )], dim=1),
        torch.zeros((token_pad, new_size))], dim=0)

    atom_1d_features = {
        'is_protein': atom_pad_zeros_1d, 'is_dna': atom_pad_zeros_1d, 'is_rna': atom_pad_zeros_1d, 'is_ligand': atom_pad_zeros_1d,
        'atom_to_token_idx': torch.full((atom_pad,), features['atom_to_token_idx'].max() + 1, dtype=features['atom_to_token_idx'].dtype),
        'atom_to_tokatom_idx': atom_pad_zeros_1d,
        'mol_id': torch.full((atom_pad,), features['mol_id'].max() + 1, dtype=torch.float32),
        'mol_atom_index': atom_pad_zeros_1d,
        'entity_mol_id': torch.full((atom_pad,), features['entity_mol_id'].max() + 1, dtype=features['entity_mol_id'].dtype),
        'ref_mask': atom_pad_zeros_1d, 'ref_charge': atom_pad_zeros_1d
    }
    for key, pad_tensor in atom_1d_features.items():
        features[key] = pad_1d_tensor(features[key], pad_tensor)

    atom_2d_features = ['ref_pos', 'ref_element']
    for key in atom_2d_features:
        features[key] = pad_2d_tensor(features[key], atom_pad)

    features['ref_atom_name_chars'] = torch.cat([
        features['ref_atom_name_chars'], 
        torch.zeros((atom_pad, 4, 64), dtype=features['ref_atom_name_chars'].dtype)], dim=0)
    
    token2atom_nums = atom_pad // token_pad
    last_atom_nums = atom_pad - token2atom_nums * (token_pad - 1)
    ref_space_uid_padding = torch.cat([
        torch.full((token2atom_nums,), i+1, dtype=torch.long) for i in range(token_pad - 1)
    ] + [torch.full((last_atom_nums,), token_pad, dtype=torch.long)])
    ref_space_uid_padding += features['ref_space_uid'][-1].item()
    features['ref_space_uid'] = torch.cat([features['ref_space_uid'], ref_space_uid_padding], dim=0)
    
    # 处理MSA特征
    features['msa'] = F.pad(features['msa'], (0, token_pad, 0, 0), mode='constant', value=21)
    features['has_deletion'] = F.pad(features['has_deletion'], (0, token_pad, 0, 0), mode='constant', value=0)
    features['deletion_value'] = F.pad(features['deletion_value'], (0, token_pad, 0, 0), mode='constant', value=0)
    features['profile'] = pad_2d_tensor(features['profile'], token_pad)
    features['deletion_mean'] = pad_1d_tensor(features['deletion_mean'], token_pad_zeros_1d)
    
    return features

def pad_1d_tensor(tensor, pad_tensor):
    return torch.cat([tensor, pad_tensor], dim=0)

def pad_2d_tensor(tensor, pad_size):
    if pad_size == 0:
        return tensor
    pad_tensor = torch.zeros((pad_size, tensor.shape[1]), dtype=tensor.dtype)
    return torch.cat([tensor, pad_tensor], dim=0)

def stack_features(features_padded, original_batch):
    all_features = {}
    stack_keys = [k for k in features_padded[0].keys() if k not in ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids']]
    for k in stack_keys:
        try:
            all_features[k] = torch.stack([f[k] for f in features_padded], dim=0)
        except:
            continue
    # 类型转换
    features = {}
    for k, v in all_features.items():
        features[k] = (v.long() if k in {'ref_element', 'ref_atom_name_chars', 'residue_index', 
                    'restype', 'asym_id', 'sym_id', 'entity_id', 'token_index', 'msa', 
                    'num_alignments'} else v)
    # 组装最终batch
    batch_data = {'input_feature_dict': features}
    other_keys = [k for k in original_batch[0].keys() if k not in ('input_feature_dict', 'sample_label')]
    sample_labels = []
    other_data = {k: [] for k in other_keys}
    for example in original_batch:
        for k in other_keys:
            other_data[k].append(example[k])
        sample_labels.append([example['sample_label']])  # 确保是二维列表
    batch_data.update(other_data)
    batch_data['sample_label'] = torch.tensor(sample_labels)
    
    return batch_data

# def pad_features_batch(features, token_pad, atom_pad, token_len, atom_len):
#     token_pad_zeros_1d = torch.zeros(token_pad, dtype=torch.float32)
#     atom_pad_zeros_1d = torch.zeros(atom_pad, dtype=torch.float32)
#     ## add seq/atom masks
#     features['seq_mask'] = torch.cat([
#         torch.ones(token_len, dtype=torch.float32),
#         token_pad_zeros_1d
#     ], dim=0)
#     features['all_atom_mask'] = torch.cat([
#         torch.ones(atom_len, dtype=torch.float32),
#         atom_pad_zeros_1d
#     ], dim=0)
    
#     token_1d_features = {
#         'residue_index': 0,
#         'asym_id': features['asym_id'].max() + 1,
#         'sym_id': 1,
#         'entity_id': features['entity_id'].max() + 1,
#         'has_frame': 0
#     }
#     for key, pad_val in token_1d_features.items():
#         features[key] = pad_1d_tensor(features[key], token_pad, pad_val)
#     features['token_index'] = pad_1d_tensor(features['token_index'], token_pad, 'increment')

#     token_2d_features = ['restype', 'frame_atom_index', 'esm_token_embedding']
#     for key in token_2d_features:
#         features[key] = pad_2d_tensor(features[key], token_pad, 0)
#     # 处理token_bonds
#     features['token_bonds'] = F.pad(features['token_bonds'], 
#                                     (0, token_pad, 0, token_pad), 
#                                     mode='constant', value=0)

#     atom_1d_features = {
#         'is_protein': 0, 'is_dna': 0, 'is_rna': 0, 'is_ligand': 0,
#         'atom_to_token_idx': features['atom_to_token_idx'].max() + 1,
#         'atom_to_tokatom_idx': 0,
#         'mol_id': features['mol_id'].max() + 1,
#         'mol_atom_index': 0,
#         'entity_mol_id': features['entity_mol_id'].max() + 1,
#         'ref_mask': 0, 'ref_charge': 0
#     }
#     for key, pad_val in atom_1d_features.items():
#         features[key] = pad_1d_tensor(features[key], atom_pad, pad_val)

#     atom_2d_features = ['ref_pos', 'ref_element']
#     for key in atom_2d_features:
#         features[key] = pad_2d_tensor(features[key], atom_pad, 0)

#     features['ref_atom_name_chars'] = torch.cat([
#         features['ref_atom_name_chars'], 
#         torch.zeros((atom_pad, 4, 64), dtype=features['ref_atom_name_chars'].dtype)], dim=0)
    
#     token2atom_nums = atom_pad // token_pad
#     last_atom_nums = atom_pad - token2atom_nums * (token_pad - 1)
#     ref_space_uid_padding = torch.cat([
#         torch.full((token2atom_nums,), i+1, dtype=torch.long) for i in range(token_pad - 1)
#     ] + [torch.full((last_atom_nums,), token_pad, dtype=torch.long)])
#     ref_space_uid_padding += features['ref_space_uid'][-1].item()
#     features['ref_space_uid'] = torch.cat([features['ref_space_uid'], ref_space_uid_padding], dim=0)
    
#     # 处理MSA特征
#     features['msa'] = F.pad(features['msa'], (0, token_pad, 0, 0), mode='constant', value=21)
#     features['has_deletion'] = F.pad(features['has_deletion'], (0, token_pad, 0, 0), mode='constant', value=0)
#     features['deletion_value'] = F.pad(features['deletion_value'], (0, token_pad, 0, 0), mode='constant', value=0)
#     features['profile'] = pad_2d_tensor(features['profile'], token_pad, 0)
#     features['deletion_mean'] = pad_1d_tensor(features['deletion_mean'], token_pad, 0)
    
#     return features

# def pad_1d_tensor(tensor, pad_size, pad_value):
#     if not torch.is_tensor(tensor):
#         tensor = torch.tensor(tensor)
    
#     if pad_size == 0:
#         return tensor
    
#     if pad_value == 'increment':
#         pad_tensor = torch.arange(pad_size) + tensor.max() + 1
#     else:
#         pad_tensor = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
    
#     return torch.cat([tensor, pad_tensor], dim=0)

# def pad_2d_tensor(tensor, pad_size, pad_value):
#     if not torch.is_tensor(tensor):
#         tensor = torch.tensor(tensor)
    
#     if pad_size == 0:
#         return tensor
    
#     pad_shape = (pad_size, tensor.shape[1])
#     pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
    
#     return torch.cat([tensor, pad_tensor], dim=0)

# def pad_3d_tensor(tensor, pad_size, pad_value):
#     if not torch.is_tensor(tensor):
#         tensor = torch.tensor(tensor)
    
#     if pad_size == 0:
#         return tensor

#     original_shape = tensor.shape
#     pad_shape = (pad_size, original_shape[1], original_shape[2])

#     pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
    
#     return torch.cat([tensor, pad_tensor], dim=0)

# def stack_features(features_padded, original_batch):
#     all_features = {}
#     for key in features_padded[0].keys():
#         try:
#             if key not in ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids']:
#                 # print(key, [item[key].shape for item in features_padded])
#                 all_features[key] = torch.stack([item[key] for item in features_padded], dim=0)
#         except:
#             # print([item[key].shape for item in features_padded])
#             # print(f"collate_padding_fn, {key} dim is inconsistent, skip")
#             continue
#     # 类型转换
#     long_keys = {'ref_element', 'ref_atom_name_chars', 'residue_index', 'restype', 
#                     'asym_id', 'sym_id', 'entity_id', 'token_index', 'msa', 'num_alignments'}
#     features = {}
#     for k, v in all_features.items():
#         if k in long_keys:
#             features[k] = v.long()
#         else:
#             features[k] = v
#     # 组装最终batch
#     batch_data = {'input_feature_dict': features}
#     for key in original_batch[0].keys():
#         if key not in ['input_feature_dict', 'sample_label']:
#             batch_data[key] = [example[key] for example in original_batch]
#         elif key == 'sample_label':
#             batch_data[key] = torch.tensor([[example[key]] for example in original_batch])
    
#     return batch_data