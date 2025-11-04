import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

from models.protenix import Protenix
from models.binder import BinderHead, AffinityHead
from utils.torch_utils import autocasting_disable_decorator


class DistilledBinder(nn.Module):
    def __init__(self, configs):
        super(DistilledBinder, self).__init__()
        self.configs = configs
        self.N_cycle = self.configs.model.N_cycle
        self.protenix_model = Protenix(configs)
        self.Binder_head = AffinityHead(**configs.model.binder_head)
        # self.Binder_head = BinderHead(**configs.model.binder_head)
        
    def forward(
        self,
        input_feature_dict: dict[str, Any],
        # label_full_dict: dict[str, Any],
        # label_dict: dict[str, Any],
    ):
        inplace_safe = not torch.is_grad_enabled()
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None
        ### AF3 pairformer 输出
        s_inputs, s, z = self.protenix_model.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size
        )
        
        # keys_to_delete = []
        # for key in input_feature_dict.keys():
        #     if "template_" in key or key in [
        #         "msa",
        #         "has_deletion",
        #         "deletion_value",
        #         "profile",
        #         "deletion_mean",
        #         "token_bonds",
        #     ]:
        #         keys_to_delete.append(key)

        # for key in keys_to_delete:
        #     del input_feature_dict[key]
        # torch.cuda.empty_cache()
    
        ### Binding预测
        pair_mask = input_feature_dict['seq_mask'].unsqueeze(2) * input_feature_dict['seq_mask'].unsqueeze(1)
        output = self.run_head(
            s_inputs,
            s,
            z,
            single_mask=input_feature_dict['seq_mask'],
            pair_mask=pair_mask,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        
        return output


    def run_head(self, *args, **kwargs):
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.Binder_head
        )(*args, **kwargs)