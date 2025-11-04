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

import json
import time
import sys, os, pickle
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union, Mapping

import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset
import torch.distributed as dist

from datasets.esm_featurizer import ESMFeaturizer
from datasets.json_to_feature import SampleDictToFeatures

from datasets.data_pipeline import DataPipeline
from datasets.featurizer import Featurizer
from datasets.tokenizer import TokenArray
from datasets.utils import (
    data_type_transform,
    get_antibody_clusters,
    make_dummy_feature,
)
from utils.logger import get_logger
from utils.torch_utils import dict_to_tensor
from utils.distributed import DIST_WRAPPER

logger = get_logger(__name__)

class DistilledDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        use_msa: bool = False,
        configs=None,
        crop_size=-1,
    ) -> None:

        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        self.configs = configs
        self.crop_size = crop_size
        
        self.json_files = []
        self.all_inputs = []
        self.all_labels = []
        # count = 0 ###
        if os.path.isdir(input_json_path):
            if os.path.exists(os.path.join(input_json_path, "id_labels.pkl")):
                id_labels = pickle.load(open(os.path.join(input_json_path, "id_labels.pkl"), 'rb'))
                for ID in id_labels:
                    json_file = os.path.join(input_json_path, ID+'.json') 
                    try:
                        with open(json_file, "r") as f:
                            file_data = json.load(f)
                            # if count > 500: break
                            # count += 1
                            self.all_inputs.extend(file_data)
                            self.json_files.append(json_file)
                            self.all_labels.append(id_labels[ID])
                    except Exception as e:
                        logger.warning(f"Failed to load {json_file}: {e}")
            else:
                for fn in os.listdir(input_json_path):
                    if '.json' in fn:
                        json_file = os.path.join(input_json_path, fn) 
                        try:
                            with open(json_file, "r") as f:
                                file_data = json.load(f)
                                # if count > 500: break
                                # count += 1
                                self.all_inputs.extend(file_data)
                                self.json_files.append(json_file)
                                self.all_labels.append(0)
                        except Exception as e:
                            logger.warning(f"Failed to load {json_file}: {e}")
        else:
            raise ValueError(f"Input path {input_json_path} is neither a file nor a directory")

        if self.all_inputs and self.configs:
            json_task_name = os.path.basename(input_json_path)
            esm_info = configs.get("esm", {})
            configs.esm.embedding_dir = f"./esm_embeddings/{configs.run_name}/{json_task_name}/{configs.esm.model_name}"
            configs.esm.sequence_fpath = (
                f"./esm_embeddings/{configs.run_name}/{json_task_name}/{json_task_name}_prot_sequences.csv"
            )
            self.esm_enable = esm_info.get("enable", False)
            if self.esm_enable:
                os.makedirs(configs.esm.embedding_dir, exist_ok=True)
                os.makedirs(os.path.dirname(configs.esm.sequence_fpath), exist_ok=True)
                if self.configs.precompute_esm:
                    ESMFeaturizer.precompute_esm_embedding(
                        self.all_inputs,  # 使用所有样本
                        configs.esm.model_name,
                        configs.esm.embedding_dir,
                        configs.esm.sequence_fpath,
                        configs.load_checkpoint_dir,
                        configs.gpuid
                    )
                else: logger.info(f"Existing esm features, skip...")

                self.esm_featurizer = ESMFeaturizer(
                    embedding_dir=esm_info.embedding_dir,
                    sequence_fpath=esm_info.sequence_fpath,
                    embedding_dim=esm_info.embedding_dim,
                    error_dir=f"./esm_embeddings/{configs.run_name}/{json_task_name}/",
                )
        # self.cache_dir = f"./cache_tensors/{configs.run_name}"
        # if DIST_WRAPPER.rank == 0:
        #     os.makedirs(self.cache_dir, exist_ok=True)


    def __len__(self) -> int:
        return len(self.all_inputs)


    def __getitem__(self, index: int):
        single_sample_dict = self.all_inputs[index]
        sample_name = single_sample_dict["name"]

        # cache_path = f"{self.cache_dir}/{sample_name}.pt"
        # if os.path.exists(cache_path):
        #     try:
        #         data = torch.load(cache_path, map_location="cpu")
        #         return data
        #     except Exception as e:
        #         logger.warning(f"Cached {cache_path} corrupt, regenerating: {e}")
        try:
            data, _, _ = self.process_one(single_sample_dict=single_sample_dict)

            data["sample_name"] = sample_name
            data["sample_index"] = index
            data["sample_label"] = torch.tensor([self.all_labels[index]], dtype=torch.float32)

            # if DIST_WRAPPER.rank == 0:
            #     try:
            #         torch.save(data, cache_path)
            #     except Exception as e:
            #         logger.warning(f"Failed to save cache {cache_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to process sample {index}: {e}")
            return None

        return data


    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        t0 = time.time()
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type

        # Crop
        try:
            (crop_method, cropped_token_array, cropped_atom_array) = self.crop(
                bioassembly_dict={'token_array':token_array, 'atom_array':atom_array},
                crop_size=self.crop_size,
                method_weights=[1.0, 0.0, 0.0],
                contiguous_crop_complete_lig=True,
                spatial_crop_complete_lig=False,
                drop_last=True,
                remove_metal=True,
            )
        except Exception as e:
            print(e)
            sys.exit(0)
        t1 = time.time()

        # all_features
        feat = self.get_feature_and_label(
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            full_atom_array=atom_array,
            is_spatial_crop="spatial" in crop_method.lower(),
            bioassembly_dict = single_sample_dict
        )

        t2 = time.time()

        # 获得输入数据
        data = {}
        data["input_feature_dict"] = feat

        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        # N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                # "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        return data, atom_array, time_tracker

    def crop(
        self,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = True,
        remove_metal: bool = True,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crops the bioassembly data based on the specified configurations.

        Returns:
            A tuple containing the cropping method, cropped token array, cropped atom array,
                cropped MSA features, and cropped template features.
        """
        return DataPipeline.crop(
            bioassembly_dict=bioassembly_dict,
            crop_size=crop_size,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

    def get_feature_and_label(
        self,
        # idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        msa_features: dict[str, Any] = {},
        template_features: dict[str, Any] = {},
        full_atom_array: AtomArray = None,
        is_spatial_crop: bool = False,
        max_entity_mol_id: int = None,
        bioassembly_dict = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get feature and label information for a given data point.
        It uses a Featurizer object to obtain input features and labels, and applies several
        steps to add other features and labels. Finally, it returns the feature dictionary, label
        dictionary, and a full label dictionary.

        Args:
            idx: Index of the data point.
            token_array: Token array representing the amino acid sequence.
            atom_array: Atom array containing atomic information.
            msa_features: Dictionary of MSA features.
            template_features: Dictionary of template features.
            full_atom_array: Full atom array containing all atoms.
            is_spatial_crop: Flag indicating whether spatial cropping is applied, by default True.
            max_entity_mol_id: Maximum entity mol ID in the full atom array.
        Returns:
            A tuple containing the feature dictionary and the label dictionary.

        Raises:
            ValueError: If the ligand cannot be found in the data point.
        """
        features_dict = {}
        # if self.constraint.get("enable", False):
        #     token_array, atom_array, features_dict, msa_features, full_atom_array = (
        #         self.get_constraint_feature(
        #             idx,
        #             atom_array,
        #             token_array,
        #             msa_features,
        #             max_entity_mol_id,
        #             full_atom_array,
        #         )
        #     )

        # Get feature and labels from Featurizer
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=True,
            lig_atom_rename=False,
        )

        features_dict.update(feat.get_all_input_features())
        # labels_dict = feat.get_labels()

        # # Permutation list for atom permutation
        # features_dict["atom_perm_list"] = feat.get_atom_permutation_list()

        # Labels for multi-chain permutation
        # Note: the returned full_atom_array may contain fewer atoms than the input
        # label_full_dict, full_atom_array = Featurizer.get_gt_full_complex_features(
        #     atom_array=full_atom_array,
        #     cropped_atom_array=atom_array,
        #     get_cropped_asym_only=is_spatial_crop,
        # )

        # # Masks for Pocket Metrics
        # if self.find_pocket:
        #     # Get entity_id of the interested ligand
        #     sample_indice = self._get_sample_indice(idx=idx)
        #     if sample_indice.mol_1_type == "ligand":
        #         lig_entity_id = str(sample_indice.entity_1_id)
        #         lig_chain_id = str(sample_indice.chain_1_id)
        #     elif sample_indice.mol_2_type == "ligand":
        #         lig_entity_id = str(sample_indice.entity_2_id)
        #         lig_chain_id = str(sample_indice.chain_2_id)
        #     else:
        #         raise ValueError(f"Cannot find ligand from this data point.")
        #     # Make sure the cropped array contains interested ligand
        #     assert lig_entity_id in set(atom_array.label_entity_id)
        #     assert lig_chain_id in set(atom_array.chain_id)

        #     # Get asym ID of the specific ligand in the `main` pocket
        #     lig_asym_id = atom_array.label_asym_id[atom_array.chain_id == lig_chain_id]
        #     assert len(np.unique(lig_asym_id)) == 1
        #     lig_asym_id = lig_asym_id[0]
        #     ligands = [lig_asym_id]

        #     if self.find_all_pockets:
        #         # Get asym ID of other ligands with the same entity_id
        #         all_lig_asym_ids = set(
        #             full_atom_array[
        #                 full_atom_array.label_entity_id == lig_entity_id
        #             ].label_asym_id
        #         )
        #         ligands.extend(list(all_lig_asym_ids - set([lig_asym_id])))

        #     # Note: the `main` pocket is the 0-indexed one.
        #     # [N_pocket, N_atom], [N_pocket, N_atom].
        #     # If not find_all_pockets, then N_pocket = 1.
        #     interested_ligand_mask, pocket_mask = feat.get_lig_pocket_mask(
        #         atom_array=full_atom_array, lig_label_asym_id=ligands
        #     )

        #     label_full_dict["pocket_mask"] = pocket_mask
        #     label_full_dict["interested_ligand_mask"] = interested_ligand_mask

        # Masks for Chain/Interface Metrics
        # if self.find_eval_chain_interface:
        #     eval_type, cluster_id, chain_1_mask, chain_2_mask = (
        #         self._get_eval_chain_interface_mask(
        #             idx=idx, atom_array_chain_id=full_atom_array.chain_id
        #         )
        #     )
        #     labels_dict["eval_type"] = eval_type  # [N_eval]
        #     labels_dict["cluster_id"] = cluster_id  # [N_eval]
        #     labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
        #     labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        # Esm features
        if self.esm_enable:
            x_esm = self.esm_featurizer(
                token_array=token_array,
                atom_array=atom_array,
                bioassembly_dict=bioassembly_dict,
                inference_mode=True,
            )
            features_dict["esm_token_embedding"] = x_esm

        # Make dummy features for not implemented features
        dummy_feats = []
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)

        features_dict = make_dummy_feature(
            features_dict=features_dict, dummy_feats=dummy_feats
        )
        # Transform to right data type
        features_dict = data_type_transform(feat_or_label_dict=features_dict)
        # labels_dict = data_type_transform(feat_or_label_dict=labels_dict)

        # Is_distillation
        # features_dict["is_distillation"] = torch.tensor([self.is_distillation])
        # if self.is_distillation is True:
        #     features_dict["resolution"] = torch.tensor([-1.0])
        # return features_dict, labels_dict, label_full_dict
        return features_dict