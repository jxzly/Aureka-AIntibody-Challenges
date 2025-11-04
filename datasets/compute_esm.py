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

import argparse
import os

import pandas as pd
import torch
from esm import FastaBatchedDataset, pretrained
from tqdm.auto import tqdm

ESM_CONFIG = {
    "esm2-3b": {
        "type": "esm2",
        "model_path": "esm2_t36_3B_UR50D.pt",
        "emb_dim": 2560,
        "n_layers": 36,
    },
    "esm2-3b-ism": {
        "type": "esm2",
        "model_path": "esm2_t36_3B_UR50D_ism.pt",
        "emb_dim": 2560,
        "n_layers": 36,
    },  # https://www.biorxiv.org/content/10.1101/2024.11.08.622579v2
}

def _load_esm2_model(model_path):
    if os.path.exists(model_path):
        model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
        # model, alphabet = pretrained.load_model_and_alphabet_local(model_path, weights_only=False)
    else:
        model, alphabet = pretrained.load_model_and_alphabet(
            os.path.splitext(os.path.basename(model_path))[0]
        )
    return model, alphabet

def load_esm_model(model_name, local_esm_dir="release_data/checkpoint", gpuid=1):
    local_model_path = os.path.join(local_esm_dir, ESM_CONFIG[model_name]["model_path"])
    if os.path.exists(local_model_path):
        print("Try to load ESM language model from ", local_model_path)

    if "ism" in model_name and not os.path.exists(local_model_path):
        raise RuntimeError(
            f"esm2-3b-ism model: {local_model_path} does not exist \n"
            + "this model can not be download from fair-esm, \n"
            + "download it from https://af3-dev.tos-cn-beijing.volces.com/release_model/esm2_t36_3B_UR50D_ism.pt"
        )
    if model_name.startswith("esm2"):
        model, alphabet = _load_esm2_model(local_model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(gpuid)

    return model, alphabet


def _check_files_exist(save_dir, labels):
    return all(
        [os.path.exists(os.path.join(save_dir, label + ".pt")) for label in labels]
    )

def compute_ESM_embeddings(
    model_name,
    model,
    alphabet,
    labels,
    sequences,
    save_dir,
    toks_per_batch=4096,
    truncation_seq_length=1022,
    gpuid=0,
):
    if model_name.startswith("esm2"):
        embeddings = compute_esm2_embeddings(
            model,
            alphabet,
            labels,
            sequences,
            save_dir,
            toks_per_batch,
            truncation_seq_length,
            gpuid=gpuid
        )
    return embeddings


# Adapt from Corso, Gabriele, et al. "Diffdock: Diffusion steps, twists, and turns for molecular docking."
# URL: https://github.com/gcorso/DiffDock/blob/main/utils/inference_utils.py
def compute_esm2_embeddings(
    model,
    alphabet,
    labels,
    sequences,
    save_dir,
    toks_per_batch=4096,
    truncation_seq_length=1022,
    gpuid=0
):
    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )
    repr_layer = model.num_layers
    embeddings = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if _check_files_exist(save_dir, labels):
                continue
            if torch.cuda.is_available():
                toks = toks.to(device=f"cuda:{str(gpuid)}", non_blocking=True)
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            representation = out["representations"][repr_layer].to(device="cpu")
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representation[i, 1 : truncate_len + 1].clone()
                save_path = os.path.join(save_dir, label + ".pt")
                torch.save(embeddings[label], save_path)
    return embeddings