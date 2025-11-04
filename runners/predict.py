import os
import time
# import wandb
import torch
import datetime
import hashlib
import logging
from argparse import Namespace
from contextlib import nullcontext
from os.path import join as opjoin
from typing import Any, Mapping
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from ml_collections.config_dict import ConfigDict
from sklearn import metrics
import numpy as np
import pickle
import time
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from configs.config import parse_configs, parse_sys_args, save_config
from utils.distributed import DIST_WRAPPER
from utils.torch_utils import to_device
from utils.seed import seed_everything
from datasets.dataloader import get_dataloader
from models.model import DistilledBinder
from runners.inference import download_infercence_cache

logger = logging.getLogger(__name__)

# Disable WANDB's console output capture to reduce unnecessary logging
# os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])


def Write_log(logFile, text, isPrint=True):
    if isPrint:
        print(text) 
    logFile.write(text) 
    logFile.write('\n')


class Trainer(object):
    def __init__(self, configs: Any):
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_model()
        self.init_loss()
        self.init_data()
        self.try_load_checkpoint()


    def init_env(self):
        """Init pytorch/cuda envs."""
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            if torch.cuda.device_count() > 1:   # use-ddp, multi-gpu
                self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            else:
                self.device = torch.device(f"cuda:{self.configs.gpuid}") 
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}], device-{self.device}"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )
        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed
        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # diff ddp process got diff seeds

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
        logging.info("Finished init ENV.")


    def init_basics(self):
        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate 

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        # self.run_name = self.configs.run_name
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.prediction_dir, exist_ok=True)
            os.makedirs(self.dump_dir, exist_ok=True)
            os.makedirs(self.error_dir, exist_ok=True)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, checkpoint_dir: "
            + f"{self.checkpoint_dir}, prediction_dir: {self.prediction_dir}, error_dir: {self.error_dir}"
        )


    def init_model(self):
        self.model = DistilledBinder(self.configs).to(self.device)
        
        checkpoint_model = (
            f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        )
        if not os.path.exists(checkpoint_model):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_model}]")
        self.print(
            f"Loading from {checkpoint_model}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_model, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items() 
            }

        self.model.protenix_model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict, 
        )
        
        for param in self.model.protenix_model.parameters():
            param.requires_grad = False
        self.model.protenix_model.eval()

        self.print(f"Finish loading checkpoint.")
        self.print(f'model: {str(self.model)}')
        if DIST_WRAPPER.rank == 0: 
            Write_log(self.log, 'Model:\n'+str(self.model)+'\n')

        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print(f"Using DDP")
            self.use_ddp = True
            # Fix DDP/checkpoint https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.model = DDP(
                self.model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        
        def count_parameters(model):
            total_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))
            return total_params / 1000.0 / 1000.0

        self.print(f"Init Model Parameters: {count_parameters(self.model)}")


    def init_loss(self):
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.MSELoss()
        
        if self.use_ddp:
            self.optimizer = torch.optim.Adam(
                filter(
                    lambda p: p.requires_grad,
                    self.model.parameters(),
                ),
                lr=self.configs.lr,
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.configs.lr
            )


    def init_data(self):
        self.predict_dl = get_dataloader(
            self.configs,
            batchsize=self.configs.batchsize, 
            shuffle=True, 
            input_json_path=self.configs.predict_json_path, 
            mode="predict",
        )
        

    def init_log(self):
        # if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
        #     wandb.init(
        #         project=self.configs.project,
        #         name=self.run_name,
        #         config=vars(self.configs),
        #         id=self.configs.wandb_id or None,
        #         # mode="offline" # online
        #     )
        if DIST_WRAPPER.rank == 0:
            self.log = open(os.path.join(self.run_dir, 'predict.log'), 'w', buffering=1)
            Write_log(self.log, 'Project:\n'+self.configs.project+'\n')
            Write_log(self.log, 'Name:\n'+self.run_name+'\n')
            Write_log(self.log, 'Config:\n'+str(self.configs)+'\n')


    def save_checkpoint(self, ema_suffix=""):
        if DIST_WRAPPER.rank == 0:
            path = f"{self.checkpoint_dir}/{self.step}{ema_suffix}.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
            }
            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")


    def try_load_checkpoint(self):
        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
        ):
            if not os.path.exists(checkpoint_path):
                raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(checkpoint_path, self.device)
            sample_key = [k for k in checkpoint["model"].keys()][0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module.") and not self.use_ddp:
                # DDP checkpoint has module. prefix
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            self.model.load_state_dict(
                state_dict=checkpoint["model"],
                strict=self.configs.load_strict,
            )
            if not load_params_only:
                if not skip_load_optimizer:
                    self.print(f"Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print(f"Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate

            self.print(f"Finish loading checkpoint, current step: {self.step}")

        # Load model
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_step=self.configs.skip_load_step,
            )

    def model_forward(self, batch: dict, mode: str = "train") -> tuple[dict, dict]:
        assert mode in ["train", "eval"]
        batch["pred_dict"] = self.model(
            input_feature_dict=batch["input_feature_dict"],
            # label_dict=batch["label_dict"],
            # label_full_dict=batch["label_full_dict"],
        )
        return batch


    def train_step(self, batch: dict):
        self.model.train()
        # FP16 training has not been verified yet
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "float16"),
        )

        with enable_amp:
            batch = self.model_forward(batch, mode="train")
            loss = self.loss(batch["pred_dict"], batch["sample_label"])
        
        scaler.scale(loss / self.iters_to_accumulate).backward()
        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            # self.print(
            #     f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            # )
            # Unscales the gradients of optimizer's assigned parameters in-place
            scaler.unscale_(self.optimizer)
            # Do grad clip only
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        torch.cuda.empty_cache()

        return batch


    def run(self):
        """
        This function handles the training process, evaluation, logging, and checkpoint saving.
        """
        predict_dict = self.predict('predict')
        if DIST_WRAPPER.rank == 0:
            with open(os.path.join(self.prediction_dir, f'predict.pkl'), 'wb') as f:
                pickle.dump(predict_dict, f)
        
    def predict(self, mode='predict'):
        self.model.eval()
        assert mode in ['predict']
        if mode == 'predict':
            dls = self.predict_dl
        
        predictions = []
        prediction_dict = {}  # 每个进程本地的预测字典
        
        for batch in tqdm(dls, disable=(DIST_WRAPPER.rank != 0)):
            batch = to_device(batch, self.device)
            batch = self.model_forward(batch, mode="eval")
            
            predictions.append(batch["pred_dict"].detach())
            
            # 记录当前进程处理的样本预测
            for idx, name in enumerate(batch["sample_name"]):
                prediction_dict[name] = batch["pred_dict"][idx].detach().cpu().numpy()
        
        
        
        # 聚合预测和标签（若需全局指标）
        predictions_tensor = torch.cat(predictions, dim=0)
        
        if DIST_WRAPPER.world_size > 1:
            # --------------------------
            # 兼容低版本PyTorch：用位置参数调用gather_object
            # --------------------------
            import torch.distributed as dist
            
            # 1. 主进程初始化接收列表（长度=总进程数），其他进程为None
            if DIST_WRAPPER.rank == 0:
                gather_list = [None] * DIST_WRAPPER.world_size  # 用于接收所有进程的字典
            else:
                gather_list = None  # 非主进程无需接收列表
            
            # 2. 调用gather_object：参数按位置传递（obj, gather_list, dst）
            # 注意：低版本中不支持关键字参数，必须按位置传参
            dist.gather_object(
                prediction_dict,  # 第1个参数：当前进程要发送的对象（必传）
                gather_list,      # 第2个参数：主进程的接收列表（非主进程传None）
                0                 # 第3个参数：目标主进程rank（必传）
            )
            
            # 3. 主进程合并所有字典
            if DIST_WRAPPER.rank == 0:
                full_prediction_dict = {}
                for d in gather_list:
                    full_prediction_dict.update(d)  # 合并所有进程的结果
                prediction_dict = full_prediction_dict  # 覆盖为完整字典
            # --------------------------
            # 聚合预测（张量用dist.gather，同样注意低版本参数可能需要位置传递）
            if DIST_WRAPPER.rank == 0:
                all_predictions = [torch.zeros_like(predictions_tensor) for _ in range(DIST_WRAPPER.world_size)]
            else:
                all_predictions = None
            # 低版本dist.gather可能也需要位置参数：(tensor, gather_list, dst)
            dist.gather(predictions_tensor, all_predictions, 0)

            # 主进程拼接
            if DIST_WRAPPER.rank == 0:
                all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
        else:
            all_predictions = predictions_tensor.cpu().numpy()
        
        return prediction_dict


    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)


    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def main() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "triattention"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    configs.model_name = "protenix_mini_esm_v0.5.0"
    model_name = configs.model_name
    configs.use_msa = False
    configs.esm.enable = True # Debug
    download_infercence_cache(configs)
    
    _, model_size, model_feature, model_version = model_name.split("_")
    logger.info(
        f"Inference by Protenix: model_size: {model_size}, with_feature: {model_feature.replace('-',', ')}, model_version: {model_version}"
    )
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    configs.update(model_specfics_configs)
    logger.info(
        f"Triangle_multiplicative kernel: {configs.triangle_multiplicative}, Triangle_attention kernel: {configs.triangle_attention}"
    )
    # download_infercence_cache(configs)
    trainer = Trainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()