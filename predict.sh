#!/bin/bash

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=. 

python data_preprocess.py

torchrun --standalone --nproc_per_node=1 runners/predict.py --project "Covid-design" --run_name "Covid-design" --base_dir "./outputs" --deterministic_seed True  --seed "2025" --dtype "fp32" --max_steps "200"  --eval_interval "1" --checkpoint_interval "1" --log_interval "1" --iters_to_accumulate "1" --precompute_esm "True" --num_workers "16" --predict_json_path "./data/Covid/Covid_design" --lr "1e-4" --batchsize "4" --load_params_only "False" --skip_load_optimizer "False" --skip_load_step "False"