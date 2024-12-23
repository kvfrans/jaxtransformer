from localutils.tpu_util import launch_tmux_jobs, launch_tmux_jobs_multi
import random

import os
os.system('sudo chmod -R 777 /nfs/jax-cache')

job_list = []
single_process = 'TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 '
debug_config = '--model.hidden_size 64 --model.depth 0 --model.num_heads 2 --model.mlp_ratio 1'
tiny_config = '--model.hidden_size 128 --model.depth 4 --model.num_heads 4 --model.mlp_ratio 4'
xsmall_config = '--model.hidden_size 144 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
small_config = '--model.hidden_size 384 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
big_config = '--model.hidden_size 768 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
xlarge_config = '--model.hidden_size 1152 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
# large_config = '--model.hidden_size 1024 --model.depth 24 --model.num_heads 16 --model.mlp_ratio 4'
# xlarge_config = '--model.hidden_size 1152 --model.depth 28 --model.num_heads 16 --model.mlp_ratio 4'

dit_config = '--model.train_type dit --dataset_name imagenet256 --fid_stats data/imagenet256_fidstats_ours.npz --model.cfg_scale 1.5 --model.class_dropout_prob 0.1'
vit_config = '--model.train_type vit --dataset_name imagenet256-augment'
gpt_config = '--model.train_type gpt --dataset_name openwebtext --model.use_stable_vae 0'


######## Some trials.
# launch_tmux_jobs([0,1,2,3], job_list, session_name="tpu1", start_dir='/nfs/ranklearn/')
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')
# launch_tmux_jobs(['i','j','k','l'], job_list, session_name="tpu4", start_dir='/nfs/ranklearn/')


# base = 'python train.py --wandb.group Nov13-LRFix --model.sharding fsdp --log_interval 1 --eval_interval 50_000 --max_steps 20_000'
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-HighLR {big_config} {gpt_config} --model.lr 0.001')
# launch_tmux_jobs([0,1,2,3], job_list, session_name="tpu1", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-HighLR {big_config} {gpt_config} --model.lr 0.001')
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-HighLR {big_config} {gpt_config} --model.lr 0.001')
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-HighLR {big_config} {gpt_config} --model.lr 0.001')
# launch_tmux_jobs(['i','j','k','l'], job_list, session_name="tpu4", start_dir='/nfs/ranklearn/')

# base = 'python train.py --wandb.group Nov13-Normalized --model.sharding fsdp --log_interval 10 --eval_interval 50_000 --max_steps 400_000'

# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-Normalize-DebugInit {big_config} {gpt_config} --model.normalize_activations 1 --model.scaling_lr_multiplier 1')
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')

# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-Normalize-HighLR {big_config} {gpt_config} --model.normalize_activations 1 --model.scaling_lr_multiplier 1 --model.lr 0.001')
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name DiT-768-Normalize-Renormalize {big_config} {dit_config} --model.normalize_activations 1 --model.scaling_lr_multiplier 1')
# launch_tmux_jobs([0,1,2,3], job_list, session_name="tpu1", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name ViT-768-Normalize {big_config} {vit_config} --model.normalize_activations 1 --model.scaling_lr_multiplier 1')
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')

# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-Normalize-Scale1 {big_config} {gpt_config} --model.normalize_activations 1 --model.scaling_lr_multiplier 1')
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-Normalize-Scale10 {big_config} {gpt_config} --model.normalize_activations 1 --model.scaling_lr_multiplier 10')
# launch_tmux_jobs(['i','j','k','l'], job_list, session_name="tpu4", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-Normalize-Scale100 {big_config} {gpt_config} --model.normalize_activations 1 --model.scaling_lr_multiplier 100')
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')


# base = 'python train.py --wandb.group Nov19-LRSweep --model.sharding fsdp --log_interval 10 --eval_interval 10_000 --max_steps 51_000 --log_effective_rank 1 --model.drop_lr_at_end 1'
# for tpu, lr in zip(['tpu1', 'tpu2', 'tpu3', 'tpu4'], [0.003, 0.001, 0.0003, 0.0001]):
#     job_list = []
#     job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-{lr} {big_config} {gpt_config} --model.lr {lr}')
#     job_list.append(base + f' --batch_size 64 --wandb.name ViT-768-{lr} {big_config} {vit_config} --model.lr {lr}')
#     job_list.append(base + f' --batch_size 64 --wandb.name DiT-768-{lr} {big_config} {dit_config} --model.lr {lr}')
#     job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-{lr}-Normalized {big_config} {gpt_config} --model.lr {lr} --model.normalize_activations 1 --model.scaling_lr_multiplier 1')
#     job_list.append(base + f' --batch_size 64 --wandb.name ViT-768-{lr}-Normalized {big_config} {vit_config} --model.lr {lr} --model.normalize_activations 1 --model.scaling_lr_multiplier 1')
#     job_list.append(base + f' --batch_size 64 --wandb.name DiT-768-{lr}-Normalized {big_config} {dit_config} --model.lr {lr} --model.normalize_activations 1 --model.scaling_lr_multiplier 1')
#     launch_tmux_jobs_multi(job_list, session_name=tpu, start_dir='/nfs/ranklearn/')



base = f'python train.py --wandb.group Nov25-FullNorm --model.sharding fsdp --log_interval 10 --eval_interval 10_000 --max_steps 51_000 --log_effective_rank 1 --model.drop_lr_at_end 1 --model.normalize_activations 1 --model.normalize_weights 1 --model.use_scale_terms 1 {big_config} {gpt_config}'

# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-HighScales --model.lr 0.0003 --model.lr_multiplier_scales 10')
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-HighOutput --model.lr 0.0003 --model.lr_multiplier_outputs 10')
# launch_tmux_jobs([0,1,2,3], job_list, session_name="tpu1", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-HighAttn --model.lr 0.0003 --model.lr_multiplier_attn 10')
# launch_tmux_jobs([0,1,2,3], job_list, session_name="tpu1", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-HighMlp --model.lr 0.0003 --model.lr_multiplier_mlp 10')
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-HighInputs --model.lr 0.0003 --model.lr_multiplier_inputs 10')
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')

job_list = []
job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-LowOutput --model.lr 0.0003 --model.lr_multiplier_outputs 0.1')
launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')
job_list = []
job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-LowAttn --model.lr 0.0003 --model.lr_multiplier_attn 0.1')
launch_tmux_jobs([0,1,2,3], job_list, session_name="tpu1", start_dir='/nfs/ranklearn/')
job_list = []
job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-LowMlp --model.lr 0.0003 --model.lr_multiplier_mlp 0.1')
launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/ranklearn/')
# job_list = []
# job_list.append(base + f' --batch_size 64 --wandb.name GPT-768-0.0003-LowInputs --model.lr 0.0003 --model.lr_multiplier_inputs 0.1')
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')