from localutils.tpu_util import launch_tmux_jobs
from jaxtransformer.configs import small_config, big_config, large_config, xlarge_config

import os
os.system('sudo chmod -R 777 /nfs/jax-cache')

job_list = []
single_process = 'TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 '
debug_config = '--tf.hidden_size 64 --tf.depth 2 --tf.num_heads 2 --tf.mlp_ratio 1'
xsmall_config = '--tf.hidden_size 128 --tf.depth 6 --tf.num_heads 4 --tf.mlp_ratio 2'
small_config = '--tf.hidden_size 384 --tf.depth 12 --tf.num_heads 12 --tf.mlp_ratio 4'
big_config = '--tf.hidden_size 768 --tf.depth 12 --tf.num_heads 12 --tf.mlp_ratio 4'
large_config = '--tf.hidden_size 1024 --tf.depth 24 --tf.num_heads 16 --tf.mlp_ratio 4'
xlarge_config = '--tf.hidden_size 1152 --tf.depth 28 --tf.num_heads 16 --tf.mlp_ratio 4'

######## Some trials.

# base = 'python examples/train_diffusion.py --train.dataset_name imagenet256 --diffusion.fid_stats data/imagenet256_fidstats_ours.npz'
# job_list = [base + f' --wandb.name DiT-B-Imagenet {big_config}']
# launch_tmux_jobs([0,1,2,3], job_list, session_name="tpu1", start_dir='/nfs/jaxtransformer/')
# job_list = [base + f' --wandb.name DiT-XL-Imagenet {xlarge_config}']
# launch_tmux_jobs([4,5,6,7], job_list, session_name="tpu2", start_dir='/nfs/jaxtransformer/')


# base = 'python examples/train_vit.py --train.dataset_name imagenet256-augment'
# job_list = [base + f' --wandb.name ViT-B-Imagenet {big_config}']
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/jaxtransformer/')
# job_list = [base + f' --wandb.name ViT-XL-Imagenet {xlarge_config}']
# launch_tmux_jobs(['i','j','k','l'], job_list, session_name="tpu4", start_dir='/nfs/jaxtransformer/')


base = 'python examples/train_llm.py --train.dataset_name openwebtext'
job_list = [base + f' --wandb.name LLM-B-OpenWebText {big_config}']
launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/jaxtransformer/')
job_list = [base + f' --wandb.name LLM-XL-OpenWebText {xlarge_config}']
launch_tmux_jobs(['i','j','k','l'], job_list, session_name="tpu4", start_dir='/nfs/jaxtransformer/')
