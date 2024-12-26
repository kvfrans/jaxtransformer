from ml_collections import config_flags
import ml_collections
from  ml_collections.config_dict import FieldReference

# General training config.
training_config = ml_collections.ConfigDict({
    'dataset_name': 'imagenet256',
    'load_dir': FieldReference(None, field_type=str),
    'save_dir': FieldReference(None, field_type=str),
    'seed': 10,
    'log_interval': 1000,
    'eval_interval': 20000,
    'save_interval': 100000,
    'max_steps': 400_000,
    'sharding': 'fsdp', # dp or fsdp.
    'use_jit_cache': 1,
    'batch_size': 64,
})

# Config for optimizer (AdamW).
optimizer_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.001,
    'warmup': 1_000,
    'schedule': 'constant', # 'constant' or 'cosine'.
    'use_ema': 0, # Keep an exponential moving average of params?
    'ema_update_rate': 0.999, # EMA update rate.
})

# Config for backbone transformer model. 
transformer_config = ml_collections.ConfigDict({
    'hidden_size': 64,
    'depth': 2,
    'num_heads': 2,
    'mlp_ratio': 1,
})
xsmall_config = '--tf.hidden_size 128 --tf.depth 6 --tf.num_heads 4 --tf.mlp_ratio 2'
small_config = '--tf.hidden_size 384 --tf.depth 12 --tf.num_heads 12 --tf.mlp_ratio 4'
big_config = '--tf.hidden_size 768 --tf.depth 12 --tf.num_heads 12 --tf.mlp_ratio 4'
large_config = '--tf.hidden_size 1024 --tf.depth 24 --tf.num_heads 16 --tf.mlp_ratio 4'
xlarge_config = '--tf.hidden_size 1152 --tf.depth 28 --tf.num_heads 16 --tf.mlp_ratio 4'

# Wandb config.
# def default_wandb_config():
wandb_config = ml_collections.ConfigDict({
    'offline': 0,
    'project': "jaxtransformer",
    'name': 'jaxtransformer-run',
    'entity': FieldReference(None, field_type=str),
})