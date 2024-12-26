import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import matplotlib.pyplot as plt
import flax.linen as nn
import sys

from jaxtransformer.utils.wandb import setup_wandb
from jaxtransformer.utils.datasets import get_dataset
from jaxtransformer.configs import training_config, optimizer_config, transformer_config, wandb_config
from jaxtransformer.utils.train_state import TrainState
from jaxtransformer.utils.checkpoint import Checkpoint
from jaxtransformer.utils.sharding import create_sharding
from jaxtransformer.transformer import TransformerBackbone
from jaxtransformer.modalities import PatchEmbed, TokenEmbed, ClassifierOutput, get_2d_sincos_pos_embed

vit_config = ml_collections.ConfigDict({
    'patch_size': 16,
    'num_classes': 1000,
})
config_flags.DEFINE_config_dict('train', training_config, lock_config=False)
config_flags.DEFINE_config_dict('optim', optimizer_config, lock_config=False)
config_flags.DEFINE_config_dict('tf', transformer_config, lock_config=False)
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('vit', vit_config, lock_config=False)
FLAGS = flags.FLAGS
FLAGS(sys.argv)

###################################
# Model Definition.
###################################
class ViTModel(nn.Module):
    @nn.compact
    def __call__(self, image):
        num_patches = (image.shape[1] // FLAGS.vit.patch_size) ** 2
        pos_embed = get_2d_sincos_pos_embed(None, FLAGS.tf.hidden_size, num_patches)
        x = PatchEmbed(FLAGS.vit.patch_size, FLAGS.tf.hidden_size)(image)
        x = x + pos_embed
        class_token = TokenEmbed(1, FLAGS.tf.hidden_size)(jnp.zeros((image.shape[0], 1), dtype=jnp.int32))
        x = jnp.concatenate([class_token, x], axis=1)
        x = TransformerBackbone(**FLAGS.tf, use_conditioning=False, use_causal_masking=False)(x)
        x = x[:, 0] # Get class token.
        x = ClassifierOutput(FLAGS.vit.num_classes)(x)
        return x
    
    def weight_decay_mask(self, params): # No clean way to define this in __call__, unfortunately.
        weight_decay_mask = {p: True for p in params.keys()}
        return weight_decay_mask
    
##############################################
## Initialization.
##############################################

if jax.process_index() == 0:
    setup_wandb(FLAGS.flag_values_dict(), **FLAGS.wandb)
np.random.seed(FLAGS.train.seed)
rng = jax.random.PRNGKey(FLAGS.train.seed)
if FLAGS.train.use_jit_cache:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache('/nfs/jax-cache')

# Load dataset and helper modules for VAE and FID.
dataset, dataset_valid = [get_dataset(FLAGS.train.dataset_name, FLAGS.train.batch_size, is_train=is_train) for is_train in (True, False)]
example_image, example_label = next(dataset)

# Initialize model, parameters, optimizer, train state.
model_def = ViTModel()
placeholder_input = (example_image,)
placeholder_params = jax.eval_shape(model_def.init, jax.random.PRNGKey(0), *placeholder_input)['params']
weight_decay_mask = model_def.weight_decay_mask(placeholder_params)
lr_schedule = optax.linear_schedule(0.0, FLAGS.optim.lr, FLAGS.optim.warmup)
tx = optax.adamw(learning_rate=lr_schedule, b1=FLAGS.optim.beta1, b2=FLAGS.optim.beta2, 
                 weight_decay=FLAGS.optim.weight_decay, mask=weight_decay_mask)
init_fn = partial(TrainState.create, model_def=model_def, model_input=placeholder_input, tx=tx, use_ema=FLAGS.optim.use_ema)
train_state_shape = jax.eval_shape(init_fn, rng=rng)
train_state_sharding, no_sharding, shard_data = create_sharding(FLAGS.train.sharding, train_state_shape)
train_state = jax.jit(init_fn, out_shardings=train_state_sharding)(rng=rng)
start_step = 0
print(nn.tabulate(model_def, jax.random.PRNGKey(0))(*placeholder_input))

if FLAGS.train.load_dir is not None:
    cp = Checkpoint(FLAGS.train.load_dir)
    train_state = train_state.replace(**cp.load_as_dict()['train_state'])
    train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
    print("Loaded model with step", train_state.step)
    train_state = train_state.replace(step=0)
    del cp

###################################
# Update Function
###################################

@partial(jax.jit, out_shardings=(train_state_sharding, no_sharding))
def update(train_state, batch):
    new_rng, key = jax.random.split(train_state.rng)

    images, labels = batch
    def loss_fn(grad_params):
        logits = train_state.call_model(images, params=grad_params)
        log_probs = jax.nn.log_softmax(logits)
        loss = jnp.mean(jnp.sum(-log_probs * jax.nn.one_hot(labels, FLAGS.vit['num_classes']), axis=-1))
        return loss, {
            'loss': loss,
            'accuracy': jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        }
    
    grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
    updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    info['grad_norm'] = optax.global_norm(grads)
    info['update_norm'] = optax.global_norm(updates)
    info['param_norm'] = optax.global_norm(new_params)
    info['lr'] = lr_schedule(train_state.step)

    train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
    train_state = train_state.update_ema(FLAGS.optim.ema_update_rate) if FLAGS.optim.use_ema else train_state
    return train_state, info

###################################
# Train Loop
###################################

for i in tqdm.tqdm(range(1, FLAGS.train.max_steps + 1),
                    smoothing=0.1,
                    dynamic_ncols=True):
    
    # Update.
    batch = shard_data(*next(dataset))
    train_state, update_info = update(train_state, batch)

    # Per-update logs.
    if i % FLAGS.train.log_interval == 0:
        update_info = jax.device_get(update_info)
        update_info = jax.tree_map(lambda x: np.array(x), update_info)
        update_info = jax.tree_map(lambda x: x.mean(), update_info)
        train_metrics = {f'training/{k}': v for k, v in update_info.items()}

        valid_batch = shard_data(*next(dataset_valid))
        _, valid_update_info = update(train_state, valid_batch)
        valid_update_info = jax.device_get(valid_update_info)
        valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
        train_metrics['training/loss_valid'] = valid_update_info['loss']
        train_metrics['training/accuracy_valid'] = valid_update_info['accuracy']

        if jax.process_index() == 0:
            wandb.log(train_metrics, step=i)

    # Save model checkpoint.
    if i % FLAGS.train.save_interval == 0 and FLAGS.train.save_dir is not None:
        train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
        if jax.process_index() == 0:
            cp = Checkpoint(FLAGS.train.save_dir+str(train_state_gather.step+1), parallel=False)
            cp.train_state = train_state_gather
            cp.save()
            del cp
        del train_state_gather