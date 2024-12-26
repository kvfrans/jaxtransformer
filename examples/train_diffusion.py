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
from jaxtransformer.utils.stable_vae import StableVAE
from jaxtransformer.utils.fid import get_fid_network, fid_from_stats
from jaxtransformer.utils.sharding import create_sharding
from jaxtransformer.transformer import TransformerBackbone
from jaxtransformer.modalities import PatchEmbed, PatchOutput, TimestepEmbed, TokenEmbed, get_2d_sincos_pos_embed

diffusion_config = ml_collections.ConfigDict({
    'patch_size': 2,
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 1.5,
    'use_stable_vae': 1,
    'fid_stats': '',
    'fid_num': 4096, # For comparable FID to literature, use 50_000.
})
config_flags.DEFINE_config_dict('train', training_config, lock_config=False)
config_flags.DEFINE_config_dict('optim', optimizer_config, lock_config=False)
config_flags.DEFINE_config_dict('tf', transformer_config, lock_config=False)
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('diffusion', diffusion_config, lock_config=False)
FLAGS = flags.FLAGS
FLAGS(sys.argv)

###################################
# Model Definition.
###################################
class DiffusionModel(nn.Module):
    @nn.compact
    def __call__(self, image, timestep, label):
        num_patches = (image.shape[1] // FLAGS.diffusion.patch_size) ** 2
        pos_embed = get_2d_sincos_pos_embed(None, FLAGS.tf.hidden_size, num_patches)
        x = PatchEmbed(FLAGS.diffusion.patch_size, FLAGS.tf.hidden_size)(image)
        x = x + pos_embed
        time_embed = TimestepEmbed(FLAGS.tf.hidden_size)(timestep)
        label_embed = TokenEmbed(FLAGS.diffusion.num_classes + 1, FLAGS.tf.hidden_size)(label) # +1 for unconditional class.
        conditioning = time_embed + label_embed
        x = TransformerBackbone(**FLAGS.tf, use_conditioning=True, use_causal_masking=False)(x, conditioning)
        x = PatchOutput(FLAGS.diffusion.patch_size, image.shape[-1], FLAGS.tf.hidden_size)(x, conditioning)
        return x
    
    def weight_decay_mask(self, params): # No clean way to define this in __call__, unfortunately.
        weight_decay_mask = {p: True for p in params.keys()}
        weight_decay_mask['TokenEmbed_0'] = False
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
if FLAGS.diffusion.fid_stats != '':
    truth_fid_stats = np.load(FLAGS.diffusion.fid_stats)
    get_fid_activations = get_fid_network()
if FLAGS.diffusion.use_stable_vae:
    vae = StableVAE.create()
    vae_decode = jax.jit(vae.decode)
    example_image = vae.encode(jax.random.PRNGKey(0), example_image)
x_shape = example_image.shape

# Initialize model, parameters, optimizer, train state.
model_def = DiffusionModel()
placeholder_input = (example_image, jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32))
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

    # Noise data to generate training pairs.
    images, labels = batch
    t_key, eps_key, labels_key, vae_key = jax.random.split(key, 4)
    if FLAGS.diffusion.use_stable_vae:
        images = vae.encode(vae_key, images)
    labels_dropout = jax.random.bernoulli(labels_key, FLAGS.diffusion.class_dropout_prob, (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.diffusion.num_classes, labels)
    t = jax.random.randint(t_key, (images.shape[0],), minval=0, maxval=FLAGS.diffusion.denoise_timesteps).astype(jnp.float32)
    t /= FLAGS.diffusion.denoise_timesteps
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]
    x_1 = images
    x_0 = jax.random.normal(eps_key, images.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    def loss_fn(grad_params):
        v_prime = train_state.call_model(x_t, t, labels_dropped, params=grad_params)
        loss = jnp.mean((v_prime - v_t) ** 2)
        return loss, {
            'loss': loss,
            'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
        }
    
    grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
    updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    info['v_magnitude_true'] = jnp.sqrt(jnp.mean(jnp.square(v_t)))
    info['grad_norm'] = optax.global_norm(grads)
    info['update_norm'] = optax.global_norm(updates)
    info['param_norm'] = optax.global_norm(new_params)
    info['lr'] = lr_schedule(train_state.step)

    train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
    train_state = train_state.update_ema(FLAGS.optim.ema_update_rate) if FLAGS.optim.use_ema else train_state
    return train_state, info

@jax.jit
def call_model(train_state, x, t, l):
    return train_state.call_model(x, t, l)

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

        if jax.process_index() == 0:
            wandb.log(train_metrics, step=i)

    # Model evaluation.
    if i % FLAGS.train.eval_interval == 0 or i in (0, 1000, 10000):
        with jax.spmd_mode('allow_all'):
            activations = []
            for fid_it in tqdm.tqdm(range(FLAGS.diffusion.fid_num // FLAGS.train.batch_size)):
                key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), fid_it), jax.process_index())
                eps_key, label_key = jax.random.split(key)
                x = jax.random.normal(eps_key, x_shape)
                labels = jax.random.randint(label_key, (x_shape[0],), 0, FLAGS.diffusion.num_classes)
                x, labels = shard_data(x, labels)
                delta_t = 1.0 / FLAGS.diffusion.denoise_timesteps
                for ti in range(FLAGS.diffusion.denoise_timesteps):
                    t = ti / FLAGS.diffusion.denoise_timesteps # From x_0 (noise) to x_1 (data)
                    t_vector = shard_data(jnp.full((x_shape[0], ), t))
                    v = call_model(train_state, x, t_vector, labels)
                    x = x + v * delta_t # Euler sampling.
                if FLAGS.diffusion.use_stable_vae:
                    vae_img = vae_decode(x) # Image is in [-1, 1] space.
                x = jax.image.resize(vae_img, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)
            
            vae_img = vae_img * 0.5 + 0.5
            img = jnp.clip(vae_img, 0, 1)
            vae_img = jax.experimental.multihost_utils.process_allgather(vae_img)
            vae_img = np.array(vae_img)

        if jax.process_index() == 0:
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            wandb.log({'fid'+str(FLAGS.diffusion.fid_num): fid}, step=i)

            fig, axs = plt.subplots(8, 8, figsize=(30, 30))
            axs_flat = axs.flatten()
            for j in range(len(vae_img)):
                axs_flat[j].imshow(vae_img[j])
                axs_flat[j].axis('off')
            wandb.log({'generated_images': wandb.Image(fig)}, step=i)
            plt.close(fig)

    # Save model checkpoint.
    if i % FLAGS.train.save_interval == 0 and FLAGS.train.save_dir is not None:
        train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
        if jax.process_index() == 0:
            cp = Checkpoint(FLAGS.train.save_dir+str(train_state_gather.step+1), parallel=False)
            cp.train_state = train_state_gather
            cp.save()
            del cp
        del train_state_gather