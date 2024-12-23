import jax.experimental
from localutils.debugger import enable_debug
enable_debug()

from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import matplotlib.pyplot as plt

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache('/nfs/jax-cache')

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding, all_gather
from utils.datasets import get_dataset
from helpers_logging import create_horizontal_bars
from model import Transformer
from helpers_weightnorm import normalize_weight

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_integer('seed', 10, 'Random seed.') # Must be the same across all processes.
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_bool('debug_overfit', False, 'Flag to overfit on small data.')
flags.DEFINE_integer('log_effective_rank', 1, 'Flag to log effective rank.')

model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'use_cosine': 0,
    'warmup': 10_000,
    'dropout': 0.0,
    'hidden_size': 64, # change this!
    'patch_size': 8, # change this!
    'depth': 2, # change this!
    'num_heads': 2, # change this!
    'mlp_ratio': 1, # change this!
    'sharding': 'dp', # dp or fsdp.
    'target_update_rate': 0.999,
    'use_ema': 0,
    'train_type': 'dit', # 'dit', 'vit', 'gpt'
    # Custom options
    'drop_lr_at_end': 1,
    'inspect_grads': 0,
    'normalize_activations': 0,
    'normalize_weights': 0,
    'use_scale_terms': 0,
    'lr_multiplier_scales': 1.0,
    'lr_multiplier_inputs': 1.0,
    'lr_multiplier_attn': 1.0,
    'lr_multiplier_mlp': 1.0,
    'lr_multiplier_outputs': 1.0,
    # Diffusion options
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 1.5,
    'use_stable_vae': 1,
    # GPT options
    'sequence_length': 256,
    'vocab_size': 50257 # Don't change if using default tiktoken
})


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'ranklearn',
    'name': 'ranklearn_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)
    
##############################################
## Training Code.
##############################################
def main(_):

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)
        
    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, is_train=True, max_sequence_length=FLAGS.model.sequence_length, debug_overfit=FLAGS.debug_overfit)
    dataset_valid = get_dataset(FLAGS.dataset_name, local_batch_size, is_train=False, max_sequence_length=FLAGS.model.sequence_length, debug_overfit=FLAGS.debug_overfit)
    example_data, example_label = next(dataset)
    if FLAGS.model.train_type == 'gpt':
        example_data = example_data[:, 1:]

    if FLAGS.model.use_stable_vae:
        assert FLAGS.model.train_type != 'gpt' # VAE should only be used for images.
        vae = StableVAE.create()
        vae_decode = jax.jit(vae.decode)
        x_shape = vae.encode(jax.random.PRNGKey(0), example_data[0:1]).shape[1:]
    else:
        x_shape = example_data.shape[1:]

    if FLAGS.fid_stats is not None:
        truth_fid_stats = np.load(FLAGS.fid_stats)

    ###################################
    # Creating Model and put on devices.
    ###################################
    transformer_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'num_classes': FLAGS.model['num_classes'] if FLAGS.model['train_type'] != 'gpt' else FLAGS.model['vocab_size'],
        'train_type': FLAGS.model['train_type'],
        'config_kwargs': {
            'normalize_activations': FLAGS.model['normalize_activations'],
            'normalize_weights': FLAGS.model['normalize_weights'],
            'use_scale_terms': FLAGS.model['use_scale_terms'],
            'dropout': FLAGS.model['dropout'],
        }
    }
    model_def = Transformer(**transformer_args)
    tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
    placeholder_input = (example_data, jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32))
    placeholder_params = jax.eval_shape(model_def.init, jax.random.PRNGKey(0), *placeholder_input)['params']
    print(tabulate_fn(*placeholder_input))
    weight_decay_mask = {p: True for p in placeholder_params.keys()}
    weight_decay_mask['TokenEmbedder_0'] = False
    if FLAGS.model.train_type == 'gpt':
        weight_decay_mask['TokenEmbedder_1'] = False

    if FLAGS.model.use_cosine:
        lr_schedule = optax.warmup_cosine_decay_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'], FLAGS.max_steps)
    elif FLAGS.model.warmup > 0:
        lr_schedule = optax.linear_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'])
    else:
        lr_schedule = lambda x: FLAGS.model['lr']
    if FLAGS.model.drop_lr_at_end:
        lr_schedule_old = lr_schedule
        lr_schedule = lambda t : jnp.where(t < FLAGS.max_steps - 10_000, lr_schedule_old(t), 0.0001)
    adam = optax.adamw(learning_rate=lr_schedule, b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], weight_decay=FLAGS.model['weight_decay'], mask=weight_decay_mask)
    tx = optax.chain(adam)
    
    def init(rng):
        param_key, dropout_key = jax.random.split(rng, 2)
        model_rngs = {'params': param_key, 'dropout': dropout_key}
        params = model_def.init(model_rngs, jnp.zeros((1, *x_shape)), jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32))['params']
        opt_state = tx.init(params)
        return TrainStateEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    # jax.debug.visualize_array_sharding(train_state.params['Block_0']['Dense_0']['kernel'])
    start_step = 0

    # if FLAGS.load_dir is not None:
    #     cp = Checkpoint(FLAGS.load_dir)
    #     replace_dict = cp.load_as_dict()['train_state']
    #     train_state = train_state.replace(**replace_dict)
    #     if FLAGS.wandb.run_id != "None": # If we are continuing a run.
    #         start_step = train_state.step
    #     train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
    #     print("Loaded model with step", train_state.step)
    #     train_state = train_state.replace(step=0)
    #     jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'])
    #     del cp

    ###################################
    # Update Function
    ###################################

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard), static_argnames=('return_activations', 'return_grads'))
    def update(train_state, batch, return_activations=False, return_grads=False):
        new_rng, dropout_key, key = jax.random.split(train_state.rng, 3)

        def loss_fn_dit(grad_params):
            info = {}
            images, labels = batch
            t_key, eps_key, labels_key, vae_key = jax.random.split(key, 4)
            if FLAGS.model.use_stable_vae:
                images = vae.encode(vae_key, images)
            labels_dropout = jax.random.bernoulli(labels_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
            labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
            info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])
            t = jax.random.randint(t_key, (images.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
            t /= FLAGS.model['denoise_timesteps']
            t_full = t[:, None, None, None] # [batch, 1, 1, 1]
            x_1 = images
            x_0 = jax.random.normal(eps_key, images.shape)
            x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
            v_t = x_1 - (1 - 1e-5) * x_0
            v_prime, activations, infos = train_state.call_model(x_t, t, labels_dropped, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
            loss = jnp.mean((v_prime - v_t) ** 2)
            info['v_magnitude_prime'] = jnp.sqrt(jnp.mean(jnp.square(v_prime)))
            info['loss'] = loss
            info.update({'infos/'+k:v for k, v in infos.items()})
            return loss, (activations, info)
        def loss_fn_vit(grad_params):
            info = {}
            images, labels = batch
            if FLAGS.model.use_stable_vae:
                images = vae.encode(key, images)
            logits, activations, infos = train_state.call_model(images, None, None, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
            log_probs = jax.nn.log_softmax(logits)
            loss = jnp.mean(jnp.sum(-log_probs * jax.nn.one_hot(labels, FLAGS.model['num_classes']), axis=-1))
            info['loss'] = loss
            info['accuracy'] = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            info.update({'infos/'+k:v for k, v in infos.items()})
            return loss, (activations, info)
        def loss_fn_gpt(grad_params):
            info = {}
            text, _ = batch
            text_input, text_target = text[:, :-1], text[:, 1:]
            logits, activations, infos = train_state.call_model(text_input, None, None, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
            log_probs = jax.nn.log_softmax(logits)
            loss = jnp.mean(jnp.sum(-log_probs * jax.nn.one_hot(text_target, FLAGS.model['vocab_size']), axis=-1))
            info['loss'] = loss
            info['accuracy'] = jnp.mean(jnp.argmax(logits, axis=-1) == text_target)
            info.update({'infos/'+k:v for k, v in infos.items()})
            return loss, (activations, info)

        loss_fn = {'dit': loss_fn_dit, 'vit': loss_fn_vit, 'gpt': loss_fn_gpt}[FLAGS.model.train_type]
        grads, (activations, info) = jax.grad(loss_fn, has_aux=True)(train_state.params)
        updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        if FLAGS.model.train_type == 'vit':
            updates = jax.tree_map(lambda x: x * 0.5, updates) # Small LR for ViT.

        # Manual learning rate scalings.
        def do_scale(path, update):
            full_path = '/'.join([p.key for p in path])
            if 'scale' in full_path:
                return update * FLAGS.model['lr_multiplier_scales']
            if 'MlpBlock' in full_path:
                return update * FLAGS.model['lr_multiplier_mlp']
            if 'Block' in full_path:
                # If it's not in MlpBlock, it's attention head params.
                return update * FLAGS.model['lr_multiplier_attn']
            if 'Embed' in full_path:
                return update * FLAGS.model['lr_multiplier_inputs']
            if 'Output' in full_path:
                return update * FLAGS.model['lr_multiplier_outputs']
            return update
        updates = jax.tree_util.tree_map_with_path(do_scale, updates)

        new_params = optax.apply_updates(train_state.params, updates)

        if FLAGS.model['normalize_weights']:
            def do_normalize(path, param):
                if path[-1].key == 'dense_weightnorm':
                    return normalize_weight(param)
                elif path[-1].key == 'embedding_weightnorm':
                    return normalize_weight(param, axis=1)
                return param
            new_params = jax.tree_util.tree_map_with_path(do_normalize, new_params)

            def get_norm(path, param):
                if path[-1].key == 'dense_weightnorm':
                    return jnp.sum(param ** 2)
                return 0
            info['param_norm_weightnorm'] = jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_util.tree_map_with_path(get_norm, new_params)))

        # Log some statistics about activations, params, etc.
        info.update({'activations/' + k : jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()})
        if return_activations:
            info.update({'activations_full/' + k : v for k, v in activations.items()})
        if return_grads:
            info['grads'] = grads

        info['grad_max'] = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(jnp.max(x), jnp.max(y)), grads)
        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_max'] = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(jnp.max(x), jnp.max(y)), new_params)
        info['param_norm'] = optax.global_norm(new_params)
        info['lr'] = lr_schedule(train_state.step)

        train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
        return train_state, info

    ###################################
    # Train Loop
    ###################################

    for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                       smoothing=0.1,
                       dynamic_ncols=True):
        
        # Update.
        if i == 1 or not FLAGS.debug_overfit:
            batch = shard_data(*next(dataset))
        train_state, update_info = update(train_state, batch)

        # if FLAGS.model['inspect_grads']:
        #     update_info_gradmax = jax.device_get(update_info['grad_max'])
        #     if update_info_gradmax > 10:
        #         update_info = jax.device_get(update_info)
        #         grads = update_info['grads']
        #         grads = jax.experimental.multihost_utils.process_allgather(grads)
        #         breakpoint()

        # Per-update logs.
        if i % FLAGS.log_interval == 0 or i == 1 and i > 10000:
            update_info = jax.device_get(update_info)
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            if not FLAGS.debug_overfit:
                valid_batch = shard_data(*next(dataset_valid))
                _, valid_update_info = update(train_state, valid_batch)
                valid_update_info = jax.device_get(valid_update_info)
                valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
                train_metrics['training/loss_valid'] = valid_update_info['loss']
                train_metrics['training/accuracy_valid'] = valid_update_info.get('accuracy', 0.0)

            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        # Evaluation logs.
        if i % FLAGS.eval_interval == 0 or i in (1, 1000, 10000):
            # if FLAGS.model['train_type'] == 'dit':
            #     from helpers_dit import plot_generated_images, eval_fid
            #     plot_generated_images(FLAGS, train_state, x_shape, shard_data, data_sharding, vae_decode, i)
            #     eval_fid(FLAGS, train_state, x_shape, shard_data, vae_decode, truth_fid_stats, i)

            # if FLAGS.model['train_type'] == 'gpt':
            #     from helpers_gpt import log_generated_data
            #     log_generated_data(FLAGS, train_state, shard_data, batch, i)

            _, update_info = update(train_state, batch, return_activations=True)
            if FLAGS.log_effective_rank != 0:
                update_info_arrays = {key:[] for key in update_info if 'activations_full' in key}
                for j in range(16):
                    batch = shard_data(*next(dataset))
                    _, update_info = update(train_state, batch, return_activations=True)
                    update_info = jax.device_get(update_info)
                    for key in update_info_arrays:
                        update_info_arrays[key].append(np.array(update_info[key]))
                if jax.process_index() == 0:
                    update_info_all = {key: np.concatenate(update_info_arrays[key]) for key in update_info_arrays}
                    block_keys = ['embed_input'] + [f'block_{j}' for j in range(FLAGS.model['depth'])]
                    effective_ranks, effective_ranks_9, max_activation, max_singular = [], [], [], []
                    for j, block_key in enumerate(block_keys):
                        x = update_info_all[f'activations_full/{block_key}']
                        flat_x = x.reshape(-1, x.shape[-1]) # [batch*sequence, hidden]
                        print('Flat_x shape:', flat_x.shape)
                        square_x = flat_x.T @ flat_x
                        u, s, v = np.linalg.svd(square_x, hermitian=True)
                        s_sum = np.sum(s)
                        s_cumsum = np.cumsum(s) / s_sum
                        effective_ranks.append(np.sum(s_cumsum < 0.99))
                        effective_ranks_9.append(np.sum(s_cumsum < 0.9))
                        max_activation.append(np.max(np.abs(x)))
                        max_singular.append(np.max(np.sqrt(s))) # Divide by sqrt(32*batch*256).
                    del update_info_all
                    print('Effective ranks are:', np.array(effective_ranks))
                    fig = create_horizontal_bars(effective_ranks, block_keys, 'Effective Rank', FLAGS.model['hidden_size'])
                    wandb.log({'effective_rank': wandb.Image(fig)}, step=i)
                    wandb.log({'block3/effective_rank': effective_ranks[3]}, step=i)
                    wandb.log({'block8/effective_rank': effective_ranks[7]}, step=i)
                    plt.close(fig)
                    fig = create_horizontal_bars(effective_ranks_9, block_keys, 'Effective Rank 0.9', FLAGS.model['hidden_size'])
                    wandb.log({'effective_rank_9': wandb.Image(fig)}, step=i)
                    wandb.log({'block3/effective_rank_9': effective_ranks_9[3]}, step=i)
                    wandb.log({'block8/effective_rank_9': effective_ranks_9[7]}, step=i)
                    plt.close(fig)
                    fig = create_horizontal_bars(max_activation, block_keys, 'Max Activation', None)
                    wandb.log({'max_activation': wandb.Image(fig)}, step=i)
                    wandb.log({'block3/max_activation': max_activation[3]}, step=i)
                    wandb.log({'block8/max_activation': max_activation[7]}, step=i)
                    plt.close(fig)
                    fig = create_horizontal_bars(max_singular, block_keys, 'Max Singular Value', None)
                    wandb.log({'max_singular': wandb.Image(fig)}, step=i)
                    wandb.log({'block3/max_singular': max_singular[3]}, step=i)
                    wandb.log({'block8/max_singular': max_singular[7]}, step=i)
                    plt.close(fig)

            if jax.process_index() == 0:
                # Additional Figures
                block_keys = [f'block_{j}' for j in range(FLAGS.model['depth'])]
                infos = [
                    ('relu_diff', 'ReLU Ratio, diference from 0.5', 0.5),
                    ('relu_zero', 'Ratio of features where relu always < 0', 1.0),
                    ('relu_positive', 'Ratio of features where relu always > 1', 1.0),
                    ('relu_norm', 'RMSNorm of intermediates', None),
                    ('max_attn_weight', 'Max Attention Weight', 1.0),
                    ('attn_norm_ratio', 'Addition-Attention / Residual RMSNorm', 1.0),
                    ('mlp_norm_ratio', 'Addition-MLP / Residual RMSNorm', 1.0),
                    ('activations', 'Activation RMSNorm', None),
                ]
                for b in block_keys:
                    update_info['infos/'+b+'_activations'] = update_info['activations/'+b]
                for info_name, info_label, plot_scale in infos:
                    info_values = [update_info['infos/'+ b + '_' + info_name].item() for b in block_keys]
                    fig = create_horizontal_bars(info_values, block_keys, info_label, plot_scale)
                    wandb.log({info_name: wandb.Image(fig)}, step=i)
                    plt.close(fig)
                    wandb.log({'block3/'+info_name: info_values[3]}, step=i)
                    wandb.log({'block8/'+info_name: info_values[7]}, step=i)

            attn_weight_scales = []
            attn_residual_scales = []
            mlp_residual_scales = []
            attn_weight_scales_max = []
            attn_residual_scales_max = []
            mlp_residual_scales_max = []
            param_norms = []
            for j in range(FLAGS.model['depth']):
                params = train_state.params[f'Block_{j}']
                params = jax.experimental.multihost_utils.process_allgather(params)
                if FLAGS.model['use_scale_terms']:
                    attn_weight_scales.append(jnp.mean(params['attn_weight_scale']))
                    attn_residual_scales.append(jnp.mean(params['attn_residual_scale']))
                    mlp_residual_scales.append(jnp.mean(params['mlp_residual_scale']))
                    attn_weight_scales_max.append(jnp.max(params['attn_weight_scale']))
                    attn_residual_scales_max.append(jnp.max(params['attn_residual_scale']))
                    mlp_residual_scales_max.append(jnp.max(params['mlp_residual_scale']))
                param_norms.append(optax.global_norm(params))
            if jax.process_index() == 0:
                infos = [
                    (param_norms, 'Param Norm', None)
                ]
                if FLAGS.model['use_scale_terms']:
                    infos += [
                        (attn_weight_scales, 'Attention Weight Scale', None),
                        (attn_residual_scales, 'Attention Residual Scale', None),
                        (mlp_residual_scales, 'MLP Residual Scale', None),
                        (attn_weight_scales_max, 'Attention Weight Scale Max', None),
                        (attn_residual_scales_max, 'Attention Residual Scale Max', None),
                        (mlp_residual_scales_max, 'MLP Residual Scale Max', None),
                    ]
                for stats, info_label, plot_scale in infos:
                    fig = create_horizontal_bars(stats, block_keys, info_label, plot_scale)
                    wandb.log({info_label: wandb.Image(fig)}, step=i)
                    plt.close(fig)
                    wandb.log({'block3/'+info_label: stats[3]}, step=i)
                    wandb.log({'block8/'+info_label: stats[7]}, step=i)


            def do_log_scale(path, param):
                if 'scale' in path[-1].key:
                    if len(path) >= 2:
                        if 'Block' in path[-2].key:
                            return None
                        if 'LayerNorm' in path[-2].key:
                            return None
                    p = jnp.mean(jax.experimental.multihost_utils.process_allgather(param))
                    if jax.process_index() == 0:
                        wandb.log({'training/'+path[-1].key: p.mean()}, step=i)
                        return None
                return None
            jax.tree_util.tree_map_with_path(do_log_scale, train_state.params)
 


        # if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
        #     train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
        #     if jax.process_index() == 0:
        #         cp = Checkpoint(FLAGS.save_dir+str(train_state_gather.step+1), parallel=False)
        #         cp.train_state = train_state_gather
        #         cp.save()
        #         del cp
        #     del train_state_gather

if __name__ == '__main__':
    app.run(main)