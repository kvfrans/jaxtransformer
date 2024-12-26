## jaxtransformer

A minimal implementation of a Transformer backbone in JAX, following the GPT-2 architecture. The same backbone can be applied to various training objectives -- e.g. language modelling, image classification, diffusion modelling.

This repo is intended to be imported and/or copied into any project. The main files are in `transformer.py`, which implements the transformer backbone from scratch as a Flax module. For use in image/text domains, `modalities.py` contains helper modules such as a patch embedder, or positional encoding. The `utils/` folder contains assorted utilities for training -- e.g. logging, checkpointing, sharding, etc.

This repo supports:
- Minimal transformer implementation in 80 lines of code.
- Example scripts for diffusion (DiT), language modelling (GPT), and image classification (ViT).
- Support for running latent diffusion with a pretrained Stable Diffusion VAE.
- Support for multi-host training using fully-sharded data parallel (FSDP).

### Using the code

This codebase is written in JAX, and was developed on TPU-v3 machines. You should start by installing the conda dependencies from `environment.yml` and `requirements.txt`. To load datasets, we use TFDS, and you can see our specific dataloaders at [https://github.com/kvfrans/tfds_builders](https://github.com/kvfrans/tfds_builders), of course you are free to use your own dataloader as well. FID stat files can be downloaded using `data/download.sh`.

To train a diffusion model:
```
python examples/train_diffusion.py --train.dataset_name imagenet256 --diffusion.fid_stats data/imagenet256_fidstats_ours.npz --wandb.name DiT-B-Imagenet --tf.hidden_size 768 --tf.depth 12 --tf.num_heads 12 --tf.mlp_ratio 4
```
To train an image classification model:
```
python examples/train_vit.py --train.dataset_name imagenet256-augment --wandb.name ViT-B-Imagenet --tf.hidden_size 768 --tf.depth 12 --tf.num_heads 12 --tf.mlp_ratio 4
```
To train an autoregressive language model:
```
python examples/train_llm.py --train.dataset_name openwebtext --wandb.name LLM-B-OpenWebText --tf.hidden_size 768 --tf.depth 12 --tf.num_heads 12 --tf.mlp_ratio 4
```

The above scripts train a Base size model. To train an XLarge size model, simply change the config flags to:
```
--tf.hidden_size 1152 --tf.depth 28 --tf.num_heads 16 --tf.mlp_ratio 4
```

When training on a single host, simply run the Python commands as described above. The code will automatically work if multiple GPUs/TPUs are present. To train on multiple hosts, run a copy of the Python commands on each host. See [this JAX guide](https://jax.readthedocs.io/en/latest/multi_process.html) for more info.