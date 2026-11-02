# LPWM on four IWS tasks

Research training snapshot used for the October 2026 LPWM experiments on PushT, Bimanual Rope, Single Grasp and Bimanual Sweep. Based on [Tal Daniel's LPWM](https://github.com/taldatech/lpwm), under the preserved MIT license. This is an independent experiment snapshot, not the original IWS model implementation.

## What changed

- IWS HDF5 loader with camera selection, episode-bounded clips and process-local file handles.
- FK-next action conditioning: frame `t` uses achieved FK pose at `min(t+1, episode_length-1)`. This is not the raw controller command. Normalization uses training-split min/max for both training and validation.
- Four task recipes and tests for temporal alignment, clip boundaries, train-only normalization and camera switching.
- Accelerate gradient accumulation and `use_ep_done_mask` configuration support.
- The inherited training runtime includes W&B metrics/media and resumable optimizer checkpoints.

The published core model, IWS loader and distributed trainer are byte-identical to the preserved training snapshot; hashes and source caveats are in [source provenance](docs/source-provenance.json). Only portable paths, integration-test discovery and publication documentation were changed during packaging. Machine-specific orchestration, credentials, datasets, weights and notebooks are excluded.

## Recorded experiments

| Task | Final camera | Action dimension |
|---|---|---:|
| pusht | camera_1_color | 4 |
| bimanual_rope | camera_0_color | 8 |
| single_grasp | camera_0_color | 4 |
| bimanual_sweep | camera_0_color | 4 |

Two A100 GPUs per task, 150 epochs, seed 42, FP32, LR 0.00008, 24 particles, 128×128 pixels. Native batch is 3 per GPU and gradient accumulation is 2 (effective global batch 12). Training uses 21-frame clips / 20 transitions, four sampled clips per episode. Built-in validation uses 4 context frames and 41 prediction frames, without future-image context. This built-in validation is distinct from the later full-episode SA image-metrics evaluation; that separate evaluator is not included here.

## Environment

The historical environment used PyTorch 2.10.0+cu126 and torchvision 0.25.0+cu126. Full recorded package versions are in [environment-packages.json](docs/environment-packages.json). The following installs the corresponding dependencies; CUDA drivers must support the chosen wheels.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-iws.txt
wandb login
```

## Data prerequisites

The experiment used `Liangguang/slotworld-iws-128`, revision `b760580b30bfe2f2fa33d102677031648cfc22d4`. Extract each task under `data/slotworld-iws-128/<task>/{train,val}`. **Raw images alone are insufficient**: the loader expects prepared achieved-FK caches and normalization files. Cache generation was external to this LPWM adaptation and is not included here; obtain the prepared data from the experiment maintainer rather than substituting raw actions.

- Each split contains `episode_*.hdf5` with `obs/images/<camera>` RGB arrays.
- PushT caches: `<split>/_fk_action_cache/episode_N.hdf5.npy`.
- Other tasks: `<split>/_fk_action_cache/<task>/episode_N.hdf5.npy`.
- Train stats: `train/action_stats_fk_next.npz` for PushT; `train/action_stats_<task>_fk_next.npz` for other tasks. Arrays are named `min` and `max`.

Never calculate normalization on validation data or concatenate clips across episodes.

## Train one task

Run from this repository root. Edit `root` and `output_dir` in a copy of the relevant recipe if needed. Set `wandb_entity` to your account/team (null uses your W&B default), and optionally change `wandb_project`/`wandb_name`. Use a different output directory and port for every concurrent job.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m accelerate.commands.launch \
  --config_file accel_iws_2gpu.yaml --main_process_port 29710 \
  train_lpwm_accelerate.py -d configs/iws_pusht.json
```

The other recipes are `configs/iws_bimanual_rope.json`, `configs/iws_single_grasp.json` and `configs/iws_bimanual_sweep.json`. Each command uses two GPUs. W&B is enabled online in every recipe. Credentials are obtained from your own W&B login and are never part of the recipe.

To resume, set `resume_checkpoint` in the runtime recipe to a full saved training checkpoint and preserve the original `wandb_run_id` (available in the run's `wandb_run.json`). Do not use a model-only checkpoint as a full optimizer resume. Keep the original output directory and recipe with each run.

## Tests

```bash
python -m unittest discover -s tests -p 'test_iws*.py' -v
```

Synthetic loader tests do not require downloaded data. Real-data integration tests use `IWS_DATA_ROOT` (default `data/slotworld-iws-128`) and skip when unavailable. The historical pilots tested finite updates, checkpoint reload and accumulation equivalence; publication itself does not rerun the expensive training experiments.
