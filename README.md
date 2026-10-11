# NODE IK

[Paper](https://arxiv.org/abs/2209.00498) · [Original implementation](https://github.com/cadop/nodeik)

Minimal offline AI Worker IK using the original conditional CNF and LeRobot Pinocchio FK.
Pose-only and Vision-conditioned models share the data split, CNF architecture and training settings.
Vision reuses LeRobot's ResNet18 + SpatialSoftmax structure, **randomly initialized and trained end-to-end**.

## Install

Reuse the existing LeRobot uv environment:

```bash
uv pip install --python ../dream_lerobot/.venv/bin/python --no-deps -e . torchdiffeq==0.2.5
```

Alternatively: `uv sync --extra worker`, then use `uv run nodeik-worker`.

## Train

```bash
# Vision first, then the identical Pose-only baseline
uv run --project ../dream_lerobot --no-sync nodeik-worker train \
  --mode compare --log-dir log/ai_worker

# Extend / resume both experiments
uv run --project ../dream_lerobot --no-sync nodeik-worker train \
  --mode compare --log-dir log/ai_worker --resume --steps 100000
```

| Setting | Default |
|---|---|
| Steps / batch | 30,000 / 64 |
| CNF / Vision LR | `3e-4` / `1e-4` |
| CNF | 4 × 256, Dopri5, adjoint, tolerance `1e-5` |
| Vision | Four separate ResNet18 + GroupNorm, 96×96, random crop 84×84 |
| Validation | Every 1,000 steps; same fixed 512 held-out frames (`--eval-frames 0` for all) |
| Best FK score | Mean position (m) + 0.1 × mean orientation (rad) |
| Early stopping | 10 evaluations without significant FK or NLL improvement; minimum 10,000 steps |

`--mode pose` / `vision` trains one model. `--steps 50000` or `100000` extends the budget.
The loss is standard-normal CNF NLL, without joint MSE or FK loss. No pretrained checkpoint is used.
ImageNet mean/std are fixed pixel normalization constants; encoder weights are random.
`--uniform-samples N --mode pose` selects paper-style uniform URDF/FK samples.
`--dims 1024-1024-1024-1024` selects the paper's hidden width.

The default dataset has **62,385 frames / 100 episodes**. Episodes 0–89 provide 56,452 training frames;
90–99 provide 5,933 validation frames. At batch 64, 30k / 50k / 100k steps equal
34.01 / 56.69 / 113.37 training epoch equivalents (samples drawn with replacement).

Resized **pixels**, not encoder features, are cached on SSD under `~/.cache/nodeik` (~6.9 GB).
The run removes its temporary cache on completion, interruption or failure. `--cache-root` changes its location.
Logs are ignored by git: `report.html`, `status.json`, `metrics.jsonl`, `best.pt`, `best_nll.pt`, `last.pt`.
`last.pt` includes both networks, optimizer and RNG states for resuming.

## Evaluate / infer

```bash
uv run --project ../dream_lerobot --no-sync nodeik-worker eval \
  --model log/ai_worker/vision/best.pt --limit 128 --latent gaussian --samples 4

uv run --project ../dream_lerobot --no-sync nodeik-worker infer \
  --device cpu --model log/ai_worker/vision/best.pt --index 0 --latent 0.1
```

Use `--input request.npz` for external poses/images and `--output result.npz` to save predictions.
`poses` is `(2,4,4)` or `(N,2,4,4)`, absolute left/right TCP SE(3) transforms in the fixed-base URDF world.
Vision also requires all `observation.images.*` camera keys, RGB CHW/BCHW float `[0,1]` or uint8 `[0,255]`.
Latents: `0`, `0.1`, `gaussian`. Output: left arm joint1–7, right arm joint1–7, Lift (rad / m).
EE conditioning follows LeRobot's `xyz + row-based rot6d`; joints retain physical units.
Relative EEF actions must first be restored with their absolute anchors. ROS and robot control are excluded.

## Visualize results

Training automatically adds an interactive FK replay to `log/<run>/report.html` after completion.
It compares both best-FK models on the same held-out episode: target/predicted EE paths,
arm/Lift motion, position/orientation errors, and joint-limit/continuity statistics.
Latents `0`, `0.1` and one seeded Gaussian vector stay fixed over the entire path, as in the official examples.
The viewer uses the URDF link skeleton and needs no rendering dependencies.

```bash
# Regenerate, or attach to an already running training job with --wait
uv run --project ../dream_lerobot --no-sync python -m nodeik.visualize_worker \
  --run-dir log/ai_worker --frames 240 --wait
```

Open `report.html` or `visualization/viewer.html` locally. Drag to orbit; select model/latent and scrub frames.
`visualization/trajectories.npz` and `metrics.json` save the replay predictions and errors.

## Citation

```bibtex
@inproceedings{park2022nodeik,
  title={NODEIK: Solving Inverse Kinematics with Neural Ordinary Differential Equations for Path Planning},
  author={Park, Suhan and Schwartz, Mathew and Park, Jaeheung},
  booktitle={2022 22nd International Conference on Control, Automation and Systems (ICCAS)},
  pages={944--949},
  year={2022}
}
```

The original CNF layers derive from [FFJORD](https://github.com/rtqichen/ffjord) and
[SoftFlow](https://github.com/ANLGBOY/SoftFlow). Legacy Warp examples remain in `examples/`.
