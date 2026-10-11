"""End-to-end CNF training, pixel cache, paired validation and local run logs."""

import html
import json
import math
import shutil
import signal
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

from nodeik.ai_worker import (
    JOINT_NAMES,
    Vision,
    encode_poses,
    evaluate,
    fk_poses,
    load_joints,
    make_flow,
    predict,
)
from nodeik.utils import standard_normal_logprob


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def report(root, status):
    """A local, self-refreshing report; JSONL remains the source of truth."""
    status["updated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    write_json(root / "status.json", status)
    panels = []
    for mode in status["modes"]:
        info = status["modes"][mode]
        file = root / mode / "metrics.jsonl"
        rows = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
        plots = []
        for key, label in (
            ("train_nll", "Training NLL"),
            ("validation_nll", "Validation NLL"),
            ("position_mean_m", "FK position (m)"),
            ("orientation_mean_rad", "FK orientation (rad)"),
        ):
            points = [(r["step"], r[key]) for r in rows if key in r]
            if not points:
                continue
            lo, hi = min(y for _, y in points), max(y for _, y in points)
            x_max = max(1, max(x for x, _ in points))
            coords = " ".join(
                f"{45 + 700 * x / x_max:.1f},{145 - 115 * (y - lo) / max(hi - lo, 1e-8):.1f}"
                for x, y in points
            )
            plots.append(
                f'<p>{label}: {points[-1][1]:.6f}</p><svg viewBox="0 0 780 175">'
                f'<path d="M45 20V145H745" fill="none" stroke="#475569"/>'
                f'<polyline points="{coords}" fill="none" stroke="#38bdf8" stroke-width="2"/>'
                f'<text x="45" y="165">0</text><text x="650" y="165">step {x_max}</text>'
                f'<text x="5" y="20">{hi:.3g}</text><text x="5" y="140">{lo:.3g}</text></svg>'
            )
        panels.append(
            f"<section><h2>{mode}</h2><pre>{html.escape(json.dumps(info, indent=2))}</pre>"
            f'<a href="{mode}/metrics.jsonl">metrics</a> · '
            f'<a href="{mode}/best.pt">best FK checkpoint</a> · '
            f'<a href="{mode}/best_nll.pt">best NLL checkpoint</a> · '
            f'<a href="{mode}/last.pt">last / resume</a>{"".join(plots)}</section>'
        )
    visualization = root / "visualization/section.html"
    refresh = '<meta http-equiv="refresh" content="20">' if status["state"] != "completed" else ""
    page = (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        + refresh
        + "<title>NODE IK · Pose / Vision</title><style>body{background:#0f172a;color:#e2e8f0;"
        "font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}section{background:#1e293b;"
        "padding:24px;margin:20px 0;border-radius:12px}pre{white-space:pre-wrap}a{color:#38bdf8}"
        "svg{width:100%}text{fill:#94a3b8;font-size:12px}table{width:100%;border-collapse:collapse}"
        "td,th{text-align:left;padding:8px;border-bottom:1px solid #475569}</style>"
        "<h1>NODE IK · Pose-only / Vision-conditioned</h1>"
        "<p>Random ResNet18 · end-to-end CNF NLL · 학습 중 20초 자동 새로고침</p>"
        f"<pre>{html.escape(json.dumps({k: v for k, v in status.items() if k != 'modes'}, indent=2))}</pre>"
        '<p><a href="config.json">설정</a> · <a href="dataset.json">데이터 / epoch</a> · '
        '<a href="train.log">학습 로그</a> · <a href="provenance.json">코드 / 환경 / 참고 자료</a></p>'
        + "".join(panels)
        + (visualization.read_text() if visualization.exists() else "")
        + "</html>"
    )
    temporary = root / "report.html.tmp"
    temporary.write_text(page)
    temporary.replace(root / "report.html")


def build_cache(root, cache, indices, episodes, keys, size, progress):
    """Adapt the referenced cache builder: native timestamp decode -> uint8 Resize."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec
    from torchvision.transforms.v2 import Resize

    required = len(indices) * len(keys) * 3 * size * size
    if shutil.disk_usage(cache).free < required + 2 * 1024**3:
        raise OSError(f"Pixel cache needs {required / 1024**3:.2f} GiB plus 2 GiB free reserve")
    meta = LeRobotDatasetMetadata(Path(root).name, root=root)
    table = pa.concat_tables(
        [
            pq.read_table(p, columns=["index", "timestamp"])
            for p in sorted((Path(root) / "data").rglob("*.parquet"))
        ]
    )
    table = table.sort_by([("index", "ascending")])
    if not np.array_equal(table["index"].to_numpy(), np.arange(len(table))):
        raise ValueError("Pixel cache expects contiguous LeRobot absolute frame indices")
    timestamps = table["timestamp"].to_numpy()[indices].astype(np.float64)
    resize = Resize((size, size), antialias=True)
    arrays = {}
    total = len(keys) * len(indices)
    completed = 0
    for key in keys:
        path = cache / (key.replace(".", "_") + ".npy")
        array = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.uint8, shape=(len(indices), 3, size, size)
        )
        decoders = VideoDecoderCache(max_size=1)
        try:
            for episode in np.unique(episodes):
                rows = np.flatnonzero(episodes == episode)
                ep = meta.episodes[int(episode)]
                video = Path(root) / meta.get_video_file_path(int(episode), key)
                offset = ep[f"videos/{key}/from_timestamp"]
                for start in range(0, len(rows), 64):
                    ids = rows[start : start + 64]
                    frames = decode_video_frames_torchcodec(
                        video,
                        (timestamps[ids] + offset).tolist(),
                        1e-4,
                        decoder_cache=decoders,
                        return_uint8=True,
                    )
                    array[ids] = resize(frames).numpy()
                    completed += len(ids)
                progress(
                    {
                        "camera": key,
                        "frames": completed,
                        "total": total,
                        "percent": round(100 * completed / total, 2),
                    }
                )
        finally:
            decoders.clear()
        array.flush()
        del array
        arrays[key] = np.load(path, mmap_mode="r")
    return arrays


def image_batch(arrays, ids, device):
    return {
        key: torch.from_numpy(np.array(value[ids], copy=True)).to(device) for key, value in arrays.items()
    }


def nll(model, joint, condition):
    z, delta = model(joint, condition, joint.new_zeros(len(joint), 1))
    return -(standard_normal_logprob(z).sum(1, keepdim=True) - delta).squeeze(1)


def validation(model, vision, robot, q, pose, target, ids, arrays, args, device):
    model.eval()
    model.chain[0].odefunc.calc_density = True
    if vision:
        vision.eval()
    values, predictions = [], []
    devices = [device.index or 0] if device.type == "cuda" else []
    # Reuse the same Hutchinson probes across evaluations and both experiments.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed + 10000)
        for start in range(0, len(ids), args.batch_size):
            batch = ids[start : start + args.batch_size]
            c = pose[batch].to(device)
            if vision:
                with torch.no_grad():
                    c = torch.cat([c, vision(image_batch(arrays, batch, device))], 1)
            with torch.no_grad():
                # ODEfunc locally enables autograd for the Jacobian trace.
                values.extend(nll(model, q[batch].to(device), c).cpu().tolist())
            predictions.append(predict(model, c, "0"))
            model.chain[0].odefunc.calc_density = True
    errors, _ = evaluate(robot, np.concatenate(predictions), target[ids])
    position = sum(errors[side]["position_mean_m"] for side in ("left", "right")) / 2
    orientation = sum(errors[side]["orientation_mean_rad"] for side in ("left", "right")) / 2
    return {
        "validation_nll": float(np.mean(values)),
        "position_mean_m": position,
        "orientation_mean_rad": orientation,
        "fk_score": position + 0.1 * orientation,
        "fk_errors": errors,
        "validation_frames": len(ids),
    }


def run_mode(mode, args, device, root, status, robot, q, pose, target, train_ids, val_ids, arrays, keys):
    directory = root / mode
    directory.mkdir(exist_ok=True)
    # Both runs start from the same seed and use an independent identical frame sampler.
    torch.manual_seed(args.seed)
    config = {
        "dims": args.dims,
        "tol": args.tol,
        "condition_dim": 18 + (64 * len(keys) if mode == "vision" else 0),
    }
    model = make_flow(config, device)
    vision_config = {"keys": keys, "size": args.image_size, "crop_ratio": 0.875} if mode == "vision" else None
    vision = Vision(vision_config).to(device) if vision_config else None
    groups = [{"params": model.parameters(), "lr": args.lr}]
    if vision:
        groups.append({"params": vision.parameters(), "lr": args.vision_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    sampler = np.random.default_rng(args.seed)
    start_step, best_fk, best_nll, stalled = 0, float("inf"), float("inf"), 0
    progress_fk, progress_nll = best_fk, best_nll
    last = directory / "last.pt"
    if args.resume and last.exists():
        saved = torch.load(last, map_location=device, weights_only=True)
        if (
            saved["config"] != config
            or saved["source"] != args.source
            or saved["vision_config"] != vision_config
            or not np.array_equal(saved["train_ids"].cpu().numpy(), train_ids)
            or not np.array_equal(saved["val_ids"].cpu().numpy(), val_ids)
        ):
            raise ValueError("Resume requires the same model configuration and dataset split")
        model.load_state_dict(saved["model"])
        if vision:
            vision.load_state_dict(saved["vision"])
        optimizer.load_state_dict(saved["optimizer"])
        for group, lr in zip(optimizer.param_groups, (args.lr, args.vision_lr)):
            group["lr"] = lr
        sampler.bit_generator.state = saved["sampler"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        start_step, best_fk, best_nll, stalled = (
            saved[k] for k in ("step", "best_fk", "best_nll", "stalled")
        )
        progress_fk, progress_nll = saved.get("progress_fk", best_fk), saved.get("progress_nll", best_nll)
    if start_step >= args.steps:
        status["modes"][mode] = {
            "state": "completed",
            "step": start_step,
            "best_fk": best_fk,
            "best_nll": best_nll,
        }
        report(root, status)
        return
    step = start_step
    started = time.monotonic()
    last_update, logged_step = started, start_step
    loss_total, loss_count = 0.0, 0
    parameters = list(model.parameters()) + (list(vision.parameters()) if vision else [])

    def checkpoint(path):
        payload = {
            "version": 2,
            "model": model.state_dict(),
            "config": config,
            "joint_names": JOINT_NAMES,
            "pose_rep": "absolute_xyz_row_rot6d_left_right",
            "urdf": robot.cfg.urdf_path,
            "source": args.source,
            "vision_config": vision_config,
            "vision": vision.state_dict() if vision else None,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_fk": best_fk,
            "best_nll": best_nll,
            "stalled": stalled,
            "progress_fk": progress_fk,
            "progress_nll": progress_nll,
            "sampler": sampler.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "train_ids": torch.from_numpy(train_ids),
            "val_ids": torch.from_numpy(val_ids),
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def record(row):
        row.update(
            mode=mode,
            step=step,
            epoch=step * args.batch_size / len(train_ids),
            recorded_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        )
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        status["modes"][mode].update(row)
        report(root, status)
        print(json.dumps(row, allow_nan=False), flush=True)

    status["modes"][mode] = {
        "state": "training",
        "step": step,
        "steps": args.steps,
        "node_lr": args.lr,
        "vision_lr": args.vision_lr if vision else None,
        "parameters": sum(p.numel() for p in parameters),
    }
    report(root, status)
    try:
        for step in range(start_step + 1, args.steps + 1):
            ids = sampler.choice(train_ids, args.batch_size)
            model.train()
            model.chain[0].odefunc.calc_density = True
            if vision:
                vision.train()
            condition = pose[ids].to(device)
            if vision:
                condition = torch.cat([condition, vision(image_batch(arrays, ids, device))], 1)
            optimizer.zero_grad(set_to_none=True)
            loss = nll(model, q[ids].to(device), condition).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training NLL at step {step}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
            if vision and step == start_step + 1:
                encoder_gradient = sum(
                    float(p.grad.abs().sum()) for p in vision.parameters() if p.grad is not None
                )
                if encoder_gradient <= 0:
                    raise RuntimeError("CNF NLL did not propagate gradients to the vision encoder")
                print(f"End-to-end encoder gradient L1: {encoder_gradient:.6f}", flush=True)
            optimizer.step()
            loss_total += loss.item()
            loss_count += 1
            if step == start_step + 1 or step % 100 == 0 or step == args.steps:
                now = time.monotonic()
                speed = (step - logged_step) / max(now - last_update, 1e-6)
                record(
                    {
                        "event": "train",
                        "train_nll": loss_total / loss_count,
                        "steps_per_second": speed,
                        "eta_seconds": (args.steps - step) / speed,
                        "elapsed_seconds": now - started,
                        "gradient_norm": float(gradient_norm),
                        "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0,
                    }
                )
                loss_total, loss_count, last_update, logged_step = 0.0, 0, now, step
            if step % args.eval_every == 0 or step == args.steps:
                metrics = validation(model, vision, robot, q, pose, target, val_ids, arrays, args, device)
                if not all(math.isfinite(metrics[k]) for k in ("validation_nll", "fk_score")):
                    raise FloatingPointError("Non-finite validation metric")
                score, value = metrics["fk_score"], metrics["validation_nll"]
                meaningful = score < progress_fk * 0.995 or value < progress_nll - 0.01
                if meaningful:
                    progress_fk, progress_nll = min(progress_fk, score), min(progress_nll, value)
                improved_fk, improved_nll = score < best_fk, value < best_nll
                best_fk, best_nll = min(best_fk, score), min(best_nll, value)
                stalled = 0 if meaningful else stalled + 1
                if improved_fk:
                    checkpoint(directory / "best.pt")
                if improved_nll:
                    checkpoint(directory / "best_nll.pt")
                checkpoint(last)
                record(
                    {
                        "event": "validation",
                        **metrics,
                        "best_fk": best_fk,
                        "best_nll": best_nll,
                        "stalled_evaluations": stalled,
                    }
                )
                if step >= args.min_steps and stalled >= args.patience:
                    status["modes"][mode]["state"] = "early_stopped"
                    break
        else:
            status["modes"][mode]["state"] = "completed"
        checkpoint(last)
    except KeyboardInterrupt:
        status["modes"][mode]["state"] = "interrupted"
        raise
    except Exception as error:
        status["modes"][mode].update(state="failed", error=str(error))
        raise
    finally:
        report(root, status)


def train(args, device):
    from lerobot.state_free.ai_worker import AIWorkerConfig, AIWorkerRobot

    if args.uniform_samples and args.mode != "pose":
        raise ValueError("Uniform FK samples have no images; use --mode pose")
    if args.resume and args.log_dir is None:
        raise ValueError("--resume requires the existing --log-dir")
    root = args.log_dir or Path("log") / datetime.now(ZoneInfo("Asia/Seoul")).strftime("nodeik_%Y%m%d_%H%M%S")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((root / mode / "last.pt").exists() for mode in ("pose", "vision")):
        raise FileExistsError("Run directory already contains checkpoints; use --resume or a new --log-dir")
    write_json(
        root / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    robot = AIWorkerRobot(AIWorkerConfig(**({"urdf_path": args.urdf} if args.urdf else {})))
    if args.uniform_samples:
        mapping = [robot._joint_q_index(name) for name in JOINT_NAMES]
        q = (
            np.random.default_rng(args.seed)
            .uniform(
                robot.model.lowerPositionLimit[mapping],
                robot.model.upperPositionLimit[mapping],
                (args.uniform_samples, 15),
            )
            .astype(np.float32)
        )
        indices, episodes = np.arange(len(q)), np.zeros(len(q), dtype=int)
    else:
        q, indices, episodes = load_joints(args.dataset, args.source, args.limit)
    target = fk_poses(robot, q)
    pose = torch.from_numpy(encode_poses(target))
    unique = np.unique(episodes)
    held_out = (
        np.isin(episodes, unique[-max(1, len(unique) // 10) :])
        if len(unique) > 1
        else np.arange(len(q)) >= max(1, int(len(q) * 0.9))
    )
    train_ids, all_val_ids = np.flatnonzero(~held_out), np.flatnonzero(held_out)
    if not len(train_ids) or not len(all_val_ids):
        raise ValueError("Need at least two frames")
    val_ids = (
        all_val_ids[np.linspace(0, len(all_val_ids) - 1, args.eval_frames, dtype=int)]
        if args.eval_frames and len(all_val_ids) > args.eval_frames
        else all_val_ids
    )
    info = json.loads((args.dataset / "meta/info.json").read_text()) if not args.uniform_samples else {}
    keys = [key for key, feature in info.get("features", {}).items() if feature["dtype"] == "video"]
    metadata = {
        "dataset": str(args.dataset.resolve()),
        "frames": len(q),
        "episodes": len(unique),
        "train_frames": len(train_ids),
        "validation_frames": len(all_val_ids),
        "evaluated_frames": len(val_ids),
        "train_episodes": np.unique(episodes[train_ids]).tolist(),
        "validation_episodes": np.unique(episodes[all_val_ids]).tolist(),
        "batch_size": args.batch_size,
        "epoch_definition": "optimizer samples / training frames (sampling with replacement)",
        "steps_per_epoch_equivalent": len(train_ids) / args.batch_size,
        "epoch_equivalents": {
            str(steps): steps * args.batch_size / len(train_ids) for steps in (30000, 50000, 100000)
        },
        "requested_epoch_equivalent": args.steps * args.batch_size / len(train_ids),
        "joint_names": JOINT_NAMES,
        "image_keys": keys,
        "image_size": args.image_size,
        "best_metric": "mean position(m) + 0.1 * mean orientation(rad)",
        "early_stop": "no FK improvement >= 0.5% and no NLL decrease >= 0.01 for patience evaluations",
    }
    write_json(root / "dataset.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    modes = ("vision", "pose") if args.mode == "compare" else (args.mode,)
    status = {
        "state": "running",
        "run": str(root),
        "device": str(device),
        "dataset": metadata,
        "modes": {mode: {"state": "queued"} for mode in modes},
        "cache_removed": False,
    }
    report(root, status)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with tempfile.TemporaryDirectory(prefix="nodeik_", dir=args.cache_root) as temporary:
            cache = Path(temporary)
            status["cache_path"] = str(cache)
            arrays = {}
            if "vision" in modes:
                if not keys:
                    raise ValueError("Vision requires RGB video features in the dataset")
                status["state"] = "building_pixel_cache"
                report(root, status)

                def progress(value):
                    status["cache_progress"] = value
                    report(root, status)
                    print(json.dumps({"cache": value}), flush=True)

                arrays = build_cache(args.dataset, cache, indices, episodes, keys, args.image_size, progress)
            status["state"] = "training"
            report(root, status)
            for mode in modes:
                run_mode(
                    mode,
                    args,
                    device,
                    root,
                    status,
                    robot,
                    torch.from_numpy(q),
                    pose,
                    target,
                    train_ids,
                    val_ids,
                    arrays if mode == "vision" else {},
                    keys,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            arrays.clear()
            status["state"] = "completed"
    except KeyboardInterrupt:
        status["state"] = "interrupted"
        raise
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        status["cache_removed"] = not Path(status.get("cache_path", args.cache_root / "absent")).exists()
        signal.signal(signal.SIGTERM, previous)
        report(root, status)
    if status["state"] == "completed" and not args.uniform_samples:
        from nodeik.visualize_worker import visualize

        try:
            visualize(root, seed=args.seed)
        except (RuntimeError, ValueError, OSError, ArithmeticError) as error:
            status["visualization"] = {"state": "failed", "error": str(error)}
            report(root, status)
            print(f"Visualization failed; retry with python -m nodeik.visualize_worker: {error}", flush=True)
