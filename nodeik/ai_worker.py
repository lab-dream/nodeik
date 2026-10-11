"""Offline AI Worker NODE IK: official CNF layers + LeRobot FK/RGB encoder."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.state_free.ai_worker import AIWorkerConfig, AIWorkerRobot
from lerobot.state_free.math import mat_to_pose10d
from torch import nn

from nodeik.utils import build_model

DATASET = Path(__file__).resolve().parents[2] / "dataset/Task_0_tape_MCAP_lerobot_v30"
JOINT_NAMES = [
    *(f"arm_l_joint{i}" for i in range(1, 8)),
    *(f"arm_r_joint{i}" for i in range(1, 8)),
    "lift_joint",
]


def load_joints(root, source="observation.state", limit=0):
    """Read named joints without decoding videos; preserve absolute frame indices."""
    import pyarrow.parquet as pq

    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"][source]["names"]
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError(f"{source} must have unique joint names in meta/info.json")
    mapping = [names.index(name) for name in JOINT_NAMES]
    rows, indices, episodes = [], [], []
    for file in sorted((root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(
            batch_size=8192, columns=[source, "index", "episode_index"]
        ):
            data = batch.to_pydict()
            rows.extend(data[source])
            indices.extend(data["index"])
            episodes.extend(data["episode_index"])
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No joint rows under {root / 'data'}")
    order = np.argsort(indices)
    if limit:
        order = order[:limit]
    q = np.asarray(rows, dtype=np.float32)[order][:, mapping]
    if not np.isfinite(q).all():
        raise ValueError("Dataset contains non-finite joint positions")
    return q, np.asarray(indices)[order], np.asarray(episodes)[order]


def fk_poses(robot, joints):
    """Return (N, 2, 4, 4) TCP transforms, left then right, in fixed-base world."""
    poses = []
    for q in np.atleast_2d(joints):
        pair = robot.forward_kinematics_bimanual(q[:7], q[7:14], float(q[14]))
        poses.append(np.stack([pair["left"], pair["right"]]))
    return np.stack(poses)


def encode_poses(poses):
    """LeRobot UMI convention: xyz + first two rotation rows, left then right."""
    return mat_to_pose10d(poses).reshape(-1, 18).astype(np.float32)


class Vision(nn.Module):
    """Randomly initialized LeRobot ResNet18 encoders, trained through CNF NLL."""

    def __init__(self, config):
        super().__init__()
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
        from torchvision.transforms.v2 import Resize

        self.keys = config["keys"]
        size = config["size"]
        cfg = DiffusionConfig(
            input_features={key: PolicyFeature(FeatureType.VISUAL, (3, size, size)) for key in self.keys},
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            use_group_norm=True,
            resize_shape=(size, size),
            crop_ratio=config["crop_ratio"],
            crop_is_random=True,
            spatial_softmax_num_keypoints=32,
            use_separate_rgb_encoder_per_camera=True,
            device="cpu",
        )
        self.encoders = nn.ModuleList([DiffusionRgbEncoder(cfg) for _ in self.keys])
        self.resize = Resize((size, size), antialias=True)
        # Same fixed VISUAL normalization as the referenced Diffusion training.
        # These constants are preprocessing statistics, not pretrained weights.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        features = []
        for key, encoder in zip(self.keys, self.encoders, strict=True):
            value = self.resize(images[key])
            value = value.float() / 255 if value.dtype == torch.uint8 else value.float()
            if value.ndim == 3:
                value = value.unsqueeze(0)
            features.append(encoder((value - self.mean) / self.std))
        return torch.cat(features, dim=-1)


def make_flow(config, device):
    args = SimpleNamespace(
        dims=config["dims"],
        num_blocks=1,
        time_length=1.0,
        train_T=False,
        layer_type="concatsquash",
        nonlinearity="tanh",
        rademacher=False,
        divergence_fn="approximate",
        solver="dopri5",
        atol=config["tol"],
        rtol=config["tol"],
    )
    return build_model(args, 15, config["condition_dim"], regularization_fns=None).to(device)


@torch.no_grad()
def predict(model, condition, latent="0", samples=1):
    """Return (target, sample, 15) physical joints. No clipping or IK refinement."""
    if samples < 1:
        raise ValueError("samples must be positive")
    model.eval()
    model.chain[0].odefunc.calc_density = False
    c = condition.repeat_interleave(samples, dim=0)
    z = (
        torch.randn(len(c), 15, device=c.device)
        if latent == "gaussian"
        else torch.full((len(c), 15), float(latent), device=c.device)
    )
    q = model(z, c, rev=True)
    if not torch.isfinite(q).all():
        raise FloatingPointError("ODE produced non-finite joints")
    return q.reshape(len(condition), samples, 15).cpu().numpy()


def evaluate(robot, joints, target):
    """Geodesic rotation errors in radians; every sample is included."""
    actual = fk_poses(robot, joints.reshape(-1, 15)).reshape(*joints.shape[:2], 2, 4, 4)
    target = np.asarray(target)[:, None]
    position = np.linalg.norm(actual[..., :3, 3] - target[..., :3, 3], axis=-1)
    relative = np.swapaxes(target[..., :3, :3], -1, -2) @ actual[..., :3, :3]
    rotation = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    metrics = {
        side: {
            "position_mean_m": float(position[..., i].mean()),
            "position_max_m": float(position[..., i].max()),
            "orientation_mean_rad": float(rotation[..., i].mean()),
        }
        for i, side in enumerate(("left", "right"))
    }
    mapping = [robot._joint_q_index(name) for name in JOINT_NAMES]
    lower, upper = (robot.model.lowerPositionLimit[mapping], robot.model.upperPositionLimit[mapping])
    metrics["joint_limit_violation_fraction"] = float(
        np.any((joints < lower) | (joints > upper), axis=-1).mean()
    )
    return metrics, actual


def load_model(path, device, urdf=None):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["joint_names"] != JOINT_NAMES or saved["pose_rep"] != "absolute_xyz_row_rot6d_left_right":
        raise ValueError("Checkpoint joint order or pose representation is incompatible")
    model = make_flow(saved["config"], device)
    model.load_state_dict(saved["model"], strict=True)
    cfg = AIWorkerConfig(urdf_path=urdf or saved["urdf"])
    robot = AIWorkerRobot(cfg)
    if saved.get("version") != 2:
        raise ValueError("Use an end-to-end v2 checkpoint; old frozen-encoder checkpoints are incompatible")
    vision = Vision(saved["vision_config"]).to(device) if saved["vision_config"] else None
    if vision:
        vision.load_state_dict(saved["vision"], strict=True)
        vision.eval()
    return model, robot, vision, saved


def read_request(path, vision):
    """NPZ poses: (2,4,4)/(N,2,4,4), plus RGB CHW/BCHW for each camera."""
    with np.load(path, allow_pickle=False) as data:
        poses = np.asarray(data["poses"], dtype=np.float64)
        if poses.shape == (2, 4, 4):
            poses = poses[None]
        if poses.ndim != 4 or poses.shape[1:] != (2, 4, 4) or not np.isfinite(poses).all():
            raise ValueError("poses must be finite (N,2,4,4) absolute left/right TCP transforms")
        rotation = poses[..., :3, :3]
        if (
            not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)
        ):
            raise ValueError("poses must contain valid SE(3) transforms")
        images = {}
        if vision:
            for key in vision.keys:
                value = torch.from_numpy(data[key].copy())
                if value.dtype == torch.uint8:
                    value = value.float() / 255
                if value.ndim == 3:
                    value = value.unsqueeze(0)
                if (
                    value.ndim != 4
                    or value.shape[:2] != (len(poses), 3)
                    or not torch.isfinite(value).all()
                    or value.min() < 0
                    or value.max() > 1
                ):
                    raise ValueError(f"{key} must be RGB BCHW float [0,1] or uint8 [0,255]")
                images[key] = value.float()
    return poses, images


def run_inference(args, device):
    model, robot, vision, saved = load_model(args.model, device, args.urdf)
    if args.command == "infer" and args.input:
        target, images = read_request(args.input, vision)
        condition = torch.from_numpy(encode_poses(target))
        if vision:
            with torch.no_grad():
                images = {key: value.to(device) for key, value in images.items()}
                condition = torch.cat([condition, vision(images).cpu()], 1)
    else:
        limit = args.limit if args.command == "eval" else args.index + 1
        q, indices, _ = load_joints(args.dataset, saved["source"], limit)
        if args.command == "infer":
            if args.index >= len(q):
                raise ValueError("Dataset index out of range")
            q, indices = (q[args.index : args.index + 1], indices[args.index : args.index + 1])
        target = fk_poses(robot, q)
        condition = torch.from_numpy(encode_poses(target))
        if vision:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset(Path(args.dataset).name, root=args.dataset, video_backend="pyav")
            features = []
            with torch.no_grad():
                for start in range(0, len(indices), args.vision_batch_size):
                    items = [dataset[int(i)] for i in indices[start : start + args.vision_batch_size]]
                    images = {
                        key: torch.stack([item[key] for item in items]).to(device) for key in vision.keys
                    }
                    features.append(vision(images).cpu())
            condition = torch.cat([condition, torch.cat(features)], 1)
    joints = predict(model, condition.to(device), args.latent, args.samples)
    metrics, actual = evaluate(robot, joints, target)
    result = {
        "frames": len(target),
        "samples": args.samples,
        "latent": args.latent,
        "joint_names": JOINT_NAMES,
        "fk_errors": metrics,
    }
    if args.command == "infer":
        result["joints_rad_lift_m"] = joints.tolist()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, joints=joints, joint_names=JOINT_NAMES, target_poses=target, fk_poses=actual)
        result["output"] = str(output)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval", "infer"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", type=Path, default=DATASET)
        sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--threads", type=int, default=4)
        sub.add_argument("--urdf", help="Override the existing AI Worker URDF")
        sub.add_argument("--vision-batch-size", type=int, default=16)
        if name == "train":
            sub.add_argument("--mode", choices=("pose", "vision", "compare"), default="vision")
            sub.add_argument("--source", choices=("observation.state", "action"), default="observation.state")
            sub.add_argument("--limit", type=int, default=0)
            sub.add_argument("--uniform-samples", type=int, default=0)
            sub.add_argument("--steps", type=int, default=30000)
            sub.add_argument("--batch-size", type=int, default=64)
            sub.add_argument("--dims", default="256-256-256-256")
            sub.add_argument("--tol", type=float, default=1e-5)
            sub.add_argument("--lr", type=float, default=3e-4)
            sub.add_argument("--vision-lr", type=float, default=1e-4)
            sub.add_argument("--eval-every", type=int, default=1000)
            sub.add_argument("--eval-frames", type=int, default=512, help="Fixed held-out subset; 0: all")
            sub.add_argument(
                "--patience",
                type=int,
                default=10,
                help="Evaluation rounds with neither FK nor NLL improvement",
            )
            sub.add_argument("--min-steps", type=int, default=10000)
            sub.add_argument("--log-dir", type=Path)
            sub.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/nodeik")
            sub.add_argument("--image-size", type=int, default=96)
            sub.add_argument(
                "--resume", action="store_true", help="Resume last.pt in the specified log directory"
            )
        else:
            sub.add_argument("--model", type=Path, required=True)
            sub.add_argument("--latent", choices=("0", "0.1", "gaussian"), default="0")
            sub.add_argument("--samples", type=int, default=1)
            sub.add_argument("--output", type=Path, help="Optional predictions .npz")
            if name == "eval":
                sub.add_argument("--limit", type=int, default=128)
            else:
                sub.add_argument(
                    "--input", type=Path, help="NPZ absolute TCP transforms and optional RGB images"
                )
                sub.add_argument(
                    "--index", type=int, default=0, help="Dataset frame used when --input is absent"
                )
    args = parser.parse_args()
    if args.threads < 1 or args.vision_batch_size < 1:
        parser.error("threads and vision-batch-size must be positive")
    if args.command == "train" and (
        args.steps < 1
        or args.batch_size < 1
        or args.eval_frames < 0
        or args.eval_every < 1
        or args.patience < 1
        or args.image_size < 32
        or args.min_steps < 0
        or args.limit < 0
        or args.uniform_samples < 0
        or args.tol <= 0
    ):
        parser.error(
            "steps/batch-size/eval-every/patience/tol must be positive; counts nonnegative; image-size >= 32"
        )
    if args.command != "train" and (
        args.samples < 1 or getattr(args, "index", 0) < 0 or getattr(args, "limit", 0) < 0
    ):
        parser.error("samples must be positive and index/limit nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.command == "train":
        from nodeik.train_worker import train

        train(args, device)
    else:
        run_inference(args, device)


if __name__ == "__main__":
    main()
