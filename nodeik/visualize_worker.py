"""Offline FK trajectory replay, using one latent per path as in the official examples."""

import argparse
import html
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from nodeik.ai_worker import JOINT_NAMES, encode_poses, evaluate, fk_poses, load_joints, load_model
from nodeik.train_worker import build_cache, image_batch, report, write_json


def skeleton(robot, joints):
    positions = []
    names = [JOINT_NAMES[-1], *JOINT_NAMES[:-1]]
    for q in joints:
        pair = robot.forward_kinematics_bimanual(q[:7], q[7:14], float(q[14]))
        positions.append(
            np.stack(
                [np.zeros(3)]
                + [robot.data.oMi[robot.model.getJointId(name)].translation.copy() for name in names]
                + [pair[side][:3, 3] for side in ("left", "right")]
            )
        )
    return np.asarray(positions)


def visualize(root, frames=240, seed=0, device="cpu"):
    """Compare best-FK checkpoints on the same contiguous held-out episode segment."""
    root = Path(root).resolve()
    status = json.loads((root / "status.json").read_text())
    if status["state"] != "completed" or not status.get("cache_removed"):
        raise ValueError("Wait for training completion and cache removal before visualizing")
    config = json.loads((root / "config.json").read_text())
    if config.get("uniform_samples"):
        raise ValueError("Recorded trajectory visualization requires the dataset training mode")
    device = torch.device(device)
    torch.manual_seed(seed)
    gaussian = torch.randn(1, 15)  # Shared across modes and held fixed over the entire trajectory.
    output = root / "visualization"
    output.mkdir(exist_ok=True)
    result, summary, arrays = {}, {}, {}
    reference_ids = None
    for mode in status["modes"]:
        model, robot, vision, saved = load_model(root / mode / "best.pt", device)
        q, indices, episodes = load_joints(Path(config["dataset"]), saved["source"], config["limit"])
        val_ids = saved["val_ids"].numpy()
        episode = int(episodes[val_ids[0]])
        ids = np.flatnonzero(episodes == episode)
        ids = ids[~np.isin(ids, saved["train_ids"].numpy())]
        if len(ids) > frames:
            start = (len(ids) - frames) // 2
            ids = ids[start : start + frames]
        if not len(ids):
            raise ValueError("No contiguous held-out frames")
        if reference_ids is not None and not np.array_equal(indices[ids], reference_ids):
            raise ValueError("Pose and Vision checkpoints must share the same trajectory")
        reference_ids = indices[ids]
        target = fk_poses(robot, q[ids])
        condition = torch.from_numpy(encode_poses(target))
        if vision:
            cache_root = Path(config["cache_root"])
            cache_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="nodeik_vis_", dir=cache_root) as directory:
                pixels = build_cache(
                    Path(config["dataset"]),
                    Path(directory),
                    indices[ids],
                    episodes[ids],
                    vision.keys,
                    saved["vision_config"]["size"],
                    lambda value: print(json.dumps({"visualization_cache": value}), flush=True),
                )
                features = []
                with torch.no_grad():
                    for start in range(0, len(ids), 16):
                        features.append(
                            vision(
                                image_batch(pixels, np.arange(start, min(start + 16, len(ids))), device)
                            ).cpu()
                        )
                condition = torch.cat([condition, torch.cat(features)], 1)
                pixels.clear()
        if not result:
            info = json.loads((Path(config["dataset"]) / "meta/info.json").read_text())
            result.update(
                episode=episode,
                indices=indices[ids].tolist(),
                fps=info["fps"],
                target=target[..., :3, 3].tolist(),
                reference_skeleton=skeleton(robot, q[ids]).tolist(),
                runs={},
                joint_names=JOINT_NAMES,
            )
            arrays.update(
                indices=indices[ids], target_poses=target, reference_joints=q[ids], gaussian=gaussian.numpy()
            )
        model.eval()
        model.chain[0].odefunc.calc_density = False
        for latent in ("0", "0.1", "gaussian"):
            z = gaussian if latent == "gaussian" else torch.full((1, 15), float(latent))
            with torch.no_grad():
                predicted = np.concatenate(
                    [
                        model(z.to(device).expand(len(batch), -1), batch.to(device), rev=True).cpu().numpy()
                        for batch in condition.split(64)
                    ]
                )
            if not np.isfinite(predicted).all():
                raise FloatingPointError(f"Non-finite trajectory: {mode}/{latent}")
            metrics, actual = evaluate(robot, predicted[:, None], target)
            actual = actual[:, 0]
            position = np.linalg.norm(actual[..., :3, 3] - target[..., :3, 3], axis=-1)
            relative = np.swapaxes(target[..., :3, :3], -1, -2) @ actual[..., :3, :3]
            rotation = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
            delta = np.abs(np.diff(predicted, axis=0))
            metrics.update(
                checkpoint_step=int(saved["step"]),
                arm_max_step_rad=float(delta[:, :14].max()) if len(delta) else 0.0,
                lift_max_step_m=float(delta[:, 14].max()) if len(delta) else 0.0,
                arm_steps_over_0_1_rad=int(np.any(delta[:, :14] > 0.1, axis=1).sum()),
            )
            key = f"{mode} / {latent}"
            summary[key] = metrics
            result["runs"][key] = {
                "skeleton": skeleton(robot, predicted).tolist(),
                "actual": actual[..., :3, 3].tolist(),
                "position_mm": (position * 1000).tolist(),
                "rotation_deg": np.rad2deg(rotation).tolist(),
                "joints": predicted.tolist(),
                "metrics": metrics,
            }
            arrays[f"{mode}_{latent}_joints"] = predicted
            arrays[f"{mode}_{latent}_fk"] = actual
            print(json.dumps({"trajectory": key, **metrics}), flush=True)
        del model, vision, saved
    np.savez_compressed(output / "trajectories.npz", **arrays)
    write_json(output / "metrics.json", summary)
    template = VIEWER.replace("__DATA__", json.dumps(result, allow_nan=False))
    (output / "viewer.html").write_text(template)
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{m['checkpoint_step']}</td>"
        f"<td>{1000 * (m['left']['position_mean_m'] + m['right']['position_mean_m']) / 2:.2f}</td>"
        f"<td>{np.rad2deg((m['left']['orientation_mean_rad'] + m['right']['orientation_mean_rad']) / 2):.2f}</td>"
        f"<td>{100 * m['joint_limit_violation_fraction']:.1f}%</td>"
        f"<td>{m['arm_steps_over_0_1_rad']}</td></tr>"
        for key, m in summary.items()
    )
    (output / "section.html").write_text(
        "<section><h2>Result visualization · FK trajectory replay</h2>"
        f"<p>Held-out episode {result['episode']}, {len(reference_ids)} contiguous frames. "
        "best FK checkpoints · latent 0 / 0.1 / fixed Gaussian (seed "
        f"{seed}). Same latent over the whole path, following the official examples. "
        "All predictions included; no joint clipping or continuity filtering.</p>"
        "<p>Offline URDF link skeleton; gray = recorded joints, blue/orange = predicted arms, "
        "dotted = target EE path. This segment comparison complements the validation metrics above.</p>"
        "<table><tr><th>Model / latent</th><th>Best step</th><th>Position (mm)</th>"
        "<th>Orientation (deg)</th><th>Limit violation</th><th>Arm steps &gt;0.1 rad</th></tr>"
        + rows
        + '</table><p><a href="visualization/viewer.html">Open interactive viewer</a> · '
        '<a href="visualization/trajectories.npz">Trajectories NPZ</a> · '
        '<a href="visualization/metrics.json">Metrics JSON</a></p>'
        '<iframe title="NODE IK trajectory" src="visualization/viewer.html" '
        'style="width:100%;height:900px;border:0;border-radius:8px"></iframe></section>'
    )
    status["visualization"] = {
        "state": "completed",
        "episode": result["episode"],
        "frames": len(reference_ids),
    }
    report(root, status)


VIEWER = r"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>NODE IK trajectories</title>
<style>body{margin:16px;background:#0f172a;color:#e2e8f0;font:14px system-ui}canvas{width:100%;background:#111e31;border-radius:8px}button,select{padding:8px;background:#334155;color:white;border:0;border-radius:5px}input{width:50%}pre{white-space:pre-wrap}p{margin:8px 0}</style>
<select id="model"></select> <button id="play">Pause</button> <input id="frame" type="range" min="0" value="0"> <span id="time"></span>
<p>Drag to orbit · scroll to zoom · dotted EE path: target · blue/orange: prediction · gray: recorded skeleton</p>
<canvas id="scene" width="1000" height="480"></canvas>
<p>EE position error (mm), left / right</p><canvas id="pos" width="1000" height="115"></canvas>
<p>EE orientation error (deg), left / right</p><canvas id="rot" width="1000" height="115"></canvas><pre id="values"></pre>
<script>
const data=__DATA__, keys=Object.keys(data.runs), model=document.getElementById('model'), slider=document.getElementById('frame'), scene=document.getElementById('scene'), ctx=scene.getContext('2d');
keys.forEach(k=>model.add(new Option(k,k)));slider.max=data.indices.length-1;
let frame=0,playing=true,yaw=0.7,pitch=0.25,zoom=1,drag=null,last=0;
const colors=['#38bdf8','#fb923c'], chains=[[1,2,3,4,5,6,7,8,16],[1,9,10,11,12,13,14,15,17]];
const cloud=[...data.reference_skeleton.flat(),...Object.values(data.runs).flatMap(r=>r.skeleton.flat())];
const lo=[0,1,2].map(a=>Math.min(...cloud.map(p=>p[a]))), hi=[0,1,2].map(a=>Math.max(...cloud.map(p=>p[a]))), center=lo.map((v,i)=>(v+hi[i])/2), scale=340/Math.max(...hi.map((v,i)=>v-lo[i]),0.5);
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],u=x*Math.cos(yaw)-y*Math.sin(yaw),v=x*Math.sin(yaw)+y*Math.cos(yaw);return[500+u*scale*zoom,240-(z*Math.cos(pitch)-v*Math.sin(pitch))*scale*zoom];}
function line(points,color,width,dash=[]){ctx.beginPath();points.forEach((p,i)=>{let[a,b]=project(p);i?ctx.lineTo(a,b):ctx.moveTo(a,b)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
function body(points,ghost=false){line([points[0],points[1]],ghost?'#64748b':'#e2e8f0',ghost?2:4);chains.forEach((chain,s)=>{line(chain.map(i=>points[i]),ghost?'#64748b':colors[s],ghost?2:4);if(!ghost)chain.forEach(i=>{let[x,y]=project(points[i]);ctx.fillStyle=colors[s];ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()})});}
function chart(id,series){let c=document.getElementById(id),g=c.getContext('2d'),max=Math.max(...series.flat(),0.001);g.clearRect(0,0,c.width,c.height);g.fillStyle='#94a3b8';g.fillText(max.toFixed(2),6,15);g.fillText('0',6,105);[0,1].forEach(s=>{g.beginPath();series.forEach((p,i)=>{let x=48+930*i/Math.max(series.length-1,1),y=103-88*p[s]/max;i?g.lineTo(x,y):g.moveTo(x,y)});g.strokeStyle=colors[s];g.lineWidth=2;g.stroke()});g.beginPath();let x=48+930*frame/Math.max(series.length-1,1);g.moveTo(x,10);g.lineTo(x,105);g.strokeStyle='#f8fafc';g.lineWidth=1;g.stroke();}
function draw(){const r=data.runs[model.value];ctx.clearRect(0,0,1000,480);for(let s=0;s<2;s++){line(data.target.map(p=>p[s]),'#94a3b8',2,[4,5]);line(r.actual.map(p=>p[s]),colors[s],2);let[x,y]=project(data.target[frame][s]);ctx.strokeStyle='#a3e635';ctx.lineWidth=2;ctx.strokeRect(x-5,y-5,10,10)}body(data.reference_skeleton[frame],true);body(r.skeleton[frame]);chart('pos',r.position_mm);chart('rot',r.rotation_deg);document.getElementById('time').textContent=`frame ${data.indices[frame]} · ${(frame/data.fps).toFixed(2)} s`;document.getElementById('values').textContent=`${model.value} | Lift ${r.joints[frame][14].toFixed(4)} m\nLeft (rad): ${r.joints[frame].slice(0,7).map(x=>x.toFixed(3)).join(' ')}\nRight (rad): ${r.joints[frame].slice(7,14).map(x=>x.toFixed(3)).join(' ')}`;}
model.onchange=draw;slider.oninput=()=>{frame=+slider.value;draw()};document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play'};
scene.onpointerdown=e=>{drag=[e.clientX,e.clientY];scene.setPointerCapture(e.pointerId)};scene.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*0.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*0.01));drag=[e.clientX,e.clientY];draw()}};scene.onpointerup=()=>drag=null;scene.onwheel=e=>{e.preventDefault();zoom=Math.max(0.2,Math.min(5,zoom*Math.exp(-e.deltaY*0.001)));draw()};
function tick(t){if(playing&&t-last>=1000/data.fps){frame=(frame+1)%data.indices.length;slider.value=frame;last=t;draw()}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wait", action="store_true", help="Wait for the background training to finish")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    torch.set_num_threads(4)
    if args.wait:
        print(f"Waiting for training completion: {args.run_dir}", flush=True)
        while True:
            status = json.loads((args.run_dir / "status.json").read_text())
            if status["state"] in ("failed", "interrupted"):
                raise RuntimeError(f"Training {status['state']}; resume it and restart this watcher")
            if status["state"] == "completed" and status.get("cache_removed"):
                break
            time.sleep(20)
    visualize(args.run_dir, args.frames, args.seed, args.device)
    print(f"Visualization complete: {args.run_dir / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
