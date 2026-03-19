#!/usr/bin/env python3
"""Run vision benchmark matrix (DA3 + PromptDA + SAM3 + SAM2_w_prompt).

Designed for Docker usage in `handrl-policy`:
  cd /workspace
  /root/miniconda3/envs/policy/bin/python dep/vision_benchmark_suite.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


def run_capture(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def parse_metrics(log_text: str) -> Dict[str, Any]:
    def last_float(pattern: str) -> float | None:
        matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
        return float(matches[-1]) if matches else None

    mean_match = re.findall(
        r"mean\s*:\s*([0-9.]+)\s*ms\s*\(\s*([0-9.]+)\s*Hz\)",
        log_text,
        flags=re.IGNORECASE,
    )
    mean_ms = float(mean_match[-1][0]) if mean_match else None
    mean_hz = float(mean_match[-1][1]) if mean_match else None

    video_match = re.findall(r"video saved\s*[→:]\s*(.+?)(?:\s*\(|$)", log_text)
    video_path = video_match[-1].strip() if video_match else ""

    return {
        "mean_ms": mean_ms,
        "mean_hz": mean_hz,
        "min_ms": last_float(r"min\s*:\s*([0-9.]+)\s*ms"),
        "max_ms": last_float(r"max\s*:\s*([0-9.]+)\s*ms"),
        "std_ms": last_float(r"std\s*:\s*([0-9.]+)\s*ms"),
        "video_path_from_log": video_path,
    }


def ensure_exists(paths: List[Path], header: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"{header}\n" + "\n".join(missing))


def build_jobs(
    *,
    python_exe: str,
    repo_root: Path,
    data_root: Path,
    color_dir: Path,
    out_root: Path,
    video_fps: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    focal: float,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    fps_s = str(int(video_fps) if float(video_fps).is_integer() else video_fps)

    jobs.append(
        {
            "id": "da3_metric_large",
            "tool": "da3",
            "model": "depth-anything/DA3METRIC-LARGE",
            "cmd": [
                python_exe,
                "-m",
                "robot_motion_interface.utils.da3_utils",
                "--model",
                "depth-anything/DA3METRIC-LARGE",
                "--frames_dir",
                str(data_root),
                "--out_dir",
                str(out_root / "da3_metric_large"),
                "--video_fps",
                fps_s,
                "--focal",
                str(focal),
                "--device",
                "cuda",
            ],
        }
    )
    jobs.append(
        {
            "id": "da3_nested_giant_large_1_1",
            "tool": "da3",
            "model": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "cmd": [
                python_exe,
                "-m",
                "robot_motion_interface.utils.da3_utils",
                "--model",
                "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                "--frames_dir",
                str(data_root),
                "--out_dir",
                str(out_root / "da3_nested_giant_large_1_1"),
                "--video_fps",
                fps_s,
                "--fx",
                str(fx),
                "--fy",
                str(fy),
                "--cx",
                str(cx),
                "--cy",
                str(cy),
                "--device",
                "cuda",
            ],
        }
    )

    for ckpt, enc, job_id in [
        ("PromptDA-s-transparent.ckpt", "vits", "promptda_s_transparent_vits"),
        ("PromptDA-s.ckpt", "vits", "promptda_s_vits"),
        ("PromptDA-l.ckpt", "vitl", "promptda_l_vitl"),
    ]:
        jobs.append(
            {
                "id": job_id,
                "tool": "promptda",
                "model": ckpt,
                "cmd": [
                    python_exe,
                    "-m",
                    "robot_motion_interface.utils.promptda_utils",
                    "--ckpt",
                    str(repo_root / "models" / "promptda" / ckpt),
                    "--encoder",
                    enc,
                    "--frames_dir",
                    str(data_root),
                    "--out_dir",
                    str(out_root / job_id),
                    "--video_fps",
                    fps_s,
                    "--device",
                    "cuda",
                ],
            }
        )

    sam3_matrix = [
        (
            "sam3_original",
            "sam3.pt",
            ["--mode", "sam3"],
        ),
        (
            "sam3_litetext_s0",
            "efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
            ["--mode", "litetext", "--text_encoder_type", "MobileCLIP-S0", "--text_encoder_context_length", "16"],
        ),
        (
            "sam3_litetext_s1",
            "efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt",
            ["--mode", "litetext", "--text_encoder_type", "MobileCLIP-S1", "--text_encoder_context_length", "16"],
        ),
        (
            "sam3_litetext_mobileclip2_l",
            "efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt",
            ["--mode", "litetext", "--text_encoder_type", "MobileCLIP2-L", "--text_encoder_context_length", "16"],
        ),
        (
            "sam3_efficient_efficientvit_b0",
            "efficient_sam3_efficientvit_s.pt",
            ["--mode", "efficient", "--backbone_type", "efficientvit", "--model_name", "b0"],
        ),
    ]
    for job_id, ckpt, extra in sam3_matrix:
        jobs.append(
            {
                "id": job_id,
                "tool": "sam3",
                "model": ckpt,
                "cmd": [
                    python_exe,
                    "-m",
                    "robot_motion_interface.utils.sam3_utils",
                    "--ckpt",
                    str(repo_root / "models" / "sam3" / ckpt),
                    "--frames_dir",
                    str(color_dir),
                    "--out_dir",
                    str(out_root / job_id),
                    "--video_fps",
                    fps_s,
                    "--compile",
                    "--device",
                    "cuda",
                    *extra,
                ],
            }
        )

    for job_id, ckpt, cfg in [
        ("sam2_tiny", "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        ("sam2_small", "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        ("sam2_base_plus", "sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
        ("sam2_large", "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    ]:
        jobs.append(
            {
                "id": job_id,
                "tool": "sam2_w_prompt",
                "model": ckpt,
                "cmd": [
                    python_exe,
                    "-m",
                    "robot_motion_interface.utils.sam2_w_prompt",
                    "--sam3_ckpt",
                    str(repo_root / "models" / "sam3" / "sam3.pt"),
                    "--sam2_ckpt",
                    str(repo_root / "models" / "sam2" / ckpt),
                    "--sam2_cfg",
                    cfg,
                    "--frames_dir",
                    str(color_dir),
                    "--out_dir",
                    str(out_root / job_id),
                    "--video_fps",
                    fps_s,
                    "--compile",
                    "--device",
                    "cuda",
                ],
            }
        )

    return jobs


def find_video(out_dir: Path, parsed_video: str, status: str) -> str:
    if status != "ok":
        return ""
    if parsed_video and Path(parsed_video).exists():
        return parsed_video
    direct = out_dir.parent / f"{out_dir.name}.mp4"
    if direct.exists():
        return str(direct)
    return ""


def write_reports(summary: Dict[str, Any], report_dir: Path) -> None:
    summary_json = report_dir / "benchmark_fps.json"
    summary_csv = report_dir / "benchmark_fps.csv"
    summary_md = report_dir / "benchmark_fps.md"
    results = summary["results"]

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = [
        "id",
        "tool",
        "model",
        "status",
        "returncode",
        "mean_ms",
        "mean_hz",
        "min_ms",
        "max_ms",
        "std_ms",
        "runtime_sec",
        "video_path",
        "log_path",
        "output_dir",
        "command",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    sorted_rows = sorted(
        results,
        key=lambda r: (r["mean_hz"] is None, -(r["mean_hz"] if r["mean_hz"] is not None else -1.0)),
    )

    preflight = summary["preflight"]
    lines: List[str] = []
    lines.append("# Vision Benchmark Report")
    lines.append("")
    lines.append(f"- Dataset: `{summary['dataset_root']}`")
    lines.append(f"- Run root: `{summary['run_root']}`")
    lines.append(f"- Timestamp: `{summary['timestamp']}`")
    lines.append("- Container: `handrl-policy`")
    lines.append(f"- Python: `{summary['python_exe']}`")
    lines.append(f"- Frames: color={preflight['color_frames']}, depth={preflight['depth_frames']}")
    lines.append("")
    lines.append("## Benchmark Matrix")
    for r in results:
        lines.append(f"- `{r['id']}`: tool={r['tool']}, model={r['model']}, status={r['status']}")
    lines.append("")
    lines.append("## FPS/Hz Table (sorted by mean_hz)")
    lines.append("| id | tool | model | status | mean_ms | mean_hz | min_ms | max_ms | std_ms | video | log |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in sorted_rows:
        fmt = lambda v: "" if v is None else f"{v:.2f}"
        lines.append(
            f"| {r['id']} | {r['tool']} | {r['model']} | {r['status']} | {fmt(r['mean_ms'])} | {fmt(r['mean_hz'])} | "
            f"{fmt(r['min_ms'])} | {fmt(r['max_ms'])} | {fmt(r['std_ms'])} | `{r['video_path']}` | `{r['log_path']}` |"
        )
    lines.append("")
    lines.append("## Per-Tool Highlights")
    for tool in ["da3", "promptda", "sam3", "sam2_w_prompt"]:
        valid = [r for r in results if r["tool"] == tool and r["status"] == "ok" and r["mean_hz"] is not None]
        if not valid:
            lines.append(f"- {tool}: no successful runs with parsed Hz.")
            continue
        best = max(valid, key=lambda x: x["mean_hz"])
        worst = min(valid, key=lambda x: x["mean_hz"])
        lines.append(f"- {tool}: best `{best['id']}` = {best['mean_hz']:.2f} Hz, worst `{worst['id']}` = {worst['mean_hz']:.2f} Hz")
    lines.append("")
    lines.append("## Failures")
    fails = [r for r in results if r["status"] != "ok"]
    if not fails:
        lines.append("- None")
    else:
        for r in fails:
            lines.append(f"- `{r['id']}` rc={r['returncode']} log=`{r['log_path']}`")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DA3/PromptDA/SAM3/SAM2_w_prompt benchmark matrix.")
    parser.add_argument("--repo_root", default="/workspace")
    parser.add_argument(
        "--dataset_root",
        default="/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807",
    )
    parser.add_argument("--python_exe", default="/root/miniconda3/envs/policy/bin/python")
    parser.add_argument("--video_fps", type=float, default=60.0)
    parser.add_argument("--run_id", default="", help="Optional suffix in bench_vision_<run_id>. Defaults to timestamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    data_root = Path(args.dataset_root)
    color_dir = data_root / "color"
    depth_dir = data_root / "depth"
    metadata_path = data_root / "metadata.json"

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = data_root / f"bench_vision_{run_id}"
    log_dir = run_root / "logs"
    out_dir = run_root / "outputs"
    rep_dir = run_root / "reports"
    for p in [run_root, log_dir, out_dir, rep_dir]:
        p.mkdir(parents=True, exist_ok=True)

    ensure_exists([data_root, color_dir, depth_dir, metadata_path], "Dataset preflight failed, missing:")
    preflight = {
        "data_root_exists": data_root.exists(),
        "color_exists": color_dir.exists(),
        "depth_exists": depth_dir.exists(),
        "metadata_exists": metadata_path.exists(),
        "color_frames": len(list(color_dir.glob("*"))),
        "depth_frames": len(list(depth_dir.glob("*"))),
    }

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    cintr = meta["color_intrinsics"]
    fx = float(cintr["fx"])
    fy = float(cintr["fy"])
    cx = float(cintr["cx"])
    cy = float(cintr["cy"])
    focal = 0.5 * (fx + fy)

    ffmpeg_ok = run_capture(["ffmpeg", "-version"], cwd=repo_root).returncode == 0
    if not ffmpeg_ok:
        raise RuntimeError("ffmpeg is not available.")
    preflight["ffmpeg_ok"] = ffmpeg_ok

    base_env = os.environ.copy()
    existing = base_env.get("PYTHONPATH", "")
    promptda_path = "/workspace/dep/PromptDA"
    base_env["PYTHONPATH"] = f"{promptda_path}:{existing}" if existing else promptda_path

    help_checks: Dict[str, Dict[str, Any]] = {}
    help_cmds = {
        "da3": [args.python_exe, "-m", "robot_motion_interface.utils.da3_utils", "--help"],
        "promptda": [args.python_exe, "-m", "robot_motion_interface.utils.promptda_utils", "--help"],
        "sam3": [args.python_exe, "-m", "robot_motion_interface.utils.sam3_utils", "--help"],
        "sam2_w_prompt": [args.python_exe, "-m", "robot_motion_interface.utils.sam2_w_prompt", "--help"],
    }
    for name, cmd in help_cmds.items():
        r = run_capture(cmd, cwd=repo_root, env=base_env)
        help_checks[name] = {"ok": r.returncode == 0, "returncode": r.returncode}
        if r.returncode != 0:
            (rep_dir / f"preflight_{name}_help.log").write_text(r.stdout, encoding="utf-8")
    if not all(v["ok"] for v in help_checks.values()):
        raise RuntimeError(f"Module --help preflight failed: {help_checks}")

    ensure_exists(
        [
            repo_root / "models" / "promptda" / "PromptDA-s-transparent.ckpt",
            repo_root / "models" / "promptda" / "PromptDA-s.ckpt",
            repo_root / "models" / "promptda" / "PromptDA-l.ckpt",
            repo_root / "models" / "sam3" / "sam3.pt",
            repo_root / "models" / "sam3" / "efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
            repo_root / "models" / "sam3" / "efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt",
            repo_root / "models" / "sam3" / "efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt",
            repo_root / "models" / "sam3" / "efficient_sam3_efficientvit_s.pt",
            repo_root / "models" / "sam2" / "sam2.1_hiera_tiny.pt",
            repo_root / "models" / "sam2" / "sam2.1_hiera_small.pt",
            repo_root / "models" / "sam2" / "sam2.1_hiera_base_plus.pt",
            repo_root / "models" / "sam2" / "sam2.1_hiera_large.pt",
        ],
        "Checkpoint preflight failed, missing:",
    )

    jobs = build_jobs(
        python_exe=args.python_exe,
        repo_root=repo_root,
        data_root=data_root,
        color_dir=color_dir,
        out_root=out_dir,
        video_fps=args.video_fps,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        focal=focal,
    )

    print(f"[bench] run_root={run_root}")
    print(f"[bench] total_jobs={len(jobs)}")

    results: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs, start=1):
        job_id = job["id"]
        log_path = log_dir / f"{job_id}.log"
        job_out_dir = out_dir / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[bench] ({idx}/{len(jobs)}) start {job_id}")
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as f:
            f.write("COMMAND:\n")
            f.write(" ".join(job["cmd"]) + "\n\n")
            proc = subprocess.run(
                job["cmd"],
                cwd=str(repo_root),
                env=base_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        dt = time.time() - t0

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = parse_metrics(log_text)
        status = "ok" if proc.returncode == 0 else "failed"
        video_path = find_video(job_out_dir, m["video_path_from_log"], status)

        rec = {
            "id": job_id,
            "tool": job["tool"],
            "model": job["model"],
            "status": status,
            "returncode": proc.returncode,
            "mean_ms": m["mean_ms"],
            "mean_hz": m["mean_hz"],
            "min_ms": m["min_ms"],
            "max_ms": m["max_ms"],
            "std_ms": m["std_ms"],
            "runtime_sec": round(dt, 3),
            "command": " ".join(job["cmd"]),
            "log_path": str(log_path),
            "output_dir": str(job_out_dir),
            "video_path": video_path,
        }
        results.append(rec)
        print(f"[bench] ({idx}/{len(jobs)}) done {job_id} status={status} mean_hz={rec['mean_hz']}")

    summary = {
        "timestamp": run_id,
        "run_root": str(run_root),
        "dataset_root": str(data_root),
        "python_exe": args.python_exe,
        "preflight": preflight,
        "help_checks": help_checks,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "focal": focal},
        "total_jobs": len(results),
        "success_jobs": sum(1 for r in results if r["status"] == "ok"),
        "failed_jobs": sum(1 for r in results if r["status"] != "ok"),
        "results": results,
    }
    write_reports(summary, rep_dir)

    print(f"[bench] summary_json={rep_dir / 'benchmark_fps.json'}")
    print(f"[bench] summary_csv={rep_dir / 'benchmark_fps.csv'}")
    print(f"[bench] summary_md={rep_dir / 'benchmark_fps.md'}")
    print("[bench] done")


if __name__ == "__main__":
    main()
