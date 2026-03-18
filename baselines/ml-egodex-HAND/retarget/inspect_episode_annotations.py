"""
Inspect EgoDex HDF5 episode annotations and schema.

Examples:
    # Scan a full task folder
    python baselines/ml-egodex-HAND/retarget/inspect_episode_annotations.py \
        --data_dir models/egodex/test/screw_unscrew_bottle_cap

    # Inspect one episode file only
    python baselines/ml-egodex-HAND/retarget/inspect_episode_annotations.py \
        --episode models/egodex/test/add_remove_lid/0.hdf5 --verbose

    # Save machine-readable output
    python baselines/ml-egodex-HAND/retarget/inspect_episode_annotations.py \
        --data_dir models/egodex/test/screw_unscrew_fingers_fixture \
        --json_out /tmp/fingers_fixture_annotations.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


LEFT_FINGER_TIPS = ["leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerTip"]
RIGHT_FINGER_TIPS = ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip"]

ANNOTATION_KEYS = [
    "annotated",
    "annotator_version",
    "task",
    "llm_type",
    "which_llm_description",
    "llm_description",
    "llm_description2",
    "llm_verbs",
    "llm_objects",
    "object",
    "environment",
    "session_name",
    "extra",
]


def _normalize_attr(v: Any) -> Any:
    """Convert HDF5 attr values to JSON/print-friendly Python types."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_normalize_attr(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_normalize_attr(x) for x in v]
    return v


def _active_instruction(attrs: dict[str, Any]) -> str | None:
    llm_type = str(attrs.get("llm_type", ""))
    if llm_type != "reversible":
        desc = attrs.get("llm_description")
        return str(desc) if desc is not None else None
    which = str(attrs.get("which_llm_description", "1"))
    key = "llm_description" if which == "1" else "llm_description2"
    desc = attrs.get(key)
    return str(desc) if desc is not None else None


def _mean_tip_pos_in_camera(
    transforms: h5py.Group,
    cam_ext: np.ndarray,
    tip_names: list[str],
) -> list[float] | None:
    if not all(name in transforms for name in tip_names):
        return None
    positions = []
    n_frames = int(cam_ext.shape[0])
    for i in range(n_frames):
        inv_cam = np.linalg.inv(cam_ext[i])
        for tip in tip_names:
            tf_cam = inv_cam @ transforms[tip][i]
            positions.append(tf_cam[:3, 3])
    if not positions:
        return None
    mean_pos = np.mean(np.asarray(positions, dtype=np.float64), axis=0)
    return [float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2])]


def inspect_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        top_keys = sorted(list(f.keys()))

        transforms = f.get("transforms")
        transform_keys = sorted(list(transforms.keys())) if transforms is not None else []
        num_frames = (
            int(transforms["camera"].shape[0])
            if transforms is not None and "camera" in transforms
            else None
        )

        confidences = f.get("confidences")
        confidence_keys = sorted(list(confidences.keys())) if confidences is not None else []

        expected_conf_keys = sorted(k for k in transform_keys if k != "camera")
        missing_confidence_keys = sorted(set(expected_conf_keys) - set(confidence_keys))
        orphan_confidence_keys = sorted(set(confidence_keys) - set(expected_conf_keys))

        inferred_centers_cam: dict[str, list[float] | None] = {}
        if transforms is not None and "camera" in transforms:
            cam_ext = transforms["camera"][:]
            inferred_centers_cam = {
                "all_6_tips": _mean_tip_pos_in_camera(
                    transforms, cam_ext, LEFT_FINGER_TIPS + RIGHT_FINGER_TIPS
                ),
                "left_3_tips": _mean_tip_pos_in_camera(
                    transforms, cam_ext, LEFT_FINGER_TIPS
                ),
                "right_3_tips": _mean_tip_pos_in_camera(
                    transforms, cam_ext, RIGHT_FINGER_TIPS
                ),
            }

        attrs = {k: _normalize_attr(v) for k, v in f.attrs.items()}
        chosen_attrs = {k: attrs.get(k) for k in ANNOTATION_KEYS if k in attrs}
        chosen_attrs["active_instruction"] = _active_instruction(attrs)

        return {
            "episode": path.name,
            "path": str(path),
            "frames": num_frames,
            "top_level_keys": top_keys,
            "transform_key_count": len(transform_keys),
            "confidence_key_count": len(confidence_keys),
            "has_confidences": confidences is not None,
            "missing_confidence_keys": missing_confidence_keys,
            "orphan_confidence_keys": orphan_confidence_keys,
            "inferred_object_centers_cam": inferred_centers_cam,
            "annotation": chosen_attrs,
            "all_attr_keys": sorted(list(attrs.keys())),
            "transform_keys": transform_keys,
            "confidence_keys": confidence_keys,
        }


def _print_episode_info(info: dict[str, Any], verbose: bool = False) -> None:
    ann = info["annotation"]
    print(f"\nEpisode: {info['episode']}")
    print(f"  frames: {info['frames']}")
    print(f"  top_level_keys: {info['top_level_keys']}")
    print(
        "  transforms/confidences: "
        f"{info['transform_key_count']}/{info['confidence_key_count']} "
        f"(has_confidences={info['has_confidences']})"
    )

    if info["missing_confidence_keys"]:
        print(
            "  missing_confidence_keys: "
            f"{len(info['missing_confidence_keys'])} (e.g. {info['missing_confidence_keys'][:6]})"
        )
    if info["orphan_confidence_keys"]:
        print(
            "  orphan_confidence_keys: "
            f"{len(info['orphan_confidence_keys'])} (e.g. {info['orphan_confidence_keys'][:6]})"
        )

    print(f"  task: {ann.get('task')}")
    print(f"  llm_type: {ann.get('llm_type')}")
    print(f"  llm_verbs: {ann.get('llm_verbs')}")
    print(f"  llm_objects: {ann.get('llm_objects')}")
    print(f"  which_llm_description: {ann.get('which_llm_description')}")
    print(f"  active_instruction: {ann.get('active_instruction')}")
    centers = info.get("inferred_object_centers_cam", {})
    if centers:
        print(
            "  inferred_object_centers_cam: "
            f"all_6={centers.get('all_6_tips')} "
            f"left_3={centers.get('left_3_tips')} "
            f"right_3={centers.get('right_3_tips')}"
        )

    if verbose:
        print("  annotation_attrs:")
        for k, v in ann.items():
            if k == "active_instruction":
                continue
            print(f"    - {k}: {v}")
        print(f"  all_attr_keys: {info['all_attr_keys']}")
        print("  transform_keys:")
        for k in info["transform_keys"]:
            print(f"    - {k}")
        if info["confidence_keys"]:
            print("  confidence_keys:")
            for k in info["confidence_keys"]:
                print(f"    - {k}")
        else:
            print("  confidence_keys: []")


def _print_summary(per_episode: list[dict[str, Any]]) -> None:
    missing_conf_eps = [x["episode"] for x in per_episode if not x["has_confidences"]]
    tasks = sorted({str(x["annotation"].get("task")) for x in per_episode})
    llm_types = sorted({str(x["annotation"].get("llm_type")) for x in per_episode})

    print("\n=== Summary ===")
    print(f"episodes_scanned: {len(per_episode)}")
    print(f"unique_task_values: {tasks}")
    print(f"unique_llm_type_values: {llm_types}")
    print(
        "episodes_missing_confidences: "
        f"{len(missing_conf_eps)}"
        + (f" -> {missing_conf_eps}" if missing_conf_eps else "")
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect EgoDex episode annotations/schema")
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing EgoDex .hdf5 episodes",
    )
    p.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Single episode .hdf5 path to inspect",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of episodes to inspect when using --data_dir",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra fields (all attr keys and annotation attrs)",
    )
    p.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional output JSON path for full structured report",
    )
    return p.parse_args()


def _resolve_episodes(args: argparse.Namespace) -> list[Path]:
    if args.episode:
        path = Path(args.episode)
        if not path.exists():
            raise FileNotFoundError(f"Episode not found: {path}")
        return [path]

    if args.data_dir is None:
        raise ValueError("Provide either --episode or --data_dir")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    episodes = sorted(data_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    if not episodes:
        raise FileNotFoundError(f"No .hdf5 files found in: {data_dir}")

    if args.limit is not None:
        episodes = episodes[: args.limit]
    return episodes


def main() -> None:
    args = parse_args()
    episodes = _resolve_episodes(args)
    per_episode = [inspect_episode(p) for p in episodes]

    for info in per_episode:
        _print_episode_info(info, verbose=args.verbose)
    _print_summary(per_episode)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"episodes": per_episode}, f, indent=2)
        print(f"\nWrote JSON report: {out_path}")


if __name__ == "__main__":
    main()
