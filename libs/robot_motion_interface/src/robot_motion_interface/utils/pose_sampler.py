"""
pose_sampler.py — Bimanual pre-grasp EE pose sampler.

Sampling geometry
-----------------
Left arm  (left_delto_base_link):
  Position  – hollow cylinder  r ∈ [0.15, 0.30] m,  z ∈ [0.05, 0.07] m
  Rotation  – z-axis points radially inward toward the world vertical axis (x=0, y=0),
               x-axis is horizontal (tangential),  y-axis is downward (≈ vertical)

Right arm (right_delto_base_link):
  Position  – solid cylinder   r ∈ [0.00, 0.10] m,  z ∈ [0.15, 0.30] m
  Rotation  – z-axis points from current position toward the world origin

Both arms get a small random rotation noise (≤ 15 °) applied on top of the
analytical frame, then near-duplicate poses are filtered out.  The caller can
build all bimanual combinations with itertools.product(left_poses, right_poses).
"""

import itertools

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_rotation_noise(rot_matrix: np.ndarray, max_angle_deg: float = 15.0) -> Rotation:
    """Apply a random small rotation to *rot_matrix* and return a Rotation."""
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0.0, np.deg2rad(max_angle_deg))
    noise = Rotation.from_rotvec(axis * angle)
    return noise * Rotation.from_matrix(rot_matrix)


def _rot_to_wxyz(r: Rotation) -> np.ndarray:
    """scipy xyzw → wxyz."""
    q = r.as_quat()          # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _wxyz_to_scipy(q_wxyz: np.ndarray) -> Rotation:
    """wxyz → scipy Rotation."""
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


# ---------------------------------------------------------------------------
# Pose filtering
# ---------------------------------------------------------------------------

def filter_similar_poses(
    poses: list[tuple[np.ndarray, np.ndarray]],
    pos_threshold: float = 0.03,
    angle_threshold_deg: float = 15.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Remove near-duplicate poses from *poses*.

    A pose is considered duplicate of an already-kept pose when BOTH
    its position distance is < *pos_threshold* (m) AND its rotation
    distance is < *angle_threshold_deg* (°).

    The position check is vectorised (fast); the quaternion check is
    applied only to the small set of position-close candidates.

    Parameters
    ----------
    poses:
        List of (pos [3,], quat_wxyz [4,]) tuples.
    pos_threshold:
        Euclidean position threshold in metres.
    angle_threshold_deg:
        Rotation angle threshold in degrees.

    Returns
    -------
    Filtered list (order-preserving, greedy).
    """
    if not poses:
        return []

    angle_threshold_rad = np.deg2rad(angle_threshold_deg)
    kept_pos  = np.empty((len(poses), 3), dtype=np.float64)
    kept_rots: list[Rotation] = []
    kept_raw : list[tuple[np.ndarray, np.ndarray]] = []

    n_kept = 0
    for pos, q_wxyz in poses:
        pos = np.asarray(pos, dtype=np.float64)

        # --- fast vector position distance to all kept poses ---
        if n_kept > 0:
            dists = np.linalg.norm(kept_pos[:n_kept] - pos, axis=1)  # [n_kept]
            close_mask = dists < pos_threshold
            if close_mask.any():
                # check rotation only for position-close candidates
                new_rot = _wxyz_to_scipy(q_wxyz)
                too_similar = False
                for idx in np.where(close_mask)[0]:
                    rel = new_rot.inv() * kept_rots[idx]
                    angle_diff = np.linalg.norm(rel.as_rotvec())  # radians
                    if angle_diff < angle_threshold_rad:
                        too_similar = True
                        break
                if too_similar:
                    continue

        kept_pos[n_kept] = pos
        kept_rots.append(_wxyz_to_scipy(q_wxyz))
        kept_raw.append((pos.astype(np.float32), q_wxyz))
        n_kept += 1

    return kept_raw


# ---------------------------------------------------------------------------
# Per-arm samplers
# ---------------------------------------------------------------------------

def generate_left_arm_candidates(
    num_samples: int = 500,
    r_range: tuple[float, float] = (0.15, 0.30),
    z_range: tuple[float, float] = (0.05, 0.07),
    noise_deg: float = 15.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample left-arm (left_delto_base_link) EE poses.

    Position : hollow cylinder, r ∈ r_range, z ∈ z_range.
    Frame    : z-axis radially inward (horizontal), x-axis tangential
               (horizontal), y-axis = [0, 0, +1] (world up / vertical).
    """
    poses = []
    for _ in range(num_samples):
        # --- position ---
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        r     = np.random.uniform(*r_range)
        z     = np.random.uniform(*z_range)
        pos   = np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=np.float32)

        # --- frame construction ---
        # z-axis: radially inward toward the vertical axis, lies in xy-plane
        z_ax = np.array([-np.cos(theta), -np.sin(theta), 0.0])
        # x-axis: tangential (horizontal).  Randomly flip sign so that y = cross(z,x)
        # is equally likely to be [0,0,+1] (up) or [0,0,-1] (down).
        sign  = np.random.choice([-1.0, 1.0])
        x_ax  = sign * np.array([ np.sin(theta), -np.cos(theta), 0.0])
        # y-axis: [0, 0, ±1]
        y_ax = np.cross(z_ax, x_ax)

        rot_mat = np.column_stack((x_ax, y_ax, z_ax))  # columns = frame axes

        rot   = _add_rotation_noise(rot_mat, noise_deg)
        q     = _rot_to_wxyz(rot)
        poses.append((pos, q))

    return poses


def generate_right_arm_candidates(
    num_samples: int = 500,
    r_max: float = 0.10,
    z_range: tuple[float, float] = (0.15, 0.30),
    noise_deg: float = 15.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample right-arm (right_delto_base_link) EE poses.

    Position : solid cylinder, r ∈ [0, r_max] (area-uniform), z ∈ z_range.
    Frame    : z-axis points from the sampled position toward world origin.
    """
    poses = []
    for _ in range(num_samples):
        # --- position (area-uniform disk sampling) ---
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        r     = np.sqrt(np.random.uniform(0.0, 1.0)) * r_max
        z     = np.random.uniform(*z_range)
        pos   = np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=np.float32)

        # --- frame construction ---
        # z-axis: from position toward origin
        z_ax = -pos.astype(np.float64)
        z_ax /= np.linalg.norm(z_ax)

        # x-axis: horizontal, perpendicular to z_ax
        # When r ≈ 0 (directly above origin), fall back to world-x
        x_candidate = np.array([-pos[1], pos[0], 0.0], dtype=np.float64)
        if np.linalg.norm(x_candidate) < 1e-5:
            x_candidate = np.array([1.0, 0.0, 0.0])
        x_ax = x_candidate / np.linalg.norm(x_candidate)

        # y-axis
        y_ax = np.cross(z_ax, x_ax)

        rot_mat = np.column_stack((x_ax, y_ax, z_ax))

        rot = _add_rotation_noise(rot_mat, noise_deg)
        q   = _rot_to_wxyz(rot)
        poses.append((pos, q))

    return poses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_bimanual_pregrasp(
    n_raw: int = 500,
    pos_threshold: float = 0.03,
    angle_threshold_deg: float = 15.0,
    noise_deg: float = 15.0,
) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    """Generate a Cartesian-product set of bimanual pre-grasp EE poses.

    Steps
    -----
    1. Sample *n_raw* left-arm poses and *n_raw* right-arm poses.
    2. Independently filter near-duplicates from each arm.
    3. Return itertools.product(left, right) — every (left, right) pair.

    Returns
    -------
    List of  ((l_pos, l_quat_wxyz), (r_pos, r_quat_wxyz)).
    """
    raw_left  = generate_left_arm_candidates(n_raw, noise_deg=noise_deg)
    raw_right = generate_right_arm_candidates(n_raw, noise_deg=noise_deg)

    left  = filter_similar_poses(raw_left,  pos_threshold, angle_threshold_deg)
    right = filter_similar_poses(raw_right, pos_threshold, angle_threshold_deg)

    return list(itertools.product(left, right))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_pregrasp_poses(
    left_poses: list[tuple[np.ndarray, np.ndarray]],
    right_poses: list[tuple[np.ndarray, np.ndarray]],
    n_sample: int = 50,
    n_plots: int = 5,
    arrow_len: float = 0.045,
    save_path: str | None = None,
) -> None:
    """Plot sampled EE poses with x/y/z frame axes in five 3D subplots.

    Each subplot shows n_sample // n_plots points (half left, half right).
    Three arrows per point show the local x (light), y (mid), z (dark) axes.

    Parameters
    ----------
    left_poses / right_poses:
        Output of generate_left/right_arm_candidates or filter_similar_poses.
    n_sample:
        Total points to sample (split evenly between left and right).
    n_plots:
        Number of 3D subplots (arranged in a 2×3 grid, last cell empty).
    arrow_len:
        Length of each axis arrow in metres.
    save_path:
        If given, save the figure to this path (in addition to plt.show()).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Colour palettes: (x, y, z) per arm — light → dark within hue family
    L_COLORS = ("lightblue", "cornflowerblue", "steelblue")       # blues
    R_COLORS = ("moccasin",  "goldenrod",       "darkorange")      # oranges
    DOT_L, DOT_R = "steelblue", "darkorange"

    n_each    = n_sample // 2
    n_per_plt = n_sample // n_plots
    n_l_pp    = n_per_plt // 2
    n_r_pp    = n_per_plt - n_l_pp

    rng = np.random.default_rng()
    l_idx = rng.choice(len(left_poses),  size=min(n_each, len(left_poses)),  replace=False)
    r_idx = rng.choice(len(right_poses), size=min(n_each, len(right_poses)), replace=False)
    l_sample = [left_poses[i]  for i in l_idx]
    r_sample = [right_poses[i] for i in r_idx]

    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        subplot_kw={"projection": "3d"},
        figsize=(5 * ncols, 5 * nrows),
    )
    axes_flat = axes.flat

    _tgrid = np.linspace(-0.35, 0.35, 8)
    _tx, _ty = np.meshgrid(_tgrid, _tgrid)
    _tz = np.zeros_like(_tx)

    # x and y axes are shorter than z to reduce visual clutter
    _axis_scale = (0.5, 0.5, 1.0)   # (x_scale, y_scale, z_scale)

    def _draw_frame(ax, pos, q, colors):
        """Draw x/y/z arrows for one pose."""
        R = _wxyz_to_scipy(q).as_matrix()   # columns: x, y, z axes in world
        for col, (color, scale) in enumerate(zip(colors, _axis_scale)):
            ax_vec = R[:, col] * (arrow_len * scale)
            ax.quiver(
                pos[0], pos[1], pos[2],
                ax_vec[0], ax_vec[1], ax_vec[2],
                color=color, linewidth=1.2, arrow_length_ratio=0.3,
            )

    for i in range(n_plots):
        ax = axes_flat[i]

        l_sub = l_sample[i * n_l_pp : (i + 1) * n_l_pp]
        r_sub = r_sample[i * n_r_pp : (i + 1) * n_r_pp]

        for pos, q in l_sub:
            ax.scatter(*pos, color=DOT_L, s=35, depthshade=True, zorder=5)
            _draw_frame(ax, pos, q, L_COLORS)

        for pos, q in r_sub:
            ax.scatter(*pos, color=DOT_R, s=35, depthshade=True, zorder=5)
            _draw_frame(ax, pos, q, R_COLORS)

        # reference: table plane + vertical axis
        ax.plot_surface(_tx, _ty, _tz, alpha=0.08, color="gray", zorder=1)
        ax.plot([0, 0], [0, 0], [0.0, 0.32], color="black",
                linestyle="--", linewidth=0.8, alpha=0.5)
        ax.scatter(0, 0, 0, color="black", s=20, marker="+")

        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(-0.02, 0.35)
        ax.set_xlabel("X", labelpad=2)
        ax.set_ylabel("Y", labelpad=2)
        ax.set_zlabel("Z", labelpad=2)
        ax.set_title(f"Sample group {i + 1}", fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n_plots, nrows * ncols):
        axes_flat[j].set_visible(False)

    handles = [
        # left arm
        Line2D([0], [0], color=L_COLORS[0], lw=2, label="left  x-axis"),
        Line2D([0], [0], color=L_COLORS[1], lw=2, label="left  y-axis"),
        Line2D([0], [0], color=L_COLORS[2], lw=2, label="left  z-axis"),
        # right arm
        Line2D([0], [0], color=R_COLORS[0], lw=2, label="right x-axis"),
        Line2D([0], [0], color=R_COLORS[1], lw=2, label="right y-axis"),
        Line2D([0], [0], color=R_COLORS[2], lw=2, label="right z-axis"),
        # reference
        Line2D([0], [0], color="black", lw=1, linestyle="--", label="vertical axis"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)
    fig.suptitle(
        f"Pre-grasp EE poses  —  light=x  mid=y  dark=z  |  "
        f"left={len(l_sample)} pts  right={len(r_sample)} pts",
        fontsize=10,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    print("Generating left-arm candidates ...")
    raw_l = generate_left_arm_candidates(1000)
    filt_l = filter_similar_poses(raw_l, pos_threshold=0.03, angle_threshold_deg=15.0)
    print(f"  Left : 1000 raw → {len(filt_l)} after dedup")

    print("Generating right-arm candidates ...")
    raw_r = generate_right_arm_candidates(1000)
    filt_r = filter_similar_poses(raw_r, pos_threshold=0.03, angle_threshold_deg=15.0)
    print(f"  Right: 1000 raw → {len(filt_r)} after dedup")

    combos = list(itertools.product(filt_l, filt_r))
    print(f"  Bimanual combinations: {len(combos)}")

    # Spot-check a few left poses
    print("\nSample left poses (pos | z-axis in world):")
    for pos, q in filt_l[:3]:
        rot = _wxyz_to_scipy(q)
        z_world = rot.as_matrix()[:, 2]   # third column = local z in world
        print(f"  pos={np.round(pos, 3)}  z_world={np.round(z_world, 3)}")

    print("\nSample right poses (pos | z-axis in world):")
    for pos, q in filt_r[:3]:
        rot = _wxyz_to_scipy(q)
        z_world = rot.as_matrix()[:, 2]
        print(f"  pos={np.round(pos, 3)}  z_world={np.round(z_world, 3)}")

    print("\nRendering pose visualisation ...")
    visualize_pregrasp_poses(filt_l, filt_r, n_sample=60, n_plots=6,
                             save_path="models/pregrasp_poses.png")
