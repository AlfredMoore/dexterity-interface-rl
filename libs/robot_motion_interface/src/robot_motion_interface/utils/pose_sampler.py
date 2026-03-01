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
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_rotation_noise(rot_matrix: np.ndarray, max_angle_deg: float = 15.0) -> Rotation:
    """Apply a small random rotation perturbation to a rotation matrix.

    Samples a uniformly random unit axis and a uniform angle in
    ``[0, max_angle_deg]``, builds the corresponding axis-angle rotation, and
    pre-multiplies it onto *rot_matrix*.

    Args:
        rot_matrix: 3×3 orthonormal rotation matrix representing the base
            orientation to perturb.
        max_angle_deg: Maximum perturbation angle in degrees.  The actual angle
            is sampled uniformly from ``[0, max_angle_deg]``.

    Returns:
        A scipy ``Rotation`` object representing the perturbed orientation.
    """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0.0, np.deg2rad(max_angle_deg))
    noise = Rotation.from_rotvec(axis * angle)
    return noise * Rotation.from_matrix(rot_matrix)


def _rot_to_wxyz(r: Rotation) -> np.ndarray:
    """Convert a scipy ``Rotation`` to a wxyz quaternion array.

    Args:
        r: Rotation in scipy's internal xyzw convention.

    Returns:
        Float32 NumPy array of shape ``(4,)`` in ``[w, x, y, z]`` order.
    """
    q = r.as_quat()          # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _wxyz_to_scipy(q_wxyz: np.ndarray) -> Rotation:
    """Convert a wxyz quaternion to a scipy ``Rotation``.

    Args:
        q_wxyz: 1-D array of shape ``(4,)`` in ``[w, x, y, z]`` order.

    Returns:
        Equivalent scipy ``Rotation`` object.
    """
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


# ---------------------------------------------------------------------------
# Pose filtering
# ---------------------------------------------------------------------------

def filter_similar_poses(
    poses: list[tuple[np.ndarray, np.ndarray]],
    pos_threshold: float = 0.03,
    angle_threshold_deg: float = 15.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Remove near-duplicate poses from a list (greedy, order-preserving).

    A pose is considered a duplicate of an already-kept pose when *both*
    its position distance is below *pos_threshold* **and** its angular
    distance is below *angle_threshold_deg*.  The position check is
    vectorised; the quaternion check is performed only for the small subset
    of position-close candidates.

    Args:
        poses: List of ``(pos, quat_wxyz)`` tuples where ``pos`` has shape
            ``(3,)`` and ``quat_wxyz`` has shape ``(4,)`` in ``[w, x, y, z]``
            order.
        pos_threshold: Euclidean distance threshold in metres.  A candidate
            pose only undergoes the additional rotation check when its distance
            to some already-kept pose is below this value.
        angle_threshold_deg: Rotation-distance threshold in degrees.  A pose
            is discarded when its rotation differs from a position-close
            already-kept pose by less than this angle.

    Returns:
        Filtered list of ``(pos, quat_wxyz)`` tuples in original order.
        All position arrays are ``float32``.
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
    """Sample left-arm end-effector poses in a hollow-cylinder workspace.

    Each pose is constructed analytically so that the local z-axis points
    radially inward toward the world vertical axis (lying in the xy-plane)
    and the local x-axis lies tangentially in the horizontal plane.  A small
    random rotation noise is then added on top.

    Args:
        num_samples: Number of candidate poses to generate before any
            filtering.
        r_range: ``(r_min, r_max)`` radial distance from the world vertical
            axis in metres.  Samples are drawn uniformly over this interval.
        z_range: ``(z_min, z_max)`` height range in metres.  Samples are drawn
            uniformly over this interval.
        noise_deg: Maximum perturbation angle in degrees applied as a random
            rotation on top of the analytical EE frame.

    Returns:
        List of *num_samples* ``(pos, quat_wxyz)`` tuples.  ``pos`` is a
        float32 array of shape ``(3,)``; ``quat_wxyz`` is a float32 array of
        shape ``(4,)`` in ``[w, x, y, z]`` order.
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
    """Sample right-arm end-effector poses in a solid-cylinder workspace.

    Each pose is constructed analytically so that the local z-axis points
    from the sampled position toward the world origin (hand faces inward /
    downward).  A small random rotation noise is then added on top.

    Args:
        num_samples: Number of candidate poses to generate before any
            filtering.
        r_max: Maximum radial distance from the world vertical axis in metres.
            Samples are drawn with area-uniform distribution over
            ``[0, r_max]``, i.e. ``r = sqrt(U) * r_max`` with
            ``U ~ Uniform(0, 1)``.
        z_range: ``(z_min, z_max)`` height range in metres.  Samples are drawn
            uniformly over this interval.
        noise_deg: Maximum perturbation angle in degrees applied as a random
            rotation on top of the analytical EE frame.

    Returns:
        List of *num_samples* ``(pos, quat_wxyz)`` tuples.  ``pos`` is a
        float32 array of shape ``(3,)``; ``quat_wxyz`` is a float32 array of
        shape ``(4,)`` in ``[w, x, y, z]`` order.
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
    """Generate all pairwise combinations of filtered bimanual pre-grasp poses.

    Sampling pipeline:

    1. Draw *n_raw* raw candidates for each arm independently.
    2. Remove near-duplicate poses from each arm's candidate set via
       :func:`filter_similar_poses`.
    3. Return the full Cartesian product of the two filtered sets.

    The total number of returned pairs is ``len(left) × len(right)`` after
    deduplication, which is typically much smaller than ``n_raw²``.

    Args:
        n_raw: Number of raw candidates to draw per arm before filtering.
        pos_threshold: Euclidean position threshold in metres passed to
            :func:`filter_similar_poses`.
        angle_threshold_deg: Rotation-distance threshold in degrees passed to
            :func:`filter_similar_poses`.
        noise_deg: Maximum random rotation noise in degrees added to each
            analytically constructed EE frame.

    Returns:
        List of ``((l_pos, l_quat_wxyz), (r_pos, r_quat_wxyz))`` tuples, one
        per bimanual pose combination.  Each ``pos`` is a float32 array of
        shape ``(3,)`` and each ``quat_wxyz`` is a float32 array of shape
        ``(4,)`` in ``[w, x, y, z]`` order.
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
    """Visualise sampled EE poses as 3-axis orientation arrows in 3-D subplots.

    Draws *n_plots* Matplotlib 3-D subplots arranged in rows of three.  Each
    subplot shows a random subset of the provided poses with three quiver
    arrows per point representing the local x (light colour), y (mid colour),
    and z (dark colour) axes.  Left-arm poses use a blue palette; right-arm
    poses use an orange palette.

    Args:
        left_poses: Left-arm candidate poses as ``(pos, quat_wxyz)`` tuples,
            typically the output of :func:`filter_similar_poses` applied to
            :func:`generate_left_arm_candidates`.
        right_poses: Right-arm candidate poses in the same format.
        n_sample: Total number of poses to display across all subplots,
            split evenly between the two arms and across *n_plots* panels.
        n_plots: Number of 3-D subplots to create.  Panels are arranged in
            rows of three; unused cells in the last row are hidden.
        arrow_len: Base length of each axis arrow in metres.  The x- and
            y-axis arrows are additionally scaled to 0.5× to reduce visual
            clutter.
        save_path: If provided, the figure is saved to this file path (PNG,
            PDF, etc.) in addition to being shown interactively.

    Returns:
        None.  Calls ``plt.show()`` and optionally writes *save_path*.
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

    # IK filtering example (requires a running CuRoboBimanualMotionPlanner):
    #
    #   from robot_motion_interface.utils.kinematics import CuRoboBimanualMotionPlanner
    #   planner = CuRoboBimanualMotionPlanner(robot_cfg_path="...", collision_activation_distance=0.025)
    #   ik_results = filter_by_ik(filt_l, filt_r, planner, max_pairs=500, verbose=True)
    #   print(f"\nFeasible bimanual pre-grasp poses: {len(ik_results)}")
    #   for (l_pos, l_quat), (r_pos, r_quat), q_sol in ik_results[:3]:
    #       print(f"  left={np.round(l_pos, 3)}  right={np.round(r_pos, 3)}")
    #       print(f"  q_sol={np.round(q_sol, 3)}")

    # IKResult: ((l_pos, l_quat_wxyz), (r_pos, r_quat_wxyz), q_solution [n_joints,])
    IKResult = tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray]

    def filter_by_ik(
        left_poses: list[tuple[np.ndarray, np.ndarray]],
        right_poses: list[tuple[np.ndarray, np.ndarray]],
        planner,  # CuRoboBimanualMotionPlanner — avoids circular import
        max_pairs: int | None = None,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> list[IKResult]:
        """Filter bimanual pose pairs by IK feasibility.

        Iterates over every (left, right) Cartesian-product combination (up to
        *max_pairs*) and keeps only those for which the bimanual IK solver
        finds a valid, self-collision-free joint configuration.

        Args:
            left_poses: Filtered left-arm candidate poses as ``(pos, quat_wxyz)``
                tuples, typically from :func:`filter_similar_poses`.
            right_poses: Filtered right-arm candidate poses in the same format.
            planner: A ready (warmed-up) ``CuRoboBimanualMotionPlanner``
                instance whose ``solve_ik`` method will be called.
            max_pairs: Maximum number of combinations to test.  ``None`` tests
                the full Cartesian product (``len(left) × len(right)`` pairs).
            shuffle: If ``True``, randomly shuffle the Cartesian product before
                testing so that feasible results are spread across the whole
                workspace rather than clustered around the first few left poses.
            verbose: If ``True``, print a progress line every 100 pairs and a
                summary line at the end.

        Returns:
            List of ``IKResult`` triples
            ``((l_pos, l_quat_wxyz), (r_pos, r_quat_wxyz), q_sol)`` where
            ``q_sol`` is a float32 NumPy array of shape ``(n_joints,)``
            containing the full joint solution (arm + gripper joints for both
            arms).
        """
        pairs: list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = list(
            itertools.product(left_poses, right_poses)
        )
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(pairs)  # type: ignore[arg-type]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        n_total = len(pairs)
        results: list[IKResult] = []

        for i, ((l_pos, l_quat), (r_pos, r_quat)) in enumerate(pairs):
            q_sol, ok = planner.solve_ik(l_pos, l_quat, r_pos, r_quat)
            if ok:
                # q_sol is a torch.Tensor on GPU; convert once, store as numpy
                q_np = q_sol.detach().cpu().numpy().astype(np.float32)
                results.append(((l_pos, l_quat), (r_pos, r_quat), q_np))

            if verbose and (i + 1) % 100 == 0:
                pct = 100.0 * len(results) / (i + 1)
                print(f"  [{i + 1:>{len(str(n_total))}}/{n_total} {(i + 1)/n_total:.1%}]  "
                    f"feasible: {len(results)}  ({pct:.1f} %)")

        if verbose:
            pct = 100.0 * len(results) / max(n_total, 1)
            print(f"IK filter done — {len(results)}/{n_total} pairs feasible ({pct:.1f} %)")

        return results

    try:
        from .kinematics import CuRoboBimanualMotionPlanner, DEFAULT_CUROBO_ROBOT_CFG_PATH
    except ImportError:
        from kinematics import CuRoboBimanualMotionPlanner, DEFAULT_CUROBO_ROBOT_CFG_PATH

    planner = CuRoboBimanualMotionPlanner(
        robot_cfg_path              = DEFAULT_CUROBO_ROBOT_CFG_PATH,
        left_ee_link                = "left_delto_base_link",
        right_ee_link               = "right_delto_base_link",
        device                      = "cuda:0",
        trajopt_tsteps              = 64,
        interpolation_steps         = 2000,
        num_ik_seeds                = 50,
        num_trajopt_seeds           = 32,
        grad_trajopt_iters          = 800,
        interpolation_dt            = 0.02,
        collision_activation_distance = 0.005,
    )
    
    feasible_pre_grasp_q = filter_by_ik(filt_l, filt_r, planner=planner, max_pairs=None, verbose=True)
    print(f"\nFeasible bimanual pre-grasp poses: {len(feasible_pre_grasp_q)}")

    if len(feasible_pre_grasp_q) > 0:
        # ((l_pos, l_quat), (r_pos, r_quat), q_sol))
        q_list = [item[2] for item in feasible_pre_grasp_q]
        
        q_tensor = torch.tensor(np.array(q_list), dtype=torch.float32)
        print(f"\n[INFO] Generated Tensor Shape: {q_tensor.shape} (Expected: N x 38)")

        save_dir = Path("models")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "pre_grasp_q_samples.pt"
        
        torch.save(q_tensor, save_path)
        print(f"[SUCCESS] Saved {len(feasible_pre_grasp_q)} feasible pre-grasp poses to {save_path.resolve()}")
    else:
        print("\n[WARN] No feasible pre-grasp poses were found. Nothing to save.")