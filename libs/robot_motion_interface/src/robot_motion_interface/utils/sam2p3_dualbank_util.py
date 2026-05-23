"""
SAM3-init + SAM2-track dual-bank streaming utility for real-time deployment.

Architecture:
  - SAM3 image model gives a text-prompted bbox on the *bootstrap frame* and
    on each subsequent *re-anchor frame*.
  - SAM2 video predictor maintains TWO banks (inference_states) that overlap
    for a warmup window before a swap, so the cond_frame is periodically
    refreshed without losing tracking continuity:

        global frame   0 ... K-W-1   K-W ... K-1    K ... 2K-W-1   2K-W ... 2K-1   2K ...
        bank_A         PRI ... PRI   PRI ... PRI    dead
        bank_B                       WARM... WARM   PRI ... PRI    PRI ... PRI     dead
        bank_C                                                     WARM... WARM    PRI ...

    PRI = primary (mask output comes from this bank).  WARM = warming up
    its memory bank in parallel; not used for output until its turn.

  - Each bank holds a pre-allocated images tensor (size = cycle_K + warmup_W
    + margin) so per-frame ingestion is an O(1) in-place write — no torch.cat
    realloc cost.

Lifecycle:
    tracker = SAM2P3DualBankTracker(
        sam2_ckpt=..., sam2_cfg=..., sam3_ckpt=...,
        prompt="bottle.",
        cycle_K=100, warmup_W=7,
    )
    tracker.warmup(first_bgr_frame)        # SAM2 encoder warmup (~3 calls)
    mask_0 = tracker.init(first_bgr_frame) # SAM3 on bootstrap frame, spawn bank_A
    for bgr in camera_stream:
        mask = tracker.step(bgr)           # ~25-30 ms / frame on 4090, single bank
                                           # ~50-60 ms during a warmup window

Returned mask is (H, W) bool at the input frame's resolution.

Smoke test (defaults to sample_20260521_232953_470):
    python sam2p3_dualbank_util.py --sample sample_20260521_232953_470
"""

from __future__ import annotations

import shutil
import tempfile
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms as tv_nms


# Prompt default is just a string and harmless; checkpoint / config paths are
# environment-specific and must be passed by the caller (no defaults).
DEFAULT_PROMPT = "bottle."

# ImageNet stats — must match what SAM2's load_video_frames applies.
_SAM2_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_SAM2_IMG_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def _bpe_path() -> str:
    """Locate the SAM3 BPE asset (bypass pkg_resources misfire on editable installs)."""
    import sam3 as _pkg
    return str(Path(list(_pkg.__path__)[0]) / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def _build_empty_state(predictor, anchor_bgr: np.ndarray, max_frames: int,
                       offload_video_to_cpu: bool) -> dict:
    """Construct a SAM2 inference_state in memory, replacing the public
    init_state(video_path=...) call.

    SAM2's init_state only takes a video_path / frames dir, so feeding a
    single in-memory anchor frame through it requires a tmpfile +
    cv2.imwrite + JPG decode round-trip (~15-25 ms per bank spawn).  Here
    we replicate init_state's field setup directly and skip disk IO.

    The anchor is written into images[0]; the rest of the (max_frames, 3,
    image_size, image_size) buffer stays zero until step() fills it.
    """
    compute_device = predictor.device
    H, W = anchor_bgr.shape[:2]
    image_size = predictor.image_size
    img_device = torch.device("cpu") if offload_video_to_cpu else compute_device

    anchor_tensor = _preprocess_bgr(anchor_bgr, image_size, img_device)
    buf = torch.zeros(
        (max_frames, 3, image_size, image_size),
        dtype=anchor_tensor.dtype, device=img_device,
    )
    buf[0] = anchor_tensor

    state = {}
    state["images"]                 = buf
    state["num_frames"]             = 1
    state["offload_video_to_cpu"]   = offload_video_to_cpu
    state["offload_state_to_cpu"]   = False
    state["video_height"]           = H
    state["video_width"]            = W
    state["device"]                 = compute_device
    state["storage_device"]         = compute_device
    state["point_inputs_per_obj"]   = {}
    state["mask_inputs_per_obj"]    = {}
    state["cached_features"]        = {}
    state["constants"]              = {}
    state["obj_id_to_idx"]          = OrderedDict()
    state["obj_idx_to_id"]          = OrderedDict()
    state["obj_ids"]                = []
    state["output_dict_per_obj"]    = {}
    state["temp_output_dict_per_obj"] = {}
    state["frames_tracked_per_obj"] = {}
    # Mirror init_state's tail: prime the image encoder + cached_features
    # for frame 0 so the first add_new_mask() doesn't pay cold start surcharge.
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    return state


def _preprocess_bgr(bgr: np.ndarray, image_size: int, target_device) -> torch.Tensor:
    """BGR (H, W, 3) uint8 → (3, image_size, image_size) float32, normalized
    the same way SAM2's `_load_img_as_tensor` + `load_video_frames` does."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((image_size, image_size))
    arr = np.array(pil, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).to(target_device)
    mean = _SAM2_IMG_MEAN.to(target_device)
    std  = _SAM2_IMG_STD.to(target_device)
    return (img - mean) / std


# ── SAM3 image-model anchor ────────────────────────────────────────────────
class _SAM3Anchor:
    """SAM3 image model held in GPU memory; top(bgr, prompt) → (mask, box)
    for the top-1 detection.

    Both prompts are returned: the *mask* is what we feed to SAM2 as the
    initial cond-frame conditioning (highest fidelity — SAM2's native
    `add_new_mask` accepts an (H, W) bool), and the *bbox* is kept for
    visualisation / external use (SAM2 cannot use both prompts on the same
    frame: its API and underlying _run_single_frame_inference enforce
    `point_inputs is None or mask_inputs is None`).

    Kept private — public users go through SAM2P3DualBankTracker.
    """

    def __init__(self, ckpt: str, device: str = "cuda",
                 conf: float = 0.3, nms_iou: float = 0.5):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            bpe_path=_bpe_path(),
            checkpoint_path=ckpt,
            device=device,
            eval_mode=True,
            load_from_HF=False,
            compile=False,
        )
        self.proc = Sam3Processor(model, device=device, confidence_threshold=conf)
        self.device = device
        self.nms_iou = nms_iou

    @torch.inference_mode()
    def top(self, bgr: np.ndarray, prompt: str) -> Optional[tuple]:
        """Returns (mask, box) on the SAM3 model's device, both torch.Tensor:
            mask: (H, W) bool
            box:  (4,)  float32 in absolute pixel coords [x0, y0, x1, y1]
        Returns None if no detection passed the confidence threshold.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.proc.set_image(Image.fromarray(rgb))
            state = self.proc.set_text_prompt(prompt=prompt, state=state)
        masks  = state.get("masks")
        boxes  = state.get("boxes")
        scores = state.get("scores")
        if boxes is None or boxes.numel() == 0:
            return None
        keep = tv_nms(boxes.float(), scores.float(), self.nms_iou)
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
        top = int(scores.argmax().item())
        mask = masks[top, 0].detach().bool()        # (H, W) bool tensor, on device
        box  = boxes[top].detach().float()          # (4,)   float tensor, on device
        return mask, box


# ── A single streaming SAM2 bank ──────────────────────────────────────────
class _StreamBank:
    """
    SAM2 inference_state seeded with one anchor frame, with a pre-allocated
    images buffer so step() is an O(1) in-place write.

    spawn_global : global frame index where this bank was created (= local 0).
    _local_pos   : -1 before any step; equals local frame index after step.
                   global_frame = spawn_global + _local_pos.
    """

    def __init__(self, predictor, spawn_global: int, anchor_bgr: np.ndarray,
                 sam3_mask: torch.Tensor, sam3_box: torch.Tensor,
                 device: str, max_frames: int,
                 prompt_type: str,
                 offload_video_to_cpu: bool):
        if prompt_type not in ("mask", "box"):
            raise ValueError(f"prompt_type must be 'mask' or 'box', got {prompt_type!r}")
        self.predictor    = predictor
        self.spawn_global = spawn_global
        self.device       = device
        self._image_size  = predictor.image_size
        self._local_pos   = -1
        self._max_frames  = max_frames
        self.prompt_type  = prompt_type
        # Frozen-mode cache: once the buffer fills, step() stops pushing and
        # returns this cached mask instead of raising. SAM3 retries continue
        # in the outer tracker, so when detection comes back a new bank takes
        # over and this one is retired.
        self._last_returned_mask: Optional[np.ndarray] = None
        self._frozen_warned: bool = False

        # Both prompts are stored regardless of which one is sent to SAM2 —
        # the unused one is still useful for visualisation / logging.  SAM2's
        # API and `_run_single_frame_inference`'s
        #   assert point_inputs is None or mask_inputs is None
        # enforce that exactly ONE of {points/box, mask} is the cond prompt
        # on a given frame; `prompt_type` selects which.
        # Both arrive as torch.Tensor (on SAM3's device) — no numpy roundtrip.
        self.sam3_mask = sam3_mask
        self.sam3_box  = sam3_box

        # In-memory state construction — avoids the tmpfile + cv2.imwrite +
        # JPG decode round-trip that init_state(video_path=...) would impose.
        # Pre-allocated images buffer prevents torch.cat realloc cost per push.
        self.state = _build_empty_state(
            predictor, anchor_bgr,
            max_frames=max_frames,
            offload_video_to_cpu=offload_video_to_cpu,
        )

        # Anchor prompt at local frame 0
        if prompt_type == "mask":
            # add_new_mask accepts a torch tensor directly; internally moves
            # it to inference_state["device"] (GPU).  No conversion needed.
            predictor.add_new_mask(
                inference_state=self.state,
                frame_idx=0, obj_id=1, mask=sam3_mask,
            )
        else:  # "box"
            # add_new_points_or_box internally cats `box_coords` (= our box)
            # with a CPU placeholder `points`.  Passing a CUDA tensor causes
            # a device-mismatch error there, so we hand it off as CPU tensor.
            predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=0, obj_id=1, box=sam3_box.cpu(),
            )
        # Consolidate temp → output_dict; runs the memory encoder on the anchor
        # so its maskmem_features is ready for downstream attention.
        predictor.propagate_in_video_preflight(self.state)

    def step(self, bgr: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance one frame; return (H, W) bool mask.

        First call: pass bgr=None — returns the anchor's cond_frame mask.
        Subsequent calls: pass the new BGR frame.
        """
        if self._local_pos == -1:
            # Anchor: cond_frame_outputs[0] already has the SAM2-image mask
            # derived from the SAM3 bbox via add_new_points_or_box + preflight.
            self._local_pos = 0
            cond_out = self.state["output_dict_per_obj"][0]["cond_frame_outputs"][0]
            pred_masks = cond_out["pred_masks"].to(self.device, non_blocking=True)
        else:
            assert bgr is not None, "bgr required after the anchor step"
            # Buffer full → freeze. Stop pushing new frames, return the last
            # computed mask. Don't raise — the outer SAM3-retry loop keeps
            # running (every cycle_K/4 frames), so when detection succeeds a
            # new bank spawns and this frozen one gets retired on swap.
            # `_last_returned_mask is None` only happens if the anchor step
            # itself didn't produce a mask, which shouldn't be possible.
            if self._local_pos + 1 >= self._max_frames:
                if not self._frozen_warned:
                    self._frozen_warned = True
                    print(
                        f"  [bank@{self.spawn_global}] buffer full at "
                        f"max_frames={self._max_frames}; mask frozen until "
                        f"SAM3 retry succeeds and a new bank replaces this one."
                    )
                return self._last_returned_mask
            self._local_pos += 1
            # In-place write into the pre-allocated buffer.
            img = _preprocess_bgr(bgr, self._image_size, self.state["images"].device)
            self.state["images"][self._local_pos] = img
            self.state["num_frames"] = self._local_pos + 1

            obj_output_dict = self.state["output_dict_per_obj"][0]
            current_out, pred_masks = self.predictor._run_single_frame_inference(
                inference_state=self.state,
                output_dict=obj_output_dict,
                frame_idx=self._local_pos,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            obj_output_dict["non_cond_frame_outputs"][self._local_pos] = current_out
            self.state["frames_tracked_per_obj"][0][self._local_pos] = {"reverse": False}

        # Up-sample low-res logits → original-resolution (H, W) mask.
        _, video_res_masks = self.predictor._get_orig_video_res_output(
            self.state, pred_masks,
        )
        mask_np = (video_res_masks[0, 0] > 0).cpu().numpy()
        self._last_returned_mask = mask_np
        return mask_np

    def release(self):
        """Drop heavy state to let GC reclaim images buffer and memory bank entries."""
        self.state = None


# ── Public tracker ─────────────────────────────────────────────────────────
class SAM2P3DualBankTracker:
    """
    Streaming dual-bank SAM2 tracker with periodic SAM3 re-anchoring.

    See module docstring for the schedule diagram.

    Args:
        sam2_ckpt, sam2_cfg, sam3_ckpt: model checkpoints / hydra config.
        prompt:                         SAM3 text prompt (e.g. "bottle.").
        cycle_K:                        primary duration between swaps (frames).
        warmup_W:                       overlap window before each swap (frames).
                                        Must be < cycle_K.
        device:                         torch device.
        offload_video_to_cpu:           keep per-bank images buffer on CPU;
                                        ~0.5 ms / frame CPU↔GPU overhead but
                                        saves ~1.8 GB GPU per bank.  Recommend
                                        True on <16 GB GPUs, False on 24 GB+.
        sam3_conf:                      SAM3 confidence threshold on init / re-anchor.
        verbose:                        print spawn / swap events.

    Public API:
        warmup(frame, n=3)              warm SAM2 encoder (one-time).
        init(frame) -> mask             bootstrap: SAM3 on this frame, spawn bank_A.
        step(frame) -> mask             process next frame; manages spawn / swap.
        reset()                         discard all banks; next init() starts fresh.

    Properties:
        last_mask, last_score           most recent output.
        global_frame                    current frame counter (init=0, +1 per step).
        primary_anchor_frame            global frame of the current primary bank's anchor.
        did_spawn_last_step             True iff step() just spawned a warmup bank.
        did_swap_last_step              True iff step() just retired the primary.
        last_sam3_box, last_sam3_frame  the most recent SAM3 anchor (for viz).
    """

    def __init__(
        self,
        sam2_ckpt: str,
        sam2_cfg: str,
        sam3_ckpt: str,
        prompt: str    = DEFAULT_PROMPT,
        cycle_K: int   = 100,
        warmup_W: int  = 7,
        device: str    = "cuda",
        prompt_type: str = "mask",                                 # "mask" | "box"
        offload_video_to_cpu: bool = True,
        sam3_conf: float = 0.3,
        verbose: bool  = True,
    ):
        """Args:
            sam2_ckpt, sam2_cfg, sam3_ckpt:
                          Required paths — no defaults, must be passed by the
                          caller (depend on deploy machine layout).
            prompt:       SAM3 text query, e.g. "bottle.".
            cycle_K:      primary duration between swaps (frames).
            warmup_W:     overlap window before each swap (frames).  Match
                          SAM2's num_maskmem-1 (= 6 by default → use W=7) —
                          larger W is wasted compute (older frames are out of
                          memory attention's reach).
            prompt_type:  selects the SAM3 → SAM2 prompt at each bank spawn:
                          "mask" (default) — feed SAM3's mask as cond, highest
                                             fidelity.  Recommended.
                          "box"            — feed SAM3's bbox; SAM2-S then
                                             re-derives the mask from the box.
                                             Matches the original baseline.
                          Both are stored on the bank either way (accessible
                          as bank.sam3_mask / bank.sam3_box) for viz / logging.
        """
        if not (cycle_K > warmup_W > 0):
            raise ValueError(f"cycle_K ({cycle_K}) must exceed warmup_W ({warmup_W}) > 0")
        if prompt_type not in ("mask", "box"):
            raise ValueError(f"prompt_type must be 'mask' or 'box', got {prompt_type!r}")

        self.prompt      = prompt
        self.prompt_type = prompt_type
        self.cycle_K     = cycle_K
        self.warmup_W    = warmup_W
        self.device      = device
        self.offload_video_to_cpu = offload_video_to_cpu
        self.verbose     = verbose

        # Bank-buffer size. Old default was cycle_K + warmup_W + 32, but that
        # leaves zero room for SAM3 spawn-frame misses (one miss freezes
        # _next_spawn_at; the primary bank then runs to the cap and trips
        # RuntimeError). Sizing it at 2 * cycle_K + warmup_W gives the fixed
        # K/4 retry stride 4 attempts (~cycle_K worth of frames) before the
        # buffer fills. VRAM cost on cycle_K=100 is +1.6 GB per bank, fits
        # easily on a 4090.
        self._max_frames_per_bank = 2 * cycle_K + warmup_W

        # Build SAM3 (image model)
        self.sam3 = _SAM3Anchor(sam3_ckpt, device=device, conf=sam3_conf)

        # Build SAM2 (video predictor)
        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)

        # Per-episode state — populated by init(), cleared by reset()
        self._banks: deque[_StreamBank] = deque(maxlen=2)
        self._global_frame   = -1
        self._next_spawn_at  = cycle_K - warmup_W
        self._next_swap_at   = cycle_K
        self._last_mask: Optional[np.ndarray] = None
        self._last_score = 0.0  # set when we get score data from SAM3
        self._did_spawn = False
        self._did_swap  = False
        self._last_sam3_box: Optional[np.ndarray] = None
        self._last_sam3_frame: Optional[int] = None
        # Exponential-backoff retry counter for SAM3 spawn-frame miss. Without
        # this, a single miss (e.g. hand occludes the bottle on the spawn
        # frame) freezes _next_spawn_at / _next_swap_at, so the primary bank
        # never retires and eventually trips _StreamBank.max_frames=139.
        # Reset to 0 on every successful SAM3 spawn (init + step's spawn path).
        self._sam3_miss_count: int = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _vprint(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def set_prompt(self, prompt: str) -> None:
        """Change the text prompt for *future* SAM3 calls (does not re-anchor now)."""
        self.prompt = prompt

    # ── Warmup ──────────────────────────────────────────────────────────
    def warmup(self, frame: np.ndarray, n: int = 3) -> None:
        """Warm the SAM2 image encoder so the first real init() is not cold.

        Re-runs SAM2's image encoder n times on `frame` (a dummy invocation
        path via init_state) to trigger any lazy CUDA kernel compilation.
        Does NOT consume any of the episode state.
        """
        h, w = frame.shape[:2]
        tmp = tempfile.mkdtemp(prefix="sam2p3_warmup_")
        for i in range(n):
            cv2.imwrite(f"{tmp}/{i:05d}.jpg", frame)
        try:
            _ = self.predictor.init_state(
                video_path=tmp,
                offload_video_to_cpu=self.offload_video_to_cpu,
                async_loading_frames=False,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._vprint(f"  [warmup] SAM2 encoder warmed up on {n} dummy frames")

    # ── Episode lifecycle ───────────────────────────────────────────────
    def reset(self) -> None:
        """Discard all active banks; require a new init() before step()."""
        for b in self._banks:
            b.release()
        self._banks.clear()
        self._global_frame    = -1
        self._next_spawn_at   = self.cycle_K - self.warmup_W
        self._next_swap_at    = self.cycle_K
        self._last_mask       = None
        self._last_score      = 0.0
        self._did_spawn       = False
        self._did_swap        = False
        self._last_sam3_box   = None
        self._last_sam3_frame = None
        self._sam3_miss_count = 0
        self._vprint("  [reset] all banks released")

    @torch.inference_mode()
    def init(self, frame: np.ndarray) -> np.ndarray:
        """Bootstrap: run SAM3 on `frame`, spawn bank_A, return mask for `frame`."""
        with torch.autocast(self.device, dtype=torch.bfloat16):
            res = self.sam3.top(frame, self.prompt)
        if res is None:
            raise RuntimeError(
                f"SAM3 missed '{self.prompt}' on the bootstrap frame; cannot init tracker."
            )
        mask, box = res
        self._last_sam3_box   = box
        self._last_sam3_frame = 0
        self._global_frame    = 0
        self._did_spawn       = True
        self._did_swap        = False
        self._vprint(f"  [spawn] frame   0  box={box.int().tolist()}  (primary, {self.prompt_type} init)")
        with torch.autocast(self.device, dtype=torch.bfloat16):
            bank = _StreamBank(
                self.predictor, spawn_global=0, anchor_bgr=frame,
                sam3_mask=mask, sam3_box=box, device=self.device,
                max_frames=self._max_frames_per_bank,
                prompt_type=self.prompt_type,
                offload_video_to_cpu=self.offload_video_to_cpu,
            )
            self._banks.append(bank)
            self._last_mask = bank.step(None)              # cond_frame mask
        return self._last_mask

    @torch.inference_mode()
    def step(self, frame: np.ndarray) -> np.ndarray:
        """Advance one frame; manages spawn / swap automatically.  Returns mask."""
        if not self._banks:
            raise RuntimeError("step() called before init()")

        self._global_frame += 1
        g = self._global_frame
        self._did_spawn = False
        self._did_swap  = False

        with torch.autocast(self.device, dtype=torch.bfloat16):
            # ── Spawn warmup bank if scheduled ────────────────────────────
            spawned_this_iter = False
            if g == self._next_spawn_at:
                res_g = self.sam3.top(frame, self.prompt)
                if res_g is not None:
                    mask_g, box_g = res_g
                    self._last_sam3_box   = box_g
                    self._last_sam3_frame = g
                    self._vprint(
                        f"  [spawn] frame {g:3d}  box={box_g.int().tolist()}  (warmup, {self.prompt_type} init)"
                    )
                    self._banks.append(_StreamBank(
                        self.predictor, spawn_global=g, anchor_bgr=frame,
                        sam3_mask=mask_g, sam3_box=box_g, device=self.device,
                        max_frames=self._max_frames_per_bank,
                        prompt_type=self.prompt_type,
                        offload_video_to_cpu=self.offload_video_to_cpu,
                    ))
                    spawned_this_iter = True
                    self._did_spawn = True
                    self._sam3_miss_count = 0  # success resets the backoff
                else:
                    # SAM3 missed on the spawn frame (e.g. hand occludes the
                    # bottle right when re-anchor was scheduled). Reschedule
                    # the next spawn at a *fixed* cycle_K // 4 retry stride
                    # and shift swap_at in lockstep so the primary retires
                    # the moment a new bank finally spawns.
                    #
                    # Fixed (not exponential) stride is intentional: with a
                    # bank lifetime of cycle_K + warmup_W + cycle_K (see
                    # _max_frames_per_bank), 4 retries fit before the buffer
                    # fills, vs only 1 retry for K/2 backoff. K/4 ≈ 1.7s at
                    # 15 Hz inference — comparable to a grasp's occlusion
                    # window, so retries land soon after the hand clears.
                    self._sam3_miss_count += 1
                    backoff = max(1, self.cycle_K // 4)
                    self._next_spawn_at = g + backoff
                    self._next_swap_at  = self._next_spawn_at + self.warmup_W
                    self._vprint(
                        f"  [spawn] frame {g:3d}  SAM3 missed (miss #{self._sam3_miss_count}) "
                        f"— retry at frame {self._next_spawn_at}, swap at {self._next_swap_at}"
                    )

            # ── Step each active bank ────────────────────────────────────
            outs = []
            for i, bank in enumerate(self._banks):
                is_just_spawned = (spawned_this_iter and i == len(self._banks) - 1)
                m = bank.step(None) if is_just_spawned else bank.step(frame)
                outs.append(m)

            # ── Output = primary (oldest in deque) ───────────────────────
            self._last_mask = outs[0]

            # ── Swap event ───────────────────────────────────────────────
            if g == self._next_swap_at and len(self._banks) >= 2:
                retiring = self._banks.popleft()
                self._vprint(
                    f"  [swap]  frame {g:3d}: retire bank@{retiring.spawn_global}, "
                    f"new primary spawned at frame {self._banks[0].spawn_global}"
                )
                retiring.release()
                self._did_swap = True
                self._next_spawn_at = self._next_swap_at + (self.cycle_K - self.warmup_W)
                self._next_swap_at  = self._next_swap_at + self.cycle_K

        return self._last_mask

    # ── Read-only properties for visualisation / monitoring ─────────────
    @property
    def last_mask(self) -> Optional[np.ndarray]:
        return self._last_mask

    @property
    def global_frame(self) -> int:
        return self._global_frame

    @property
    def primary_anchor_frame(self) -> Optional[int]:
        return self._banks[0].spawn_global if self._banks else None

    @property
    def n_active_banks(self) -> int:
        return len(self._banks)

    @property
    def did_spawn_last_step(self) -> bool:
        return self._did_spawn

    @property
    def did_swap_last_step(self) -> bool:
        return self._did_swap

    @property
    def last_sam3_box(self) -> Optional[torch.Tensor]:
        """(4,) float tensor on SAM3's device, abs pixel xyxy.  Call .cpu().numpy()
        externally if you need numpy."""
        return self._last_sam3_box

    @property
    def last_sam3_frame(self) -> Optional[int]:
        return self._last_sam3_frame


# ── Visualisation ─────────────────────────────────────────────────────────
def draw_result(bgr: np.ndarray, tracker: SAM2P3DualBankTracker) -> np.ndarray:
    """Overlay the current tracker mask + the latest SAM3 anchor box onto a frame."""
    vis = bgr.copy()
    m = tracker.last_mask
    if m is not None and m.any():
        ov = vis.copy()
        ov[m] = (0, 0, 255)
        cv2.addWeighted(ov, 0.45, vis, 0.55, 0, vis)
        contours, _ = cv2.findContours(
            m.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    # If SAM3 anchored on THIS frame, draw the bbox
    if tracker.last_sam3_frame == tracker.global_frame and tracker.last_sam3_box is not None:
        x0, y0, x1, y1 = tracker.last_sam3_box.cpu().numpy().astype(int)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 1)
        cv2.putText(vis, "SAM3", (x0, max(y0 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    # Red bar on swap frame
    if tracker.did_swap_last_step:
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, h - 3), (w, h), (0, 0, 255), -1)
    return vis


# ── Smoke test entry ──────────────────────────────────────────────────────
def _main():
    import argparse
    import subprocess
    import time

    # smoke-test-only defaults (this machine's layout); tracker itself has none
    COTRAIN_ROOT     = "/workspace/robotics/isaaclab_repos/cotrain"
    DEFAULT_SAMPLE   = "sample_20260521_232953_470"
    DEFAULT_OUT_ROOT = "/workspace/robotics/isaaclab_repos/distill/runs/sam2p3_dualbank_util"
    DEFAULT_SAM3_CKPT = "/workspace/robotics/isaaclab_repos/distill/models/sam3.pt"
    DEFAULT_SAM2_CKPT = "/workspace/robotics/isaaclab_repos/distill/sam2/checkpoints/sam2.1_hiera_small.pt"
    DEFAULT_SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

    ap = argparse.ArgumentParser(description="Smoke test for SAM2P3DualBankTracker")
    ap.add_argument("--sample",    default=DEFAULT_SAMPLE)
    ap.add_argument("--cycle_K",   type=int, default=100)
    ap.add_argument("--warmup_W",  type=int, default=7)
    ap.add_argument("--prompt",    default="bottle.")
    ap.add_argument("--prompt_type", choices=("mask", "box"), default="mask",
                    help="SAM3 → SAM2 prompt at each spawn")
    ap.add_argument("--sam2_ckpt", default=DEFAULT_SAM2_CKPT)
    ap.add_argument("--sam2_cfg",  default=DEFAULT_SAM2_CFG)
    ap.add_argument("--sam3_ckpt", default=DEFAULT_SAM3_CKPT)
    ap.add_argument("--out_root",  default=DEFAULT_OUT_ROOT)
    ap.add_argument("--video_fps", type=float, default=30.0)
    ap.add_argument("--no_offload",  action="store_true",
                    help="Disable offload_video_to_cpu (faster on big GPUs)")
    args = ap.parse_args()

    sample_dir = Path(COTRAIN_ROOT) / args.sample
    out_dir    = Path(args.out_root) / args.sample
    vis_dir    = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir(parents=True)

    print(f"[load] {sample_dir}/data.npz")
    rgb_lores = np.load(sample_dir / "data.npz")["rgb_lores"]
    N, H, W = rgb_lores.shape[:3]
    print(f"[load] {N} frames @ {H}x{W}  cycle_K={args.cycle_K}  warmup_W={args.warmup_W}")

    tracker = SAM2P3DualBankTracker(
        sam2_ckpt=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg,
        sam3_ckpt=args.sam3_ckpt,
        prompt=args.prompt,
        prompt_type=args.prompt_type,
        cycle_K=args.cycle_K,
        warmup_W=args.warmup_W,
        offload_video_to_cpu=(not args.no_offload),
    )

    # ── Warmup + init ──────────────────────────────────────────────────────
    tracker.warmup(rgb_lores[0], n=3)
    t_init = time.perf_counter()
    mask0 = tracker.init(rgb_lores[0])
    print(f"[init] done in {(time.perf_counter() - t_init)*1000:.1f} ms")

    # ── Process frame 0 ────────────────────────────────────────────────────
    vis0 = draw_result(rgb_lores[0], tracker)
    cv2.putText(vis0, f"000/{N}  init  dualbank-util",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(vis_dir / "00000.jpg"), vis0)

    # ── Stream frames 1..N-1 ───────────────────────────────────────────────
    latencies_ms = [0.0]                                       # frame 0 not timed here
    for i in range(1, N):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mask = tracker.step(rgb_lores[i])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(ms)

        vis = draw_result(rgb_lores[i], tracker)
        cv2.putText(vis, f"{i:03d}/{N}  {1000/max(ms,1e-3):.1f}Hz  dualbank-util",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(vis_dir / f"{i:05d}.jpg"), vis)

    steady = latencies_ms[1:]
    mean_ms = sum(steady) / len(steady)
    print(f"\n[hz] {len(steady)} streamed frames  mean {mean_ms:.2f} ms ({1000/mean_ms:.1f} Hz)  "
          f"min {min(steady):.2f}  max {max(steady):.2f}")

    mp4 = out_dir / "dualbank_util.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(args.video_fps),
        "-i", str(vis_dir / "%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(mp4),
    ], check=True)
    print(f"[mp4] {mp4}")


if __name__ == "__main__":
    _main()
