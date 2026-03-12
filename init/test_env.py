import torch
import sys
import os
from pathlib import Path

# Resolve project root relative to this file (init/test_env.py -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT   = PROJECT_ROOT / "models"

def test_cuda_torch():
    print("--- 1. PyTorch & CUDA Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if CUDA is visible to PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Compiled Version: {torch.version.cuda}")
        
        # Simple Tensor Operation on GPU to verify memory allocation
        try:
            x = torch.rand(5, 3).cuda()
            print("GPU Tensor Operation: SUCCESS")
        except Exception as e:
            print(f"GPU Tensor Operation FAILED: {e}")
            return False
    else:
        print("ERROR: PyTorch cannot find GPU. Check nvidia-smi.")
        return False
    print("")
    return True

def test_pinocchio():
    print("--- 2. Pinocchio Kinetics Check ---")
    try:
        import pinocchio as pin
        import numpy as np
        print(f"Pinocchio Imported Successfully! Path: {pin.__file__}")
        
        # Build a sample 6-DOF manipulator model
        model = pin.buildSampleModelManipulator()
        data = model.createData()
        
        # Compute Forward Kinematics (FK)
        q = pin.randomConfiguration(model)
        pin.forwardKinematics(model, data, q)
        
        print(f"Model Name: {model.name} | Joints: {model.njoints}")
        print("Forward Kinematics Test: SUCCESS")
        
    except Exception as e:
        print(f"Pinocchio Test FAILED: {e}")
        return False
    print("")
    return True

def test_curobo():
    print("--- 3. cuRobo & CUDA Kernels Check ---")
    try:
        import curobo
        from curobo.geom.types import WorldConfig
        from curobo.util_file import get_robot_configs_path
        print(f"cuRobo Imported Successfully! Version: {curobo.__version__}")
        print(f"cuRobo Path: {curobo.__file__}")

        # Test if CUDA-accelerated modules can be initialized
        # This checks if the .so files (C++ extensions) are correctly linked
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        print("cuRobo CUDA Robot Model Import: SUCCESS")
        
        # Check if the build is optimized for the current GPU (RTX 4090)
        # cuRobo uses JIT or pre-compiled kernels; this triggers a basic load
        from curobo.rollout.rollout_base import RolloutBase
        print("cuRobo Rollout Engine Load: SUCCESS")

    except ImportError as e:
        print(f"ERROR: cuRobo not found. Did you run 'pip install -e . --break-system-packages'?")
        print(f"Technical Error: {e}")
        return False
    except Exception as e:
        print(f"cuRobo Functional Test FAILED: {e}")
        return False
    print("")
    return True

def test_promptda():
    print("--- 4. PromptDA Metric Depth Check ---")
    try:
        import numpy as np
        from robot_motion_interface.utils.promptda_utils import PromptDAInference
        print("PromptDAInference Import: SUCCESS")
    except Exception as e:
        print(f"PromptDAInference Import FAILED: {e}")
        return False

    ckpt = MODEL_ROOT / "pda-s-trans-model.ckpt"
    if not ckpt.exists():
        print(f"SKIP: checkpoint not found at {ckpt}")
        print("")
        return True   # not a code failure, just missing model file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pda = PromptDAInference(ckpt_path=str(ckpt), encoder="vits", device=device)
        print(f"PromptDA loaded on {device}")

        # Synthetic 640x480 color + depth frame
        bgr   = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)

        out = pda.infer(bgr, depth)
        print(f"PromptDA output shape: {tuple(out.shape)}  dtype: {out.dtype}")
        assert out.ndim == 2, f"Expected 2-D output, got shape {out.shape}"
        print("PromptDA Inference Test: SUCCESS")
    except Exception as e:
        print(f"PromptDA Functional Test FAILED: {e}")
        return False
    print("")
    return True


def test_sam3():
    print("--- 5. SAM 3 Semantic Segmentation Check ---")
    try:
        import numpy as np
        from robot_motion_interface.utils.sam3_utils import (
            SAM3Inference, CONCEPT_LEFT_ARM, CONCEPT_RIGHT_ARM, CONCEPT_CUP
        )
        print("SAM3Inference Import: SUCCESS")
    except Exception as e:
        print(f"SAM3Inference Import FAILED: {e}")
        return False

    ckpt = MODEL_ROOT / "sam3.pt"
    if not ckpt.exists():
        print(f"SKIP: checkpoint not found at {ckpt}")
        print("")
        return True   # not a code failure, just missing model file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # compile=False avoids torch.compile warm-up delay during testing
        sam3 = SAM3Inference(ckpt_path=str(ckpt), device=device, compile=False)
        print(f"SAM 3 loaded on {device}")

        # Synthetic 640x480 BGR frame
        bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        results = sam3.infer(bgr)

        # Verify all 3 concept entries are present
        assert set(results.keys()) == {CONCEPT_LEFT_ARM, CONCEPT_RIGHT_ARM, CONCEPT_CUP}, \
            f"Unexpected result keys: {results.keys()}"
        for obj_id, name in [(CONCEPT_LEFT_ARM, "left_arm"),
                              (CONCEPT_RIGHT_ARM, "right_arm"),
                              (CONCEPT_CUP, "cup")]:
            r = results[obj_id]
            masks_info = f"shape={r['masks'].shape}" if r["masks"] is not None else "no det"
            print(f"  {name}: {masks_info}  scores={r['scores']}")

        print("SAM 3 Inference Test: SUCCESS")
    except Exception as e:
        print(f"SAM 3 Functional Test FAILED: {e}")
        return False
    print("")
    return True


if __name__ == "__main__":
    print("==========================================")
    print("   Robotics Environment Health Check v3   ")
    print("==========================================\n")

    torch_ok   = test_cuda_torch()
    pin_ok     = test_pinocchio()
    curobo_ok  = test_curobo()
    pda_ok     = test_promptda()
    sam3_ok    = test_sam3()

    all_ok = torch_ok and pin_ok and curobo_ok and pda_ok and sam3_ok
    if all_ok:
        print("ALL CLEAR: Torch, Pinocchio, cuRobo, PromptDA, and SAM 3 are ready!")
    else:
        print("FAILED: One or more components are missing or misconfigured.")
        sys.exit(1)