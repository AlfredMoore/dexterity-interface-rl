import torch
import sys
import os

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

if __name__ == "__main__":
    print("==========================================")
    print("   Robotics Environment Health Check v2   ")
    print("==========================================\n")
    
    torch_ok = test_cuda_torch()
    pin_ok = test_pinocchio()
    curobo_ok = test_curobo()
    
    if torch_ok and pin_ok and curobo_ok:
        print("🎉 ALL CLEAR: Torch, Pinocchio, and cuRobo are ready for RTX 4090!")
    else:
        print("❌ FAILED: One or more components are missing or misconfigured.")
        sys.exit(1)