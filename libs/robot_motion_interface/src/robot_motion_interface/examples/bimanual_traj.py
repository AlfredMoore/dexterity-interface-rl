"""
Local Data Player for Bimanual Robot Interface.
Reads .npy or .pt files and executes trajectory on the robot.

Usage:
    python local_playback.py --file motion_data.npy --freq 30.0 --loop
"""

import time
import argparse
import numpy as np
from pathlib import Path

# Import the Bimanual Interface
from robot_motion_interface.bimanual_interface import BimanualInterface

# ----------------- Configuration ----------------- #
CONFIG_FILENAME = "bimanual_arm_config.yaml"
# ------------------------------------------------- #

def get_args():
    parser = argparse.ArgumentParser(description="Local Trajectory Player")
    parser.add_argument("--freq", type=float, default=30.0, help="Control frequency in Hz (default: 30.0)")
    parser.add_argument("--loop", action="store_true", help="Loop the playback")
    return parser.parse_args()

import numpy as np

def generate_sine_trajectory(home_pos, duration=5.0, dt=1/30.0, freq=0.5, active_joints=None, amp=0.01):
    """
    Generates a [steps, dim] trajectory. Only specified joints will move.
    
    Args:
        home_pos: [dim] array of home positions.
        active_joints: List of indices that should move (e.g., [0, 2, 5]). 
                       If None, all joints move.
        amp: Scalar or [dim] array of amplitudes.
    """
    num_steps = int(duration / dt)
    num_joints = len(home_pos)
    
    # 1. 构造振幅向量 [num_joints]
    # 初始化为全 0
    amp_vector = np.zeros(num_joints)
    
    if active_joints is None:
        # 如果没指定，全部关节都用传入的 amp
        amp_vector[:] = amp
    else:
        # 只给选中的关节赋值
        for idx in active_joints:
            amp_vector[idx] = amp

    # 2. 生成时间向量 [num_steps, 1]
    t = np.linspace(0, duration, num_steps).reshape(-1, 1)
    
    # 3. 计算基础的正弦波形 [num_steps, 1]
    base_wave = np.sin(2 * np.pi * freq * t)
    
    # 4. 利用广播机制计算偏移 [num_steps, num_joints]
    # [num_steps, 1] * [num_joints] -> [num_steps, num_joints]
    offsets = base_wave * amp_vector
    
    # 5. 叠加到起始位置
    trajectory = home_pos + offsets
    
    return trajectory


def main():
    args = get_args()

    # 1. Setup Config
    config_dir = Path(__file__).resolve().parents[3] / "config"
    config_path = config_dir / CONFIG_FILENAME
    
    if not config_path.exists():
        if Path(CONFIG_FILENAME).exists():
            config_path = Path(CONFIG_FILENAME)
        else:
            print(f"Error: Config file not found at {config_path}")
            return

    # 2. Initialize Interface
    print(f"Initializing BimanualInterface from: {config_path}")
    interface = BimanualInterface.from_yaml(str(config_path))
    interface.start_loop()
    interface.home()
    time.sleep(2.0)  # wait for the robot to stabilize
    
    # 3. Load Traj
    try:
        home_position = interface._home_joint_positions
        trajectory = generate_sine_trajectory(home_position, duration=5.0, dt=1/args.freq, freq=0.5, 
                                              active_joints=[2, 3, 10, 14, 17, 18], amp=0.01)
    except Exception as e:
        print(f"Error loading data: {e}")
        interface.stop_loop()
        return

    # 4. Playback Loop
    period = 1.0 / args.freq
    print(f"Starting playback at {args.freq} Hz (Period: {period:.4f}s)")
    print("Press Ctrl+C to stop.")

    try:
        while True: # Outer loop for --loop functionality
            
            # back to starting point
            print("Moving to start position...")
            interface.set_joint_positions(trajectory[0])
            time.sleep(2.0) # wait to settle

            start_time_global = time.time()
            
            for i, q in enumerate(trajectory):
                loop_start = time.time()
                
                # --- Send Command ---
                # Assume q's dimensions match the number of joints defined in the config
                print(f"q: {q}")
                interface.set_joint_positions(q)
                
                # --- Frequency Control ---
                # Calculate processing time, only sleep the remaining time to ensure stable frequency
                elapsed = time.time() - loop_start
                sleep_time = period - elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Optional: Print progress
                if i % int(args.freq) == 0:
                    print(f"Progress: {i}/{len(trajectory)} frames")

            print("Trajectory finished.")
            
            if not args.loop:
                break
            
            print("Looping enabled. Restarting...")

    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    except Exception as e:
        print(f"\nRuntime Error: {e}")
    finally:
        print("Stopping interface...")
        interface.stop_loop()

if __name__ == "__main__":
    main()