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

def generate_sine_trajectory(home_pos, duration=5.0, dt=1/30.0, freq=0.5, amp=0.1):
    """
    Generates a [steps, dim] trajectory oscillating around home_position.
    """
    num_steps = int(duration / dt)
    num_joints = len(home_pos)
    
    # Create time vector [steps, 1]
    t = np.linspace(0, duration, num_steps).reshape(-1, 1)
    
    # Calculate offsets [steps, dim]
    # We use sin(2 * pi * f * t) starting at 0 to ensure a smooth takeoff
    offsets = amp * np.sin(2 * np.pi * freq * t)
    
    # Broadcast home_position to [steps, dim] and add offsets
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
    home_position = interface.home_joint_positions
    try:
        trajectory = generate_sine_trajectory(home_position, duration=5.0, dt=1/args.freq, freq=0.5, amp=0.1)
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