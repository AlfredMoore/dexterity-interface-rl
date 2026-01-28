"""
ZMQ Subscriber for Bimanual Robot Interface.
Listens for commands with topic "ACTION".

Usage:
    python bimanual_zmq.py --ip 127.0.0.1 --port 5555 --verbose
"""

import zmq
import time
import argparse
import json
import numpy as np
from pathlib import Path

# Import the Bimanual Interface
from robot_motion_interface.bimanual_interface import BimanualInterface

# ----------------- Configuration ----------------- #
CONFIG_FILENAME = "bimanual_config.yaml"
DTYPE = np.float64  # Data type for joint positions
# data -> np.tobytes(DTYPE) -> send & recv -> np.frombuffer(data, dtype=DTYPE) -> data 
# ------------------------------------------------- #

def get_args():
    """Parses command line arguments for ZMQ connection."""
    parser = argparse.ArgumentParser(description="ZMQ Subscriber for Bimanual Robot Interface")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="ZMQ Publisher IP")
    parser.add_argument("--port", type=str, default="5555", help="ZMQ Publisher Port")
    parser.add_argument("--topic", type=str, default="ACTION", help="ZMQ Topic")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    # 1. Parse Arguments
    args = get_args()

    # 2. Setup Configuration Path
    config_dir = Path(__file__).resolve().parents[3] / "config"
    config_path = config_dir / CONFIG_FILENAME

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        if Path(CONFIG_FILENAME).exists():
            config_path = Path(CONFIG_FILENAME)
        else:
            return

    # 3. Initialize Bimanual Interface
    print(f"Initializing BimanualInterface from: {config_path}")
    interface = BimanualInterface.from_yaml(config_path)

    # 4. Initialize ZMQ Subscriber
    print(f"Initializing ZMQ Subscriber connecting to tcp://{args.ip}:{args.port}...")
    print(f"Subscribing to TOPIC: '{args.topic}'")
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    try:
        print(f"Connecting to tcp://{args.ip}:{args.port}...")
        socket.connect(f"tcp://{args.ip}:{args.port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
        socket.setsockopt(zmq.CONFLATE, 1)  # Conflate to keep only latest message
        print("Subscriber initialized. Waiting for data...")
        
    except zmq.ZMQError as e:
        print(f"Failed to connect to ZMQ: {e}")
        return

    # 5. Start the Interface Loop
    interface.start_loop()
    print(f"Interface started. Waiting for '{args.topic}' commands...")

    try:
        while True:
            try:
                # Frame 0: "ACTION", Frame 1: binary data
                parts = socket.recv_multipart()
                
                if len(parts) < 2:
                    continue

                topic = parts[0].decode('utf-8')
                payload = parts[1]

                # Double-check Topic (although ZMQ already filters, it's safer to check)
                if topic != args.topic:
                    print(f"Received message with invalid topic: {topic}")
                    continue
                
                q = np.frombuffer(payload, dtype=DTYPE)
                
                if args.verbose:
                    print(f"Received topic: '{topic}', data length: {len(q)}")
                
                # if len(q) != 38: continue

                # Send to Robot Interface
                if q is not None:
                    # data sequence is left hand, left finger, right hand, right finger
                    interface.set_joint_positions(q)

            except zmq.ZMQError as e:
                print(f"ZMQ Error: {e}")
                time.sleep(0.001)
            except json.JSONDecodeError:
                print("Error: Received invalid JSON data")
            except Exception as e:
                print(f"Error processing command: {e}")

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected.")
        
    finally:
        print("Stopping Bimanual Interface...")
        interface.stop_loop()
        socket.close()
        context.term()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()