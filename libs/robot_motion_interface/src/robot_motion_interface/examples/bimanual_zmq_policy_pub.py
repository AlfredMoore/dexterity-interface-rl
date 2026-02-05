"""
ZMQ Publisher for Robot Policy (Simulated).
Generates [T, joint_num] torch tensors and publishes them via ZMQ.

Usage:
    python bimanual_policy_pub.py --port 5555 --rate 30
"""

import zmq
import time
import argparse
import torch
import numpy as np

# ----------------- Configuration ----------------- #
# 必须与 Subscriber 的 DTYPE 保持一致 (np.float64)
# PyTorch 默认是 float32，发送前必须转换
TARGET_DTYPE = np.float64 
# ------------------------------------------------- #

def get_args():
    parser = argparse.ArgumentParser(description="ZMQ Policy Publisher")
    parser.add_argument("--ip", type=str, default="*", help="Bind IP ('*' for all interfaces)")
    parser.add_argument("--port", type=str, default="5555", help="ZMQ Publisher Port")
    parser.add_argument("--topic", type=str, default="ACTION", help="ZMQ Topic name")
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate in Hz")
    parser.add_argument("--joints", type=int, default=14, help="Number of joints (e.g. 14 for bimanual)")
    parser.add_argument("--T", type=int, default=10, help="Trajectory horizon T")
    return parser.parse_args()

def generate_policy_output(step_counter, T, num_joints):
    """
    模拟 Policy 网络输出。
    返回: torch.Tensor, shape [T, num_joints]
    """
    # 模拟生成一个随时间变化的正弦波轨迹
    # 这里的逻辑只是为了生成动态数据，实际使用替换为你的 Policy Inference
    
    # 时间轴 [T, 1]
    t_seq = torch.linspace(0, 2*torch.pi, T).unsqueeze(1) 
    
    # 关节相位偏移 [1, J]
    joint_phases = torch.arange(num_joints).unsqueeze(0) * 0.2
    
    # 随 step 移动的波形
    global_t = step_counter * 0.05
    wave = torch.sin(t_seq + joint_phases + global_t)
    
    # 缩放幅度 (例如 +/- 0.1 rad)
    action = wave * 0.1 
    
    # 加上一个 offset (模拟 Home Position，防止机器人在 0 位乱撞)
    # 假设前7个是左臂，后7个是右臂
    home_pose = torch.zeros(1, num_joints)
    # home_pose[:, :7] = ... # 左臂 home
    # home_pose[:, 7:] = ... # 右臂 home
    
    return action + home_pose

def main():
    args = get_args()
    
    # 1. 初始化 ZMQ Context 和 Socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # Publisher 通常使用 bind
    address = f"tcp://{args.ip}:{args.port}"
    print(f"[INFO] Binding ZMQ Publisher to {address} ...")
    socket.bind(address)
    
    # 等待连接建立 (防止第一条消息丢失)
    time.sleep(1.0)
    
    print(f"[INFO] Starting loop at {args.rate} Hz. Press Ctrl+C to stop.")
    
    period = 1.0 / args.rate
    step_counter = 0

    try:
        while True:
            loop_start = time.time()
            
            # 2. 生成 Policy 输出 (Torch Tensor)
            # Shape: [T, joint_num] (例如 [10, 14])
            # Device: 假设你的 Policy 在 GPU 上
            action_tensor_gpu = generate_policy_output(step_counter, args.T, args.joints).cuda() if torch.cuda.is_available() else generate_policy_output(step_counter, args.T, args.joints)
            
            # ==========================================
            # 关键步骤：处理 [T, J] 数据以匹配 Subscriber
            # ==========================================
            
            # 方案 A: 你的 Subscriber 也是 MPC，需要一次性接收整个 T 序列
            # 发送整个扁平化数组，Subscriber 需要自己 reshape(T, J)
            # data_to_send = action_tensor_gpu
            
            # 方案 B (最常用): Policy 输出 T 步预测，但我们只发送第 0 步给底层驱动执行
            # 这种是典型的 MPC (Model Predictive Control) 逻辑
            data_to_send = action_tensor_gpu[0, :] # 取出第一帧 [joint_num]
            
            # 3. 序列化
            # Torch (GPU/CPU) -> Numpy (CPU) -> Bytes
            # 必须转为 float64 (double) 才能匹配 Subscriber 的 DTYPE = np.float64
            payload = data_to_send.double().cpu().numpy().tobytes()
            
            # 4. 发送
            # Frame 0: Topic (utf-8 bytes)
            # Frame 1: Payload (binary)
            socket.send_multipart([
                args.topic.encode('utf-8'),
                payload
            ])
            
            # 打印调试信息
            if step_counter % int(args.rate) == 0: # 每秒打印一次
                print(f"[PUB] Sent step {step_counter}, Shape: {data_to_send.shape}")

            # 5. 频率控制
            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            step_counter += 1

    except KeyboardInterrupt:
        print("\n[INFO] User stopped publisher.")
    finally:
        socket.close()
        context.term()
        print("[INFO] Cleaned up ZMQ context.")

if __name__ == "__main__":
    main()