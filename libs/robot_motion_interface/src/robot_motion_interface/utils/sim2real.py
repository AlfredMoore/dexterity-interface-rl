import numpy as np
import torch

def joint_mapping(
        policy_joint_names: list[str], 
        real_joint_names: list[str],
    ) -> tuple[list[int], list[int]]:
    """
    Create a mapping from simulation joint names to policy action indices.

    Args:
        policy_joint_names (list[str]): List of joint names in the simulation.
        real_joint_names (list[str]): List of joint names in the real robot.
    Returns:
        policy2real_indexing (list[int])
            Usage: real_action = policy_action[policy2real_indexing].
        real2policy_indexing (list[int])
            Usage: policy_action = real_action[real2policy_indexing].
    """
    policy2real_indexing = [policy_joint_names.index(name) for name in real_joint_names]
    real2policy_indexing = [real_joint_names.index(name) for name in policy_joint_names]
    
    return policy2real_indexing, real2policy_indexing