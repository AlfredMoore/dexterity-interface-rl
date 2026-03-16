
'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Contains a function for computing the best-of-K distance metrics from the EgoDex paper.
'''
import torch

@torch.no_grad()
def evaluate_distance(gt_actions, gt_padding, pred_actions, best_k=[1,5,10]):
    '''
    Calculates the best-of-K distance metrics from the paper for a single batch of data. 
    In practice this should be computed over all data points in the test set. 
    Currently assumes 48-dimensional actions but can be modified as desired.

    gt_actions: shape B x T x 48 torch tensor for ground truth action chunks, where B is batch size and T is chunk size
    gt_padding: shape B x T binary torch tensor indicating action chunk padding (1 for padding, 0 for no padding)
    pred_actions: shape N x B x T x 48 torch tensor for N model predictions. N = max(best_k)
    best_k: list of k values for which to compute best-of-k
    '''
    avg_values, final_values = {}, {}
    no_pad = (gt_padding.sum(dim=1) == 0) # we only eval on data points without padding
    assert gt_actions.shape[-1] == 48 # change if act space is different
    assert pred_actions.shape[-1] == 48
    action = gt_actions[:,:, list(range(3)) + list(range(9,27)) + list(range(33,48))] # assumes this indexing gets all 12 positions and skips 6d rotations
    pts1 = action.view(action.shape[0], action.shape[1], 12, 3)
    N = max(best_k)
    assert N <= pred_actions.shape[0], "insufficient number of predictions"

    final_dists = torch.zeros(action.shape[0], N)
    avg_dists = torch.zeros(action.shape[0], N)
    for i in range(N): 
        pred = pred_actions[i,:,:, list(range(3)) + list(range(9,27)) + list(range(33,48))]
        pts2 = pred.view(pred.shape[0], pred.shape[1], 12, 3)
        final_dist = torch.norm(pts1[:,-1,:,:] - pts2[:,-1,:,:], dim=-1) # euclidean distances
        final_dists[:,i] = final_dist.mean(dim=1) # average over num keypoints -> tensor of shape (B,)
        avg_dist = torch.norm(pts1 - pts2, dim=-1) # euclidean distances
        avg_dists[:,i] = avg_dist.mean(dim=(1,2)) # average over T and num keypoints -> tensor of shape (B,)

    for k in best_k:
        min_firstk, _ = torch.min(avg_dists[no_pad][:,:k], dim=1)
        avg_values[k] = min_firstk.mean().item()
        min_firstk, _ = torch.min(final_dists[no_pad][:,:k], dim=1)
        final_values[k] = min_firstk.mean().item()
        print("k = {}".format(k))
        print("Average: ", avg_values[k])
        print("Final: ", final_values[k])

    return avg_values, final_values