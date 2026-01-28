"""
Loss functions for ZipRerank training.

Stage 1: RankNet loss (pairwise ranking)
Stage 2: Soft Ranking Loss (knowledge distillation from GPT ranking)
"""

from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rank_net(
    y_pred,
    y_true,
    weighted=False,
    use_rank=False,
    weight_by_diff=False,
    weight_by_diff_powed=False,
):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    
    Args:
        y_pred: predictions from the model, shape [batch_size, slate_length]
        y_true: ground truth labels, shape [batch_size, slate_length]
        weighted: flag indicating whether to use relevance feedback weighting.
        use_rank: flag indicating whether to use rank-based true values.
        weight_by_diff: flag indicating whether to weight by ground truth differences.
        weight_by_diff_powed: flag indicating whether to weight by squared differences.
        
    Returns:
        loss value, a torch.Tensor
    """
    if use_rank is None:
        y_true = torch.tensor(
            [[1 / (np.argsort(y_true)[::-1][i] + 1) for i in range(y_pred.size(1))]]
            * y_pred.size(0)
        ).cuda()

    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weighted:
        values, indices = torch.sort(y_true, descending=True)
        ranks = torch.zeros_like(indices)
        ranks.scatter_(
            1,
            indices,
            torch.arange(1, y_true.numel() + 1).to(y_true.device).view_as(indices),
        )
        pairs_ranks = ranks[:, document_pairs_candidates]
        rank_sum = pairs_ranks.sum(-1)
        weight = 1 / rank_sum[the_mask]  # Relevance Feedback
    else:
        if weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(
                pairs_true[:, :, 1], 2
            )
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def soft_ranking_loss(y_pred, rank_labels, gt_indices, decay=0.5, reduction="mean"):
    """
    Soft Ranking Loss using cross-entropy with position-decayed target distribution.
    
    This loss:
    1. Uses the GPT-distilled ranking as soft targets (not hard labels)
    2. GUARANTEES GT is at position 0 (highest target probability)
    3. Naturally prevents overconfidence (GT target is ~50% with decay=0.5)
    4. Has a single interpretable parameter (decay)
    5. Is equivalent to knowledge distillation from GPT ranking
    
    Target distribution is created by:
    1. Placing GT at position 0 (highest weight)
    2. Following GPT ranking for remaining positions
    3. Assigning exponentially decaying weights by position
    4. Normalizing to a probability distribution
    
    Args:
        y_pred: Model scores [batch_size, num_candidates]
        rank_labels: GPT ranking for each sample. rank_labels[i][j] = candidate index at position j
        gt_indices: List of lists of GT indices per sample (GT is forced to position 0)
        decay: Exponential decay factor for position weights (default: 0.5)
               - decay=0.5: pos0=50%, pos1=25%, pos2=12.5%, ... (moderate confidence)
               - decay=0.3: pos0=70%, pos1=21%, pos2=6%, ... (higher confidence)
               - decay=0.7: pos0=30%, pos1=21%, pos2=15%, ... (lower confidence, more uniform)
        reduction: "mean" or "sum" for loss aggregation
        
    Returns:
        Cross-entropy loss between model's softmax and target distribution
    """
    device = y_pred.device
    batch_losses = []
    
    for batch_idx in range(y_pred.shape[0]):
        scores = y_pred[batch_idx]
        n_total = scores.shape[0]
        ranking = rank_labels[batch_idx] if batch_idx < len(rank_labels) else list(range(n_total))
        gt_idx_list = gt_indices[batch_idx] if batch_idx < len(gt_indices) else [0]
        gt_set = set(gt_idx_list)
        
        # Identify valid (non-padded) candidates: scores that are not -inf
        valid_mask = ~torch.isinf(scores)
        valid_indices = torch.where(valid_mask)[0].tolist()
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            continue
        
        # Build final ranking: GT first, then GPT ranking for non-GT (only valid candidates)
        final_ranking = []
        valid_set = set(valid_indices)
        
        # Add GT candidates at the top (position 0, 1, ... for multiple GTs)
        for gt_idx in gt_idx_list:
            if gt_idx in valid_set:
                final_ranking.append(gt_idx)
        
        # Add non-GT candidates in GPT ranking order
        for cand_idx in ranking:
            if cand_idx in valid_set and cand_idx not in gt_set:
                final_ranking.append(cand_idx)
        
        if len(final_ranking) == 0:
            continue
        
        # Extract valid scores only for softmax computation
        valid_scores = scores[valid_mask]
        
        # Create target distribution with exponential decay by position
        # Map from original index to valid-only index
        idx_to_valid_pos = {idx: i for i, idx in enumerate(valid_indices)}
        target_weights = torch.zeros(n_valid, device=device)
        for pos, cand_idx in enumerate(final_ranking):
            valid_pos = idx_to_valid_pos.get(cand_idx)
            if valid_pos is not None:
                target_weights[valid_pos] = decay ** pos
        
        # Handle edge case: no valid candidates in ranking
        weight_sum = target_weights.sum()
        if weight_sum < 1e-10:
            continue
            
        # Normalize to probability distribution
        target_dist = target_weights / weight_sum
        
        # Cross-entropy over valid candidates only: -sum(target * log(softmax(pred)))
        log_pred = F.log_softmax(valid_scores, dim=0)
        loss = -torch.sum(target_dist * log_pred)
        batch_losses.append(loss)
    
    if not batch_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    stacked = torch.stack(batch_losses)
    if reduction == "mean":
        return stacked.mean()
    elif reduction == "sum":
        return stacked.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# Loss function registry
loss_dict = {
    "ranknet": rank_net,
    "soft_ranking": soft_ranking_loss,
}

