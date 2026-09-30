"""Shared training / evaluation metrics for dual-head and unified models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from brahim_chess.moves import legal_move_mask


def dual_head_batch_metrics(start_logits, end_logits, start_targets, end_targets):
    start_pred = start_logits.argmax(dim=1)
    end_pred = end_logits.argmax(dim=1)
    start_correct = (start_pred == start_targets).sum().item()
    end_correct = (end_pred == end_targets).sum().item()
    exact_correct = ((start_pred == start_targets) & (end_pred == end_targets)).sum().item()
    return start_correct, end_correct, exact_correct


def dual_head_masked_exact(start_logits, end_logits, fen_strs, start_targets, end_targets):
    start_probs = F.softmax(start_logits, dim=1)
    end_probs = F.softmax(end_logits, dim=1)
    exact_correct = 0
    for i in range(start_probs.size(0)):
        joint = torch.outer(start_probs[i], end_probs[i]).reshape(-1)
        mask = legal_move_mask(fen_strs[i]).to(joint.device)
        joint = joint * mask
        pred_flat = int(joint.argmax().item())
        pred_start = pred_flat // 64
        pred_end = pred_flat % 64
        if pred_start == start_targets[i] and pred_end == end_targets[i]:
            exact_correct += 1
    return exact_correct


def unified_masked_exact(move_logits, fen_strs, move_targets):
    exact_correct = 0
    for i in range(move_logits.size(0)):
        logits = move_logits[i]
        mask = legal_move_mask(fen_strs[i]).to(logits.device)
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        pred_flat = int(masked_logits.argmax().item())
        if pred_flat == move_targets[i]:
            exact_correct += 1
    return exact_correct


def topk_masked_dual(start_logits, end_logits, fen_str: str, k: int = 5):
    """Return top-k legal UCI strings from dual-head logits for one position."""
    from brahim_chess.moves import make_uci

    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    joint = torch.outer(start_probs, end_probs).reshape(-1)
    mask = legal_move_mask(fen_str).to(joint.device)
    joint = joint * mask
    topk = torch.topk(joint, k=min(k, int((mask > 0).sum().item()) or 1))
    results = []
    for score, flat_idx in zip(topk.values.tolist(), topk.indices.tolist()):
        start_idx = flat_idx // 64
        end_idx = flat_idx % 64
        results.append((make_uci(start_idx, end_idx), float(score)))
    return results
