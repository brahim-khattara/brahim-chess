"""Shared dual-head training epoch loop."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from brahim_chess.metrics import dual_head_batch_metrics, dual_head_masked_exact


def run_dual_head_epoch(model, dataloader, optimizer, device, masked_eval: bool = False, input_key: str = "input_ids"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_start = 0
    total_end = 0
    total_exact = 0
    total_exact_masked = 0
    total_samples = 0

    for batch in dataloader:
        inputs = batch[input_key].to(device)
        start_targets = batch["start_target"].to(device)
        end_targets = batch["end_target"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        start_logits, end_logits = model(inputs)
        loss = F.cross_entropy(start_logits, start_targets) + F.cross_entropy(end_logits, end_targets)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        start_correct, end_correct, exact_correct = dual_head_batch_metrics(
            start_logits, end_logits, start_targets, end_targets
        )
        total_start += start_correct
        total_end += end_correct
        total_exact += exact_correct
        total_samples += batch_size

        if masked_eval:
            total_exact_masked += dual_head_masked_exact(
                start_logits,
                end_logits,
                batch["fen_str"],
                start_targets.cpu().tolist(),
                end_targets.cpu().tolist(),
            )

    n = max(total_samples, 1)
    exact_masked_acc = 100.0 * total_exact_masked / n if masked_eval else 0.0
    return (
        total_loss / n,
        100.0 * total_start / n,
        100.0 * total_end / n,
        100.0 * total_exact / n,
        exact_masked_acc,
    )
