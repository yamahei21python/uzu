#!/usr/bin/env python3
"""Standalone reproduction script for QAD scale collapse & loss divergence.

Executes a 32-condition grid (4 lr x 2 weight_decay x 2 log/raw x 2 clamp).
Does not require GPU or external dependencies (runs with numpy).
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
import numpy as np


def run_single_simulation(mode="buggy", scale_lr=1e-4, weight_decay=0.0,
                          use_log_scale=False, use_clamp=True, steps=50, seed=42):
    rng = np.random.default_rng(seed)
    in_dim, out_dim, group_size = 256, 1024, 32
    num_groups = (in_dim * out_dim) // group_size

    # Teacher weight
    w_teacher = rng.normal(0.0, 1.0 / math.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)

    # Initial scale: absmax / 7 (INT4 symmetric)
    w_grouped = w_teacher.reshape(-1, group_size)
    absmax = np.max(np.abs(w_grouped), axis=1)
    init_scales = np.maximum(absmax / 7.0, 1e-6).astype(np.float32)

    scales = np.log(init_scales.copy()) if use_log_scale else init_scales.copy()
    w_master = w_teacher.copy()

    # Adam states
    m_scale = np.zeros_like(scales)
    v_scale = np.zeros_like(scales)
    b1, b2, eps = 0.9, 0.95, 1e-8

    loss_history = []
    scale_min_threshold = 1e-5

    for step in range(1, steps + 1):
        x = rng.normal(0.0, 1.0, size=(16, in_dim)).astype(np.float32)

        curr_scales = np.exp(scales) if use_log_scale else scales
        curr_scales_g = curr_scales[:, None]

        # Quantize forward
        w_g = w_master.reshape(-1, group_size)
        scaled_w = w_g / curr_scales_g
        int_codes = np.clip(np.round(scaled_w), -8, 7)
        w_dequant = (int_codes * curr_scales_g).reshape(in_dim, out_dim)

        # Logits
        logits_t = x @ w_teacher
        logits_s = x @ w_dequant

        # KL loss (T=1.0)
        p_t = np.exp(logits_t - np.max(logits_t, axis=-1, keepdims=True))
        p_t /= np.sum(p_t, axis=-1, keepdims=True)
        log_p_t = np.log(np.maximum(p_t, 1e-12))

        p_s = np.exp(logits_s - np.max(logits_s, axis=-1, keepdims=True))
        p_s /= np.sum(p_s, axis=-1, keepdims=True)
        log_p_s = np.log(np.maximum(p_s, 1e-12))

        kl = float(np.sum(p_t * (log_p_t - log_p_s), axis=-1).mean())
        loss_history.append(kl)

        # Gradients
        grad_out = (p_s - p_t) / 16.0
        grad_w_dequant = x.T @ grad_out
        grad_w_g = grad_w_dequant.reshape(-1, group_size)

        if mode == "buggy":
            # Lalamo bug: stop_gradient(scales) inside round() leaves only int_codes
            grad_scale_raw = np.sum(grad_w_g * int_codes, axis=1)
        elif mode == "correct":
            # True LSQ STE: round(w/s) - w/s
            quant_error = int_codes - scaled_w
            grad_scale_raw = np.sum(grad_w_g * quant_error, axis=1)
        elif mode == "frozen":
            grad_scale_raw = np.zeros_like(curr_scales)
        else:
            raise ValueError(f"Unknown mode {mode}")

        grad_scale = (grad_scale_raw * curr_scales) if use_log_scale else grad_scale_raw

        if weight_decay > 0.0:
            grad_scale += weight_decay * scales

        # Adam update
        m_scale = b1 * m_scale + (1.0 - b1) * grad_scale
        v_scale = b2 * v_scale + (1.0 - b2) * (grad_scale ** 2)
        m_hat = m_scale / (1.0 - b1 ** step)
        v_hat = v_scale / (1.0 - b2 ** step)
        update = scale_lr * m_hat / (np.sqrt(v_hat) + eps)
        scales -= update

        if use_clamp and not use_log_scale:
            scales = np.maximum(scales, 1e-6)

    final_scales = np.exp(scales) if use_log_scale else scales
    collapsed_fraction = float(np.mean(final_scales <= scale_min_threshold))

    return {
        "mode": mode,
        "scale_lr": scale_lr,
        "weight_decay": weight_decay,
        "use_log_scale": use_log_scale,
        "use_clamp": use_clamp,
        "loss_start": round(loss_history[0], 6),
        "loss_final": round(loss_history[-1], 6),
        "scale_min": float(np.min(final_scales)),
        "scale_med": float(np.median(final_scales)),
        "scale_max": float(np.max(final_scales)),
        "collapsed_ratio": round(collapsed_fraction, 4),
        "loss_history": [round(l, 6) for l in loss_history],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run full 32-condition grid")
    parser.add_argument("--mode", choices=("buggy", "correct", "frozen"), default="buggy")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--clamp", action="store_true")
    args = parser.parse_args()

    if args.all:
        lrs = [1e-6, 1e-5, 1e-4, 1e-3]
        wds = [0.0, 1e-4]
        logs = [False, True]
        clamps = [False, True]

        results = []
        print(f"{'ID':2s} | {'Mode':7s} | {'lr':7s} | {'WD':5s} | {'Log':5s} | {'Clamp':5s} | {'Loss Init':9s} -> {'Loss Fin':9s} | {'Scale Min':9s} | {'Collapse %':10s}")
        print("-" * 90)

        idx = 1
        for lr, wd, ls, cl in itertools.product(lrs, wds, logs, clamps):
            res = run_single_simulation(mode="buggy", scale_lr=lr, weight_decay=wd,
                                        use_log_scale=ls, use_clamp=cl)
            res["condition_id"] = idx
            results.append(res)
            print(f"{idx:02d} | {'BUGGY':7s} | {lr:7.0e} | {wd:5.0e} | {str(ls):5s} | {str(cl):5s} | "
                  f"{res['loss_start']:9.6f} -> {res['loss_final']:9.6f} | {res['scale_min']:9.2e} | {res['collapsed_ratio']*100:9.1f}%")
            idx += 1

        out_path = Path("research/qad_scale_repro_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved {len(results)} results to {out_path}")
    else:
        res = run_single_simulation(mode=args.mode, scale_lr=args.lr, weight_decay=args.wd,
                                    use_log_scale=args.log_scale, use_clamp=args.clamp)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
