#!/usr/bin/env python3
"""scripts/bit_alloc.py - Mixed-Precision Layer Bit Allocation for Qwen3.5 (9B, 4B, 0.8B)

Calculates optimal 4/6/8-bit allocation under memory budget constraints using a
greedy knapsack optimization based on Hessian/trace sensitivity and condition numbers.

Usage:
  python scripts/bit_alloc.py --budget-gb 5.0 --model 9b
  python scripts/bit_alloc.py --budget-gb 5.5 --model 9b --mode full
  python scripts/bit_alloc.py --budget-gb 4.5 --model 9b --mode linear --output research/9b_bit_allocation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from typing import Literal


# Model architecture specifications
MODEL_SPECS = {
    "9b": {
        "name": "Qwen3.5-9B",
        "layers": 32,
        "hidden": 4096,
        "intermediate": 12288,
        "vocab": 248320,
        "full_attn_interval": 4,
        "tied_embedding": False,
        "gdn_in": (4096, 12352),
        "gdn_out": (4096, 4096),
        "ga_qkvg": (4096, 10240),
        "ga_out": (4096, 4096),
        "mlp_up": (4096, 24576),  # gate + up
        "mlp_down": (12288, 4096),
    },
    "4b": {
        "name": "Qwen3.5-4B",
        "layers": 32,
        "hidden": 2560,
        "intermediate": 7680,
        "vocab": 248320,
        "full_attn_interval": 4,
        "tied_embedding": True,
        "gdn_in": (2560, 7712),
        "gdn_out": (2560, 2560),
        "ga_qkvg": (2560, 6400),
        "ga_out": (2560, 2560),
        "mlp_up": (2560, 15360),
        "mlp_down": (7680, 2560),
    },
    "0.8b": {
        "name": "Qwen3.5-0.8B",
        "layers": 24,
        "hidden": 1024,
        "intermediate": 3584,
        "vocab": 248320,
        "full_attn_interval": 4,
        "tied_embedding": True,
        "gdn_in": (1024, 8224),
        "gdn_out": (2048, 1024),
        "ga_qkvg": (1024, 5120),
        "ga_out": (2048, 1024),
        "mlp_up": (1024, 7168),
        "mlp_down": (3584, 1024),
    },
}

# Quantization unit costs (bytes per parameter for asymmetric group size 32)
# unit_cost = b / 8 (payload) + 2 / 32 (FP16 scale) + 1 / 32 (INT8 zero-point) = b / 8 + 3 / 32
UNIT_COSTS = {
    4: 0.50000 + 0.09375,  # 0.59375 B/param
    6: 0.75000 + 0.09375,  # 0.84375 B/param
    8: 1.00000 + 0.09375,  # 1.09375 B/param
}

# Step size squared relative to 4-bit (uniform quantization: Delta = Range / (2^b - 1))
STEP_SQUARED_NORM = {
    4: 1.0,
    6: (15.0 / 63.0) ** 2,   # ~0.056689
    8: (15.0 / 255.0) ** 2,  # ~0.003460
}

# Structural sensitivity weights based on 0.8B calibration condition numbers and architectural roles
TYPE_WEIGHTS = {
    "lm_head": 5.0,     # Direct logit cross-entropy mapping (vocab 248,320)
    "ga_out": 3.0,      # Condition ~1.16e6; direct residual stream write; global attention
    "ga_qkvg": 2.5,     # Condition ~1.03e5; global routing query/key/value/gate
    "mlp_down": 2.5,    # Condition ~6.02e5; direct residual stream write
    "embed": 1.2,       # Lookup table only; static input representation
    "mlp_up": 1.0,      # Condition ~4.94e4; SwiGLU activation non-linearity absorbs noise
    "gdn_in": 0.8,      # Condition ~5.21e4; recurrent input projection
    "gdn_out": 0.7,     # Condition ~8.50e5; bounded linear recurrence (alpha/beta in [0, 1])
}


def get_depth_weight(layer: int, total_layers: int) -> float:
    """U-shaped depth sensitivity curve reflecting early feature extraction and late classification heads."""
    if layer < 0 or layer >= total_layers:
        return 1.0
    # Shallow layers (first 4 layers)
    if layer < 4:
        return 1.6 - 0.15 * layer
    # Deep layers (last 4 layers)
    elif layer >= total_layers - 4:
        return 1.15 + 0.15 * (layer - (total_layers - 4))
    # Middle layers
    return 1.0


@dataclass
class Component:
    name: str
    layer_idx: int
    comp_type: str
    in_dim: int
    out_dim: int
    params: int
    type_weight: float
    depth_weight: float
    sensitivity: float  # sensitivity = type_weight * depth_weight * params (proportional to trace * w)


def build_components(model_key: str, mode: Literal["full", "linear", "tied"]) -> list[Component]:
    spec = MODEL_SPECS[model_key]
    layers_count = spec["layers"]
    comps: list[Component] = []

    # Embedding & LM Head handling
    if mode == "full":
        # Untied or tied as per spec
        comps.append(
            Component(
                name="embed_tokens",
                layer_idx=-1,
                comp_type="embed",
                in_dim=spec["hidden"],
                out_dim=spec["vocab"],
                params=spec["hidden"] * spec["vocab"],
                type_weight=TYPE_WEIGHTS["embed"],
                depth_weight=1.0,
                sensitivity=TYPE_WEIGHTS["embed"] * (spec["hidden"] * spec["vocab"]),
            )
        )
        if not spec["tied_embedding"]:
            comps.append(
                Component(
                    name="lm_head",
                    layer_idx=layers_count,
                    comp_type="lm_head",
                    in_dim=spec["hidden"],
                    out_dim=spec["vocab"],
                    params=spec["hidden"] * spec["vocab"],
                    type_weight=TYPE_WEIGHTS["lm_head"],
                    depth_weight=1.0,
                    sensitivity=TYPE_WEIGHTS["lm_head"] * (spec["hidden"] * spec["vocab"]),
                )
            )
    elif mode == "tied":
        comps.append(
            Component(
                name="embed_and_head_tied",
                layer_idx=-1,
                comp_type="lm_head",
                in_dim=spec["hidden"],
                out_dim=spec["vocab"],
                params=spec["hidden"] * spec["vocab"],
                type_weight=TYPE_WEIGHTS["lm_head"],
                depth_weight=1.0,
                sensitivity=TYPE_WEIGHTS["lm_head"] * (spec["hidden"] * spec["vocab"]),
            )
        )

    # 128 (or 96) Linear layers
    for l in range(layers_count):
        is_ga = (l % spec["full_attn_interval"] == (spec["full_attn_interval"] - 1))
        dw = get_depth_weight(l, layers_count)

        # MLP Up (gate + up)
        mlp_up_in, mlp_up_out = spec["mlp_up"]
        mlp_up_params = mlp_up_in * mlp_up_out
        comps.append(
            Component(
                name=f"layers.{l}.mlp.up_proj",
                layer_idx=l,
                comp_type="mlp_up",
                in_dim=mlp_up_in,
                out_dim=mlp_up_out,
                params=mlp_up_params,
                type_weight=TYPE_WEIGHTS["mlp_up"],
                depth_weight=dw,
                sensitivity=TYPE_WEIGHTS["mlp_up"] * dw * mlp_up_params,
            )
        )

        # MLP Down
        mlp_down_in, mlp_down_out = spec["mlp_down"]
        mlp_down_params = mlp_down_in * mlp_down_out
        comps.append(
            Component(
                name=f"layers.{l}.mlp.down_proj",
                layer_idx=l,
                comp_type="mlp_down",
                in_dim=mlp_down_in,
                out_dim=mlp_down_out,
                params=mlp_down_params,
                type_weight=TYPE_WEIGHTS["mlp_down"],
                depth_weight=dw,
                sensitivity=TYPE_WEIGHTS["mlp_down"] * dw * mlp_down_params,
            )
        )

        # Mixer (GA vs GDN)
        if is_ga:
            ga_in_in, ga_in_out = spec["ga_qkvg"]
            ga_in_params = ga_in_in * ga_in_out
            comps.append(
                Component(
                    name=f"layers.{l}.mixer.qkvg_proj",
                    layer_idx=l,
                    comp_type="ga_qkvg",
                    in_dim=ga_in_in,
                    out_dim=ga_in_out,
                    params=ga_in_params,
                    type_weight=TYPE_WEIGHTS["ga_qkvg"],
                    depth_weight=dw,
                    sensitivity=TYPE_WEIGHTS["ga_qkvg"] * dw * ga_in_params,
                )
            )

            ga_out_in, ga_out_out = spec["ga_out"]
            ga_out_params = ga_out_in * ga_out_out
            comps.append(
                Component(
                    name=f"layers.{l}.mixer.out_proj",
                    layer_idx=l,
                    comp_type="ga_out",
                    in_dim=ga_out_in,
                    out_dim=ga_out_out,
                    params=ga_out_params,
                    type_weight=TYPE_WEIGHTS["ga_out"],
                    depth_weight=dw,
                    sensitivity=TYPE_WEIGHTS["ga_out"] * dw * ga_out_params,
                )
            )
        else:
            gdn_in_in, gdn_in_out = spec["gdn_in"]
            gdn_in_params = gdn_in_in * gdn_in_out
            comps.append(
                Component(
                    name=f"layers.{l}.mixer.in_proj",
                    layer_idx=l,
                    comp_type="gdn_in",
                    in_dim=gdn_in_in,
                    out_dim=gdn_in_out,
                    params=gdn_in_params,
                    type_weight=TYPE_WEIGHTS["gdn_in"],
                    depth_weight=dw,
                    sensitivity=TYPE_WEIGHTS["gdn_in"] * dw * gdn_in_params,
                )
            )

            gdn_out_in, gdn_out_out = spec["gdn_out"]
            gdn_out_params = gdn_out_in * gdn_out_out
            comps.append(
                Component(
                    name=f"layers.{l}.mixer.out_proj",
                    layer_idx=l,
                    comp_type="gdn_out",
                    in_dim=gdn_out_in,
                    out_dim=gdn_out_out,
                    params=gdn_out_params,
                    type_weight=TYPE_WEIGHTS["gdn_out"],
                    depth_weight=dw,
                    sensitivity=TYPE_WEIGHTS["gdn_out"] * dw * gdn_out_params,
                )
            )

    return comps


def greedy_allocate(
    components: list[Component],
    budget_bytes: float,
) -> tuple[dict[str, int], float, float, bool, float]:
    """Greedy knapsack bit allocation from 4-bit baseline up to 6-bit and 8-bit.

    Returns:
      (allocation_map, final_size_bytes, total_proxy_error, is_feasible, deficit_bytes)
    """
    alloc: dict[str, int] = {c.name: 4 for c in components}
    base_size = sum(c.params * UNIT_COSTS[4] for c in components)

    if base_size > budget_bytes:
        deficit = base_size - budget_bytes
        total_error = sum(c.sensitivity * STEP_SQUARED_NORM[4] for c in components)
        return alloc, base_size, total_error, False, deficit

    current_size = base_size

    while True:
        best_cand = None
        best_efficiency = -1.0

        for c in components:
            curr_b = alloc[c.name]
            next_b = 6 if curr_b == 4 else (8 if curr_b == 6 else None)
            if next_b is None:
                continue

            delta_bytes = c.params * (UNIT_COSTS[next_b] - UNIT_COSTS[curr_b])
            if current_size + delta_bytes > budget_bytes:
                continue

            delta_err = c.sensitivity * (STEP_SQUARED_NORM[curr_b] - STEP_SQUARED_NORM[next_b])
            eff = delta_err / delta_bytes

            if eff > best_efficiency:
                best_efficiency = eff
                best_cand = (c, next_b, delta_bytes)

        if best_cand is None:
            break

        c, next_b, delta_bytes = best_cand
        alloc[c.name] = next_b
        current_size += delta_bytes

    total_error = sum(c.sensitivity * STEP_SQUARED_NORM[alloc[c.name]] for c in components)
    return alloc, current_size, total_error, True, 0.0


def main():
    parser = argparse.ArgumentParser(description="Layer Bit Allocation for Qwen3.5")
    parser.add_argument("--budget-gb", type=float, required=True, help="Memory budget in GB")
    parser.add_argument("--model", choices=["9b", "4b", "0.8b"], default="9b", help="Model scale")
    parser.add_argument(
        "--mode",
        choices=["linear", "full", "tied"],
        default="linear",
        help="Allocation scope: 'linear' (128 transformer linears), 'full' (all text parameters), 'tied' (with tied embeddings)",
    )
    parser.add_argument(
        "--unit",
        choices=["decimal", "binary"],
        default="decimal",
        help="'decimal' (1 GB = 1e9 B) or 'binary' (1 GiB = 1024^3 B)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    multiplier = 1e9 if args.unit == "decimal" else float(1024**3)
    unit_str = "GB" if args.unit == "decimal" else "GiB"
    budget_bytes = args.budget_gb * multiplier

    components = build_components(args.model, args.mode)
    alloc, final_size, total_error, is_feasible, deficit = greedy_allocate(components, budget_bytes)

    total_params = sum(c.params for c in components)
    bit_counts = {4: 0, 6: 0, 8: 0}
    type_breakdown: dict[str, dict[int, int]] = {}

    for c in components:
        b = alloc[c.name]
        bit_counts[b] += 1
        if c.comp_type not in type_breakdown:
            type_breakdown[c.comp_type] = {4: 0, 6: 0, 8: 0}
        type_breakdown[c.comp_type][b] += 1

    print("=" * 70)
    print(f"Qwen3.5-{args.model.upper()} Mixed-Precision Bit Allocation")
    print("=" * 70)
    print(f"Scope Mode:       {args.mode.upper()} ({len(components)} components, {total_params:,} parameters)")
    print(f"Memory Budget:    {args.budget_gb:.2f} {unit_str} ({budget_bytes:,.0f} bytes)")
    print(f"Status:           {'FEASIBLE' if is_feasible else 'INFEASIBLE (Budget Deficit)'}")
    print(f"Estimated Size:   {final_size / multiplier:.4f} {unit_str} ({final_size:,.0f} bytes)")

    if not is_feasible:
        print(f"Deficit:          +{deficit / multiplier:.4f} {unit_str} ({deficit:,.0f} bytes) above budget")
        print(f"Note: Minimum all-4bit size exceeds budget. Consider INT2/D4S4, linear-only mode, or higher budget.")
    print(f"Estimated Error:  {total_error:.4e} (arbitrary proxy error units)")
    print("-" * 70)
    print(f"Bit Counts:       4-bit: {bit_counts[4]:2d} | 6-bit: {bit_counts[6]:2d} | 8-bit: {bit_counts[8]:2d}")
    print("-" * 70)
    print("Breakdown by Component Type:")
    for ct, counts in sorted(type_breakdown.items()):
        total_c = sum(counts.values())
        print(f"  {ct:15s} (N={total_c:2d}):  4-bit: {counts[4]:2d} | 6-bit: {counts[6]:2d} | 8-bit: {counts[8]:2d}")
    print("=" * 70)

    # Detailed export
    if args.output:
        alloc_records = []
        for c in components:
            b = alloc[c.name]
            size_b = c.params * UNIT_COSTS[b]
            alloc_records.append(
                {
                    "name": c.name,
                    "layer": c.layer_idx,
                    "type": c.comp_type,
                    "in_dim": c.in_dim,
                    "out_dim": c.out_dim,
                    "params": c.params,
                    "bits": b,
                    "size_bytes": size_b,
                    "size_gb_est": size_b / 1e9,
                    "type_weight": c.type_weight,
                    "depth_weight": c.depth_weight,
                    "sensitivity": c.sensitivity,
                    "proxy_error": c.sensitivity * STEP_SQUARED_NORM[b],
                }
            )

        output_data = {
            "metadata": {
                "model": args.model,
                "model_name": MODEL_SPECS[args.model]["name"],
                "mode": args.mode,
                "unit": args.unit,
                "budget_gb": args.budget_gb,
                "is_feasible": is_feasible,
                "deficit_gb": deficit / multiplier if not is_feasible else 0.0,
                "total_params": total_params,
                "total_size_bytes_est": final_size,
                "total_size_gb_est": final_size / multiplier,
                "total_proxy_error_est": total_error,
                "bit_counts": bit_counts,
                "type_weights": TYPE_WEIGHTS,
                "unit_costs_bytes_per_param": UNIT_COSTS,
                "formula_size": "params * (bits/8 + 3/32)",
                "formula_error": "sum(weight_type * weight_depth * params * (15 / (2^b - 1))^2)",
                "note": "All 9B numbers are estimates based on 0.8B empirical traces and theoretical formulas.",
            },
            "type_summary": {ct: counts for ct, counts in sorted(type_breakdown.items())},
            "components": alloc_records,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Exported allocation table to {args.output}")


if __name__ == "__main__":
    main()
