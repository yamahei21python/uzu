#!/usr/bin/env python3
"""bit_alloc_proxy.py

0.8B の calibration.json から Hessian trace ベースの量子化誤差代理指標 (proxy) を計算し、
site ランキング（上位/下位）と、実測検証実験用の --layer-bits-json 設定を出力するスクリプト。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


# ステップ幅の二乗の相対比 (4bit を 1.0 とする)
# 4bit: 2^-8 = 1/256
# 6bit: 2^-12 = 1/4096 (4bit 比 1/16 = 0.0625)
# 8bit: 2^-16 = 1/65536 (4bit 比 1/256 = 0.00390625)
STEP2_RATIO = {
    4: 1.0,
    6: 1.0 / 16.0,
    8: 1.0 / 256.0,
}


def compute_site_proxy(
    layer: dict[str, Any],
    bits: int = 4,
    metric: str = "trace",
) -> tuple[float, float]:
    """site ごとの代理誤差とソートキーを計算する。

    metric:
      - 'trace': input_trace (タイブレークに input_condition を使用)
      - 'frob': input_trace * out_dim
      - 'cond': input_condition
      - 'trace_log_cond': input_trace * log10(input_condition)
    """
    trace = float(layer.get("input_trace", 0.0))
    out_dim = int(layer.get("out_dim", 1))
    cond = float(layer.get("input_condition", 1.0))
    step2 = STEP2_RATIO.get(bits, 1.0)

    if metric == "trace":
        # 主スコア: trace, タイブレーク: cond
        base_score = trace
        sort_key = (round(trace, 1), cond)
    elif metric == "frob":
        base_score = trace * out_dim
        sort_key = (base_score, cond)
    elif metric == "cond":
        base_score = cond
        sort_key = (cond, trace)
    elif metric == "trace_log_cond":
        import math
        base_score = trace * math.log10(max(cond, 1.0))
        sort_key = (base_score, cond)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return base_score * step2, sort_key


def calculate_weight_bytes(in_dim: int, out_dim: int, bits: int, group_size: int = 32) -> tuple[int, int]:
    """重みのバイト数を計算する (オーバーヘッド込み, 純粋重み)。
    group 32 非対称量子化:
      - 重み: bits / 8 バイト/パラメータ
      - scale (fp16, 2B) + zero (fp16, 2B) = 4B per 32 params = 1.0 bit/param (0.125 B/param)
    """
    params = in_dim * out_dim
    overhead_bits_per_param = 32.0 / group_size  # 1.0 bit
    total_bytes = int(params * (bits + overhead_bits_per_param) / 8.0)
    pure_bytes = int(params * bits / 8.0)
    return total_bytes, pure_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="0.8B bit allocation proxy calculator")
    parser.add_argument(
        "--calibration-json",
        type=Path,
        required=True,
        help="Path to clean-enonly calibration.json",
    )
    parser.add_argument(
        "--metric",
        choices=["trace", "frob", "cond", "trace_log_cond"],
        default="trace",
        help="Sensitivity metric (default: trace)",
    )
    parser.add_argument(
        "--num-promoted",
        type=int,
        default=24,
        help="Number of sites to promote to 6bit (default: 24)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for control condition (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/bit_alloc_configs"),
        help="Directory to dump layer-bits-json files",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="Optional path to output validation plan JSON",
    )
    args = parser.parse_args()

    if not args.calibration_json.exists():
        raise SystemExit(f"File not found: {args.calibration_json}")

    with open(args.calibration_json, encoding="utf-8") as f:
        calib_data = json.load(f)

    layers = calib_data.get("layers", [])
    if not layers:
        raise SystemExit("No layers found in calibration.json")

    print(f"Loaded {len(layers)} sites from {args.calibration_json}")
    print(f"Sensitivity Metric: {args.metric}")
    print(f"Sites to promote to 6bit: {args.num_promoted} / {len(layers)}")

    # 各 site の基本感度スコア (4bit 基準)
    for l in layers:
        score, sort_key = compute_site_proxy(l, bits=4, metric=args.metric)
        l["sensitivity_score"] = score
        l["sort_key"] = sort_key

    # ランキングソート
    sorted_layers = sorted(layers, key=lambda x: x["sort_key"], reverse=True)

    print("\n" + "=" * 80)
    print(f"=== TOP 10 SITES (Most Sensitive -> Priority for 6bit) ===")
    print("=" * 80)
    print(f"{'Rank':<5} {'Path':<55} {'Score':<10} {'Trace':<8} {'Cond':<10} {'in x out'}")
    print("-" * 80)
    for i, l in enumerate(sorted_layers[:10]):
        print(
            f"{i+1:<5} {l['path']:<55} {l['sensitivity_score']:<10.1f} "
            f"{l['input_trace']:<8.1f} {l['input_condition']:<10.1f} "
            f"{l['in_dim']}x{l['out_dim']}"
        )

    print("\n" + "=" * 80)
    print(f"=== BOTTOM 10 SITES (Least Sensitive -> Keep 4bit) ===")
    print("=" * 80)
    print(f"{'Rank':<5} {'Path':<55} {'Score':<10} {'Trace':<8} {'Cond':<10} {'in x out'}")
    print("-" * 80)
    for i, l in enumerate(sorted_layers[-10:]):
        rank = len(sorted_layers) - 10 + i + 1
        print(
            f"{rank:<5} {l['path']:<55} {l['sensitivity_score']:<10.1f} "
            f"{l['input_trace']:<8.1f} {l['input_condition']:<10.1f} "
            f"{l['in_dim']}x{l['out_dim']}"
        )

    # 4条件の定義
    # 条件 1: uniform4 (全層 4bit)
    # 条件 2: proxy_top (上位 num_promoted を 6bit)
    # 条件 3: proxy_bottom (下位 num_promoted を 6bit)
    # 条件 4: random (ランダム num_promoted を 6bit)

    top_paths = set(l["path"] for l in sorted_layers[: args.num_promoted])
    bottom_paths = set(l["path"] for l in sorted_layers[-args.num_promoted :])

    rng = random.Random(args.random_seed)
    random_sample = rng.sample(layers, args.num_promoted)
    random_paths = set(l["path"] for l in random_sample)

    conditions = [
        {
            "id": "cond1_uniform4",
            "label": "Uniform 4bit (基準)",
            "description": "全 96 site を 4bit で量子化。ベースライン。",
            "promoted_paths": set(),
        },
        {
            "id": f"cond2_proxy_top{args.num_promoted}_6b",
            "label": f"Proxy 上位 {args.num_promoted} site 6bit (敏感層昇格)",
            "description": f"proxy 指標上位 {args.num_promoted} site を 6bit に昇格。最も PPL 改善が期待される。",
            "promoted_paths": top_paths,
        },
        {
            "id": f"cond4_random{args.num_promoted}_6b",
            "label": f"ランダム {args.num_promoted} site 6bit (対照実験)",
            "description": f"シード {args.random_seed} で無作為に {args.num_promoted} site を 6bit に昇格。",
            "promoted_paths": random_paths,
        },
        {
            "id": f"cond3_proxy_bottom{args.num_promoted}_6b",
            "label": f"Proxy 下位 {args.num_promoted} site 6bit (鈍感層昇格)",
            "description": f"proxy 指標下位 {args.num_promoted} site を 6bit に昇格。改善は最小にとどまるはず。",
            "promoted_paths": bottom_paths,
        },
    ]

    base_proxy = sum(compute_site_proxy(l, bits=4, metric=args.metric)[0] for l in layers)

    # 実験サマリ出力
    print("\n" + "=" * 95)
    print(f"=== 4 EXPERIMENTAL CONDITIONS EVALUATION (0.8B) ===")
    print("=" * 95)
    print(
        f"{'Condition':<26} {'Proxy Error':<14} {'Rel to 4b':<12} "
        f"{'4b/6b Sites':<12} {'Linear MB (inc ovhd)':<20} {'Pure W MB'}"
    )
    print("-" * 95)

    plan_conditions = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        promoted = cond["promoted_paths"]
        cond_proxy = 0.0
        total_bytes = 0
        pure_bytes = 0
        rules = []

        # rht_int4_convert.py 用のルール作成
        # site パスは .decoder/... なので、先頭ドットを除去した文字列で完全一致または部分一致を作る
        for l in layers:
            p = l["path"]
            clean_path = p.lstrip(".")
            bits = 6 if p in promoted else 4
            score, _ = compute_site_proxy(l, bits=bits, metric=args.metric)
            cond_proxy += score
            tb, pb = calculate_weight_bytes(l["in_dim"], l["out_dim"], bits)
            total_bytes += tb
            pure_bytes += pb
            if bits == 6:
                # rht_int4_convert.py は match in path で判定するため、一意な match 文字列を生成
                rules.append({"match": clean_path, "bits": 6})

        rel_error = (cond_proxy / base_proxy) * 100.0
        delta_error = rel_error - 100.0
        n_4b = len(layers) - len(promoted)
        n_6b = len(promoted)
        total_mb = total_bytes / 1e6
        pure_mb = pure_bytes / 1e6

        print(
            f"{cond['id']:<26} {cond_proxy:<14.1f} {rel_error:>5.1f}% ({delta_error:>+5.1f}%) "
            f"{n_4b:>2} / {n_6b:<2}        {total_mb:>6.2f} MB            {pure_mb:>6.2f} MB"
        )

        # layer-bits-json ファイルの保存
        layer_bits_spec = {
            "default": 4,
            "rules": rules,
        }
        spec_file = args.output_dir / f"{cond['id']}_layer_bits.json"
        with open(spec_file, "w", encoding="utf-8") as f:
            json.dump(layer_bits_spec, f, indent=2, ensure_ascii=False)

        plan_conditions.append(
            {
                "condition_id": cond["id"],
                "label": cond["label"],
                "description": cond["description"],
                "layer_bits_spec_file": str(spec_file),
                "layer_bits_rules_count": len(rules),
                "rules": rules,
                "proxy_error": round(cond_proxy, 2),
                "proxy_error_relative_percent": round(rel_error, 2),
                "proxy_error_delta_percent": round(delta_error, 2),
                "sites_4bit": n_4b,
                "sites_6bit": n_6b,
                "linear_weight_mb_total": round(total_mb, 2),
                "linear_weight_mb_pure": round(pure_mb, 2),
            }
        )

    print("-" * 95)
    print("判定式 (Proxy 正当性基準):")
    print("  期待される品質 (PPL / BPC 劣化量):")
    print(f"    PPL(cond2_proxy_top) < PPL(cond4_random) < PPL(cond3_proxy_bottom) < PPL(cond1_uniform4)")
    print(f"  期待される proxy 誤差:")
    print(
        f"    E(cond2: {plan_conditions[1]['proxy_error_relative_percent']:.1f}%) < "
        f"E(cond4: {plan_conditions[2]['proxy_error_relative_percent']:.1f}%) < "
        f"E(cond3: {plan_conditions[3]['proxy_error_relative_percent']:.1f}%) < "
        f"E(cond1: 100.0%)"
    )

    if args.plan_json:
        plan_doc = {
            "title": "0.8B Bit Allocation Proxy Validation Plan",
            "metric": args.metric,
            "base_proxy_error": round(base_proxy, 2),
            "total_sites": len(layers),
            "num_promoted": args.num_promoted,
            "random_seed": args.random_seed,
            "criteria_formula": "PPL(cond2_top) < PPL(cond4_random) < PPL(cond3_bottom) < PPL(cond1_uniform4)",
            "conditions": plan_conditions,
        }
        args.plan_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.plan_json, "w", encoding="utf-8") as f:
            json.dump(plan_doc, f, indent=2, ensure_ascii=False)
        print(f"\nSaved validation plan JSON to {args.plan_json}")


if __name__ == "__main__":
    main()
