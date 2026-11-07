# 無検閲モデル uzu化調査結果

計測日: 2026-10-28

## 結論（先に書く）

**uzu化可能な無検閲モデルは存在する**。
ただし、全て「公式の `qwen3.5:9b-M`（投機ON・62 t/s）」より遅くなる（投機speculator無し・量子化なし）。
「速度 or 無検閲」でトレードオフ。

---

## A. uzu化可能なsafetensors候補

### 1. `nDimensional/Qwen3.5-9B-Uncensored-Safetensors` ⭐ 最有力

| 項目 | 内容 |
|---|---|
| **形式** | safetensors (シャードあり) |
| **サイズ** | ~18GB (FP16/BF16) |
| **ベース** | HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive + Qwen3.5-9B merge |
| **tags** | `uncensored`, `unfiltered`, `safetensors`, `qwen3_5` |
| **gated** | ❌ 公開 |
| **lalamo変換** | ✅ 可能。`HFQwen35Config` で読める。ModelSpecはqwen.pyに1行追加 |
| **uzu投機** | ❌ なし → 素のまま ~15 t/s |
| **現実性** | ⚠️ 変換時JAXが18GB展開 → 16GB Macはswap地獄。完了まで1-2時間 |

### 2. `GitMylo/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-safetensors`

| 項目 | 内容 |
|---|---|
| **形式** | safetensors |
| **ベース** | Qwen3.5-9B-Base をfinetune |
| **dl数** | 965 |
| **特徴** | 比較的新しい。config.json/モデル構造は標準的 |
| **lalamo変換** | ✅ 同上 |
| **投機** | ❌ なし |

### 3. `DavidAU/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED` ⭐ 品質高

| 項目 | 内容 |
|---|---|
| **形式** | safetensors (4シャード、計18.8GB) |
| **ダウンロード** | 2,984 |
| **ベース** | trohrbaugh/Qwen3.5-9B-heretic-v2 |
| **品質** | 「HighIQ」タグで推論・創作特化。Claude系SFTデータ使用 |
| **lalamo変換** | ✅ |
| **投機** | ❌ |
| **問題** | 18.8GB → 変換時メモリが厳しい |

---

## B. MLX形式（lalamoが直接変換可能かも）

lalamo READMEに `mlx-community/` のモデルがlistされている → **MLX safetensorsも変換対象**。

| モデル | サイズ | 量子化 | uzu化 |
|--------|--------|--------|-------|
| `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4` | ~5GB | 4bit | ✅ 変換時間短縮 |
| `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp8` | ~9GB | 8bit | ✅ |
| `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-bf16` | ~18GB | BF16 | ⚠️ 変換重い |
| `TheCluster/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-MLX-bf16` | ~4GB | BF16 | ✅ **最軽量** |
| `avitals/Qwen3.5-4B-Uncensored-MLX-4bit` | ~2.5GB | 4bit | ✅ 変換楽 |

**MLX版の利点**: 既に量子化済みなので、lalamo変換時のメモリ圧力が小さい。

---

## C. uzu化の手順（実践編）

lalamo は `MODEL_REPO` 文字列で変換する。ModelSpec登録されていなくても、`qwen.py` に1行追加すればOK：

```python
# lalamo/model_import/model_specs/qwen.py のQwen3.5セクションに追加
LanguageModelSpec(
    vendor="Custom",
    family="Qwen3.5",
    name="Qwen3.5-9B-Uncensored-Custom",
    size="9B",
    origin=HuggingFaceOrigin(repo="nDimensional/Qwen3.5-9B-Uncensored-Safetensors"),
    config_type=HFQwen35Config,
    response_format=QWEN_RESPONSE_FORMAT,
    reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    configs=ConfigMap(
        generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
    ),
),
```

→ `uv run lalamo convert nDimensional/Qwen3.5-9B-Uncensored-Safetensors` で変換開始。

---

## D. 実行可能性マトリクス（M3 16GB）

| モデル | 変換時間 | 実行速度 | 投機 | 実用性 |
|--------|---------|---------|------|--------|
| `trymirai/Qwen3.5-9B-M` (公式) | 不要 | **62 t/s** | ✅ | ⭐⭐⭐ |
| `nDimensional/...safetensors` (FP16) | 1-2h | **15 t/s** | ❌ | ⭐⭐ |
| `TheCluster/...MLX-mxfp4` | 20-30min | **18-20 t/s** | ❌ | ⭐⭐⭐ |
| `TheCluster/...2B-MLX-bf16` | 10-15min | **50+ t/s** | ❌ | ⭐⭐⭐ (小さいが無検閲) |

---

## E. 推奨アクション

** speed優先 ** → 公式 `qwen3.5:9b-M` のまま(62 t/s)。無検閲版はOllamaで `huihui_ai/qwen3.5-abliterated:9b` を併用。

** uzuで無検閲を試したい ** → 
1. `TheCluster/Qwen3.5-2B-Uncensored...MLX-bf16` を試す（変換15分で50+ t/s）
2. その後 `nDimensional/9B-safetensors` を変換（1-2hかかるが9Bが使える）

** 今すぐやる？ **
- `A` → 2B MLX-bf16をDL→変換→計測（15分〜30分）
- `B` → 9B MLX-mxfp4をDL→変換（30分〜1時間）
- `C` → 調査のみで終了（後日別タイミングで）
