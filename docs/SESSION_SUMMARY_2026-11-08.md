# セッションまとめ（2026-11-08）

対象: Huihui Qwen3.5（0.8B 中心 / 4B・9B は方針決め）。
main = `53d1a1b0`、テスト **435 passed**（エンジン 434 + bench 1）。

前日までの経緯は `docs/SESSION_SUMMARY_2026-11-07.md`、数値の正本は
`docs/0.8B_RESULTS.md` と `docs/STATUS.md`。

---

## 1. 確定した結果（0.8B / 100 chunks / ΔBPC vs BF16）

BF16 baseline は全条件で EN 35.36 / JA 43.43。

| モデル | EN | JA | 速度（M3） | uzu で動く |
|---|---:|---:|---:|---|
| 公式 `Qwen3.5-0.8B-M` | +1.21% | +3.83% | 142.9 t/s | ✅ |
| wide8 + emb4 **asymmetric** int8 | **+1.13%** | **+1.24%** | 未測定 | ❌（未実装だった） |
| wide8 + emb4 **symmetric** int8 | +1.73% | +2.10% | 124.4 t/s | ✅ |
| wide8 + emb4 symmetric + damping 1.0 | +1.80% | +1.93% | 未測定 | ✅ |
| down-out8 + emb4 asym（d1e-1） | +1.39% | +1.70% | — | ❌ |
| down-out8 + emb4 asym（d1e0） | +1.27% | +1.49% | — | ❌ |
| uniform4 + emb4 | +1.99% | +2.26% | 146.1 t/s | ✅ |

**読み方**

- **日本語では我々が公式を 1.7pp 上回る**（runnable 同士: 3.83 vs 2.10）
- **英語では公式が上**（1.21 vs 1.73）。差 0.52pp は seed 幅 0.44pp を超える
- asymmetric の 1.13 / 1.24 は **uzu が動かせなかった数字**。symmetric にすると +0.6/+0.86pp 悪化する

### wide8 の seed（asymmetric）

| seed | EN | JA |
|---|---:|---:|
| 0 | +1.13% | +1.24% |
| 1 | +0.69% | +1.22% |
| 2 | +1.32% | +1.39% |
| 平均 | +1.05% | +1.28% |

JA は 3 seed とも down-out8（+1.70%）を下回る。EN は seed 幅内で有意でない。

### embedding の値段（down-out8 + d1e-1 ベース）

| embedding | EN | JA |
|---|---:|---:|
| BF16 | 1.39% | 1.70% |
| int8 | 1.38% | 1.73% |
| int4 | 2.02% | 2.56% |

emb8 はタダ。emb4 は EN +0.64 / JA +0.83pp。

### QAD（Muse / 最弱 student / 50 step）

EN +7.51 → +6.06（−1.46pp）、JA +8.34 → +7.46（−0.88pp）。
BF16 基準は完全再現。**弱い較正を QAD が補う**方向。強い student での効果は別途。

## 2. 速度（M3 16GB / 512 tokens / 2 runs）

| モデル | メモリ | Generate |
|---|---:|---:|
| 公式 0.8B-M | 0.521GB | 142.9 t/s |
| uniform4 + emb4 | 0.521GB | 146.1 t/s |
| down-out8 sym | 0.566GB | 128.4 t/s |
| wide8 sym | 0.597GB | 124.4 t/s |

9B（公式モデルを target にした draft の比較）:

| 構成 | メモリ | Generate |
|---|---:|---:|
| 投機なし | 5.00GB | 16.2 t/s |
| + 公開 abliterated draft（5層 / bf16） | 7.05GB | 41.2 t/s |
| + 同 draft を 4bit 化（632MB） | 5.60GB | 49.0 t/s |
| + 公式 draft（8層 / Weaver / 4bit） | 5.74GB | 63.4 t/s |

**draft を 4bit にするだけで +19% と −1.45GB。** 公式との差は Weaver の木・層数・量子化。

## 3. 見つけて直したバグ

| バグ | 影響 | 修正 |
|---|---|---|
| 8bit を asymmetric で export | uzu は 4bit 以外の zero-point を未コンパイル → **wide8 は一度も起動しなかった** | `--symmetric-bits`（既定 8） |
| QAD export に `encoding.json` が無い | 学習後のモデルが開けない | export で student から複製 |
| eval copy が古い `.bak` を優先 | 「after」のつもりで before を測る | stamp 方式 |
| `.bak` の鮮度判定が逆 | 常に stale と誤判定 | 同上 |
| kit の pin が source commit | `Revision not found` で全カーネル失敗 | HF dataset の sha を記録 |
| FSDP を 0.8B で使用 | サイズ 1 の配列が割れず即死 | TP に戻す |
| int8 ZP のカーネル | decode が 4bit 固定、zp 詰め幅固定、split-k で bias 二重計上 | 3 箇所修正 |

## 4. エンジン拡張（int8 zero-point GEMM）

asymmetric int8 を uzu で動かすための改修。

| 変更 | 内容 |
|---|---|
| `gemm.metal` | ZP を int8 でも許可（A が int8 のときは除外） |
| `gemv.metal` | ZP を 4/8 bit に |
| `quant_unpack.h` | `decode_zero_point<8>`（1 byte = 1 個） |
| `operands.h` | `ZERO_POINTS_PER_BYTE` を `8 / BITS` に |
| `selection.rs` | (ZP, 8) を symmetric と同じ tile に。**split-k は 1 固定**（bias 二重計上の回避） |

- parity テスト 14 件 + matmul 204 件 + 全ユニット **434 passed**
- **まだ実モデルで走らせていない**（Kaggle でビルド中）

## 5. 投機（speculator）の調査

- 公式 speculator = **DFlash 8 層 + Weaver**（755MB）。draft は本体の層 1,5,...,29 を読む
- 公開 draft を全件棚卸し: 5 層（guglxni / unieai / blaj）と 6 層（z-lab / modal-labs）。**9B 用 Weaver は非公開**
- 実測: 5 層 4bit（49.0）は 6 層（46.2）より速い
- `lalamo speculator convert` で uzu 形式に変換可能。**`shapes.json` は自前で用意が必要**
- `scripts/quantize_speculator.py` を追加（2.18GB → 632MB、22 秒）
- **自前学習は必須**（無検閲モデルなので公式 draft の教師分布が違う）

## 6. 再現性の整備

| 追加 | 内容 |
|---|---|
| `docs/REPRODUCE.md` | 実験ごとの再現コマンドと、自動化していないものの一覧 |
| `scripts/make_kaggle_kernel.py` | lane spec からカーネル生成（kit の版を固定） |
| `scripts/eval_external_model.py` | 変換せずに既存モデルを測る |
| `scripts/bench_protocol.py` | 負荷ゲート + 中央値（load が高ければ測らない） |
| `lanes/sym8-*.json` / `asym8-*.json` | 今日の実験を spec 化 |
| `docs/ENGINE_INT8_ZP.md` | エンジン拡張の設計と現状 |

## 7. 実行状態（終了時点）

| レーン | 状態 |
|---|---|
| `uzu-asym8-wide8`（Kaggle CPU） | RUNNING（wide8 asymmetric の実モデル） |
| アングラ prefill 分析 | 実行中 |
| アングラ split-k 設計 | 実行中 |
| QAD v6（Muse / GPU） | 依頼済み |
| Kaggle GPU クォータ | **週次 30h を使い切り**（週明けまで不可） |
| ローカル bench | **不可**（別プロジェクト `crypPI` が CPU/GPU を占有、load 40〜50） |

## 8. 次にやること

1. `uzu-asym8-wide8` を uzu で**ロード・生成の正しさ**確認（速度は後回し）
2. アングラ 2 件の成果物を検証（prefill の理論値、split-k の diff）
3. QAD v6 の結果で「強い student に効くか」を判定
4. 4B を 1 条件（g64 / emb8 / wide8 / d1e0）— Modal
5. 9B 本体（g64 / wide8 / emb8）— Modal
6. 公開 draft を Huihui 9B で fine-tune（5層 / Argmax / 4bit）
7. bench は `docs/REPRODUCE.md` の手順で、負荷ゲートを通してから

## 9. 判断が変わったこと

- **「公式より少し遅いが品質は上」は JA のみ**。EN は公式が上
- **8bit 配分の価値は小さい**: runnable 比較で uniform4 との差は EN 0.26 / JA 0.16pp、速度 −15%
- **エンジン拡張で 0.6pp 戻せる**見込み（asymmetric int8）
- **draft の改善が速度の本命**（エンジンではなく重み）
- **T4 で draft 学習は不可**（flash-attn が入らない、bf16 0.25 TFLOP/s）
