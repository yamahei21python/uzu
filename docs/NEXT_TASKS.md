# 今後のタスク（2026-11-09 更新 / 0.8B クローズ）

優先度順。各タスクに「何が決まるか」「どこで回すか」「完了条件」を書く。

## P0: 0.8B は完了（2026-11-09）

**最終構成: wide8 + emb8 + damping 1.0（symmetric）**
EN 1.11% / JA 1.16% / 113.8 t/s / 0.615GB（4 seed、`docs/0.8B_RESULTS.md` §13）

| 決定 | 内容 |
|---|---|
| 8bit 配分 | **採用**（uniform4 比 EN 1.0pp / JA 1.5pp 改善） |
| embedding | **8bit**（4bit は大幅劣化） |
| asymmetric int8 | **不採用**（symmetric と差なし） |
| damping | 1.0（JA 優先） |
| QAD | **保留**（emb8 で天井が上がったので再評価は後） |

## P0.5: 旧・結果待ち（残り）

| # | タスク | 状態 | 何が決まるか |
|---|---|---|---|
| 1 | `uzu-asym8-wide8-f32`（Kaggle CPU） | 実行中 | **8bit 配分を採るか捨てるか** |
| 2 | QAD v2（Muse / GPU, muon + data） | QUEUED | **QAD が強い student に効くか** |
| 3 | MXU sk=4 の実行確認 | サブ委任 | M5 で int8 ZP が正しいか |
| 4 | F32 スクラッチの性能測定 | サブ委任 | split-k 復活のコスト |

### 1 の判定

- `1.13 / 1.24` 前後 → 8bit を**採用**（uniform4 比 EN 0.86 / JA 1.02pp の利得）
- `1.6 / 2.1` 前後 → 8bit を**捨てる**（利得 0.4 / 0.2pp では −12% に見合わない）

### 2 の判定

- EN 1.5% 以下 → QAD を全モデルに適用
- 変化なし → QAD は弱い較正専用と結論

## P1: 次に回す（GPU クォータ回復後 = 週明け）

| # | タスク | 場所 | 完了条件 |
|---|---|---|---|
| 5 | **4B を 1 条件**（g64 / emb8 / wide8 / d1e0）+ eval | Modal A100 | 設定が 4B でも効く |
| 6 | **9B 本体**（g64 / wide8 / emb8）+ eval | Modal A100 | EN/JA の ΔBPC と t/s |
| 7 | 9B の emb4 vs emb8（readout のみ 4bit 等） | Modal | 速度と品質の釣り合い |

4B は「設定の妥当性確認」のみ。seed は振らない。

## P2: draft（速度の本命）

| # | タスク | 場所 | 前提 |
|---|---|---|---|
| 8 | プロンプト 1.9 万本 → 応答生成（7.6M tok） | Modal A100 + vLLM | `data/mtp_prompts/` 完成済み |
| 9 | 公開 draft を我々の 9B で fine-tune（5層 / 4bit） | Modal or Kaggle GPU | 8 のデータ |
| 10 | 受理長 τ を計測（カテゴリ別） | ローカル Mac | draft と 9B 本体 |
| 11 | Weaver 学習（57M / top-64 候補） | **Kaggle T4 可** | 9 の τ が伸びてから |

順序厳守: **draft が先、Weaver は後**。τ が 2.5 未満なら Weaver に進まない。

## P2.5: 27B の DFlash（原因確定済み）

受理 0 は vLLM 0.32.0 の回帰（PR #60692）。詳細と切り分けは `docs/VLLM_DFLASH_032.md`。

| # | タスク | 場所 | 完了条件 |
|---|---|---|---|
| 12 | `PIECEWISE` + `--no-enable-prefix-caching` で 1 本 | Muse | 受理率 >20% なら原因確定 |
| 13 | `patches/vllm-0320-dflash-kernel-block-table.patch` を当てたイメージ | Muse / Modal | FULL グラフのまま受理 >0.3 |

MTP（31.3 t/s）は影響なし。サービスは MTP で維持する。

## P3: 整備

| # | タスク | 内容 |
|---|---|---|
| 12 | kit の再公開 | `make_kaggle_kernel` / `build_mtp_prompts` / `eval_external_model` を含める |
| 13 | `docs/0.8B_RESULTS.md` の最終表 | 1〜2 の結果を反映 |
| 14 | `run_lane` の passthrough にテスト | `symmetric_bits` / `layer_bits_json` が未固定 |

## 制約（全タスク共通）

| 制約 | 内容 |
|---|---|
| GPU クォータ | 週 30h。使い切ると push 不可（現在は回復待ち） |
| Modal | **最終手段**。Kaggle で回せる形を先に探す |
| ローカル | 重い処理を回さない。uzu の bench は 1〜2 分のみ、`bench_protocol` の負荷ゲートを通す |
| dtype | 評価は **f32 に統一**（bf16 と混ぜると比較できない。今回踏んだ） |
| kit | カーネルは **HF dataset の commit** で pin する |

## やらないこと

| 項目 | 理由 |
|---|---|
| RHT block 64/128 | 次元が割れず変換不能（`8224 = 32×257`） |
| prefill 最適化 | 既に fp16 ピークの 50%。残りは +20% が上限 |
| Weaver をゼロから | 57M のアダプタなので、draft の後に回す |
| speculator の 9B 自前学習を急ぐ | まず draft の fine-tune |
