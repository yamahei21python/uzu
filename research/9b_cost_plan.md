# 9B 変換のコスト試算（Modal / ローカル）

最終更新: 2026-11。価格は Modal 公式（2026-11 取得）。

## 1. 単価

| 資源 | 単価 |
|---|---:|
| CPU | $0.0000131 / core / s = **$0.04716 / core-h** |
| Memory | $0.00000222 / GiB / s = **$0.007992 / GiB-h** |
| A100 80GB | $0.000694 / s = **$2.4984 / h** |
| A100 40GB | $0.000583 / s = **$2.0988 / h** |
| H100 80GB | $0.001097 / s = **$3.9492 / h** |
| A10 24GB | $0.000306 / s = **$1.1016 / h** |
| T4 16GB | $0.000164 / s = **$0.5904 / h** |
| Egress | $0.04 / GiB（Starter は 1TiB/月まで無料） |
| Starter 無料枠 | **$30 / 月** |

## 2. 9B の前提

| 項目 | 値 |
|---|---:|
| BF16 重み | 16.68 GiB |
| FP32 重み | 33.36 GiB |
| INT4 text-only 重み | 5.04 GB |
| INT4 + lm_head INT8 等（推奨 B） | 6.99 GB |
| H_I（96 Linear 合計） | 12.59 GiB |
| H_O（全層） | 48.60 GiB |
| 推論時（uzu, 32k） | 10.72〜12.01 GB |

## 3. シナリオ別コスト

### A. ローカルでストリーム RTN 変換

- 入力: HF から層単位でストリーム（18GB を保存しない）
- 出力: INT4 5〜7GB
- ピーク RAM: 約 2.5GB
- 時間: 25〜40 分
- コスト: **$0**
- 必要条件:
  - 空きディスク 7GB 以上（現在 11GB）
  - ストリーム変換スクリプト（未実装）
- リスク: ディスク不足、safetensors の層分割実装

### B. Modal CPU で RTN 変換

- 16 cores / 64 GiB / 40 分
- CPU: 16 × 2400 × 0.0000131 = **$0.503**
- Mem: 64 × 2400 × 0.00000222 = **$0.341**
- 合計: **約 $0.84**
- 入力は Modal 内で HF から取得（アップロード不要）
- 出力 6GB の egress は無料枠内
- 実装: lalamo を Modal image に入れるだけ

### C. Modal A100 80GB で YAQA H_I（層分割）

- model f32 36GB + grads + H + activation は 1 回で 80GB を超える
- 4層ずつ 8 グループ、16 samples、512 token
- 8 グループ × 16 = 128 backward
- 想定 1.5〜2.5 時間
- A100 80GB 2h: 2 × 2.4984 = **$5.00**
- host CPU 8 cores / 64 GiB 2h: 8×7200×0.0000131 + 64×7200×0.00000222 = **$1.78**
- 合計: **約 $6.8**
- 出力 6GB egress: 無料枠内

### D. Modal H100 80GB で YAQA H_I

- 1.5 時間想定
- H100: 1.5 × 3.9492 = **$5.92**
- host: **$1.33**
- 合計: **約 $7.3**
- A100 より速いが、コストは同等〜やや高い

### E. Modal CPU 128GB / 64 cores で YAQA

- 4 時間想定
- CPU: 64 × 14400 × 0.0000131 = **$12.07**
- Mem: 128 × 14400 × 0.00000222 = **$4.09**
- 合計: **約 $16.2**
- GPU より遅く高い。非推奨

### F. QAD（必要になった場合）

- A100 80GB、200〜1000 step、batch 2〜4、seq 512
- rollout 生成 + 蒸留で 1〜3 時間
- **$5〜15** 程度（未検証）

## 4. 推奨パイプライン（最安）

1. **ローカル CPU（$0）**
   - config / tokenizer / calibration corpus / shape 検査
   - 評価ハーネスと比較スクリプト
   - Modal へ送るのは **スクリプトと corpus のみ（数MB）**
2. **9B RTN**
   - 第一候補: ローカルストリーム変換（$0、要実装・要ディスク）
   - 次点: Modal CPU 64GB（**$0.84**、実装容易）
3. **RTN 9B を評価**
   - KL / PPL / 指示追従。許容ならここで終了
4. **不足なら YAQA H_I**
   - Modal A100 80GB、層分割（**約 $6.8**）
5. **さらに不足なら QAD**
   - Modal A100 80GB（**$5〜15**）

**合計見込み: $0〜8（QAD 込みで $15前後）。Starter の $30 無料枠でほぼ賄える。**

## 5. アップロード最小化

| 送るもの | サイズ | 方法 |
|---|---:|---|
| 変換スクリプト | < 1MB | Modal image / mount |
| calibration corpus | < 5MB | Volume |
| HF token | secret | Modal Secret |
| 9B BF16 重み | 0 | Modal 内で HF から取得 |
| 出力 INT4 | 5〜7GB | ダウンロード（egress 無料枠） |

## 6. 注意

- Modal の CPU は「physical core = 2 vCPU」。コア数指定に注意。
- 非 preemptible は 3x 価格。長時間 YAQA は preemptible を検討。
- リージョン指定は 1.15〜1.75x。
- volume は $0.09/GiB/月（1TiB まで無料）。
- 18GB の BF16 をローカルからアップロードしない。HF から直接取得する。

## 7. 計算資源の優先順位（2026-11 決定）

1. **ローカル M3 16GB（最優先）**
   - 0.8B: RTN / YAQA / QAD の小規模検証、logits / PPL / 生成評価
   - 4B: RTN 変換、評価
   - 制約: swap 6.7GB、GPU wired limit ~10.7GB。9B の f32 校正は不可

2. **Kaggle（無料・次点）**
   - CPU カーネル: **GPU クォータ非消費**、32GB RAM。4B YAQA（bf16）、9B の層分割校正
   - GPU カーネル: T4 x2（16GB × 2）、30h/週。4B YAQA、0.8B QAD、9B 層分割
   - 制約: 1セッション12h、GPU 同時1、週次リセット（09:00 JST）
   - 2026-11-07 時点: GPU 残 0h（30.27/30h）、09:00 JST 回復

3. **Modal（大規模のみ）**
   - 9B の全層 YAQA / QAD、Kaggle で収まらないとき
   - A100 80GB: 約 $6.8/run（Starter $30 無料枠）
   - 18GB BF16 をアップロードしない。HF から Modal 内で取得

### 使い分けの目安

| 作業 | 第一候補 | 代替 |
|---|---|---|
| 0.8B PTQ/評価 | ローカル | Kaggle CPU |
| 0.8B QAD | Kaggle T4 | Modal A10 |
| 4B RTN/評価 | ローカル | Kaggle CPU |
| 4B YAQA | Kaggle CPU/GPU (32GB) | Modal A100 40GB |
| 9B RTN | ローカル stream（要実装） | Modal CPU |
| 9B YAQA | Kaggle 層分割 (32GB) | Modal A100 80GB |
| 9B QAD | Modal A100 80GB | – |

## 8. Kaggle カーネルの RAM 制限（訂正）

| カーネル | RAM | GPU | 4B YAQA |
|---|---:|---|---|
| CPU のみ | **16GB** | なし | ✕（model 8.4 + grads 8.4 > 16） |
| GPU T4 x2 | **29〜32GB** | 15.6GB × 2 | ○（TP で ~12GB/GPU） |

- Kaggle CPU カーネルで 4B 全層・4層グループを試したが **SIGKILL (OOM)**
- CPU 32GB という前提は誤り。32GB は GPU カーネル側
- 4B YAQA は GPU カーネル `uzu-quant-yaqa-4b-gpu` で実行する
  - `--parallel tensor`、`--calib-dtype bfloat16`
  - GPU クォータは 2026-11-07 09:00 JST に回復
- CPU カーネルは 0.8B 向け（model 1.6 + grads 1.6 + H 1 ≒ 4GB）

## 9. Kaggle T4 x2 のメモリ解釈（重要）

- Kaggle `NvidiaTeslaT4` = **2 × T4、各 15.6GB、合計 31.3GB**
- DDP / データ並列はモデルを各 GPU に複製 → **1枚あたり 15.6GB が上限**
- 2枚をメモリプールとして使うには **tensor parallel / model parallel / FSDP** が必要
- kaggleAI の vLLM は TP=2 で2枚を活用済み
- 我々の JAX calibration は `ShardingConfig.replicated()` → 現状は1枚分のみ

| 作業 | 1枚 | 2×T4 tensor parallel（実装済み） | 備考 |
|---|---|---|---|
| 0.8B YAQA | ○ ~10GB | ◎ ~3GB/GPU | 2枚は速度目的 |
| 0.8B QAD | ○ ~13GB | ◎ ~7GB/GPU | DDP/TP |
| 4B RTN | GPU不要 | – | CPU/disk |
| 4B YAQA | ✕ ~19GB | ○ ~12GB/GPU | 4層グループ不要、全層可 |
| 4B QAD | ✕ | △ ~17GB/GPU | gradient checkpointing 次第 |
| 9B RTN | ✕ 18GB | ○ ~12GB/GPU | 重み9GB + 出力2.5GB |
| 9B YAQA | ✕ | ○ 4層×8 + TP ~11GB/GPU | 時間は長い |
| 9B QAD | ✕ | ✕ | Modal A100 |

→ `--parallel tensor` で 2枚を1つのメモリプールとして使える。
→ 4B YAQA / 9B RTN / 9B YAQA（層分割）は 2×T4 で実行可能。
→ 4B QAD / 9B QAD は依然として A100 が必要。
