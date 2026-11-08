# 実験の再現手順（2026-11-08 時点）

「何をどう回したら同じ数字が出るか」を種別ごとにまとめる。
**正本は lane spec**。カーネルは `scripts/make_kaggle_kernel.py` で生成する。

```
spec (JSON) → kernel 生成 → Kaggle → run_lane.py → calibration → convert → eval copy → eval → result.json
```

## 1. lane spec で再現できるもの

### 0.8B wide8 + symmetric int8 + emb4 + damping 1.0

```bash
python scripts/make_kaggle_kernel.py \
  --spec lanes/sym8-wide8-d1e0-08b.json \
  --kernel-id yamahei21python/uzu-sym8-wide8-d1e0 \
  --dataset yamahei21python/uzu-bf16-08b-lalamo \
  --platform cpu --push
```

- 校正: s32 / q128 / f32 / mixed50 / damping 1.0
- 変換: bits4 / g32 / block32 / **symmetric_bits 8** / wide8 / emb4
- eval: EN/JA 100 chunks / 128 tokens / f32

### 同じ構成で damping 1e-1（symmetric の基準）

```bash
python scripts/make_kaggle_kernel.py --spec lanes/sym8-wide8-d1e-1-08b.json \
  --kernel-id yamahei21python/uzu-sym8-wide8-d1e-1 \
  --dataset yamahei21python/uzu-bf16-08b-lalamo --platform cpu --push
```

### embedding 4bit vs 8bit

```bash
python scripts/make_kaggle_kernel.py --spec lanes/sym8-emb-bits-08b.json \
  --kernel-id yamahei21python/uzu-sym8-emb-bits \
  --dataset yamahei21python/uzu-bf16-08b-lalamo --platform cpu --push
```

条件は `emb8` / `emb4` の 2 本。base は down-out8 + d1e-1。

### 公式モデルの品質（変換せずに測る）

```bash
python scripts/make_kaggle_kernel.py --kind external-eval \
  --spec lanes/sym8-wide8-d1e0-08b.json \
  --model-repo trymirai/Qwen3.5-0.8B-M \
  --kernel-id yamahei21python/uzu-official-eval \
  --dataset yamahei21python/uzu-bf16-08b-lalamo --platform gpu --push
```

`eval_external_model.py` が BF16 と候補の eval copy を作り、同じ ΔBPC を出す。

## 2. spec 化していないが、コマンドで再現できるもの

### speculator の変換 → 4bit 化 → 実測（ローカル Mac）

```bash
lalamo speculator convert guglxni/Qwen3.5-9B-abliterated-DFlash --output-dir /tmp/spec
python scripts/quantize_speculator.py --source /tmp/spec --output /tmp/spec-int4
# モデルディレクトリに speculator/ として置き、uzu の cli bench で測る
```

### QAD（0.8B）

lane には入っていない。`qad_finetune.py` を直接叩く。

```bash
python kit/scripts/make_qad_dataset.py --bf16 <bf16> --out <work> \
  --corpus data/calibration_mixed_en_ja_dedup.txt \
  --eval-corpus data/eval_en_wikitext2_dedup.txt --eval-corpus data/eval_ja_wikipedia_dedup.txt \
  --samples 32 --max-tokens 128 --damping 0.1 --parallel tensor \
  --layer-bits-json lanes/bits-wide-8.json --quantize-embedding --embedding-bits 4

python kit/scripts/qad_finetune.py --teacher-dir <bf16> --student-dir <work>/student \
  --corpus ... --eval-corpus ... --output-dir <work>/run \
  --steps 50 --seq 128 --optimizer adamw --lr 5e-4 --freeze-scales \
  --skip-checkpoint --parallel tensor        # GPU は OOM。CPU で回す
```

**既知の落とし穴**（すべて修正済みだが、古い kit を使うと再発する）

| 症状 | 原因 |
|---|---|
| `any variant of untagged enum` で起動不可 | lalamo の入れ子 HybridSpec。`rewrite_lalamo_hybrid_spec.py` が必要 |
| `encoding.json` が無い | QAD export の欠陥（`48bcc1d5`） |
| after のつもりで before を測る | `.bak` が古い（`1350d022` で stamp 方式に） |
| 8bit モデルが起動しない | asymmetric int8 は uzu に無い。`--symmetric-bits 8` |

### 速度の実測（ローカル Mac のみ）

```bash
./target/release/cli bench <model_dir> /tmp/task_uzu.json /tmp/bench.json
```

**load average < 2 で測る。** 負荷があると ±9 t/s 動く。

タスクは repo に置いた: `experiments/bench/task_count_512.json`（512 tokens × 2 runs, greedy）。

## 3. まだ自動化していないもの

| 項目 | 状況 |
|---|---|
| QAD のカーネル | 手書き（`uzu-qad-best`）。spec 化していない |
| T4 で draft 学習が回るかのプローブ | 一回限りの使い捨てカーネル |
| seed 反復（wide8 の 3 seed） | `conditions` で表現可能だが spec 未作成。Muse 側で実行 |
| 4B / 9B | spec 未作成。Modal のスクリプトも未整備 |
| speculator の fine-tune | 未着手 |
| Kaggle dataset の版固定 | 未対応。`dataset_sources` は版を指定できず、常に最新がマウントされる |
| 生成カーネルの動作確認 | dry-run と生成物の検証のみ。実際に push して回してはいない |
| `run_lane` の `symmetric_bits` / `layer_bits_json` | dry-run で確認したが、テストでは未固定 |

## 4. 再現性の現状

| 種別 | spec 化 | 実測一致の確認 |
|---|---|---|
| 0.8B の変換 + eval | ✅ 3 本 | ❌ 未検証（dry-run のみ） |
| 公式モデルの eval | ✅ コマンド化 | ✅ 実測（`uzu-official-eval`） |
| QAD | ❌ | ❌ |
| speculator | ❌（コマンドのみ） | ✅ 実測 |
| 速度 | ❌（手順のみ） | ✅ 実測 |

**「spec がある」＝「同じ数字が出る」ではない。** 変換は seed 固定で決定的だが、
eval のチャンク分割やライブラリ版が変われば動く。lane の `result.json` には
環境情報（`env`, `eval_dtype`）が入るので、比較はそれごと行う。
