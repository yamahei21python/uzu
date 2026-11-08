# 実験の再現手順（2026-11-08 時点）

「何をどう回したら同じ数字が出るか」を種別ごとにまとめる。
**正本は lane spec**。カーネルは `scripts/make_kaggle_kernel.py` で生成する。
生成物は **kit の commit** と **JAX の版**を固定する（両方とも数値を変える）。

```
spec (JSON) → kernel 生成 → Kaggle → run_lane.py → calibration → convert → eval copy → eval → result.json
```


## 0. この水準で足りる理由

実験の再現性には段階がある。厳密には「同じハッシュの環境で同じコマンドを実行して
同じバイト列の結果を得る」のが理想だが、それは毎週変わるクラウド上では維持できない。
代わりに、こちらでは**判断に使える再現性**を目標にしている。

具体的には、次の 3 つが揃っていれば「同じ実験をやり直した」と見なす。

1. **入力が固定されている** — BF16 モデル、コーパス、lane spec、乱数 seed
2. **コードが固定されている** — kit の commit と lalamo の commit を明示的に指定する
3. **結果に環境が添付されている** — `result.json` の `env` / `eval_dtype` / `finished`

この 3 つがあれば、数字が動いたときに「入力が違うのか」「コードが違うのか」
「環境が違うのか」を切り分けられる。切り分けられない再現性は、記録していないのと
同じなので、ここを最重要としている。

逆に、次のものは意図的に追っていない。

- **JAX の完全な依存関係**（numpy / hub など）。固定すると Kaggle の
  Python 更新で動かなくなる。JAX 本体だけ固定して、他は実行時の記録に残す
- **Kaggle dataset の版**。`dataset_sources` に版を書けない仕様なので、
  BF16 データセットを差し替えたら数字が動く。差し替えた事実は自分で記録する
- **バイト単位の一致**。GPU のカーネル選択やスレッド数で末尾の桁は動く

## 0.1 新しいモデルで実験するときの手順

モデルが変わっても、やることは同じである。

1. `lanes/` に spec を書く。校正コーパス、convert の設定（bits / group /
   `symmetric_bits` / `layer_bits_json` / `quantize_embedding`）、eval の条件を入れる
2. `make_kaggle_kernel.py` でカーネルを生成する。kit の commit は自動で固定される
3. 実行して `result.json` を回収する。`env` が入っていることを確認する
4. 数値を docs に書き、`result.json` は Kaggle 出力に残す

**手書きのカーネルを増やさない**ことが唯一の規律。手書きが増えた時点で、
その実験は再現できない。今日書いた QAD のカーネルはこの規律から外れており、
spec 化するまでの暫定である。

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
| numpy / huggingface_hub の版 | 未固定（JAX と kit のみ固定） |

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
