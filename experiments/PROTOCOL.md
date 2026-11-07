# Experiment Protocol（0.8B → 4B/9B への事故防止）

0.8B で起きた事故を 4B/9B に持ち込まないための必須ルール。実験前にこれを読む。

## 0.8B で起きた事故

| # | 事故 | 原因 |
|---|---|---|
| 1 | mixed50 だけ 50 chunks、他は 100 chunks | `--max-sequences` の不一致 |
| 2 | BF16 baseline が 40.81/36.58 と 37.18/42.17 | eval chunk 数の違い |
| 3 | E3 と比率系で校正設定が違う | E3=16/512/1e-2、比率系=32/128/1e-3 |
| 4 | 旧 candidate が再評価不能 | `/tmp` 消滅。model dir を保存していない |
| 5 | QAD の `step_s` が 0.02s と異常 | JAX 非同期 dispatch を計測 |
| 6 | 4B eval が lalamo で失敗 | uzu 正規化済み config を lalamo に渡した |

## 必須ルール

### 1. モデルごとに config を固定
- `experiments/<model>/config.json` を唯一の正とする
- モデル repo / revision / tied/untied / 量子化仕様 / calibration / eval / QAD を書く
- 実行スクリプトはこの config を読む。コマンドラインで上書きしない
- 出力 JSON に `config_hash` を必ず入れる

### 2. 評価プロトコルを統一
- 1モデル内の全条件で **同じ `max_sequences` / `max_tokens` / corpus**
- 0.8B の比較は **100 chunks / 128 tokens** に統一
- 50 chunks の旧結果は archive。新しい表に混ぜない
- BF16 baseline は全条件で同一値でなければならない。違えば表を作らない

### 2.5 汚染防止は入口で止める
- `yaqa_calibration.py` / `qad_finetune.py` は `--eval-corpus` 必須
- 校正/QAD コーパスと評価コーパスの exact line 重複を検出したら即停止
- `--allow-overlap` は解析目的のみ。通常実験では使わない
- `make_mixed_corpus.py` も `--eval-corpus` を受け取り、生成直後に検査
- 重複除去は `scripts/dedup_calibration.py`。`*_dedup.txt` を使う
- **eval corpus 自体も `scripts/dedup_lines.py` で重複除去する**
- `eval_quantized.py` は eval corpus 内の重複 non-empty line を検出したら停止
- `check_experiment.py` で事後監査も行う

### 3. 校正プロトコルを統一
- 1モデル内の全 YAQA 条件で同じ `samples` / `max_tokens` / `output_block` / `damping`
- 4B/9B は layer-grouped。group filter と merge 元を `calibration.json` に残す
- `yaqa_calibration.py` の `__meta__`（site/shape/corpus/seed/samples）を resume 時に検証
- 校正設定が違う条件を同じ表に載せない

### 4. モデル dir の契約
```
<model>/
  uzu/      # uzu 正規化済み。config.json + encoding.json + model.safetensors
  lalamo/   # raw config。eval / 変換入力
```
- lalamo に渡すのは `lalamo/` のみ。uzu 正規化済みを渡さない
- `encoding.json` は uzu のみ。lalamo には不要
- 変換後は `model.safetensors.lalamo.bak` を保持
- 4B/9B の model dir は `/tmp` に置かず、repo か Modal Volume に保存

### 5. QAD
- `step_s` は `jax.block_until_ready(params)` 後の時間で測る
- 固定 validation chunks の `val_loss` を必ず記録
- `--save-every` で checkpoint、`--resume auto` で再開
- ランダム chunk の training loss だけで良し悪しを判断しない

### 6. 4B/9B 固有
- 0.8B のハイパーパラメータは**事前分布**。そのまま使わない
- 4B: tied。TP=2 + AdamW / Muon reshard。A100 40GB か CPU 64GB
- 9B: untied。TP/FSDP + AdamW。A100 80GB。teacher offload / layer filter
- 4B/9B の calibration は layer-grouped。group ごとに progress / partial を残す
- Kaggle は 4B/9B の calibration/QAD に使わない（OOM）

## Preflight チェックリスト

実験前に必ず確認:

- [ ] `config.json` の `config_hash` を計算した
- [ ] 全条件で同じ eval config（chunks/tokens/corpus）
- [ ] BF16 baseline が全条件で一致
- [ ] calibration 設定が全条件で一致
- [ ] lalamo には `lalamo/`、uzu には `uzu/` を渡している
- [ ] `uzu/encoding.json` がある
- [ ] candidate model dir が保存されている（`/tmp` だけにしない）
- [ ] QAD は `val_loss` と `step_s`（block_until_ready）を記録
- [ ] 4B/9B は Modal Volume に checkpoint を残す
- [ ] 0.8B の結果を 4B/9B に直接コピーしていない

## 機械チェック（必須）
```bash
python scripts/check_experiment.py \
  --config experiments/qwen35-0.8b/config.json \
  --root experiments/qwen35-0.8b
```

- 50/100 chunks 混在、BF16 baseline 不一致、校正設定不一致を検出
- 1つでも error なら表を作らない
- 4B/9B も同じコマンドで config を差し替えて実行

## 集計ルール
- 表を作る前に `config_hash` を group by
- 異なる `config_hash` が混ざったら **表を作らずエラー**
- 旧 50 chunks の結果は `archive/` に退避
- 比較表には「モデル / 校正設定 / eval設定 / commit」を必ず併記
