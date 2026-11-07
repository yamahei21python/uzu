# Lane specs

実験をデータとして定義する。owner が spec を書き、Kaggle 実行担当は `run_lane.py` を回すだけ。

```bash
python scripts/run_lane.py --spec lanes/clean-suite-08b.json --dry-run   # コマンド確認
python scripts/run_lane.py --spec lanes/clean-suite-08b.json             # 実行
python scripts/run_lane.py --spec lanes/clean-suite-08b.json --hf-results <user>/uzu-results
```

## 仕様

| key | 意味 |
|---|---|
| `lane` | 出力ディレクトリ名。`<work>/out/<lane>/` |
| `model.spec_name` / `model.hf_repo` | BF16 が無い時の import 元 |
| `model.bf16_dir` | BF16 lalamo の場所（`kit/` 相対 or 絶対） |
| `model.bf16_source` | BF16 をコピーしてくる場所（Kaggle dataset など） |
| `paths.kit` | スクリプトとコーパスのルート |
| `paths.work` | 出力ルート。`--work` で上書き可 |
| `paths.jax_platform` | `cpu` / `gpu`。`JAX_PLATFORM_NAME` に渡す |
| `eval.corpora` | 言語 → eval コーパス。必ず `*_dedup.txt` |
| `eval.max_sequences` | chunk 数。監査で 100 を要求 |
| `condition_defaults` | calibration の既定値 |
| `conditions[]` | `name`, `calibration_corpus`, 上書き値 |
| `conditions[].raw_output` | 生 Hessian を保存（balanced H 用） |
| `conditions[].skip_convert` | calibration だけ実行して npz を残す |
| `conditions[].force_convert` | raw でも変換まで回す |
| `merge` | `en`, `ja`, `ratios`, `damping`, `name_template` |
| `convert` | `bits`, `group_size`, `block_size`, `parallel` |

## 挙動

- calibration には必ず全 eval コーパスを `--eval-corpus` で渡す（重複を入口で拒否）
- `result.json` が `status: complete` なら skip（12h 制限対策の resume）
- 最後に `manifest.json` を書く
- 検証で落ちる条件:
  - `chunks` が `eval.max_sequences` と違う
  - 条件間で `bf16_ppl` が 0.05 以上ずれる（BF16 baseline は共通のはず）
  - `bf16_ppl` が無い
- 検証に落ちたら exit 1。manifest は残る

## シャード実行

1 セッション 12h を跨ぐ条件列は、spec を分けるか `--shard` で切る。

```bash
python scripts/run_lane.py --spec lanes/clean-suite-08b.json --shard 0/3
# lane 名は clean-suite-08b-shard1of3 になる
```

- `--shard i/n` は条件を `position % n == i` で抜く
- `spec.merge` と併用不可（merge は分割できない）
- 各シャードは別 lane ディレクトリなので、終わったシャードは resume で skip される

## スイープ spec の生成

27 条件（samples × seq × damping）を 3 シャードに分ける。

```bash
python scripts/make_sweep_spec.py --base lanes/clean-suite-08b.json --out-dir lanes \
  --name robustness-08b --corpus data/calibration_mixed_en_ja_dedup.txt \
  --samples 16 32 64 --seq 128 256 512 --damping 1e-4 1e-3 1e-2 --shards 3
```

生成物: `lanes/robustness-08b-shard{1,2,3}of3.json`（各 9 条件）。
条件名は `s32_q128_d1e-4` 形式。

## 既存

| spec | 内容 |
|---|---|
| `clean-suite-08b.json` | mix65 / mix35 / jaonly の正規化 calibration + INT4 eval |
| `balanced-h-08b.json` | raw EN/JA → 50/65/35 merge → INT4 eval |
