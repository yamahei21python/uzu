# Modal（4B / 9B の逃がし先）

T4 16GB では 4B / 9B の calibration が載らない。9B は BF16 18GB で eval すら不可。
ローカルでも回さない（`docs/LOCAL_POLICY.md`）。

ただし **Modal は最終手段**（下記「位置づけ」）。まず Kaggle に収める工夫をする。

## 前提

```bash
modal secret create huggingface HF_TOKEN=...
modal volume create uzu-quant      # INT4 出力
modal volume create uzu-models     # BF16 ソース（任意）
```

- イメージは lalamo `e401e177` を pin
- スクリプトと dedup コーパスだけ焼き込む。BF16 は Modal 内で HF から取る
- 各ステップは冪等（`COMPLETE` か出力の存在で skip）
- **クライアントは venv のものを使う**: グローバルの `modal`（1.2.6 / Python 3.9）は image builder `2026.10` を知らず、起動直後に `VersionError` で落ちる。`.deps/venv-lalamo/bin/python -m modal run ...`（1.6.3）で実行する

## 実行

`--size 4b` / `--size 9b`。`run_name` の既定は `huihui-4b` / `huihui-9b`。

### 量子化フラグ（2026-11-09 追加）

`rtn` / `yaqa` は 0.8B で確定した構成を既定にした。

| フラグ | 既定 | 意味 |
|---|---|---|
| `--bits` | 4 | ベースビット幅 |
| `--quant-group-size` | 32 | 量子化グループ（校正の `--group-size` とは別物） |
| `--block-size` | 32 | RHT ブロック |
| `--parallel` | replicated | JAX デバイス分割 |
| `--symmetric-bits` | 8 | このビット幅は symmetric int8（uzu は 4bit のみ ZP を持つ） |
| `--embedding-bits` | 0（未指定） | embedding/readout のビット幅（8 推奨） |
| `--layer-bits` | 空 | 層別配分 JSON の**コンテナ内パス**。`lanes/` は `/workspace/lanes` に焼いてある |
| `--quantize-embedding` | False | embedding を量子化する |

```bash
# 9B の採用レシピ（wide8 + emb8 + d1e0）
modal run scripts/modal_pipeline.py --command yaqa --size 9b \
  --quantize-embedding --embedding-bits 8 \
  --layer-bits /workspace/lanes/bits-wide-8.json
```

`--symmetric-bits ""` で全ビット幅 asymmetric になる（非推奨、`docs/ENGINE_INT8_ZP.md`）。

### 基本手順

```bash
# 1. BF16 import（CPU、~10 分）
modal run scripts/modal_pipeline.py --command import --size 9b

# 2. RTN（比較用のベースライン）
modal run scripts/modal_pipeline.py --command rtn --size 9b --quantize-embedding

# 3. calibration（A100、layer group ごとに分割）
modal run scripts/modal_pipeline.py --command calibrate --size 9b \
  --samples 8 --max-tokens 512 --group-size 4 \
  --corpus calibration_en_dedup.txt

# 4. YAQA 変換
modal run scripts/modal_pipeline.py --command yaqa --size 9b --quantize-embedding

# 5. eval（BF16 vs INT4、dedup 100 chunks、EN/JA）
modal run scripts/modal_pipeline.py --command eval --size 9b --candidate int4-yaqa

# 6. QAD（clean student が必要）
modal run scripts/modal_pipeline.py --command qad --size 9b \
  --student student --steps 10 --optimizer adamw --lr 1e-3 --scale-lr 1e-6 \
  --freeze-scales --warmup-steps 5 --save-every 10

# 7. 回収
modal run scripts/modal_pipeline.py --command fetch --size 9b --remote int4-yaqa
```

## 9B の採用レシピ（2026-11-09）

0.8B で 4 seed 回して確定した構成（`docs/0.8B_RESULTS.md` §13）を 9B に移す。

| 項目 | 値 |
|---|---|
| 校正コーパス | `calibration_mixed_dedup.txt`（EN/JA 混合） |
| 校正 | s32 / 128 tok / **small(2048)** / **damping 1.0** / bf16 / group 4 |
| 8bit 層 | `bits-wide-8.json`（down_proj / out_proj / qkvg / out_projection） |
| embedding | 8bit（symmetric） |
| 量子化グループ | 32（RHT block 32） |

```bash
R=9b-abliterated
modal run scripts/modal_pipeline.py --command import    --size 9b --run-name $R
modal run scripts/modal_pipeline.py --command calibrate --size 9b --run-name $R \
  --samples 32 --max-tokens 128 --group-size 4 \
  --corpus calibration_mixed_dedup.txt --damping 1.0 --output-block small
modal run scripts/modal_pipeline.py --command yaqa      --size 9b --run-name $R \
  --quantize-embedding --embedding-bits 8 \
  --layer-bits /workspace/lanes/bits-wide-8.json
modal run scripts/modal_pipeline.py --command eval      --size 9b --run-name $R \
  --candidate int4-yaqa --max-sequences 100
```

**順番を守る**: `check` → `import` で落ちないことを確認してから GPU（`calibrate`）に進む。
`calibrate` は group ごとに `COMPLETE` 相当の npz を残すので、途中で落ちても同じコマンドで再開する。

## バッチ実行（timeout 必須）

`modal run` にはクライアント側の timeout が無い。起動時に落ちる（mount 漏れ / import エラー）と、remote の traceback を出したまま**クライアントが無応答になる**（2026-11-07 の smoke がこの状態で数時間戻らなかった）。

`modal_batch.py` が 1 コマンドごとに上限を被せる。

```bash
python scripts/modal_batch.py --sizes 4b,9b --commands import,rtn,eval \
  --run-name 'smoke-{size}' --eval-candidate int4-rtn --max-sequences 20 \
  --timeout 1800 --log logs/modal_smoke.log
```

出力:

```
| size | command | rc | seconds | last |
|---|---:|---:|---:|---|
```

- 最初の失敗で停止（`--keep-going` で継続）
- 失敗・timeout が 1 つでもあれば exit 1
- `--run-name 'smoke-{size}'` を忘れると 4B と 9B が同じ volume ディレクトリを共有する
- 起動失敗マーカーの検出は**保険**。実際の停止は timeout が担う
- timeout で落ちたときはクライアントだけが死ぬ。remote app は `modal app list` で確認し、不要なら `modal app stop <app-id>`

## 注意

- **calibration は必ず `UZU_ATTENTION_IMPL=xla`**（既定）。Pallas flash カーネルは微分できない（`pallas_call` の JVP が `axis_frame` で assert する）。CPU 校正も XLA なので数値が揃う。`--attention pallas` は calibration では失敗する
- calibration の既定は config 由来（9B は `none` / 1e-3）。0.8B の clean 設定は `small` / 1e-3、9B の採用レシピは `small` / **1.0**。比較するときは必ず明示する
- QAD の student は `make_qad_dataset.py` で作る。`provenance.json` が無いと QAD が拒否する
- **QAD の lr は 1e-3 前後**。2e-5 は finetuning の値で、4-bit の量子化刻み（0.8B の中央値 4.7e-4）より 1 桁小さい。量子化器が更新を丸めて落とし、val_kl が 1 ビットも動かない（`uzu-qad-cpu` がこれで early stop した）。`qad_finetune.py` は起動時に `median int4 step` と lr を比べて警告する
- `eval` は `A100-80GB`。BF16 18GB + INT4 5GB + 実行時バッファで 40GB だと足りない場合がある
- 冪等なので、タイムアウトしたら同じコマンドを再実行する。途中から再開する
- Modal の GPU 時間は課金対象。**smoke をローカルで済ませてから**投げる

## 既知の制約

- 9B の calibration は layer group 分割が必須（1 層ずつでも H が大きい）
- 4B は `--group-size 8` くらいまで大きくできる見込み（未検証）
- QAD は単一 GPU。FSDP は未対応（`--parallel` は replicated のまま）

## 位置づけ（2026-11-08 決定）

**Modal は最終手段。** まず Kaggle（CPU 5 並列 / GPU T4×2）で工夫する。

| 順序 | 手段 |
|---|---|
| 1 | Kaggle CPU（無料・週次クォータ無し） |
| 2 | Kaggle GPU T4×2（30h/週） |
| 3 | 構成を変えて 1・2 に収める（fsdp / 量子化教師 / seq 短縮 / freeze） |
| 4 | **Modal A100**（従量課金） |

Modal を使う前に「なぜ Kaggle で回らないか」を docs に書く。
