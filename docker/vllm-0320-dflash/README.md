# vLLM 0.32.0 + DFlash 修正イメージ（Muse の 27B 用）

受理 0 の原因は vLLM 0.32.0 の回帰（`docs/VLLM_DFLASH_032.md`）。
上流の修正 #64059 は**どのリリースにも入っていない**ので、イメージに backport する。

## 1. まずワークアラウンドで原因を確定（ビルド不要・10 分）

```bash
vllm serve <27B target> \
  --speculative-config '{"method":"dflash","model":"<dflash2 draft>","num_speculative_tokens":8}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --no-enable-prefix-caching
```

`PIECEWISE` は DFlash draft を eager で走らせる。`--enforce-eager` と同じ診断になり、
target のグラフは残る。`--no-enable-prefix-caching` は別バグ（#58894）を交絡から外す。

**待つログ**: `Application startup complete`（`Capturing dflash2 CUDA graphs (FULL)` が
**出ない**ことを確認する）。

計測:

```bash
python scripts/vllm_acceptance.py --url http://127.0.0.1:8000/metrics --watch 10 --duration 300
```

| 結果 | 判断 |
|---|---|
| `acceptance` 0.20 以上 | 原因確定（#60692）。→ 2 へ |
| 0.00 のまま | 別件。draft の checkpoint か #58894 を疑う |

## 2. 修正イメージをビルド（FULL グラフのまま 20.6 req/s）

```bash
docker build -f docker/vllm-0320-dflash/Dockerfile -t vllm:0.32.0-dflash .
```

ビルド時の `patch --dry-run` と `hasattr(DraftModelSpeculator, "_map_kernel_block_tables")`
が通らなければ失敗する（壊れたイメージは出ない）。

serve:

```bash
docker run --gpus all -p 8000:8000 vllm:0.32.0-dflash \
  <27B target> \
  --speculative-config '{"method":"dflash","model":"<dflash2 draft>","num_speculative_tokens":8}' \
  --no-enable-prefix-caching
```

`FULL_AND_PIECEWISE` のままでよい（既定）。ログに
`Capturing dflash2 CUDA graphs (FULL)` が出ることを確認する。

計測は同じ `scripts/vllm_acceptance.py`。

## 期待値（B200 での上流報告）

| 構成 | acceptance | req/s |
|---|---:|---:|
| 0.32.0 既定 | 0.004 | 3.7 |
| + PIECEWISE | 0.382 | 11.6 |
| + 本パッチ | 0.552 | 20.6 |
| 0.31.0（参考） | 0.531 | 14.6 |

## 留保

- 修正が効くのは **kernel block table が分割される構成**（FlashAttention v4 hd256 / Blackwell）。
  Hopper / Ada では分割が起きないので、`PIECEWISE` で直ってもパッチで直らない可能性がある
- `--speculative-config '{"enforce_eager": true}'` は**効かない**（runner が target の
  cudagraph mode を draft に渡すため）
- このリポジトリから GPU 実行はしていない。パッチは v0.32.0 のタグ実ソースで
  `patch --dry-run` と `ast.parse` まで検証済み
