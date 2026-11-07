# 層別ビット配分

`rht_int4_convert.py` は既定で全 Linear を同じビット数にする。層ごとに変えたい場合は
`--layer-bits-json` で規則を渡す。根拠は `research/9b_bit_allocation.md`。

## JSON

```json
{
  "default": 4,
  "rules": [
    {"match": "layers/[0]/mlp/down_projection", "bits": 8},
    {"match": "mlp/down_projection", "bits": 6},
    {"match": "mixer/out_projection", "bits": 6}
  ]
}
```

- `match` は**層パスへの部分一致**。`decoder/.transformer/.layers/[3]/.mlp/.up_projection` のような文字列に対して判定する
- **先に書いた規則が勝つ**。具体的な規則を先に置く
- `bits` は 4 / 6 / 8 のみ
- 一致しなければ `default`（省略時は `--bits`）

## 実行

```bash
python scripts/rht_int4_convert.py \
  --load-from /path/to/bf16-lalamo \
  --output-dir /path/to/int4-mixed \
  --preconditioners calibration.npz \
  --calibration-json calibration.json \
  --layer-bits-json bits.json
```

変換ログに分布が出る:

```
layer bits overrides: {"default": 4, "rules": [...]}
bits distribution: {4: 59, 6: 69}
```

`provenance.json` に `layer_bits`（規則）と `layer_bits_used`（層パス → ビット）が入る。
**どの層を何ビットにしたかはモデルに同梱される**ので、後から追跡できる。

## 注意

- 6bit は 4bit より大きい。合計サイズを先に計算する（`research/9b_bit_allocation.md` の表）
- 8bit を混ぜると group 32 のままだとサイズが跳ねる。必要層だけにする
- `--quantize-embedding` / `--embedding-bits` は別系統。層別規則の対象外
- 配分を変えたら、必ず同じ eval 条件（dedup 100 chunks）で BF16 と比較する
