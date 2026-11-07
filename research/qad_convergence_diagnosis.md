# QAD 発散の原因調査（コード精読 + 合成再現）

対象: `scripts/qad_finetune.py`、lalamo `e401e177` の量子化 Linear。

> **重要な訂正**
> 委任先（Antigravity）は sandbox のファイルシステム障害で成果物を保存できず、最後に「再現結果」を文章で返してきた。
> その数値（loss 0.155912 → 1.741509 等）をこちらで再実行したところ**再現しなかった**（実測は 0.004708 → 0.004569）。
> したがって**数値は破棄**し、コード精読の知見だけを採用している。以下、数値はすべてこちらの実測。

## 1. 結論

1. **lalamo の量子化は scale に対して `stop_gradient` を掛けている**（`.deps/lalamo-e401/lalamo/compressed/int.py:146`）。
   QAD は `scales` を学習対象に含めている（`qad_finetune.py:185-193`）ので、**scale の勾配は dequant 経路だけ**になる。
   この勾配は `int_codes - zero_point`（概ね `w/s`、絶対値 0〜8）で、量子化誤差（±0.5）より 1 桁以上大きい。
2. scale は生値で保持され、AdamW で直接減算される（`qad_finetune.py:215, 237`）。
   下限 clamp は更新後に 1e-6 で切るだけ（`249-257`）なので、一度潰れると戻れない。
3. 合成再現では collapse を再現できなかった（後述）。**scale 学習そのものが危険**という傍証はあるが、原因の断定はまだできない。
4. 次の一手は「scale を凍結して weights だけ学習」。これで loss が下がれば scale 起因、下がらなければ蒸留の設計側。

## 2. コード精読（行番号は main 時点）

| # | 項目 | 判定 | 根拠 |
|---|---|---|---|
| 1 | KL の向き・温度・T² | 問題なし | `260-269`: `teacher_probs * (log p_t - log p_s)` に `T**2` |
| 2 | 学習対象 | **要注意** | `185-193`: `master_weights` / `scales` / `master_zero_points` を同時に学習 |
| 3 | scale のパラメータ化 | **要注意（生値）** | 生 float。AdamW で絶対量を減算するため 0 に落ちやすい |
| 4 | STE / scale への勾配 | **要注意（ライブラリ側）** | `int.py:146` の `stop_gradient(scales)`。dequant 経路のみ |
| 5 | weight decay が scale に掛かる | 問題なし | `215, 237`: `weight_decay=0.0` を明示 |
| 6 | 勾配クリップ / clamp | **要注意** | `244`: 全体ノルムクリップが scale を巻き込む。`249-257`: 下限のみ、モーメンタムは残る |
| 7 | validation が train と同じ forward | 問題なし | `260-269` の `loss_fn` を共用 |
| 8 | warmup / schedule / `--scale-lr` | **要注意** | warmup 無しで step 1 から全開。`--scale-lr` 既定 1e-6 |

### 最重要: scale 勾配の形

```python
# lalamo/compressed/int.py:144-153
int_scale_weights = _master_weights_to_int_scale_weights(
    weights,
    stop_gradient(scales),           # <- scale の微分がここで切れる
    stop_gradient(int_scale_zero_points),
    group_size,
    bits,
)
int_weights = round_fn(int_scale_weights)
return _integer_scaled_weights_to_master_weights(int_weights, scales, ...)
```

- 素直な LSQ の STE は `round(w/s) - w/s`（±0.5）を scale に流す
- ここでは `int_codes - zero_point`（`≈ w/s`、最大 ±8）が流れる
- 大きさが 1 桁以上違ううえ、符号が重みと揃うため、**scale を一方向に押し続ける**

ライブラリの意図としては「量子化は微分しない」設計なので、**QAD 側で scales を学習するのが前提違い**の可能性が高い。

## 3. 合成再現（実測・再現できず）

`scripts/qad_scale_repro.py` をローカルで実行した（numpy、256×1024、group32、50 step、32 条件）。

```bash
python scripts/qad_scale_repro.py --all
```

実測（`research/qad_scale_repro_results.json`）:

| 条件 | loss 初期 | loss 最終 | scale collapse |
|---|---:|---:|---:|
| lr 1e-6 | 0.004708 | 0.0047 前後 | 0% |
| lr 1e-5 | 0.004708 | 0.0047〜0.0048 | 0% |
| lr 1e-4 | 0.004708 | 0.004569 | 0% |
| lr 1e-3 | 0.004708 | 0.00566 | 0% |

- **collapse は起きない**。loss の変化も 1% 未満
- 合成モデルが小さすぎる（256×1024、16 batch）ため、scale 更新量が初期値に対して無視できる
- 実機の loss 0.1559 は YAQA 直後の KL。合成の 0.0047 とはスケールが違うので、この実験で実機を語れない

→ **この再現は「陰性」。原因の特定には使えない。** 数値を根拠にしない。

## 4. 次に試すこと（優先順）

### P0: scale を凍結して weights だけ学習
`collect_params` から `scales` と `master_zero_points` を外す。

```python
if "master_weights" in key and isinstance(leaf, jax.Array):
    params[key] = leaf
```

- コマンド: `--optimizer adamw --lr 2e-5 --steps 10 --scale-lr 0 --save-every 0`
- 期待: loss が単調減少。**下がれば scale 起因**、下がらなければ蒸留 loss 側
- 失敗時に分かること: 蒸留の forward（teacher/student の logits 経路）を疑う

### P1: log-scale 化 + warmup
- `scales` を `log(scales)` で保持し、`exp` して使う。乗算更新になり 0 に落ちない
- 最初の 10% を warmup
- 期待: scale の min が初期値の 10% を下回らない

### P2: クリップの分離
- `clip_by_global_norm` を weights と scales で別々に掛ける
- scale 側は `clip_by_value` で 1 step の更新幅を制限する

### P3: STE を自前で当てる
lalamo の `stop_gradient` を外すのはライブラリ改変になるため、QAD 側で scale を直接最適化しない（P0 と同じ結論）。

## 5. 再現

```bash
python scripts/qad_scale_repro.py --all
```

出力: `research/qad_scale_repro_results.json`、`research/qad_repro_stdout.txt`。

実機の再現は clean student が必要（provenance 付き）。現状はブロック中。
