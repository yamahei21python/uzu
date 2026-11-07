# ローカル実行ポリシー

ローカル（M3 16GB）は**編集・テスト・短い解析**に使う。重い処理は逃がす。

理由: 長いジョブが 1 本走ると他のタスクが止まる。実際に 0.8B の 1 条件（約 35 分）で作業が詰まった。

## ローカルでやる

| 種類 | 目安 |
|---|---|
| ファイル編集、git 操作 | — |
| `pytest -q tests/` | 4 秒 |
| `build_kit.py --check` | 1 秒 |
| `run_lane.py --dry-run` / `--shard` のコマンド確認 | 1 秒 |
| ログ・JSON の集計 | 数十秒 |
| commit 済み JSON の BPC 再計算（100 chunks、tokenizer のみ） | 1〜2 分 |
| corpus 監査 `--no-public` | 0.4 秒 |

## ローカルでやらない

| 種類 | 実測 | 逃がし先 |
|---|---|---|
| `yaqa_calibration.py`（0.8B / 32 samples） | 約 10 分 | Kaggle 枠 2（検証）または Muse |
| `rht_int4_convert.py`（0.8B） | 2〜10 分（CPU だと 9.5 分） | Kaggle |
| `eval_quantized.py`（100 chunks × 2 言語） | 約 20 分 | Kaggle |
| `qad_finetune.py` | 17〜80 秒/step | Kaggle（OOM するなら Modal） |
| smoke lane 全体 | 約 12 分 | **Kaggle の検証枠** |
| 27 条件スイープ | 18 時間 | Kaggle |
| 4B / 9B の calibration・eval・QAD | T4 に載らない | Modal |

モデルをロードする処理は、たとえ 1 条件でもローカルでやらない。

## 例外

- デバッグで **明示的に 5 分以内**と分かっているもの（`--dry-run`、`--samples 1`、2 chunks）
- 失敗の再現にどうしても要る場合。走らせる前に「何分か」を見積もり、ログに残す
- 走らせたら**他の作業を並行しない**（CPU を食ってテストや編集が止まる）

## 判断フロー

```
そのコマンド、モデルをロードする？
  yes -> Kaggle か Modal
  no  -> 5 分以内？
           yes -> ローカル可
           no  -> Kaggle（lane spec を作る）
```

## 逃がし方

1. 条件を `lanes/*.json` に書く（必要なら `make_sweep_spec.py`）
2. `run_lane.py --dry-run` でコマンド確認（ローカル、1 秒）
3. Kaggle の検証枠で smoke（5〜12 分）
4. 通ったら本実験枠 or Muse に渡す

`docs/KAGGLE_SLOTS.md` の枠割り当てに従う。owner は常に 1 枠を空ける。

## 現在のローカル常駐

| プロセス | 負荷 | 備考 |
|---|---|---|
| `ag_queue.py watch` | ほぼ 0 | 120 秒ごとに状態確認 |
| `ag_fetch.py` | 待機中のみ | 503 の再試行 |
| Kaggle 監視ループ | ほぼ 0 | 60 秒ごとに status |

どれも CPU をほぼ使わない。**モデルをロードするプロセスは常駐させない。**
