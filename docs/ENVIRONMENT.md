# 環境一致と provenance

**同じ version 文字列でも中身が違う。** これを防ぐための規約と検査。

## 実際に起きたこと

| 事故 | 何が起きたか | 検査 |
|---|---|---|
| lalamo | PyPI `0.17.0`（2026-09-24）と git `e401e177`（2026-11-02）が両方 `0.17.0`。PyPI には `compressed/incoherence.py` が無く RHT が動かない | code hash |
| Kaggle dataset | `kaggle datasets version` の既定 `--dir-mode skip` で `scripts/` `data/` が落ち、`kit_manifest.json` だけになった | publish 後の listing 検証 |
| kit 更新 | in-place 更新なのに external kernel が exact hash を assert → 更新のたびに停止 | 能力判定 / immutable revision |
| Modal mount | `modal_targets` / `lalamo_forward` を image に積み忘れ、起動前に死亡 | mount 網羅テスト |
| argparse | `args.layer_filter` を使うが宣言が無い | 宣言テスト |

## ローカルの中間ファイル

`/tmp/uzu-kernels`（`watch_kernels.py` のダウンロード先）、`/tmp/kout`（評価出力のコピー）、
変換済みモデルは**すべて再取得・再生成できる**。

- 数値の正本は `docs/STATUS.md` と `docs/0.8B_RESULTS.md`。カーネル出力はその材料でしかない
- 変換済みモデルは Kaggle 出力から再取得、または `kit` のスクリプトで数十秒〜10 分で再生成
- したがって容量が苦しくなったら消してよい。ただし**消す前に**「docs に載っているか」を確認する

計測をやり直すときは `load average < 2` を確認する。`cli bench` は負荷で ±9 t/s 動く。

## 規約

1. **lalamo は git pin**。`git+https://github.com/trymirai/lalamo@e401e17761f2fcd337b0cf20218d67d656e6ecba` を単独で入れる。PyPI を使わない。
2. **比較は version ではなく code hash**。`environment_fingerprint` の `lalamo_*_sha256` が一致したときだけ数値を混ぜる。
3. **kit 公開は `publish_kit.py` 経由**。Kaggle は必ず `--dir-mode zip`、公開後に scripts/ data/ configs/ を読み戻して検証する。
4. **外部カーネルの assert は exact hash にしない**。能力（`--layer-filter` の有無、`preflight.py` の存在）か、不変 revision（Kaggle dataset version / HF commit SHA）で固定する。
5. **manifest の必須項目**: `environment_fingerprint`, `kit_commit`, `config_hash`, `corpus_sha256`, `seed`, `eval_dtype`。
6. **外部カーネルは `kit_manifest.json` の `capabilities` を見る**。例: `rht_int4_convert.layer_filter`, `preflight.sdpa_xla`, `env_fingerprint`。commit hash は rebuild のたびに変わるが、能力は変わらない。

## コマンド

```bash
# 環境の同一性を確認（lalamo の source と RHT ファイルの hash）
python scripts/env_fingerprint.py

# 実行前の能力確認
python scripts/preflight.py --json /tmp/preflight.json

# kit のビルド・公開・検証（HF + Kaggle）
python scripts/publish_kit.py \
  --dataset-dir /Users/kohei/Myproject/kaggleAI/datasets/uzu-quant-kit-public \
  --message "Rebuild: <commit>"
```

## 外部実行環境（Muse など）に渡す確認

kit は in-place 更新なので `built_from_commit` は rebuild のたびに変わる。**exact hash を assert しない。**

### 1. kit の能力を確認（起動時）

```python
import json, pathlib
manifest = json.loads(pathlib.Path("/kaggle/working/kit/kit_manifest.json").read_text())
caps = set(manifest.get("capabilities", []))
need = {"rht_int4_convert.layer_filter", "preflight", "env_fingerprint"}
assert need <= caps, f"kit too old: missing {sorted(need - caps)}"
```

### 2. lalamo が RHT 付きか確認

```python
import json
from importlib.metadata import distribution
info = json.loads(distribution("lalamo").read_text("direct_url.json") or "{}")
print((info.get("vcs_info") or {}).get("commit_id"))  # e401e177... であること
import lalamo.compressed.incoherence                     # ImportError なら PyPI 版
```

同等の確認は `python scripts/preflight.py --json /tmp/pf.json` でもできる:
`environment.lalamo_commit` と `probes.quant_roundtrip` を見る。

### 3. 結果を混ぜてよい条件

- `lalamo_commit` が `e401e17761f2fcd337b0cf20218d67d656e6ecba`
- `probes.quant_roundtrip == "pass"`
- manifest の `environment_fingerprint.lalamo_hash_complete == "true"`

1 つでも欠けたら **owner の数値と比較しない**。内部比較のみ。

## 判定

- `lalamo_compressed_incoherence_sha256` が両方に存在し一致 → 比較可能
- 片方に無い → PyPI。その結果は使わない
- `PUBLISH INVALID` → dataset が壊れている。kernel を push しない
