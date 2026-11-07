# Guard tests

モデル不要の再発防止テスト。重み / private dataset / HF token は使わない。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements-test.txt
pytest -q tests/
```

対象:
- calibration/eval corpus の重複拒否
- dedup
- PPL config の chunk / baseline / calibration ドリフト
- layer-group merge の欠落・重複・範囲外
- atomic save
- BPC 正規化（`compute_bpc.py`）の式と token 数再現
- 2-device sharding probe（`tiny_sharding_probe.py`、`FORCE_2_CPU=1`）
- lane spec のコマンド生成・resume・検証（`run_lane.py`）
- シャード分割とスイープ生成（`--shard`, `make_sweep_spec.py`）
- シャード manifest の統合（`combine_lane_manifests.py`）
- QAD student の検証（`make_qad_dataset.py`）
- アングラ fetch の再試行とログ復元（`ag_fetch.py`）
- 環境容量の掃除と UTC 時刻の扱い（`ag_queue.py cleanup-envs`）
- 生成評価の採点（`eval_generation.py`）と結果回収（`pull_results.py`）
- 完了ログからの復帰と pid 猶予（`ag_queue.py`）
- Kaggle 使用量の集計（`kaggle_usage.py`）
- T4 向け attention 切り替え（`lalamo_forward.py`）
- docs と argparse の乖離（`check_docs.py`）
- 層別ビット規則の解決（`rht_int4_convert.py --layer-bits-json`）
- eval の BPC/chars 出力、merge のメタデータ引き継ぎ
- kit の完全性（`build_kit.py`、遅延 import の欠落検出）
- 公開コーパス監査の交差判定（`audit_public_corpus.py --no-public`）
- アングラ委任の記録・回収（`ag_tasks.py`）
- 委任結果の数値検証（実測と一致しない数値は捨てる）

モデル重み・private dataset は不要。公開 tokenizer のみ使う。
