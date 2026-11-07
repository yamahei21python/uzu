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
- kit の完全性（`build_kit.py`、遅延 import の欠落検出）

モデル重み・private dataset は不要。公開 tokenizer のみ使う。
