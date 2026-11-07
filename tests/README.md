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
