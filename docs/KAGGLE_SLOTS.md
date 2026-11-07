# Kaggle スロット運用

## 割り当て

| 枠 | owner（yamahei21python） | Muse（kentasatomuse） |
|---|---|---|
| 1 | 本実験 | 本実験 |
| 2 | **空けておく**（テスト調整用） | 本実験 |

- GPU は 1 アカウント **2 枠**。owner は常に **1 枠を空ける**
- 空き枠は「Muse に渡す前のスクリプト検証」専用。本実験を入れない
- 合計の同時実行は最大 4 本

## なぜ 1 枠空けるか

1. **手渡し前の検証**を本実験と同じ条件でやる必要がある。Kaggle でしか出ない差（config 正規化、dataset の attach、HF token の有無）が毎回出る
2. 本実験が 2 本埋まると、修正版の再投入が**待ち**になる。待ち時間が一番高いコスト
3. 12h 制限・クォータ（30h/週）は枠を埋めるほど無駄が出る。失敗 1 本で数百秒が飛ぶ

> ローカル実行の可否は `docs/LOCAL_POLICY.md`。モデルをロードする処理はローカルで流さない。

## 手順

### 1. smoke で確認（枠 2）
`run_lane` の smoke spec を使う。2 sample / 32 token / 2 chunks。
T4x2 で **5〜12 分**。

> 参考として M3 / CPU で 1 回だけ測った値: calibration 2 分、変換 9.5 分、eval 各 12 秒 = 約 12 分。
> この計測はもう繰り返さない（`docs/LOCAL_POLICY.md`）。smoke は**必ず Kaggle の検証枠**で回す。

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/trymirai/lalamo@e401e17761f2fcd337b0cf20218d67d656e6ecba",
    "jax[cuda12]", "huggingface_hub", "tokenizers"], check=True)
subprocess.run(["hf", "download", "yamahei21/uzu-quant-kit", "--repo-type", "dataset",
    "--local-dir", "/kaggle/working/kit"], check=True)
subprocess.run([sys.executable, "/kaggle/working/kit/scripts/build_kit.py",
    "--check", "/kaggle/working/kit"], check=True)
subprocess.run([sys.executable, "/kaggle/working/kit/scripts/run_lane.py",
    "--spec", "/kaggle/working/kit/lanes/smoke-08b.json"], check=True)
```

合格条件:
- `KIT OK`
- `LANE COMPLETE`
- `out/smoke-08b/mix65/result.json` が `status: complete`
- `ppl_en.json` / `ppl_ja.json` の `chunks` が 2
- BF16 が EN 35.38 / JA 43.42 付近（2 chunks なので値はぶれる。桁と大小関係だけ見る）

落ちたら**Muse に渡さない**。原因はほぼ次のどれか:
- kit にファイルが無い → `build_kit.py --check`
- BF16 の config 不整合 → `run_lane` の `bf16-lalamo` 経路
- `--include` 形式 → 使わない
- dataset の attach 漏れ → `bf16_source` のパス

### 2. 本実験（枠 1）
smoke が通ってから、`clean-suite-08b` などを**同じセッションのまま** spec だけ差し替えて流す。

### 3. Muse へ
kit の commit と spec 名を Issue #31 に書く。Muse は自分の 2 枠で流す。

## 現在の状態（2026-11-07）
ポリシー適用前の 2 本が走っている:
- `uzu-eval-4b-rtn-probe`（4B RTN eval v3、100 chunks）
- `uzu-emb8-08b-gpu`（emb8 v4、dedup）

**この 2 本は完走させる**（途中で切るとクォータが無駄）。次の投入から 1 枠空けを守る。

## T4 の制約（重要）

T4 は compute capability 7.5（Turing）で、**BF16 のテンソルコアが無い**。
BF16 matmul を要求すると:

```
UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32
```

対策は jax のバージョンを変えることではなく、**float32 で走らせる**こと。

- `run_lane` は cc < 8.0 を検出すると eval / calibration を自動で float32 に落とす
- `eval_quantized.py --dtype float32` で手動指定も可能
- 使った dtype は `manifest.json` / `result.json` の `eval_dtype` に入る

| 対象 | T4 で可能か |
|---|---|
| 0.8B calibration + eval（float32） | 可（FP32 3.2GB + INT4 0.55GB） |
| 0.8B eval のみ | 可 |
| 4B eval | **不可**（FP32 16GB + INT4 2.4GB で溢れる）→ Modal |
| 9B | 不可 → Modal |

**数値の注意**: float32 で測った BF16 baseline は、bf16 で測った値と一致しない。
run 内の BF16 vs INT4 比較は有効、過去の bf16 run との絶対値比較は不可。

## 確認コマンド

```bash
kaggle kernels list --user yamahei21python -m --sort-by dateRun | head -10
```

RUNNING が 2 本ある状態で新しい本実験を積まない。
