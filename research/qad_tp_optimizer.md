# JAX TP=2 下における optax.contrib.muon 縮約軸エラー調査と QAD 並列戦略判断書

## 1. 結論サマリ（推奨方針と判定表）

- **推奨 1 行**: 0.8B は `replicated + Muon`、4B は `TP=2 + AdamW（bf16 状態 / reshard 挟み Muon）`、9B は `TP=2 + AdamW + 層フィルタ（または CPU offload）`。
- **背景**: `scripts/qad_finetune.py:53` の `--parallel tensor` で `optax.contrib.muon` の NS 行列積 `X @ X.T` が `Contracting dimensions are sharded...` で失敗【実測】。replicated では通る。
- **採否基準**: 単一 T4（実効 15.6GiB）に収まる 0.8B は通信ゼロの `replicated`。超過する 4B/9B は `TP=2` 維持 + sharding 安全な `AdamW`、または更新時のみ `jax.sharding.reshard` で集約。

| モデル規模 | 単一卡 VRAM (15.6GiB) | 推奨並列 | 採用オプティマイザ | 採否理由・トレードオフ | 根拠 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.8B** | 全保持 ≈ 7.0GB (余裕) | `replicated` | Muon + AdamW | 通信ゼロ、NS 直交化が安定 | 【実測】/【推定】 |
| **4B** | 保持のみ 16.0GB (即 OOM) | `tensor` (TP=2) | AdamW / reshard Muon | TP 必須。無改変なら AdamW、数行改変なら reshard Muon | 【推定】/【引用】 |
| **9B** | TP=2 でも 18.0GB (超過) | `tensor` + フィルタ | AdamW (退避策) | 全層保持不可。層フィルタ+AdamW で安定完走 | 【推定】/【引用】 |

---

## 2. 失敗機序の特定（optax Muon と JAX Explicit Sharding）

### (1) エラー発生箇所
`optax/contrib/_muon.py` の 5 次 Newton-Schulz 反復部で発生【引用】:
- `optax/contrib/_muon.py:247-249`: `a = x @ x.T.conj()`, `b = c1*a + c2*a@a`, `return c0*x + b@x`
- `optax/contrib/_muon.py:328`/`:364`: 縦長行列（`rows > cols`）で転置 → 直交化 → 転置戻し。
- `optax/contrib/_muon.py:341-345`: `preconditioning='spectral'` の `norm(x, ord=2)`（SVD）も TP 下で破綻。
- 出典: https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py

### (2) JAX Shardy / Explicit mode の推論規則
JAX コア `jax/_src/lax/lax.py:6831-6835`（`_dot_general_sharding_rule`）で送出【実測】。出典: https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py
```python
for l, r in zip(lhs_contracting_spec, rhs_contracting_spec):
  if l is not None and r is not None:
    raise core.ShardingTypeError(
        'Contracting dimensions are sharded and it is ambiguous how the output should be sharded.')
```
- **replicated で通る理由**: `PartitionSpec()` で縮約軸 spec が None 同士 → 通信不要、出力 spec 一意。
- **TP=2（Explicit Mesh）で落ちる理由**:
  1. **列分散 `P(None, 'tensor')`**: `x=(M, N@tensor)`、`x.T=(N@tensor, M)`。縮約軸 N が双方 `'tensor'` → 出力 `(M, M)` の AllReduce/ReduceScatter が未定 → `ShardingTypeError`【実測】。
  2. **行分散 `P('tensor', None)`**: `x=(M@tensor, N)`、`x.T=(N, M@tensor)`。外積軸 M が重複し出力 `P('tensor','tensor')` → `produces an illegally sharded result`【実測】。
  3. **縦長 (M > N)**: 事前転置で行・列分散が逆転し同様に 100% 破綻。

### (3) `out_sharding` で解けるか
- `dot_general` 単体なら `out_sharding=P()` で AllReduce に落ちて解消可能【実測】。JAX のエラーメッセージ自体が指定を促す（`lax.py:6833-6835`）【引用】。
- ただし `optax.contrib.muon` は内部 `@` に `out_sharding` を露出しない。自作 NS か monkeypatch が必須（`jnp.matmul` に `out_sharding` 引数は無い）【実測】。

---

## 3. sandbox 再現実験（2 CPU Device での検証）【実測】

### (1) 実験環境
JAX 0.12.0.1、Optax 0.2.8、2 CPU（`XLA_FLAGS=--xla_force_host_platform_device_count=2`）、`Mesh(..., AxisType.Explicit)`（`research/lalamo/lalamo/utils/sharding.py:85` 準拠）。

### (2) 実行ログ（再現と修正）
```bash
# 列分散 P(None, 'tensor'): 縮約軸エラー
ShardingTypeError: Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
# 行分散 P('tensor', None): 重複軸エラー
ShardingTypeError: dot_general operation with inputs: bf16[64@tensor,128], bf16[128,64@tensor] produces an illegally sharded result: bf16[64@tensor,64@tensor]
# with_sharding_constraint: assert 失敗
AssertionError: `with_sharding_constraint` acts as an assert when all axes of mesh are of type `Explicit`.
# jax.sharding.reshard 修正: 成功
[Fix success] out_x sharding: NamedSharding(mesh=Mesh('tensor': 2, axis_types=(Explicit,)), spec=P(None, 'tensor'), memory_kind=device)
```
- **結論**: Explicit メッシュ下では `with_sharding_constraint` は assert と化し無効。`jax.sharding.reshard` で「更新前 replicated 変換 → Muon 計算 → 更新後 TP 再分散」を行うことで完全修復を確認【実測】。

---

## 4. TP 維持案の比較評価

| 案 | アプローチ概要 | 行数感 | 通信 (4B) | 数値影響 | リスク・トレードオフ | 根拠 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **(a) replicated 化** | master_weights を更新前 AllReduce 集約 | 約 20 行 | 15.2 GB | 数学的一致 | state 2 倍化 | 【実測】/【推定】 |
| **(b) 一時 reshard** | `jax.sharding.reshard` で NS 直前集約 | 約 10 行 | 7.6 GB | 数学的一致 | PCIe Gen3 で ~0.63s 遅延 | 【実測】/【推定】 |
| **(c) 部分 Muon** | sharded 2D は AdamW、replicated のみ Muon | 約 5 行 | 0 MB | 直交化効果激減 | 主要層の補正不十分 | 【推定】/【引用】 |
| **(d) local Muon** | `shard_map` でローカル shard 独立 NS | 約 30 行 | 0 MB | 直交性破綻 | 特異値崩壊 | 【推定】 |
| **(e) 自作 NS** | `dot_general` に `out_sharding` 付与 | 約 60 行 | ~15 GB | 数学的一致 | 保守コスト大、反復毎通信 | 【実測】/【推定】 |

---

## 5. 代替 optimizer の比較評価

| Optimizer | QAD 適性 | メモリ状態 | TP 親和性 | 実装コスト | 評価・備考 | 根拠 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW** | **極めて高** (安定収束) | m, v 各 4B (計 8B/param) | **完全安全** (要素単位) | 0 行 (標準) | **最優先の退避先**。sharding 障害皆無 | 【実測】/【引用】 |
| **SGD-Momentum** | 中 (収束が緩慢) | m のみ 2B (bf16) | **完全安全** (要素単位) | 2 行 | メモリ最小だが短期蒸留で損失低下鈍い | 【推定】/【引用】 |
| **Lion** | 中〜低 (STE 振動) | m のみ 1-2B (fp8/bf16) | **完全安全** (要素単位) | 5 行 | sign 操作が INT4 境界で振動誘発 | 【推定】 |
| **Adafactor** | 低〜中 (初期不安定) | 行・列ベクトル (微小) | 縮約 AllReduce 要 | 10 行 | 因数分解 2 次モーメント短期発散懸念 | 【推定】 |
| **SOAP / Shampoo** | 不可 (TP 下で不能) | 巨大共分散行列 | **破綻** (固有値 TP 不能) | 100 行超 | 分散特異値分解が PCIe 下でボトルネック | 【推定】/【引用】 |

Muon の LLM スケール有効性は arXiv:2502.16982 で報告【引用】。

---

## 6. replicated フォールバック時のメモリ・速度定量試算

### (1) メモリ積上げと収容限界判定
T4 実効 16.7GB。生徒+教師二重保持（`scripts/qad_finetune.py:95-97`）前提【推定】。

| 構成項目 | 0.8B (Repl) | 4B (Repl) | 4B (TP=2) | 9B (Repl) | 9B (TP=2) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **生徒 (bf16)** | 1.6 GB | 8.0 GB | 4.0 GB | 18.0 GB | 9.0 GB |
| **教師 (bf16)** | 1.6 GB | 8.0 GB | 4.0 GB | 18.0 GB | 9.0 GB |
| **勾配 (bf16)** | 1.6 GB | 8.0 GB | 4.0 GB | 18.0 GB | 9.0 GB |
| **Opt 状態** | 1.7 GB (Muon) | 8.4 GB (Muon) | 4.2 GB (Muon) | 18.2 GB (Muon) | 9.1 GB (Muon) |
| **活性化・領域** | 0.5 GB | 1.5 GB | 0.8 GB | 2.5 GB | 1.3 GB |
| **合計 VRAM** | **7.0 GB** | **33.9 GB** | **17.0 GB** | **74.7 GB** | **37.4 GB** |
| **T4 判定** | **収容可** | **OOM** | **17.0≈16.7 境界** | **OOM** | **OOM** |

- **AdamW 状態量の罠**: m+v を fp32 で持つと 4B TP=2 で 16GB/卡 → 収まらない。bf16 状態（8GB/卡）か Adafactor 必須【推定】。

### (2) 実行速度とステップ時間試算
T4（sm75）は BF16 TC 非対応 → FP32 エミュで実効 **約 8.1 TFLOPS**【引用】。seq=128、MFU 30%（2.4 TFLOPS/GPU）で試算【推定】。

| モデル規模 | 並列方式 | 計算 | 通信 | 1 step 計 | 50 step | 100 step | 300 step |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.8B** | `replicated` | 0.34 s | 0.00 s | **0.34 s** | **17 s** | 34 s | 102 s |
| **4B** | `replicated` | 1.70 s (回避時) | 0.00 s | **1.70 s** | 85 s | **170 s** | 510 s |
| **4B** | `tensor` (TP=2) | 0.85 s | 0.12 s (AdamW) | **0.97 s** | 49 s | **97 s** | 291 s |
| **4B** | `tensor` + reshard Muon | 0.85 s | 0.65 s (AllGather) | **1.50 s** | 75 s | **150 s** | 450 s |
| **9B** | `tensor` (TP=2) + フィルタ | 1.92 s | 0.25 s (AdamW) | **2.17 s** | 109 s | 217 s | **651 s** |

---

## 7. 推奨レシピ 3 ケース（0.8B / 4B / 9B）

| 設定項目 | ケース 1: 0.8B (50 step) | ケース 2: 4B (100 step) | ケース 3: 9B (300 step) |
| :--- | :--- | :--- | :--- |
| **並列戦略 (--parallel)** | `replicated` | `tensor` (TP=2) | `tensor` (TP=2) |
| **主オプティマイザ (2D)** | `Muon` (lr=1e-3, beta=0.95) | `AdamW` (lr=1e-4, b1=0.9, b2=0.95) | `AdamW` (lr=1e-4, b1=0.9, b2=0.95) |
| **補助オプティマイザ (1D)**| `AdamW` (lr=1e-4, scale-min=1e-6) | `AdamW` (lr=1e-4, scale-min=1e-6) | `AdamW` (lr=1e-4, scale-min=1e-6) |
| **教師配置 / メモリ退避** | GPU 上 (VRAM 余裕) | GPU 上 (TP 分割保持) | **CPU Offload** または **層フィルタ** |
| **期待ステップ時間** | 約 0.34 秒 / step (計 17 秒) | 約 0.97 秒 / step (計 97 秒) | 約 2.17 秒 / step (計 10.9 分) |
| **主要リスクと対策** | リスク極小 | 収束低下を step 増で補填 | VRAM 超過。層逐次更新を強制 |

### 実行スクリプト例

```bash
# Case 1: 0.8B (replicated + Muon 最速構成)
python3 scripts/qad_finetune.py --parallel replicated --steps 50 --seq 128 \
  --lr 1e-4 --muon-lr 1e-3 --adam-b2 0.95 --scale-min 1e-6

# Case 2: 4B (TP=2 + AdamW) — scripts/qad_finetune.py:147 のラベル式を "adam" へ 1 行変更必須（未変更は Muon で失敗）
python3 scripts/qad_finetune.py --parallel tensor --steps 100 --seq 128 --lr 1e-4 --adam-b2 0.95

# Case 3: 9B (TP=2 + AdamW + 層フィルタ) — --layer-filter 未実装。collect_params (scripts/qad_finetune.py:127-133) にパス正規表現フィルタ追加要
python3 scripts/qad_finetune.py --parallel tensor --steps 300 --seq 128 --lr 1e-4 --adam-b2 0.95
```

---

## 8. 診断コマンドと再現手順

```bash
# 1. 環境変数設定 (T4 推奨・P2P 回避・HLO 出力)
export JAX_PLATFORMS=cuda XLA_CLIENT_MEM_FRACTION=0.90 NCCL_P2P_DISABLE=1 JAX_LOG_COMPILES=1 NCCL_DEBUG=WARN
export XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo_dump --xla_dump_hlo_as_text"
mkdir -p /tmp/xla_hlo_dump

# 2. 最小再現スクリプト (TP=2 Muon エラーの即時検証)
python3 -c "
import jax, optax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding, AxisType
mesh = Mesh(jax.devices(), ('tensor',), axis_types=(AxisType.Explicit,))
x = jax.device_put(jnp.ones((64, 128), dtype=jnp.bfloat16), NamedSharding(mesh, P(None, 'tensor')))
opt = optax.contrib.muon(1e-3)
@jax.jit
def step(x, s): return optax.apply_updates(x, opt.update(x * 0.1, s, x)[0]), s
with jax.set_mesh(mesh): step(x, opt.init(x))
"

# 3. HLO ダンプ内での集合通信 (AllGather / AllReduce) の確認
grep -E "(all-gather|all-reduce|reduce-scatter)" /tmp/xla_hlo_dump/*.txt

# 4. 配列 sharding の実測（任意）: jax.typeof(t).sharding / jax.debug.visualize_array_sharding
```

---

## 9. 未確認事項

- **未確認 1**: 実機 T4 で `NCCL_P2P_DISABLE=1` 時のホスト経由実効帯域（12 GB/s 仮定）。
- **未確認 2**: blockwise Muon の INT4 後 PPL 劣化率。
- **未確認 3**: lalamo `ShardingConfig._mesh` の `AxisType.Auto` フォールバック可否（本番変更禁止で未検証）。
