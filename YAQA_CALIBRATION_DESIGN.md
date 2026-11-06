# YAQA Calibration 汎用設計

対象: lalamo（量子化・変換）、uzu（推論）。
前提コミット: lalamo `e401e17761f2fcd337b0cf20218d67d656e6ecba`。

## 1. 結論

RHT + INT4 Group32 + YAQA のパイプラインは、モデル非依存部分とモデル依存部分に分離できる。

| 処理 | モデル依存 | 共通化 |
|---|---|---|
| RHT（Hadamard block 32） | なし | 可能 |
| INT4 Group32（非対称、output_input） | なし | 可能 |
| YAQA 丸め（`yaqa_round_blockwise`） | なし | 可能 |
| Sketch B 統計（H_I / H_O） | なし（数学は同一） | 可能 |
| Preconditioner 生成・Damping・PSD 検査 | なし | 可能 |
| HybridMatrix 保存・復元・uzu metadata 変換 | なし | 可能 |
| 量子化対象 Linear の抽出 | あり | アダプタ |
| Forward / Backward の実行 | ほぼなし | 共通（lalamo Model API） |
| MoE / 共有 Expert / 特殊層の統計集約 | あり | アダプタ |

要点: **アルゴリズムは共通、対象層の選び方だけモデル別**。

## 2. 3層アーキテクチャ

```
┌──────────────────────────────────────────┐
│ Layer 1: Model Adapter                   │
│  - LinearSite の列挙・フィルタ           │
│  - MoE の component 軸の扱い             │
│  - 量子化可否（shape 整除性）判定        │
└──────────────────────────────────────────┘
                    ↓ LinearSite[]
┌──────────────────────────────────────────┐
│ Layer 2: Universal YAQA Calibration      │
│  - Forward → pseudo target サンプリング  │
│  - per-sample gradient G_b の取得        │
│  - H_I = mean(G_b^T G_b)                 │
│  - H_O = mean(G_b G_b^T)                 │
│  - Damping / PSD / Cholesky 検査         │
│  - PreconditionerDict の生成・保存       │
└──────────────────────────────────────────┘
                    ↓ PreconditionerDict
┌──────────────────────────────────────────┐
│ Layer 3: Universal Quantizer             │
│  - RHT (HadamardSpec 32, input_output)   │
│  - YAQA rounding (blockwise)             │
│  - INT4 Group32 (asymmetric)             │
│  - HybridSpec 平坦化 + encoding.json     │
└──────────────────────────────────────────┘
```

## 3. インターフェース

### 3.1 LinearSite

モデル非依存の「量子化対象」。

```python
@dataclass(frozen=True)
class LinearSite:
    key: tuple[str, ...]      # PreconditionerDict のキー（lalamo のパス）
    linear: Linear            # lalamo.modules.linear.Linear
    out_dim: int
    in_dim: int
    components: tuple[int, ...]  # MoE の expert 軸。通常は ()
    quantizable: bool          # 32 整除・layout 対応など
```

lalamo 標準 API で大半は作れる。

```python
from lalamo.utils.surgery import select_nodes_of_type
sites = [LinearSite(path=path, linear=linear, ...) for path, linear in select_nodes_of_type(Linear, model)]
```

### 3.2 ModelAdapter（最小実装）

```python
class ModelAdapter(Protocol):
    def load(self, directory: Path) -> Model: ...
    def sites(self, model: Model) -> list[LinearSite]: ...
    def sample_texts(self) -> list[str]: ...
```

`forward` と `loss` は lalamo の `Model.decoder` / `logits` で共通化できるため、
アダプタに必須なのは **sites と校正テキストだけ**。

### 3.3 共通エンジン

```python
def collect_sketch_b(
    model: Model,
    sites: list[LinearSite],
    token_batches: list[list[int]],
    *,
    max_tokens: int,
    damping_ratio: float,
) -> PreconditionerDict:
    ...

def quantize_model(
    model: Model,
    preconditioners: PreconditionerDict,
    spec: HybridSpec,
) -> Model:
    ...
```

## 4. Sketch B の数学

サンプル `b`、トークン `t`、Linear への入力 `x_{b,t}`、出力側勾配 `g_{b,t}`。

```
G_b = Σ_t g_{b,t} x_{b,t}^T
H_I = (1/N) Σ_b G_b^T G_b
H_O = (1/N) Σ_b G_b G_b^T
```

`lalamo` の `G_b` は「その Linear の重みに関する損失勾配」と一致する。
したがって、per-sample loss を用意して `jax.grad` を重みに取ればよい。

```python
G_b = jax.grad(loss_for_weight)(linear.weights.weights)  # (out, in)
```

これにより、活性化の記録フックを自作せずに Sketch B を実装できる。

## 5. メモリと数値安定性

- `H_O` は `out_dim^2`。`in_proj` 等では数GBになる。
  - 全モデル両側は非現実的。まず **1〜2 層で両側**、全モデルは **入力側のみ** を基本候補にする。
- `H_I` は `in_dim^2`。`1024〜3584` 程度なら 4〜51 MB/層。
- `block_ldl` は Cholesky を使うため正定値必須。
  - `H += λ I`, `λ = mean(diag(H)) * 1e-3` を既定候補。
  - 固有値・Cholesky・NaN を事前検査。
- `IntSpec` は `yaqa_round_blockwise` を呼ぶ。`ConvergenceError` は `yaqa_round_fixpoint` のみなので、現経路の成否判定には使わない。
- Pseudo target はモデル自身の logits からサンプリング（公式 Sketch B 準拠）。
  - Calibration データと評価データは分離。

## 6. モデル別の差分

| モデル種別 | 差分 | 対応箇所 |
|---|---|---|
| Qwen / Llama / Mistral Dense | なし | Layer 2/3 をそのまま |
| Gemma Dense | 量子化対象の取捨（norm/embedding） | Layer 1 のフィルタ |
| MoE（Qwen3-MoE, DeepSeek） | expert 軸の G_b → expert ごとの H | Layer 1 + Layer 2 の component 対応 |
| Hybrid Attention / DeltaNet | 対象 Linear の分類 | Layer 1 のフィルタ |
| 特殊 readout / tied embedding | 量子化しない | Layer 1 |

重要なのは **量子化アルゴリズムをモデル別にしないこと**。
モデル差は「どの Linear を、どの軸で統計するか」に閉じ込める。

## 7. 検証ゲート

1. 単一 Linear（小さい out_proj）で Sketch B → H_I/H_O → RHT → YAQA → export が通る
2. RTN と YAQA の重み誤差・重み付き目的関数を比較
3. 0.8B 全 Linear を入力側 YAQA で量子化し、logits 指標を RTN と比較
4. 0.8B 両側 YAQA は選択層のみ（メモリ制約）
5. 別系列 Dense モデル（Llama 等）で Layer 2/3 が無変更で動くことを確認
6. 最後に 9B

## 8. 現行スクリプトとの関係

- `scripts/rht_int4_convert.py`: Layer 3 に相当。`--no-rht` で E0（RHT なし）を生成できる。`preconditioner` 引数を追加すれば YAQA 対応。
- `scripts/compare_logits.py`: 評価。変更不要。
- `scripts/verify_rht_int4_model.py`: 保存データの健全性検査。変更不要。
- `scripts/verify_rht_roundtrip.py`: RHT 往復の回帰テスト。
- `scripts/yaqa_calibration.py`: Layer 2。Sketch B の全 Linear 一括逆伝播、Damping、固有値検査、`preconditioners.npz` 出力。
- `scripts/eval_quantized.py`: teacher-forced PPL / KL / Top-1 / Top-5 / cosine / margin。
- `scripts/fetch_eval_corpus.py`: HF データセットを固定コーパスとして保存。
- `scripts/experiment_record.py`: 実験の JSONL 記録。

## 9. 実験マトリクス（固定条件）

| 実験 | RHT | YAQA | 目的 |
|---|---|---|---|
| E0 | なし | なし | INT4 そのものの下限 |
| E1 | あり | なし | RHT の効果（現行 RTN） |
| E2 | あり | H_in のみ | 一方向 YAQA |
| E3 | あり | H_in + H_out | 両方向 YAQA（Sketch B） |

固定する条件:
- モデル revision、lalamo commit、uzu commit
- 量子化対象層、layout、group size、bits、symmetric
- Calibration / Validation / Test データ
- 乱数シード（quantization key, calibration sampling, target sampling）
- 評価プロンプト、トークン位置

## 10. データ分割

| 用途 | 内容 | 使途 |
|---|---|---|
| Calibration | テキストコーパス（数百〜数千サンプル） | G_b, H_I, H_O の推定 |
| Validation | 別コーパス / プロンプト | Damping・設定選択のみ |
| Test | 評価プロンプト30〜50件 | 最終評価。調整に使わない |

Calibration の擬似ターゲットはモデル自身の logits からサンプリングする（Sketch B 準拠）。
Test を見ながら Damping を選ばない。

## 11. 記録する指標

層ごと:
- weight RMSE / relative error
- 重み付き目的関数 `tr(E H_I E^T H_O)`
- RTN に対する改善率
- Preconditioner の最小固有値・最大固有値・条件数
- Cholesky 成否

モデル全体:
- Top-1 一致率 / Top-5 重複 / KL / logits cosine（固定トークン列）
- Perplexity（BF16 と INT4 の差分）
- 生成品質（指示追従、日本語、長文）
- Calibration 時間、量子化時間、ピークメモリ、保存容量、推論速度

実験記録は `scripts/experiment_record.py` で JSONL に追記する。
失敗・中断も `status=failed|aborted` と `failure_reason` を残す。

## 12. 段階評価

1. 統計の正しさ（G_b の対応、対称性、PSD、Damping 感度）
2. 量子化品質（E0〜E3、層別誤差）
3. 再現性（量子化直後 → 保存再ロード → uzu ロードの logits 一致）
4. 実用性能（Perplexity、生成、メモリ、速度）
5. 汎用化（Llama 系 Dense で Layer 2/3 無変更）
6. 高度化（層別精度選択、MoE、メモリ最適化）
