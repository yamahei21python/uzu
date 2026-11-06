# Mirai / YAQA 量子化再現 技術調査レポート

作成日: 2026-11-07 / 対象: Qwen3.5-0.8B (BF16) → lalamo RHT + INT4 G32 + YAQA → uzu
制約: 本番コード未変更。本レポートは `research/` のみ。

## 0. サマリ

| # | 結論 | 根拠 |
|---|---|---|
| 1 | 公式 Sketch B は「per-sample の完全な `G_b^T G_b` / `G_b G_b^T`」を 1 backward で取得。我々の式は一致。ただし**損失が token sum**、公式は **token mean**。H のスケールが `(B·T)^2` ずれる | `custom_linear_B.py` L52-100, `scripts/yaqa_calibration.py` L127-183 |
| 2 | 我々の既定 `--output-block none` は **H_O を保存せず LDLQ 相当**。YAQA の出力側フィードバックが消える | `calibration.json`, `yaqa_calibration.py` L216-218 |
| 3 | 公式量子化は `H /= mean(diag(H))` 後、`sigma_reg=1e-2` を Cholesky 成功まで加算。我々は固定 `1e-3 × mean(diag)` を保存前に加算 | `lib/algo/finetune.py` L160-186, `quantize_finetune_llama.py` L32 |
| 4 | 公式 0.8B-M との差は **embedding の量子化のみ**。tensor 617 vs 614、metadata 194 vs 193。線形層の shape / dtype / spec は完全一致 | safetensors header 比較 |
| 5 | raw logits は `Decoder::encode(output_range)` で取得可能。**stream.rs 無変更**、`eval.rs` 新規 + `mod.rs` 1 行が最小。メモリは rows×248320×2B | `decoder.rs` L147-210, `embedding.rs` L191 |
| 6 | QAD は lalamo に STE 部品あり、学習ループなし。Muon + AdamW + KL 蒸留の外付け実装が必要 | `lalamo/compressed/utils/rounding.py` L49-90 |

---

## 1. YAQA Sketch B: 論文式と公式実装

### 1.1 論文の定義（arXiv:2505.22988v3, ICML 2026）

Kronecker 近似 `H ≈ H_O ⊗ H_I`。Power iteration（Eq.12/13）:

```
(H_I)_i = E_s[ (∇_W ℓ)^T (H_O)_{i-1} (∇_W ℓ) ] / ||(H_O)_{i-1}||_F^2
(H_O)_i = E_s[ (∇_W ℓ) (H_I)_{i-1} (∇_W ℓ)^T ] / ||(H_I)_{i-1}||_F^2
```

初期値 `H_I = H_O = I` の 1 ラウンドが Sketch B:

```
H_I = E_s[ G_s^T G_s ]        (G_s = ∇_W ℓ, per sequence)
H_O = E_s[ G_s G_s^T ]
```

Sketch B は「1 backward で両方」。Sketch A は token 独立仮定で分散 O(1/tokens)。
丸めは Eq.5/6 の Kronecker 版 LDLQ（`L_O'^T Δ L_I' + L_O'^T Δ + Δ L_I'`）。
出典: https://arxiv.org/abs/2505.22988 / https://arxiv.org/html/2505.22988v3 §3.1-3.2

### 1.2 公式実装（Cornell-RelaxML/yaqa-quantization）

`hessian_llama/custom_linear_B.py` の `it == 0` 分岐:

```python
in_hess  = einsum('btm,btn,bsm,bsk->nk', g, x, g, x)   # = Σ_b G_b^T G_b
out_hess = einsum('btm,btn,bsk,bsn->mk', g, x, g, x)   # = Σ_b G_b G_b^T
# reduce(AVG) 後、buffer rank が加算。最後に /ct（ct = 累積 sample 数）
```

- t,s が独立和 → cross-token 項を含む完全な per-sample 外積。**token 独立近似ではない**。
- 正規化: `hin /= ct`, `hout /= ct`。`ct += bs`。→ **sample 平均**、token 和。
- dtype: `grad_output.float()` / `input.float()`、`op_dtype = float32`。`--fp64_accum` は存在するが `get_hess_llama.py` L88 で **強制的に False**（実装バグ/未使用）。
- pseudo target: `Categorical(logits=logits).sample()`、`torch.manual_seed(i)`（iteration 単位）。
- 損失: `F.cross_entropy(logits_flat, target)` → 既定 **mean（B·T で割る）**。
- power iteration: `--power_iters 1` が既定。2 以上は Eq.12/13 を einsum で交互更新、`/||H||_F^2` 正規化。README も Sketch B を推奨（ctx 2048, 65,536 seq）。
- 対象層: q/k/v/o/up/gate/down のみ。embedding / lm_head は対象外。
出典: https://github.com/Cornell-RelaxML/yaqa-quantization/blob/main/hessian_llama/custom_linear_B.py / `get_hess_llama.py` / `hessian_llama/README.md`

### 1.3 `scripts/yaqa_calibration.py` との差分

| 項目 | 公式 | 我々 | 影響 |
|---|---|---|---|
| 損失 | `mean` over B·T tokens | `-Σ_t log p`（sum） | H が `(B·T)^2` 倍。相対 damping 不変だが seq 長でスケール変動 |
| G_b | 自動微分の per-sample grad | `jax.grad` per sample | 一致 |
| H_I | `Σ_b G_b^T G_b / N` | 同式（`accumulator/count`） | 一致 |
| H_O | 常に取得 | `--output-block none` で**未取得**（L216） | 既定で LDLQ 化。YAQA 再現に必須 |
| 正規化 | sample 平均 | sample 平均 | 一致 |
| 計算 dtype | fp32（grad/input を cast） | 重み fp32、G→fp64 で蓄積、保存 fp32 | 我々の方が高精度。0.8B では問題なし |
| damping | なし（get_hess）→ 量子化時 `σ_reg=1e-2` を成功まで加算 | 保存前に `mean(diag)*1e-3` 固定 | 我々は 10 倍小さい。PD 保証なし |
| power iteration | 1（=Sketch B） | なし（単位初期化相当） | 一致 |
| pseudo target | model softmax から 1 sample / iteration | `jax.random.categorical(seed+idx)` | 分布同じ。seed 粒度のみ差 |
| データ | RedPajama 2048 token × 65k seq | wikitext 行 × 32 sample × 128 token | 分散大・系列長短い。H の biased |
| 層選択 | 7 線形のみ | lalamo の全 `Linear` | lm_head 等を含む可能性（要確認） |
| RHT | 量子化時に H へ適用 | lalamo が `process_preconditioner` で適用 | 一致（H は原基底で保存可） |

出典: `/Users/kohei/Myproject/uzu/scripts/yaqa_calibration.py` L127-218, L253
`/Users/kohei/Myproject/uzu/experiments/qwen35-0.8b/yaqa-hin-int4-g32/calibration.json`（`output_block: false`）

### 1.4 数値安定化の候補（優先順）

| 優先 | 施策 | 理由 |
|---|---|---|
| P0 | `--output-block all`（または small=2048）を既定化 | H_O なしは YAQA でない |
| P0 | 損失を token 平均に変更 or G_b を `T_b` で除算 | 公式とのスケール一致。seq 長依存を除去 |
| P1 | `σ_reg` 適応ループ（1e-2 から ×10、Cholesky 成功まで） | 公式と同じ。固定 damping より安全 |
| P1 | 固有値クリップ（負値→`ε·mean(diag)`）＋ PSD チェック | 数値誤差で非 PD になるのを防ぐ |
| P2 | `H_O` も `mean(diag)` 正規化して保存 | 公式の量子化前処理と一致。damping の相対性を保証 |
| P2 | calibration を 512-2048 token の連続系列に変更 | Sketch B の分散低減（論文 §3.2.2） |
| P3 | sample 数 32 → 128 以上 | 論文は 2K でも SOTA、公式既定 65k |

補足: RHT は直交変換なので `H + λI` の damping は変換後も等方に保たれる。保存前 damping は問題ない。

---

## 2. QAD（Quantization-Aware Distillation）

### 2.1 Mirai ブログの記述

- PTQ: 独自 JAX 実装の YAQA + 数値安定化の minor modification。calibration は内部データセットで Fisher 推定。
- QAD: 未量子化 teacher の rollout 上で軽量蒸留。
  - **Muon + deterministic STE** で量子化重みを最適化
  - **AdamW** で group scales / zero points を最適化
  - データ品質・ドメイン被覆に敏感
- 形式: 4-bit asymmetric integer、4-bit zero point、**BF16 scales**、group 32、block-diagonal RHT block 32（GPU warp 幅）。
出典: https://trymirai.com/blog/quantization

### 2.2 公開コードから見た再現性

`lalamo/compressed/utils/rounding.py`:

```python
# L49: forward = round(clip(values, 0, qmax))
# L31-40: backward mask
inside = (v >= 0) & (v <= qmax)
below_ok = (v < 0) & (grad > 0)      # 範囲へ押し戻す勾配は通す
above_ok = (v > qmax) & (grad < 0)
grad = where(inside | below_ok | above_ok, grad, 0)
```

- `jax.custom_vjp` による **deterministic STE**。stochastic rounding ではない。
- clip 外周で「戻す方向」の勾配のみ許可 → dead gradient を回避。
- `lalamo/compressed/utils/yaqa.py` に YAQA 固定点反復（`yaqa_round_fixpoint`）と block LDL あり。
- **学習ループ・Muon・AdamW・蒸留損失は lalamo に存在しない**（`grep -rni "muon|adamw|distill|finetune"` で該当なし）。QAD 実装は非公開と推定（未確認）。

出典: `/tmp/lalamo-direct/lalamo/compressed/utils/rounding.py`, `yaqa.py`, `preconditioner.py`
https://github.com/trymirai/lalamo

### 2.3 最小 QAD 実験計画（0.8B / 単一 GPU 想定）

| 段階 | 内容 | 設定案 |
|---|---|---|
| 0 | checkpoint | PTQ 済み lalamo モデル（RHT + INT4 G32 + YAQA H） |
| 1 | teacher | BF16 uzu または lalamo fp32。rollout 生成（temperature 0.6, top_p 0.95） |
| 2 | データ | 公開 chat/SFT 1-5k prompt、teacher 出力 512-1024 token。KL 評価用とは分離 |
| 3 | 損失 | `KL(softmax(z_t/T) || softmax(z_s/T))`、T=1。token 平均。必要なら CE 併用 |
| 4 | 最適化 | latent weights: Muon（lr 1e-3〜3e-3, momentum 0.95）; scale/zp: AdamW（lr 1e-3） |
| 5 | STE | `deterministic_round_to_unsigned_grid` を affine 量子化の `round` に適用。scale/zp は連続のまま |
| 6 | 期間 | 200-1000 step、batch 2-4、seq 512。0.8B なら数 GPU 時間（未検証） |
| 7 | 評価 | mean KL（100 prompt）、MMLU-Pro（任意）。BF16 logits の noise floor `8e-4` に注意 |

実装候補: JAX + optax（`optax.contrib.muon` の有無は未確認）。lalamo の `QuantizedSpec.compress/decompress` を微分可能ラッパで包む。蒸留のみなら PyTorch 版 YAQA 実装を流用する選択肢もある。

---

## 3. 公式 0.8B-M と我々の変換の差分

### 3.1 モデル構造（両者一致）

| 項目 | 値 |
|---|---|
| layers | 24（DeltaNet 18 + Attention 6、4 層周期） |
| model_dim / hidden | 1024 / 3584 |
| vocab | 248,320 |
| embedding | tied |
| tokenizer SHA256 先頭 | `87a7830d63fcf43b`（一致） |
| `decoder_config` | **完全一致** |

出典: `/tmp/mirai_08b_model/config.json`, `/tmp/qwen35_0.8b_rht_int4_lalamo/config.json`

### 3.2 safetensors / metadata 差分

| 項目 | Mirai 0.8B-M | 我々 | 判定 |
|---|---|---|---|
| tensor 数 | 617 | 614 | 差 3 |
| metadata 数 | 194 | 193 | 差 1 |
| dtype 内訳 | U8 194 / I32 193 / F32 133 / BF16 97 | U8 192 / I32 192 / F32 133 / BF16 97 | embedding 分のみ |
| embedding | INT4 G32 asym + BF16 scale + U8 zp + RHT(output) | **BF16 [248320,1024]** | **唯一の差** |
| embedding spec | `HybridSpec` + `IntSpec{input_output}` | `FullPrecisionSpec` | 同上 |
| 線形 weights | U8 `[out, in/2]` | 同一 | 一致 |
| 線形 scales | BF16 `[in/32, out]` | 同一 | 一致 |
| 線形 zero_points | U8 `[in/32, out/2]` | 同一 | 一致 |
| layout | `output_input` | 同一 | 一致 |
| incoherence | block 32, `input_output`（embedding は `output`） | 同一 | 一致 |
| モデルサイズ | 418.8 MB（uzu 形式） | 800.7 MB（lalamo 形式） | 差 ≈ 361 MB ≒ embedding BF16 分 |

metadata の値差分は **embedding の 1 キーのみ**。線形 96 層 × 2 spec（`weights.spec` / `weights.quantized.spec`）は完全一致。
出典: `/tmp/mirai_08b_model/model.safetensors` header, `/tmp/qwen35_0.8b_rht_int4_lalamo/model.safetensors` header

### 3.3 config 差分（量子化と無関係）

| キー | Mirai | 我々 |
|---|---|---|
| `generation.temperature` | 0.6 | 0.8 |
| `generation.top_k` | 20 | 40 |
| `generation.repetition_penalty` | 1.1 | 1.15 |
| `token_codec.end_of_thinking_tag` | あり | `null` |
| `token_codec.output_parser_regex` | あり | `null` |

→ 推論品質比較では generation config を揃える必要あり。量子化差分ではない。

### 3.4 含意

1. 線形層の変換は公式と**完全再現**できている。
2. 残差は embedding 量子化。埋め込みは語彙 248k × 1024 で全パラメータの約 60%。ここを BF16 のままにするとサイズ・帯域で大きく不利。
3. embedding は `layout: input_output` + RHT `output` のみ（入力側変換なし）。線形と処理が異なる点に注意。
4. 我々の `H_O` 未取得はモデル形式には現れないが、量子化品質に影響する（§1.3）。

---

## 4. uzu raw logits 取得の最小パッチ案（実装しない）

### 4.1 現状の経路

| 箇所 | 動作 |
|---|---|
| `stream.rs` L232 | `split_logits_row = prefill_cache_skips_trailing_layers()` |
| L236-242 | 最終チャンクを「本文」と「最後の 1 token」に分割 |
| L256-263 | `decoder.encode(..., output_range = Some(len-1..len), ...)` |
| L264-267 | `logits.unwrap()` → sampling へ |
| L649-657 | decode も `output_range = Some(0..batch)` で logits 取得 |
| `decoder.rs` L147-210 | `output_range: Option<Range<u32>>` の行だけ readout を射影 |
| `embedding.rs` L191 | `encode_readout` が `[rows, vocab]` を `self.data_type` で確保 |

→ **`output_range` を広げるだけで全位置 logits が取れる**。ストリームを迂回するのが最小。

### 4.2 案 A（推奨）: `eval.rs` 新規 + `mod.rs` 1 行

```rust
// crates/uzu-engine/src/engine/language_model/eval.rs
impl<B: Backend> LanguageModel<B> {
    /// Teacher-forced logits. 呼び出し側が state を所有。
    /// row_range は 0-based、max 128 行推奨（メモリ制約）。
    pub async fn teacher_forced_logits(
        &self,
        token_ids: &[u64],
        row_range: Range<u32>,
        state: &mut LanguageModelState<B>,
    ) -> Result<Vec<f32>, LanguageModelStreamError<B>> {
        // 1. state.transformer_state.prepare(context_len, batch)
        // 2. TrieNode::flat + BatchTopology::new（prefill L244-255 と同一）
        // 3. command_buffer = create_command_buffer("eval-logits", pool)
        // 4. decoder.encode(token_ids, &batch, Some(row_range), None, &mut state.transformer_state, &mut cb)
        // 5. logits を copyout::<u16>() → half::bf16 → f32
        // 6. cb.end_encoding().submit().await
    }
}
```

`mod.rs` への差分は `pub mod eval;` の 1 行のみ。

| 観点 | 評価 |
|---|---|
| stream.rs 変更 | **0 行**（generate 経路に影響なし） |
| 追加 LOC | 180-250（エラー処理・チャンク分割含む） |
| 影響範囲 | `LanguageModel` の additive API のみ |
| メモリ | `rows × 248,320 × 2B`（BF16）。128 行で 63.6 MB |
| 速度 | 1 forward = prefill と同等。readout は rows に比例 |
| テスト | 小型モデル + 既存 generate との logits 一致（最後の行）で検証可能 |
| 見積 | 0.5-1.5 人日 |

### 4.3 案 B: stream にフック（非推奨）

`LanguageModelStreamOptions` に `logits_sink: Option<Sender<...>>` を追加し、L266 の `sample_last` 分岐で `output_range` を全行に変更。

- 長所: 既存 API で済む。
- 短所: `split_logits_row` の前提（末尾 1 行のみ readout）を壊す。speculator / KV cache / sampling へ波及。1024 token で 508 MB を一度に確保。
- 判定: 評価専用なら案 A が安全。

### 4.4 メモリ見積（vocab 248,320）

| rows | BF16 | F32 |
|---|---|---|
| 32 | 15.9 MB | 31.8 MB |
| 64 | 31.8 MB | 63.6 MB |
| 128 | 63.6 MB | 127.2 MB |
| 1024 | 508.6 MB | 1,017.3 MB |

### 4.5 リスク

| リスク | 内容 | 対策 |
|---|---|---|
| 精度 | `LanguageModel` は `data_type = BF16` 固定（`mod.rs` L78）。logits も BF16 | readout のみ F32 にする `data_type` 引数追加、または KL noise floor を許容 |
| 状態 | `encode` 後の `encode_accept` を忘れると KV 長がずれる | eval 内で accept を呼ぶ or 毎回新規 state |
| チャンク | 1024 token 超は既存 prefill と同様に分割 | `max_batch_size=1024` を踏襲 |
| 位置 | `token_positions` は state から暗黙 | 連続した prefix のみ評価可能（任意位置は未対応） |

---

## 5. 追加の公開情報

| 情報 | 内容 | URL / パス |
|---|---|---|
| 論文 | v3（2026-06-03 改訂）。Sketch B は 65k seq / 30 GPUh（10B）。2k seq でも SOTA | https://arxiv.org/abs/2505.22988 |
| 公式実装 | QTIP 派生。GPLv3。Hessian 収集と量子化が分離 | https://github.com/Cornell-RelaxML/yaqa-quantization |
| 公式 H/量子化モデル | RelaxML の YAQA collection | https://huggingface.co/collections/relaxml/yaqa-6837d4c8896eb9ceb7cb899e |
| Mirai ブログ | 4-bit asym、4-bit zp、BF16 scale、G32、RHT 32、PTQ+QAD、KL noise floor 8e-4 | https://trymirai.com/blog/quantization |
| モデルカード | PTQ + QAD、4-bit asym、BF16 scales、G32、block-diagonal RHT。apache-2.0 | https://huggingface.co/trymirai/Qwen3.5-0.8B-M |
| モデル collection | Qwen3.5 8 モデル（0.8B/2B/4B × M/L） | https://huggingface.co/collections/trymirai/qwen35 |
| lalamo | YAQA 固定点反復、block LDL、deterministic STE、RHT preconditioner 変換 | https://github.com/trymirai/lalamo |
| uzu | 推論エンジン。`output_range` による readout 制御 | https://github.com/trymirai/uzu |
| 元モデル | Qwen3.5-0.8B（hybrid DeltaNet/Attention） | https://huggingface.co/Qwen/Qwen3.5-0.8B |

補足:
- 論文 §3.1 の丸めは LDLQ の Kronecker 一般化。`H_O=I` で LDLQ に一致。
- 公式 `hessian_llama/README.md` は「FP64 蓄積は必須ではない」としつつ、`get_hess_llama.py` が `fp64_accum` を False 固定する矛盾あり。
- QAD の公式コードは公開されていない（未確認）。Mirai ブログ以上の詳細は不明。

---

## 6. 未確認事項

| # | 項目 | 状態 |
|---|---|---|
| 1 | Mirai の「numerical stability の minor modifications」の具体内容 | 未確認（ブログに記述なし） |
| 2 | QAD の正確な損失（KL / CE / top-k）、Muon の設定、step 数 | 未確認 |
| 3 | lalamo に QAD 学習ループが存在するか | 公開 repo には無し。非公開の可能性 |
| 4 | 我々の calibration が lm_head / embedding を含むか | `select_nodes_of_type(Linear)` の実装依存。要確認 |
| 5 | `--output-block none` の calibration.json が示す `samples: 6` と引数 `8` の不一致 | 実行ログ未確認 |
| 6 | uzu の BF16 logits が KL 評価の noise floor に与える影響 | 未測定 |
| 7 | embedding 量子化を lalamo で再現する際の `layout: input_output` の実装差 | 未検証 |
| 8 | `optax.contrib.muon` の安定性（JAX 版） | 未確認 |

---

## 付録: 参照ファイル一覧

- `/Users/kohei/Myproject/uzu/scripts/yaqa_calibration.py`
- `/Users/kohei/Myproject/uzu/crates/uzu-engine/src/engine/language_model/stream/stream.rs`
- `/Users/kohei/Myproject/uzu/crates/uzu-engine/src/engine/language_model/stream/mod.rs`
- `/Users/kohei/Myproject/uzu/crates/uzu-engine/src/engine/language_model/mod.rs`
- `/Users/kohei/Myproject/uzu/crates/uzu-engine/src/engine/language_model/state.rs`
- `/Users/kohei/Myproject/uzu/crates/uzu-engine/src/encodable_block/decoder.rs`
- `/Users/kohei/Myproject/uzu/crates/uzu-engine/src/encodable_block/embedding.rs`
- `/tmp/mirai_08b_model/{config.json,model.safetensors}`
- `/tmp/qwen35_0.8b_rht_int4_lalamo/{config.json,model.safetensors}`
- `/tmp/lalamo-direct/lalamo/compressed/utils/{rounding.py,yaqa.py}`
- `/tmp/lalamo-direct/lalamo/compressed/incoherence.py`
- `/tmp/yaqa_repo/`（公式 repo の取得コピー）
