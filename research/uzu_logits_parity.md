# uzu ↔ lalamo logits parity ハーネス設計（QAD export 無音破壊の検出）

対象: `uzu` @ `2aba75d4`（`pr-24-mtp`）／`trymirai/lalamo` @ `66e0b2af`。本番変更0。§2 example は検証後削除済み。

## 0. 結論

1. **uzu 追加実装不要**【実測】。API は `crates/uzu-engine/src/engine/language_model/eval.rs:31` に既存（commit `0d7cfdf2`、`cargo check -p uzu-engine --lib` 通過）。呼び出し元 0 件、CLI / server / bindings / example 非露出【実測】。作業は consumer 1 ファイル（§2）。QAD 0.8B INT4 から logits 取得成功（1.21 s / 12 token）。
2. 閾値は lalamo 既存実測 11 本から導出すべき。固定「top-1 ≥ 95%」は正常 INT4 でも不能（実測最大 85.6%）で誤り（§5）。
3. 軸入替は shape で落ちる（静かでない）。無音なのは **signs 反転 / N 方向の行順入替 / block_size 変更**。主軸は「同一重み・別エンジン」（Phase 1）、PPL は従（§8）。

## 1. 現状: uzu の logits 取得経路【実測】

| 経路 | 位置 | raw logits | 備考 |
|:--|:--|:--|:--|
| Engine API | `crates/uzu-engine/src/engine/language_model/eval.rs:31` | **可** | 全 position を `Vec<f32>`。呼び出し元 0 |
| 生成ループ | `crates/uzu-engine/src/engine/language_model/stream/stream.rs:740` | 不可 | sampling 後 `drop(logits)` |
| 非対話 CLI | `crates/legacy/cli/src/interactive/non_interactive.rs:23` | 不可 | 引数は model/message/effort/seed（`crates/legacy/cli/src/main.rs:15-25`） |
| HTTP server | `crates/legacy/cli/src/server/chat_completions.rs:93-127` | 不可 | リクエスト型に `logprobs` 無し |
| binding | `crates/legacy/shoji/src/types/session/classification/output.rs:16` | 分類のみ | LM 側に相当型なし |

内部挙動【実測】: `output_range=0..len`（`eval.rs:74`）→ 全行 readout（`crates/uzu-engine/src/encodable_block/decoder.rs:183-195`）→ BF16（`eval.rs:90`）→ f32（`eval.rs:117-123`）。1024 token/回（`eval.rs:36`）。`state.tokens` が position を進める（`eval.rs:69,105`）。

## 2. 最小追加点: example 1 ファイル（prod 0 行）

`backends::metal` は `pub(crate)`（`crates/uzu-engine/src/backends/mod.rs:4`）で型名を直接書けない。公開 `BackendSelection`（`:14`）/ `select_backend`（`:21`）経由。uzu 本体も同型（`crates/legacy/uzu/src/engine/bridge/backend.rs:34`）。以下は **compile + 実行検証済み**（`rows=12`）。引数処理省略。
```rust
use std::path::PathBuf;
use uzu_engine::{backends::{Backend, BackendSelection, select_backend}, engine::Engine};

struct Dump { model: PathBuf, tokens: Vec<u64>, output: PathBuf }

impl BackendSelection for Dump {
    type Output = usize;
    type Error = String;
    async fn select<B: Backend>(self) -> Result<usize, String> {
        let engine = Engine::<B>::new().await.map_err(|e| e.to_string())?;
        let model = engine.load_language_model(&self.model).await.map_err(|e| e.to_string())?;
        let mut state = model.create_empty_state(None, 0).map_err(|e| e.to_string())?;
        let mut logits = Vec::new();
        for chunk in self.tokens.chunks(1024) {
            logits.extend(model.teacher_forced_logits(chunk, &mut state).await.map_err(|e| e.to_string())?);
        }
        let bytes = logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(&self.output, &bytes).map_err(|e| e.to_string())?;
        Ok(self.tokens.len())
    }
}
```

依存追加不要（`serde_json` / `tokio` full は `crates/uzu-engine/Cargo.toml:69-79,126`）。出力は raw LE f32、shape `[len(tokens), vocab]`、vocab は `size / (4*len)` で導出（実測 4,966,400 byte ÷ 5 token = 248,320）。

## 3. teacher-forced と KV / position

一括 prefill。`Z = X‖Y`（`N = P + C`）を 1 回流し、行 `i` が `Z[i+1]` を予測。継続部の必要行は `[P-1, N-2]`。lalamo は `state=None, return_updated_state=False` の stateless forward が最も安全（`lalamo/modules/decoder.py:221`、`scripts/eval_quantized.py:86-98` と同形）。`return_suffix_tokens`（`:231`）は末尾 k 行のみ返す最適化で全行比較では不要。uzu は `output_range=0..len` 固定。C > 1024 は chunk 分割 + state 継続。

| 落とし穴 | 症状 | 対策 |
|:--|:--|:--|
| 1024 上限 | `ContextOverflow`（`eval.rs:41-43`） | `chunks(1024)` + state 継続 |
| 位置 0 の OOD | 先頭 token の予測だけ崩れる【実測】 | chat template 適用後の ID（BOS 含む）を先頭に |
| P の取り違え | 1 token シフトで全行不一致 | `argmax(logits[P-1]) == ids[P]` を assert |
| 行数不一致 | 静かに truncate | `rows == len(tokens)` を assert |

【実測】位置 0 の罠: `"The capital of Japan is"`（BOS 無し）で pos0 ` capital` は rank 585、pos1 ` of` / pos3 ` is` は rank 0。文中 `"The"`（pos7）なら ` capital` は rank 0。

## 4. tokenizer 差異の排除

両者は同じ `tokenizer.json` を読む（uzu `crates/uzu-engine/src/engine/language_model/mod.rs:71`、lalamo `lalamo/model.py`）。**テキストを入力にしない**のが唯一の正解。token id 列を JSON で渡す。RHT 数値比較は同一入力・同一重みでのみ意味を持つ【引用】[1][3]。

【実測】QAD / BF16 dir の `tokenizer.json` は md5 一致（`8439b77c1caca73b57280e363c83b334`）、`vocab_size = 248320`。

| リスク | 実体 | 対策 |
|:--|:--|:--|
| special token 付与 | lalamo `add_special_tokens=False`（`lalamo/token_codec.py:27`）、既存 script は `True`（`scripts/eval_quantized.py:49`） | ID 列を SSOT 化、encode 1 回 |
| chat template 差 | uzu は `config.json` の `prompt_template`、lalamo は独自 codec | template 適用後 ID を uzu で 1 回生成 |
| Unicode 正規化 | テキスト経路のみ | テキストを捨てる |
| tokenizer 差し替え | QAD save 時に再出力 | md5 一致を assert |
| vocab 不一致 | readout 形状差で無言の列ずれ | `size % (4*len) == 0` と `vocab == config.vocab_size` を assert |

## 5. 指標と閾値

定義は既存実装準拠（`scripts/eval_quantized.py:128-151`）: top-1 / top-5 overlap / KL（logsumexp 後）/ cosine（raw）/ margin delta（`|(t1-t2)_a-(t1-t2)_b|` 平均）。【実測】既存 lalamo 測定（6350 token）:

| 系列 | corpus | top-1 | cosine | KL | margin |
|:--|:--|--:|--:|--:|--:|
| official-0.8b-m | ja | 0.7724 | 0.98010 | 0.1539 | 0.489 |
| yaqa-hin-hout | ja | 0.7690 | 0.97893 | 0.1695 | 0.501 |
| yaqa-emb-int4 | ja | 0.7403 | 0.97570 | 0.1992 | 0.551 |
| yaqa-mixed | ja | 0.8014 | 0.98182 | 0.1363 | 0.457 |
| official-0.8b-m | en | 0.8561 | 0.99133 | 0.0590 | — |
| yaqa-hin-hout | en | 0.8444 | 0.99067 | 0.0696 | 0.325 |

帯域: ja top-1 0.74–0.80 / cos 0.9757–0.9818 / KL 0.136–0.199、en 0.81–0.86 / 0.9858–0.9913 / 0.059–0.101。**top-1 ≥ 0.95 は正常値より上**で判定不可。

【実測】uzu ノイズ床（同一重み・batch prefill vs 1 token × 12）: cosine 0.99976–0.99996、`max|Δlogit|` 0.281 = 実装差ゼロ下限。

| Phase | 比較 | 合格（暫定） |
|:--|:--|:--|
| 0 | uzu BF16 vs lalamo BF16（同一 dir） | cos ≥ 0.9995, top-1 ≥ 0.99 |
| 1 | uzu INT4 vs lalamo INT4（同一 dir） | cos ≥ 0.9995, top-1 ≥ 0.99, max\|Δ\| ≤ 0.5 |
| 2 | uzu INT4 vs lalamo BF16 | ja: top-1 0.74–0.86, cos 0.976–0.992, KL 0.13–0.21 |
| 3 | 不合格時 | §8 の表で原因特定 |

Phase 1 が本命【推定】。Phase 2 は checkpoint 品質確認でバグ検出力低（正常帯域広い）。閾値は Phase 0 実測で上書き（§9）。

## 6. 実装手順（QAD 0.8B）

```bash
git clone https://github.com/trymirai/lalamo /tmp/lalamo-direct && (cd /tmp/lalamo-direct && uv sync --extra cpu)
cp -Rc /tmp/qwen35_0.8b_yaqa_hin_hout_mix65_lalamo_eval /tmp/uzu_parity_model
python3 scripts/rewrite_lalamo_hybrid_spec.py /tmp/uzu_parity_model/model.safetensors --no-backup
python3 scripts/normalize_lalamo_config.py /tmp/uzu_parity_model   # 実測 96 entry/-1/+2 key
python3 -c "import json;from tokenizers import Tokenizer as T;t=T.from_file('/tmp/uzu_parity_model/tokenizer.json');json.dump(t.encode('The capital of Japan is',False).ids,open('/tmp/parity/tokens.json','w'))"
cargo run --release --example dump_logits -- /tmp/uzu_parity_model /tmp/parity/tokens.json /tmp/parity/uzu.f32
JAX_PLATFORM_NAME=cpu python3 tools/dump_lalamo_logits.py --model /tmp/qwen35_0.8b_yaqa_hin_hout_mix65_lalamo_eval --tokens /tmp/parity/tokens.json --out /tmp/parity/lalamo.npy
python3 scripts/compare_engines.py --uzu /tmp/parity/uzu.f32 --lalamo /tmp/parity/lalamo.npy --tokens /tmp/parity/tokens.json --vocab 248320
```

【実測】uzu 側は model load 込み 1.21 s（12 token, release）。出力: `elems = 12 × 248320`、全 finite、`absmax = 18.63`、`argmax(last) = " Tokyo"`。lalamo dumper の中核は `scripts/eval_quantized.py:86-98` の `model.decoder(token_ids=..., token_positions=..., state=None, ...)` そのまま。

## 7. `eval_quantized.py` の再利用可否

| 部分 | 判定 | 理由 |
|:--|:--|:--|
| モデルロード / `logits_for`（`:86-98`） | 不可 | lalamo 固定。uzu ローダ無し |
| 指標計算（`:128-151`） | **コピー再利用** | 定義一致で過去値と直接比較可 |
| corpus chunking（`:48-52`） | 可 | ID 列生成のみ流用 |
| `compare_logits.py` | 不可 | lalamo 2 モデル前提（`:79-81`） |

結論: 新規 1 本（`scripts/compare_engines.py`）が必要。= 2 ファイル読込 + 既存指標のコピー。既存 2 script は量子化誤差用、エンジン一致用ではない（用途直交）。

## 8. 障害モード判定表

| 破壊 | shape | 一次指標 | 切り分け |
|:--|:--|:--|:--|
| `incoherence_signs` 反転（出力 norm） | 不変 `[dim]` I32（`embedding_table.rs:230-236`） | cos ≈ 0、top-1 < 0.05 | **無音**。Phase 1 のみ検出 |
| scales の N 方向行順入替 | 不変 | cos 0.90–0.98 | **無音**。cos ≥ 0.9995 で初めて落ちる |
| scales 転置 | 変化（0.8B in_proj `[32,8224]`） | `read_buffer` 即エラー | 有音（`weight_matrix.rs:126-131`） |
| `block_size` 32→64 | 不変 | cos 0.3–0.8 | 無音。signs 順序が変わる |
| Hadamard 二重適用【引用】[2][3] | 不変 | cos 0.3–0.7 | 無音 |
| position / KV ずれ | — | 短系列合格・長系列悪化 | 位置別 cos スキャン |

【実測】`QuantParamsLayout` は `[N,G]`（OutputGroup）/ `[G,pad4(N)]`（GroupOutput）の 2 種（`crates/uzu-engine/src/backends/common/kernel/matmul/mod.rs:31-35,70-104`）。0.8B は `output_input` で `[G,pad4(N)]` を読む。転置は**必ず** shape 不一致（軸入替は静か、は 0.8B で不成立）。

## 9. 未確認事項

- **Phase 0 実測なし**【推定】。本環境に jax なし（`ModuleNotFoundError: jax`）。lalamo logits 未取得。§5 の cos ≥ 0.9995 / top-1 ≥ 0.99 は uzu ノイズ床 0.99976 からの推定。Phase 0 後に上書き。
- `eval.rs` はテスト 0 件・呼び出し元 0 件。実測は 5 / 12 token のみ。1024 近傍、複数 chunk 継続、`ContextOverflow` 境界は未検証。
- uzu の `output_signs` 経路（`embedding.rs:98-99` → `decoder.rs:109-112` → `transformer.rs:113`）と lalamo RHT の適用順序・Hadamard 並びの一致は未確認【推定】。同一テンソルを読むが、順序一致は実測でしか示せない。
- BOS の要否。`add_special_tokens=True` でも BOS は付かない（実測 `[760,6511,314,6124,369]`）。pos0 OOD に影響しうる。
- 十数 position では行順入替を棄却できない。2,000 position 以上が必要【推定】。
- lalamo server は `logprobs` を受ける（`lalamo/server.py:138-139,278`）が返るのは生成 token のみ。teacher forcing の代替にならない。

## 参考文献

- [1] Tseng et al., "Model-Preserving Adaptive Rounding" (YAQA), ICML 2026. https://arxiv.org/abs/2505.22988
- [2] Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference". https://arxiv.org/abs/2404.00456
- [3] Tseng et al., "QuIP#", ICML 2024. https://arxiv.org/abs/2402.04396
- [4] https://github.com/trymirai/uzu ／ [5] https://github.com/trymirai/lalamo
