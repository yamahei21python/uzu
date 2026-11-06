# Qwen3.5-9B 投機的デコーディング実現性（検証済み版）

検証日: 2026-11-07 / 対象: Qwen3.5-9B + uzu + Apple M3 16GB
検証手段: ローカル実ファイル、HuggingFace API（モデル一覧・sizes・config.json）、`MODEL_CONVERSION_MANUAL.md`

---

## 1. 前提と現状値

| 項目 | 値 | 根拠 |
| :--- | :--- | :--- |
| 対象モデル | Qwen3.5-9B（text 32層、untied lm_head） | `Qwen/Qwen3.5-9B/config.json`【実測】 |
| 推論 RSS | 7.2〜8.6GB | ユーザー提示【実測】 |
| 速度 | 7.5〜10.5 tok/s | ユーザー提示【実測】 |
| 空きメモリ | 2〜4GB | ユーザー提示【引用】 |
| 帯域 | 約100GB/s | M3 公式仕様【引用】 |

理論上限: `100GB/s ÷ 5.19GB ≈ 19.3 tok/s`【推定】。実測 7.5〜10.5 tok/s は帯域効率 39〜54%【推定】。投機デコードは「1回の重み読み出しで複数トークン確定」により、この効率の壁を崩す。

---

## 2. uzu の Speculator 実装（実ファイル確認）

| 項目 | 事実 | 根拠 |
| :--- | :--- | :--- |
| 配置 | `crates/uzu-engine/src/speculators/mod.rs` は `pub mod dflash_tfm;` のみ（1行） | ローカル【実測】 |
| 本体 | `crates/uzu-engine/src/speculators/dflash_tfm.rs`（406行） | ローカル【実測】 |
| 構成 | `DFlashTfmSpeculator { dflash, weaver: Option<Weaver>, config, shape }` | 同 L87-93【実測】 |
| ロード | `model_path/config.json` → `model_path/shapes.json` → `model_path/model.safetensors` | 同 L101-125【実測】 |
| データ型 | speculator 重みは **BF16 固定**（`DataType::BF16`） | 同 L113【実測】 |
| DFlash | 非自己回帰ブロック拡散。`block_size=16` 並列予測 | speculator `config.json`【実測】 |
| Weaver | 1層ARアダプタ。候補プール上位512のみ評価（全語彙射影を回避） | `speculator_config.weaver_config`【実測】 |
| 検証 | 木構造トークンを1パス検証。`shapes.json` の budget/depth で形状決定 | `dflash_tfm.rs` L399【実測】 |
| 有効化 | `speculator_path` が渡された時のみロード。モデル側の `speculation_supported()` を assert | `engine/language_model/mod.rs` L83-95【実測】 |

### 2.1 実効設定値（`trymirai/Qwen3.5-9B-M` 同梱 speculator）

| キー | DFlash | Weaver |
| :--- | :--- | :--- |
| 次元 | `model_dim=4096`, `hidden_dim=12288` | `model_dim=2048` → `target_model_dim=4096` |
| 語彙 | `vocab_size=248320` | 同左を共有 |
| 層 | 6層（sliding 5 + full 1） | 1層、16 heads |
| 入力特徴 | `target_layer_ids=[1,5,9,13,17,21,25,29]`（32層中8層） | — |
| 並列数 | `block_size=16`, `mask_token_id=248077` | `candidate_pool_size=512`, `max_depth=15` |

【実測: `speculator/config.json`（16,958B）】

### 2.2 shapes.json と M3 の罠

HF 同梱 `speculator/shapes.json` のプロファイル:

| プロファイル | tree_budget | max_tree_depth |
| :--- | :---: | :---: |
| `default` | 8 | 8 |
| `Apple M5` / M5 Pro / M5 Max / M5 Ultra / M6 | 16 | 16 |

【実測】M3 用エントリは**存在しない**。`dflash_tfm.rs` L106-110 は `context.device_name()` の完全一致 → 無ければ `default`。よって M3 では **budget 8 / depth 8** が使われる【実測＋推定】。M3 で 16 を使うには shapes.json に `"Apple M3"` を追加（`default` の複製で可）。

---

## 3. 公開モデル・入手性（HF API 実測）

| リポジトリ | 内容 | サイズ | 状態 |
| :--- | :--- | ---: | :--- |
| `trymirai/Qwen3.5-9B-M` | INT4 本体 + `speculator/` 同梱 | 本体 5,189,121,136B / speculator 755,228,072B | 公開・DL可【実測】 |
| `z-lab/Qwen3.5-9B-DFlash` | BF16 DFlash draft 単体 | 2,583,816,465B | 公開【実測】 |
| `trymirai/weaver` | Qwen3.6-27B 用 Weaver | 226,788,447B | 公開（9B用なし）【実測】 |
| `trymirai/Qwen3.5-9B-M/speculator/shapes.json` | 木形状 | 1,432B | 同梱【実測】 |

重要: **9B 用 DFlash + Weaver は変換不要で配布済み**。`uzu` 側に speculator 変換 CLI は存在しない（`python3 -m uzu.tools.convert_speculator` は**存在しない**。リポジトリ内 `find` で該当なし【実測】）。`MODEL_CONVERSION_MANUAL.md` §7.2 も「公式モデルには同梱済み / `uzu run trymirai/Qwen3.5-9B-M`」と記載【引用】。

---

## 4. ターゲット仕様と Draft 適合性

### 4.1 Qwen3.5-9B 実config（`Qwen/Qwen3.5-9B`）

| 項目 | 値 |
| :--- | :--- |
| 層 | 32（`linear_attention` 24 + `full_attention` 8、interval 4） |
| 次元 | hidden 4096 / intermediate 12288 / head_dim 256 |
| Heads | Q 16 / KV 4（GQA）、linear value heads 32 / key heads 16 |
| 語彙 | 248,320、`tie_word_embeddings=false`（untied） |
| 位置 | 262,144、RoPE θ=10,000,000 |

【実測: `config.json`】※公式リポジトリは `vision_config` も持つ。配布 INT4 は text 運用。

### 4.2 候補比較

| 候補 | 隠れ次元 / 語彙 | 追加メモリ | 適合条件 | 判定 |
| :--- | :--- | :--- | :--- | :---: |
| Qwen3.5-0.8B（AR） | 1024 / 248,320 | INT4 約0.5GB【推定】 | 次元不一致。逐次forwardで帯域2重消費 | △ |
| Qwen3.5-2B（AR） | 2048 / 248,320 | INT4 約1.2GB【推定】 | 同上＋空き2〜4GBを圧迫 | ✕ |
| EAGLE-3 | 4096（特徴量） / 248,320 | 約0.8GB【推定】 | 9B公式重み未公開。GDN(linear attention)未検証 | △ |
| Medusa | 4096 / 248,320 | ヘッド4で約2.0GB（INT4）【推定】 | 248k×4 の lm_head が過大。BF16 で 8GB超 | ✕ |
| **DFlash + Weaver** | 4096 / 248,320 | **755MB（BF16）**【実測】 | uzu ネイティブ。語彙射影を512候補に限定 | **◎** |

DFlash は target の 8層隠れ状態を入力に取るため、**hidden=4096 と vocab=248320 の一致が必須**【実測】。小型 AR モデルをそのまま draft にする方式是外（次元変換層が別途必要）。

---

## 5. 期待 speedup とメモリ

### 5.1 モデル式

期待確定長 `E[τ] = (1-α^(k+1))/(1-α)`、draft コスト比 `c ≈ 0.755/5.19 ≈ 0.145`【推定】。
実効 speedup `S ≈ E[τ] / (1+c)`（帯域理想）。実機では木検証の compute 増で効率 `η≈0.6`【推定】。

| α | k=4 理想 | k=8 理想 | k=8 実機(η=0.6) |
| :---: | :---: | :---: | :---: |
| 0.50 | 1.69x | 1.74x | 1.05x |
| 0.60 | 2.01x | 2.16x | 1.30x |
| 0.70 | 2.42x | 2.79x | 1.68x |
| 0.80 | 2.94x | 3.78x | 2.27x |
| 0.90 | 3.58x | 5.35x | 3.21x |

【推定】α<0.55 では減速。α≥0.7 が採用閾値。DFlash 系の報告 α は 0.7〜0.85 帯【引用: Weaver README, arXiv:2607.06763】。

| 指標 | 投機なし | 投機あり | 差分 |
| :--- | :---: | :---: | :---: |
| 重み | 5.19GB【実測】 | 5.19 + 0.755 = 5.95GB【実測】 | +0.755GB |
| KV/特徴バッファ | — | +0.1〜0.3GB【推定】 | +0.2GB |
| RSS 合計 | 7.2〜8.6GB【実測】 | 8.2〜9.7GB【推定】 | +0.9〜1.1GB |
| 速度 | 7.5〜10.5 tok/s【実測】 | 12〜19 tok/s（α0.7-0.8）【推定】 | 1.7〜2.3x |

`MODEL_CONVERSION_MANUAL.md` は 9B-M を **62 t/s**【引用】と記載するが、測定ハード・条件は未記載（M3 16GB の値ではない可能性大）。M3 では帯域上限 19.3 tok/s【推定】があるため 62 t/s は不可能。

---

## 6. 実運用手順（変換なし）

```bash
# 1. 公式 INT4 + speculator を取得（755MB 同梱）
hf download trymirai/Qwen3.5-9B-M --local-dir ./models/Qwen3.5-9B-M

# 2. そのまま実行（speculator は model ディレクトリ配下を自動検出）
uzu run ./models/Qwen3.5-9B-M
```

ディレクトリ構成【実測】:

```
models/Qwen3.5-9B-M/
├── config.json
├── model.safetensors          # 5,189,121,136B
├── tokenizer.json
└── speculator/
    ├── config.json            # DFlash + Weaver 定義（16,958B）
    ├── model.safetensors      # 755,228,072B
    └── shapes.json            # チップ別 tree_budget
```

設定変更は `speculator/shapes.json` のみ（コード変更不要）。M3 用:

```json
"Apple M3": {
  "tree_budget": 8,
  "max_tree_depth": 8,
  "dflash_depth_override": 8,
  "construction_method": {"type": "Weaver", "rounds": 8, "expand_per_round": 4, "expand_width": 4}
}
```

budget 16 も試す価値あり（メモリ +数十MB【推定】、検証 compute 増）。

---

## 7. フォールバック

| 戦略 | 発動条件 | 動作 | 効果 |
| :--- | :--- | :--- | :--- |
| 通常 AR | α<0.5 が16step継続【推定】 | speculator を外す | 減速回避 |
| budget 縮小 | RSS > 10.5GB【推定】 | shapes.json を budget 4/8 に | メモリ・検証コスト削減 |
| batch | 同時要求 B≥2【推定】 | 投機停止しバッチ | 総スループット優先 |
| KV 量子化 | ctx > 4k【推定】 | KV を INT8/INT4 | キャッシュ 50-75%削減【推定】 |

speculator ロード失敗時は `DFlashTfmSpeculator::new` が `Ok(None)` を返し通常デコードへ（shapes.json 不一致時）【実測: L106-110】。ただし config/重み不整合は `Err` で起動失敗【実測】。

---

## 8. ベンチマーク設計

| 指標 | 定義 | 目標（M3 16GB） |
| :--- | :--- | :--- |
| Acceptance α | 採択 draft token / 提案 token | 0.70〜0.85 |
| E[τ] | 1検証あたり確定トークン | 2.5〜4.0 |
| tok/s | 256 token 生成 / 経過秒 | 12〜19 |
| TTFT | prefill 512 の初手 | <350ms【推定】 |
| Peak RSS | `footprint` 最大 | ≤10GB |
| Swap | `vm_stat` swapouts | 0 |

計測コマンド:

```bash
/usr/bin/time -l uzu run ./models/Qwen3.5-9B-M --prompt "def quicksort(arr):" --max-tokens 256
sudo powermetrics --samplers gpu_power -i 1000 -n 5
```

A/B は speculator ディレクトリを一時退避して同一プロンプト・同一 seed で比較。最低 5 プロンプト × 3 回、中央値で評価。

---

## 9. 未確認セクション

| 項目 | 状態 | 検証方法 |
| :--- | :--- | :--- |
| M3 での実測 α / tok/s | 未測定 | 上記 A/B |
| M3 の `device_name` 完全一致文字列 | 未確認（"Apple M3" と推定） | `MTLDevice.name` をログ出力 |
| 62 t/s の測定ハード | 不明（M5 系と推定） | 出典ブログ確認 |
| 9B 専用 Weaver の有無 | 27B のみ公開【実測】 | HF 検索継続 |
| linear attention 24層の木検証コスト | 未検証 | Metal trace |
| untied lm_head INT4 の候補512精度 | 未検証 | logit 比較 |
| Qwen3.5-9B vision tower の扱い | text-only 運用前提 | config 確認済み、未使用 |

**結論**: 変換不要。`trymirai/Qwen3.5-9B-M` を DL するだけで DFlash+Weaver が動く。M3 では `shapes.json` に M3 プロファイルが無く default(budget 8) に落ちる点のみ要注意。追加メモリ +0.76GB（実測重み）+ バッファ ≈ +0.9〜1.1GB【推定】。α≥0.7 なら 1.7〜2.3x（12〜19 tok/s）【推定】。α<0.55 なら即フォールバック。
