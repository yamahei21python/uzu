# ライセンス監査の対応記録（2026-11-07）

`research/license_audit.md` の指摘に対する実施内容。**未対応も明記する。**

## 実施済み（検証済み）

| 対象 | 変更 | 検証 |
|---|---|---|
| HF `huihui-qwen3.5-4b-int4-rht-rtn` | card YAML に `license: apache-2.0` / `base_model` を追加。abliterated の安全告知を追記。`LICENSE`（Apache-2.0 全文）を同梱 | HF API の `cardData.license == apache-2.0` |
| HF `huihui-qwen3.5-9b-int4-rht-rtn` | 同上 | 同上 |
| HF dataset `uzu-quant-kit` | README に dual license（scripts MIT / data CC BY-SA 4.0）と出所（WikiText-2、日本語 Wikipedia）を明記 | HF API の `cardData.license == cc-by-sa-4.0` |
| `models/huihui-*` ローカル | 上記 README と LICENSE を commit | — |

## 未対応（判断待ち・ブロック中）

### 1. Kaggle `yamahei21python/uzu-ja-corpora` の CC0 宣言

- 実測: `kaggle datasets version` でメタデータを CC-BY-SA-4.0 に変えても、**API は `CC0: Public Domain` のまま**（version 6）。Kaggle は既存 dataset のライセンス変更を受け付けない
- 正しい対応は **削除して作り直す**か、非公開化
- ただし 4B RTN eval v3 と emb8 v4 がこの dataset を attach して走っている。終了を待って削除する
- 削除コマンド（実行は保持者の判断で）:
  ```bash
  kaggle datasets delete yamahei21python/uzu-ja-corpora
  ```
- 代替はすでに `uzu-quant-kit-public`（CC-BY-SA-4.0）にある。旧カーネルの `dataset_sources` からは外した

### 2. 上流 `huihui-ai/*` の LICENSE

- 監査は「base model 側に LICENSE 本文が無い」と指摘。これは**上流の持ち物**で、こちらでは直せない
- 対応: 自リポジトリに Apache-2.0 全文を同梱し、`base_model` で帰属を明示した

### 3. Jackrong 由来の蒸留データ

- ライセンス条文が確認できない（`unknown`）。蒸留データの出所を主張しない。QAD の model card には書かない

## 再発防止

- HF に上げる前に `license:` frontmatter と `LICENSE` の有無を確認する
- dataset は「元データのライセンス」を継承する。コーパスに CC0 を付けない
- 監査は公開前チェックリストに含める（`docs/` に追記予定）
