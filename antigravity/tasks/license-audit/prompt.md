タスク: 公開成果物のライセンス・model card 監査（uzu の HF / Kaggle 配布物）

## 背景
uzu は量子化モデルとコーパスを public で配布している。二次配布の条件、base model の継承、abliterated モデルの扱い、コーパスのライセンス表示が正しいか確認していない。
誤りがあると公開停止・削除要求になる。公開前に一度通したい。

## 監査対象（公開 URL のみ）
1. HF モデル
   - https://huggingface.co/yamahei21/huihui-qwen3.5-4b-int4-rht-rtn
   - https://huggingface.co/yamahei21/huihui-qwen3.5-9b-int4-rht-rtn
2. base model
   - https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated
   - https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated
   - 上流の Qwen3.5 のライセンス（HF の model card / LICENSE）
3. HF dataset
   - https://huggingface.co/datasets/yamahei21/uzu-quant-kit（現在 CC-BY-SA-4.0 を宣言）
   - コーパスの出所: WikiText-2（CC BY-SA）、日本語 Wikipedia（CC BY-SA 4.0）
4. Kaggle dataset（公開）
   - yamahei21python/uzu-quant-kit-public
   - yamahei21python/uzu-ja-corpora（CC0 を宣言している。要確認）
   - yamahei21python/uzu-bf16-08b-lalamo（Apache-2.0 を宣言）
5. lalamo（量子化実装）と uzu 本体のライセンス。配布物に同梱すべき表示

## やること
### A. 事実確認
各 URL のライセンス名・制約（商用可否、再配布、変更表示、 attribution の要否）を**原文の該当箇所を引用**して整理する。
- 判断できない項目は "unknown" と書き、何が不足しているかを書く
- 推測で「Apache だから自由」と書かない

### B. 現状の不整合
以下を 1 件ずつ判定する:
1. 量子化モデルのライセンス表示は base model と整合しているか
2. model card に base model 名・ライセンス・変更点（量子化手法、abliterated 由来）が書かれているか
3. `provenance.json` は再配布に必要な情報を含むか
4. dataset のライセンス宣言は正しいか（特に `uzu-ja-corpora` の CC0 と、WikiText-2 / Wikipedia 由来の CC BY-SA）
5. Kaggle dataset のライセンス宣言（`uzu-bf16-08b-lalamo` の Apache-2.0 は妥当か）
6. コーパス再配布時の attribution が足りているか

### C. 修正案
- 各 model card / dataset card に**そのまま貼れる修正文**を書く（日本語 + 英語の短い版）
- ライセンス表記の追加・削除を、ファイル名と該当箇所つきで示す
- 削除・非公開にすべきものがあれば明記（判断が割れる場合は両論併記）

## 成果物
1. `research/license_audit.md` — A〜C
2. `research/license_audit.json` — 対象 × ライセンス × 判定（ok / fix / unknown）
3. `research/license_citations.txt` — 参照 URL と、引用した箇所の要約（1 行ずつ）

## 制約
- **repo を clone しない。**必要なファイルは raw.githubusercontent.com から個別に取る
- 実在しない条文・URL・ライセンス名を書かない
- 法的助言ではないと明記する（判断材料の整理まで）
- 有料記事の本文転載はしない
- 日本語で書く。20 分以内
- sandbox に書き込めない場合は、**数値や本文を作らずにエラーを報告する**

## 完了報告
- `fix` 判定の件数と、最優先で直す 1 件
- `unknown` の件数と、確認に必要な情報
