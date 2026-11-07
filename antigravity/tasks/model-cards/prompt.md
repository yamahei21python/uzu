タスク: 公開 model card / dataset card の最終稿（HF 4 本 + Kaggle 3 本）

## 背景
HF と Kaggle で public 配布している。ライセンス監査で「card に base model・変更点・リスク告知が足りない」と指摘され、
HF モデル 2 本には最低限を足した。しかし、何が入っていて、どう使い、何が壊れているかの記述が薄い。
第三者が再現・利用判断できる card に仕上げたい。

## 入力（公開のみ）
- モデル
  - https://huggingface.co/yamahei21/huihui-qwen3.5-4b-int4-rht-rtn
  - https://huggingface.co/yamahei21/huihui-qwen3.5-9b-int4-rht-rtn
  - base: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated
  - base: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated
- dataset
  - https://huggingface.co/datasets/yamahei21/uzu-quant-kit
  - https://huggingface.co/datasets/yamahei21/uzu-results
- Kaggle: `yamahei21python/uzu-quant-kit-public`, `yamahei21python/uzu-bf16-08b-lalamo`, `yamahei21python/uzu-qad-08b-models`
- 実測値（main に committed）:
  - `research/bpc_fertility_audit.md`（EN/JA BPC）
  - `research/license_audit.md`（ライセンス判定）
  - `research/lalamo_quant_hotspots.md`（速度の支配項）
  - `research/9b_bit_allocation.md`（配分の推定）

## やること
### A. model card（HF 4B / 9B）
各 card に次を含める。**実測値だけを書き、推定は "estimated" と明記**:
1. 何か（base model、量子化方式、サイズ、レイヤ構成）
2. 使い方（ダウンロード、uzu での実行、必要なメモリ）
3. 実測性能（M3 16GB の TTFT / prefill / gen / メモリ。EN と JA）
4. 品質（PPL と BPC。BF16 との差分。**比較可能な範囲**を明記）
5. 制限（YAQA/QAD の有無、abliterated の安全告知、JA の事実誤り）
6. ライセンスと帰属（Apache-2.0、base model、lalamo、uzu）
7. 再現（量子化コマンド、commit）

### B. dataset card（`uzu-quant-kit` / `uzu-results`）
- 何が入っているか（ファイル一覧と役割）
- ライセンスの区分（scripts MIT / data CC BY-SA 4.0）
- 出所（WikiText-2、日本語 Wikipedia）と attribution
- 使い方（`hf download`、`build_kit.py --check`）
- 含まないもの（重み）
- 既知の注意（stale kit、旧 eval の汚染）

### C. Kaggle dataset の説明文
- 3 本それぞれに、1 段落の description（英語）と license の妥当性
- `uzu-ja-corpora` は削除予定であることを明記（新規利用を止める）

### D. 差分
- 現行 card との差分を、そのまま貼れる形で出す
- HF の frontmatter（license / base_model / tags）も含める

## 成果物
1. `research/model_cards/final_4b.md`
2. `research/model_cards/final_9b.md`
3. `research/model_cards/final_dataset_quant_kit.md`
4. `research/model_cards/final_dataset_results.md`
5. `research/kaggle_cards.md`
6. `research/model_cards_diff.md`（現行との差分）

## 制約
- 実測していない数値を書かない。"estimated" と明記するか省略する
- 安全告知（abliterated）を必ず入れる
- ライセンス名を正確に（apache-2.0 / cc-by-sa-4.0）
- 日本語と英語の両方を用意（各 card の冒頭は英語、詳細は日本語で可）
- 20 分以内。書き込めない場合は内容を作らずエラー報告

## 完了報告
- 4B / 9B card で「これは書ける / 書けない（未実測）」の境界
- ライセンス表記で直した点
- 最も重要な制限 1 つ
