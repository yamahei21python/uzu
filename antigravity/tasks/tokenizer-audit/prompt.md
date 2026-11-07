タスク: 日本語 tokenizer の監査（fertility・BPC の解釈・評価セットの偏り）

## 背景
uzu の評価は EN / JA の PPL と BPC で見ている。BPC は `log2(PPL) * tokens/char` で、
tokenizer の分割に強く依存する。Qwen3.5 の tokenizer が日本語をどう分割しているかを把握していない。
「JA の BPC が 3.78 で EN の 1.28 より高い」のが、言語の情報量なのか、tokenizer の都合なのかを切り分けたい。

## 入力（公開のみ）
- 公開 tokenizer: `Qwen/Qwen3.5-0.8B` の `tokenizer.json`（`hf download` で取得）
- コーパス（HF dataset `yamahei21/uzu-quant-kit` の `data/` 6 ファイル）
- 既存の BPC 実測: https://raw.githubusercontent.com/yamahei21python/uzu/main/research/bpc_fertility_audit.md
- 既存の clean 結果: `experiments/qwen35-0.8b/clean-enonly-int4-g32/ppl_{en,ja}.json`

## やること
### A. fertility の分解
コーパスを文字種で分けて fertility を測る:
- 漢字のみ / ひらがなのみ / カタカナのみ / 英数字 / 記号 / 混在
- それぞれ「1 文字あたり何 token か」「1 token あたり何文字か」
- 文字種ごとの寄与を出す（合計が全体の fertility に一致することを確認）

### B. 語彙の使われ方
- JA コーパスで実際に使われた token の種類数、頻度上位 50（ID と文字列）
- 1 token が 3 文字以上を占める割合（漢字熟語がまとまっているか）
- byte-level fallback の発生回数（1 token が 1 byte しか表さないケース）

### C. BPC の解釈
- 同じテキストを「文字あたり」「byte あたり」「token あたり」で正規化した 3 つの指標を並べる
- JA の BPC が EN より高いのは、どれで見ても高いのか / 文字基準だけで高いのか
- 評価に使うべき指標を 1 つ提案（理由つき）

### D. 偏り
- eval コーパスの文字種構成（EN/JA）を出し、実利用（会話文）とどれだけ違うかを議論
- 現在の eval が苦手とする日本語（口語、短い文、記号混じり）を列挙

## 成果物
1. `research/ja_tokenizer_audit.md`
2. `research/ja_tokenizer_audit.json`（fertility 分解・語彙統計）
3. `scripts/tokenizer_audit.py`（再実行可能。`--tokenizer`, `--corpus-dir`）

## 制約
- 推測と実測を区別する
- モデル重みは使わない（tokenizer とコーパスだけ）
- repo を clone しない。raw URL と `hf download` のみ
- 日本語で書く。20 分以内
- 書き込めない場合は数値を作らずエラー報告

## 完了報告
- 文字種別の fertility 上位 3
- BPC の差は tokenizer 由来か言語由来かの結論
- 評価指標の提案 1 つ
