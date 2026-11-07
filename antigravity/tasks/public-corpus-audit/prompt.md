タスク: uzu 実験用コーパスの公開データ汚染監査（model-free, 重いI/O主体）

## 背景
uzu は Qwen3.5 系モデルを RHT + INT4 に量子化し、YAQA（Hessian校正）と QAD（蒸留微調整）で評価している。
これまで calibration と eval の重複（EN 16行、mixed 5-10行、EN eval 17行、JA eval 754行）が見つかり、dedup 済みファイルを作った。
次に確認したいのは「dedup 後コーパスが、公開データセット側の訓練 split とどれだけ重複しているか」。評価の漏洩リスクを定量化する。

## 入力（公開・認証不要）
HF dataset: https://huggingface.co/datasets/yamahei21/uzu-quant-kit
以下の6ファイルを取得せよ（`hf download yamahei21/uzu-quant-kit --repo-type dataset --include 'data/*'` でよい）:
- data/eval_en_wikitext2_dedup.txt
- data/eval_ja_wikipedia_dedup.txt
- data/calibration_en_wikitext2_dedup.txt
- data/calibration_ja_wikipedia_dedup.txt
- data/calibration_mixed_en_ja_dedup.txt
- data/calibration_mixed_ja65_en35_dedup.txt

## 監査対象（公開データ）
1. WikiText-2 raw（train / validation / test）
   - HF `Salesforce/wikitext`, config `wikitext-2-raw-v1` を streaming で読む
2. Japanese Wikipedia の公開サンプル
   - HF `wikimedia/wikipedia`, config `20231101.ja` を streaming で読み、先頭 200,000 行程度に制限
   - 全件は重いのでサンプルでよい。サンプル数は明記すること
3. uzu コーパス同士の交差
   - eval EN vs calibration EN
   - eval JA vs calibration JA
   - eval EN/JA vs mixed calibration
   - eval EN vs eval JA（言語混入の確認）

## 手法
- 行の正規化は 2 段階で行う:
  - raw: 完全一致
  - norm: 小文字化 → 連続空白を1個に → 前後 strip
- 長さフィルタ: 正規化後 40 文字未満の行は重複判定から除外（定型文ノイズ除去）
- 比較は set 交差。件数と割合（eval/calibration の行数に対する %）を出す
- 代表例を最大 5 件ずつ（長すぎる行は先頭 120 文字に切る）
- トークンではなく行単位でよい。文字数ベースの汚染率も出す

## 成果物（sandbox 内に作成）
1. `research/public_corpus_audit.md`
   - 手法、正規化、閾値、サンプル数
   - 表: コーパス × 監査対象 × 一致行数 × 割合
   - 代表例
   - 結論: どの eval/calibration が漏洩リスクを持つか、dedup で足りているか
   - 再現コマンド全文
2. `research/public_corpus_audit.json`
   - 上記の数値（機械可読）
3. `scripts/audit_public_corpus.py`
   - 上記を再実行できるスクリプト。`--max-ja-rows` 等の引数付き
4. `research/audit_stdout.txt`
   - 実行ログ全文

## 制約
- 秘密情報・認証は使わない。すべて公開データ
- 入力コーパスは書き換えない（読み取り専用）
- 実行時間は 20 分以内を目標。超えそうなら JA のサンプル数を下げ、下げた事実を md に明記
- 数値を捏造しない。取得できなかった項目は "not measured" と明記
- 日本語で書く

## 完了報告
- 作成したファイル一覧とサイズ
- 主要な数値 3〜5 個（最も重複率が高い組み合わせ）
- 実行コマンド
- 取得成功/失敗したデータソース
