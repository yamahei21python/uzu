タスク: ドキュメントと実装の乖離監査（uzu）

## 背景
同じ種類の事故が続いている。
- `--include` 形式の違いで kit 取得に失敗
- `rewrite_lalamo_hybrid_spec.py` が kit に無い
- `run_lane.py` の旧版が kit に残り、修正が反映されていなかった
- 手順書のコマンドが現行の CLI と一致していない可能性

ドキュメントの記述と実装（argparse、既定値、出力ファイル名、必須フラグ）を突き合わせ、**そのまま実行すると失敗する箇所**を洗い出す。

## 入力（公開のみ。clone しない）
- https://raw.githubusercontent.com/yamahei21python/uzu/main/README.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/experiments/PROTOCOL.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/experiments/qwen35-0.8b/RUNBOOK.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/lanes/README.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/docs/KAGGLE_SLOTS.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/antigravity/README.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/tests/README.md
- 実装:
  - https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/run_lane.py
  - https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/yaqa_calibration.py
  - https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/rht_int4_convert.py
  - https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/eval_quantized.py
  - https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/qad_finetune.py
  - https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/build_kit.py
  - https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/ag_queue.py

## やること
### A. コマンドの照合
ドキュメントに載っているコマンドを全部抜き出し、次を確認する:
1. フラグ名が実装に存在するか（`argparse` の `add_argument` と照合）
2. 必須フラグが書かれているか
3. 既定値が一致するか（samples / max_tokens / damping / output_block / group size）
4. 出力パス・ファイル名が実装と一致するか

結果は表: `doc:line | 記載コマンド | 判定(OK/NG) | 実装側の根拠(file:line) | 修正案`

### B. 手順の抜け
手順書を上から実行したと仮定して、**足りない手順**を探す:
- 環境変数（`LALAMO_PATH`, `JAX_PLATFORM_NAME` など）
- 前提ファイル（kit、dataset、tokenizer）
- 事前チェック（`build_kit.py --check`, corpus guard）
- 失敗時の分岐

### C. 用語と数値の不整合
- chunk 数の表現（20 / 50 / 100）が場所によって違わないか
- 「clean」「dedup」の定義が統一されているか
- BF16 baseline の値（EN 35.38 / JA 43.42）が古い値（21.42 / 34.75）と混在していないか

## 成果物
1. `research/docs_code_drift.md` — A〜C、修正案つき
2. `research/docs_code_drift.json` — `[{doc, line, snippet, verdict, evidence, fix}]`
3. `research/docs_fix_list.md` — 修正すべきファイルごとの箇条書き（そのまま作業できる粒度）

## 制約
- 実装の行番号を必ず添える。推測で「たぶん違う」と書かない
- 実行して確かめられないものは "未検証" と明記
- 日本語。20 分以内
- 書き込めない場合は内容を作らずエラー報告

## 完了報告
- NG 件数（コマンド / 手順 / 数値）
- 最優先で直す 3 件
