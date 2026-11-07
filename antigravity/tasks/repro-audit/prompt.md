タスク: クリーンルーム再現監査（手順書だけで環境を作れるか）

## 背景
手順書のコマンドが実装とずれていて、そのまま実行すると落ちる箇所が見つかっている。
RUNBOOK の E2/E3 は `--eval-corpus` が無く、corpus_guard で即終了する。
ドキュメントを直す前に、「新しい環境で、書いてある順に実行して、どこで何が起きるか」を実測してほしい。

## 入力（公開のみ）
- https://raw.githubusercontent.com/yamahei21python/uzu/main/experiments/qwen35-0.8b/RUNBOOK.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/experiments/PROTOCOL.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/docs/LOCAL_POLICY.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/docs/KAGGLE_SLOTS.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/lanes/README.md
- HF dataset `yamahei21/uzu-quant-kit`（scripts/ data/ lanes/）
- 公開モデル: `Qwen/Qwen3.5-0.8B`

## やること
### A. 環境の再現
- 何も無い sandbox に venv を作る
- 手順書の「前提」に書かれていることを、書かれた順に実行する
- `hf download` で kit を取得し、`build_kit.py --check` を通す

### B. RUNBOOK を上から実行
- RUNBOOK の各ステップを**書かれているとおりに**実行する
- モデルの DL と calibration は重いので、`--samples 1 --max-tokens 16` に落としてよい（その事実を明記）
- 各ステップで: コマンド、所要、exit code、最初のエラー 20 行
- 「書いてあるとおりでは動かない」箇所を全部拾う

### C. 修正案
- 各 NG について、正しいコマンド全文を書く（推測ではなく、実行して通ったもの）
- 前提手順の抜け（環境変数、eval copy、calibration-json、eval-corpus など）を列挙
- 修正後の RUNBOOK 差分を markdown で提示（そのまま適用できる形）

### D. 自動チェックの提案
- 手順書のコマンドを機械的に検証する方法（`--dry-run` の有無、argparse との照合）
- 実装するならどこに置くか（`scripts/check_docs.py` 案）

## 成果物
1. `research/repro_audit.md` — A〜D、コマンドと exit code の実測つき
2. `research/repro_audit.json` — step / command / exit_code / error_head / fix
3. `research/repro_audit_stdout.txt` — 実行ログ全文

## 制約
- **repo を clone しない**。raw URL と kit のみ
- 実行して通ったコマンドだけを「修正案」に書く（未検証は "未検証"）
- モデル DL は 1 回だけ。重い calibration は samples 1 に落とす
- 日本語で書く。20 分以内
- 書き込めない場合は内容を作らずエラー報告

## 完了報告
- 「書いてあるとおりでは動かない」件数
- 最初に落ちたステップと原因
- 修正が必要なファイル 1 つ
