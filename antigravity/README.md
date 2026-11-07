# Antigravity tasks（アングラ委任の記録）

委任タスクは **prompt と状態を repo に残す**。sandbox は 7 日で消えるが、ここは残る。

## 配置

```
antigravity/tasks/<slug>/
  prompt.md     # 実際に送ったプロンプト（そのまま再実行できる）
  meta.json     # state パス、ログ、取得先、成果物、状態
```

`meta.json` の例:

```json
{
  "slug": "vram-budget",
  "label": "vram-budget",
  "state": "/tmp/ag_vram_budget.json",
  "log": "logs/2026-11-07/local/antigravity_vram_budget.log",
  "fetch_dir": "/private/tmp/ag_vram_budget_fetch",
  "deliverables": ["research/vram_budget_4b_9b.md", "scripts/memory_model.py"],
  "status": "running",
  "started": "2026-11-07T13:10:00",
  "note": "4B YAQA OOM の主因特定"
}
```

## コマンド

```bash
python scripts/ag_tasks.py list              # 状態一覧（status / 取得済みファイル数）
python scripts/ag_tasks.py list --json
python scripts/ag_tasks.py show vram-budget  # meta + prompt 全文
python scripts/ag_tasks.py collect vram-budget
python scripts/ag_tasks.py collect --all
```

`collect` は `<fetch>/extracted/workspace/<deliverable>` を repo の同じ相対パスへコピーする。
`research/` `scripts/` `lanes/` `tests/` 以外は `research/antigravity/<slug>/` に落とす。

## 状態の見方

| status | 意味 |
|---|---|
| `running` | 実行中。state ファイルは `/tmp/ag_<slug>.json` |
| `started` | state はあるが完了報告が無い |
| `fetched` | 取得したが repo へ未コピー |
| `collected` | repo に取り込み済み。`collected_at` が入る |
| `failed` | 429 / timeout。state を残して `continue` で再開できる |

## 再開

同じ state で継続する。新しいセッションを立てない。

```bash
python3 ~/.config/opencode/antigravity/antigravity.py --state /tmp/ag_vram_budget.json \
  --timeout 1800 continue --prompt "欠けている実測を追加して"
```

## ルール

- 成果物は**必ずローカルで検証**してから採用する（数値の再現、コマンドの実行、テスト）
- 検証に使ったコマンドと結果を PR 本文かタスクの note に残す
- プロンプトには「捏造禁止 / 推定と実測を区別」を必ず入れる
- コード修正は owner。アングラには調査・生成・sandbox 検証までを渡す
