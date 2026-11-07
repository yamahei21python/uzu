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

## チケットキュー（常時 2 本 + 自由枠 1 本）

`ag_queue.py` がチケットを回す。容量 3、常時 2 本を自動で流し、1 本は手動用に空ける。

```bash
# 積む
python scripts/ag_queue.py add --slug <slug> --prompt prompt.txt \
  --deliverables research/x.md scripts/y.py --priority 60 --note "一行"

# 見る
python scripts/ag_queue.py list

# 1 回だけ回す（cron でも可）
python scripts/ag_queue.py pump --max-active 2

# 常駐（120 秒ごと）
nohup python scripts/ag_queue.py watch --max-active 2 --interval 120 \
  > logs/$(date +%F)/local/ag_queue_watch.log 2>&1 &
```

- `pump` は「完了検知 → fetch → collect」と「空き枠への dispatch」を 1 回でやる
- 完了判定は state ファイルの `interaction_id`。ラッパーは成功時にしか state を書かないため、実行中は pid の生存で判定する
- 失敗（pid 死亡 + state なし）は `Rate limit` を含むログなら再キュー、それ以外は `failed`
- `manual` のチケットはキューが触らない。ただし**実行中は枠を 1 つ占有**する（state が無い間）
- `watch` は `antigravity/.queue.lock` を握る。常駐中に手動 `pump` すると「another queue process holds the lock」で終わる（二重起動防止）

### 429 と環境の自動削除

429 は QPS だけでなく**環境の飽和**（実測 ~50 環境 / ~1GiB）でも起きる。`pump` は 429 を検知すると、古い環境を自動で消してから再キューする。

```bash
python scripts/ag_queue.py cleanup-envs --older-than-days 1 --dry-run   # 確認
python scripts/ag_queue.py cleanup-envs --older-than-days 1             # 実行
python scripts/ag_queue.py cleanup-envs --older-than-days 1 --keep <env_id>
```

- 既定は **1 日より古い**環境を削除
- `environment_id` が state / sessions / pending に残っている環境は**絶対に消さない**
- 作成日時が読めない環境も消さない（安全側）
- 実行中タスクは state をまだ書いていないので、**TTL 1 日**がその防御になる。短くしすぎない
- 削除 ID は `logs/<date>/antigravity_deleted_envs.txt` に追記
- 削除は不可逆。`--dry-run` で必ず確認する

### 状態

| queue_status | 意味 |
|---|---|
| `queued` | 順番待ち。priority 降順、同時刻は作成順 |
| `running` | キューが起動。pid 生存中 |
| `manual` | 手で起動。キューは触らないが枠は占有 |
| `done` | fetch + collect 済み |
| `failed` | pid 死亡 + state 無し。`retry` で戻せる |
| `cancelled` | `cancel` 済み |

