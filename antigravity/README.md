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

## fetch の 503 対策

`fetch` は `503 Environment snapshot is temporarily unavailable` をよく返す。**一時的なので、待てば取れる**。

```bash
python scripts/ag_fetch.py --state /tmp/ag_<slug>.json --out /tmp/ag_<slug>_fetch \
  --log logs/$(date +%F)/local/antigravity_<slug>.log --retries 8 --base-delay 15
```

- 15s → 30s → 60s → 120s → 180s と指数バックオフ（±10% ジッター）
- 503 / 502 / 504 / timeout / connection reset だけ再試行する。403 などは即停止
- 全滅したら**ログから最終メッセージを復元**して `final.md` に保存する
  - エージェントは sandbox に書けなくても、最後の応答本文を返してくることがある（QAD で実際に起きた）
- `ag_queue.py` の `pump` もこの経路を通る（`--fetch-retries`, `--fetch-base-delay`）
- 手動復元: `python scripts/ag_tasks.py recover <slug>`

復元した `final.md` は**数値の検証が必要**。本文中の数値は sandbox で実行できた保証がない。

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
- 失敗判定は慎重にやる。ラッパーは**終了後にしか state を書かない**ので、pid 死亡 + state なしは失敗の証明にならない
  - ログに `"status": "completed"` があれば**完了扱い**で fetch に進む
  - なければ `--pid-grace-seconds`（既定 300 秒）待ってから `Rate limit` / 容量超過を判定
- 失敗（pid 死亡 + state なし + 完了ログなし）は環境圧迫のログなら再キュー、それ以外は `failed`
- `manual` のチケットはキューが触らない。ただし**実行中は枠を 1 つ占有**する（state が無い間）
- `watch` は `antigravity/.queue.lock` を握る。常駐中に手動 `pump` すると「another queue process holds the lock」で終わる（二重起動防止）

### 429 / 容量超過と環境の自動削除

429 は QPS だけでなく**環境の飽和**でも起きる。実測では 12 環境 / 1.925GiB で
`Project environment storage quota exceeded` が返り、新しいタスクが全部即死した（quota は約 1GiB）。

`pump` は次の文字列を「環境の圧迫」として扱い、自動で掃除してから再キューする:
`Rate limit` / `too_many_requests` / `storage quota` / `quota exceeded` / `environment limit` / `RESOURCE_EXHAUSTED`

掃除の既定は **`--mode auto`**: 終了済みタスクの環境を大きい順に消し、合計が `--target-gib`（既定 0.8）以下になるまで続ける。

**自動で 3 段階までエスカレートする**（`cleanup_under_pressure`）:

1. 通常（sessions / running を保護）
2. それでも減らなければ sessions 保護を外す
3. 最後は `--force`（`--keep` 以外すべて対象）

`watch --preemptive-cleanup`（既定 ON）は、dispatch 前に環境の合計が `--env-cleanup-gib`（既定 0.9 GiB）を超えていたら先に掃除する。
上限は実測 ~1GiB。

```bash
# 容量超過のとき（大きい順に削る）
python scripts/ag_queue.py cleanup-envs --mode auto --target-gib 0.8 --dry-run
python scripts/ag_queue.py cleanup-envs --mode auto --target-gib 0.8

# 古いものをまとめて消す
python scripts/ag_queue.py cleanup-envs --mode age --older-than-days 1

# 追加で守りたい環境
python scripts/ag_queue.py cleanup-envs --mode auto --keep <env_id>
```

- `age` モードは 1 日より古い環境を削除。`auto` モードは容量優先
- `sessions/` `pending/` が指す環境は**絶対に消さない**
- 終了済み（done / failed / cancelled / collected）タスクの環境は**消してよい**。`continue` したくなったら取り直す
- 実行中タスクは state をまだ書いていないので、**作成から 10 分未満は消さない**（`--min-age-minutes`）
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

