# サブエージェント / Antigravity 委任チェックリスト

## 1. 起動時プロンプト

必ず含める:

- 目的・背景・対象ファイル・禁止事項
- 出力先: **`/workspace/research/<name>.md`**
  - Antigravity sandbox の cwd は `/`。`/research` は fetch 対象外
- `target_dir=/Users/kohei/Myproject/uzu/research`
- 数値の扱い:
  - `実測` / `推定` / `引用` を明記
  - 推定は計算根拠を書く
  - 未確認は「未確認」セクションに列挙
- 引用 URL / ファイルパス必須

## 2. 完了条件（これを満たすまで「完了」と報告しない）

1. sandbox の `/workspace/research/` に成果物が存在
2. `antigravity_fetch` でローカル `research/` に回収
3. fetch 失敗時は agent に `cat` を分割実行させ、内容を欠損なく再構成
4. **ローカルファイルを read** し、以下を確認
   - バイト数が想定と一致
   - 見出し・表・コードフェンスが閉じている
   - 途中で切れていない
5. 完了報告に以下を含める
   - ローカル絶対パス
   - バイト数
   - 検証結果
   - 未確認件数
   - Antigravity session_id / environment_id

## 3. 既知の落とし穴

| 問題 | 対策 |
|---|---|
| sandbox cwd が `/` | `/workspace/research/` に保存させる |
| fetch が新規ファイルを含めない | agent に `cat` 全文を分割出力させて再構成 |
| `target_dir` 結合で `research/research/` になる | ローカルで親ディレクトリへ移動して確認 |
| 推定値を実測のように書く | `実測/推定/引用` ラベルを必須化 |
| 環境 quota 蓄積 | `GET /v1beta/environments` → `DELETE /v1beta/environments/{id}` で削除 |

## 4. ローカル検証コマンド例

```bash
ls -l research/
python3 - <<'EOF'
from pathlib import Path
p = Path("research/<name>.md")
text = p.read_text()
print(p, len(text.encode()), len(text.splitlines()))
print("code fences:", text.count("```"))
EOF
```
