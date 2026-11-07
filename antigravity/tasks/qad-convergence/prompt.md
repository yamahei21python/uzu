タスク: QAD（量子化蒸留）の loss 上昇と scale collapse の原因を、コード精読 + sandbox 再現で特定する

## 観測されている事実
uzu の QAD（`scripts/qad_finetune.py`）で、0.8B の蒸留が壊れる。

- 50 step: loss 0.1559 → 1.7415（単調増加）。破棄
- 10 step: val KL 0.1175 → 0.2176 → 0.2269（悪化）
- AdamW で 17〜18 秒/step（T4x2）。Muon は約 80 秒/step（非同期計測バグで `step_s` は信用できない）
- `--scale-lr` を入れても改善せず
- NaN / scale collapse の停止条件に引っかかるケースがある
- 4step では完走、10step で SIGKILL になった回がある（メモリかも）

## 入力
- GitHub: https://github.com/yamahei21python/uzu main
  - `scripts/qad_finetune.py`（本体）
  - `research/qad_slow_loss_diagnosis.md`（既存の仮説）
  - `experiments/PROTOCOL.md`（前提ルール）
- HF dataset: https://huggingface.co/datasets/yamahei21/uzu-quant-kit
  - `scripts/qad_finetune.py` のコピー、`configs/*.json`
- lalamo: https://github.com/trymirai/lalamo commit e401e17761f2fcd337b0cf20218d67d656e6ecba
  - `lalamo/compressed/hybrid.py`, `lalamo/model.py` など量子化 Linear の実装

## やること

### A. コード精読（最優先）
`qad_finetune.py` を通読し、loss が上がる原因になりうる箇所を**行番号付き**で列挙する。必ず次を確認:
1. 蒸留 loss の向き（teacher/student の KL 方向、温度 T の掛け方、T² の有無）
2. 学習対象。量子化 scale を学習しているのか、重みそのものか、両方か
3. scale のパラメータ化。log-scale か生値か。optimizer の lr と初期値の整合
4. quantize/dequantize の straight-through の有無。勾配が scale に流れているか
5. AdamW の weight decay が scale に掛かっていないか（scale が 0 に潰れる典型）
6. 勾配クリップ、scale clamp、初期化（min/max からの算出）の妥当性
7. validation の計算が train と同じ forward を使っているか（eval モード漏れ、dropout）
8. lr schedule / warmup、`scale-lr` の実装が実際に効いているか

各項目について「バグ候補 / 仕様 / 問題なし」を判定し、根拠の行を示す。

### B. sandbox 再現
GPU 不要。小さな線形層（例: 256×1024）を INT4 group32 で疑似量子化し、同じ loss 形（KL + 温度）で scale だけを学習する:
- scale の初期化と lr を振る（1e-6 / 1e-5 / 1e-4 / 1e-3）
- weight decay あり / なし
- log-scale / 生値
- clamp あり / なし
各条件 50 step、loss と scale の統計（min/median/max、0 に潰れた割合）を記録する。
**loss が上がる条件を再現できたら、どの設定が原因かを特定する**。

### C. 既存診断との突き合わせ
`research/qad_slow_loss_diagnosis.md` の仮説を、A/B の結果で採点する（支持 / 否定 / 不明）。新しい仮説があれば追加する。

### D. 検証計画
owner が次に試すべき実験を、優先順に 3〜5 個。各項目に:
- 変更するファイル・関数・行
- コマンド（`--scale-lr`, `--save-every`, `--max-steps` など具体的に）
- 期待される観測（loss が下がる / 横ばい / scale が安定）
- 失敗した場合に何が分かるか

「loss が下がらなければ蒸留の設計自体が間違い」といった切り分けを明確に。

## 成果物
1. `research/qad_convergence_diagnosis.md` — A〜D
2. `scripts/qad_scale_repro.py` — B の再現スクリプト（引数で条件切替）
3. `research/qad_scale_repro_results.json` — 全条件の数値
4. `research/qad_repro_stdout.txt` — 実行ログ全文

## 制約
- 推測と実測を区別する。コードの行番号を必ず添える
- モデル重みの DL は不要（合成データでよい）
- 秘密情報は使わない
- 日本語で書く
- 実行時間 20 分以内

## 完了報告
- loss が上がる主因の候補トップ 2（行番号付き）
- sandbox で再現できた条件（lr / weight decay / scale の持ち方）
- 次に試す 1 手
