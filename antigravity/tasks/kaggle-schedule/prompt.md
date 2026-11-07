タスク: Kaggle / アングラの実行スケジュール設計（クォータと枠の配分）

## 背景
回すべき実験が増えた。資源は限られている。
- Kaggle GPU クォータ: 30h/週/アカウント
- 同時 2 枠/アカウント（owner と Muse で合計 4）
- 1 セッション最大 12h
- owner は **1 枠を本実験、1 枠を検証用に空ける**（`docs/KAGGLE_SLOTS.md`）
- 4B YAQA calibration / 9B eval / 4B·9B QAD は T4 16GB に載らない → Modal か保留

この制約下で、何をどの順で回すかの現実的な計画を作る。

## 入力
- https://raw.githubusercontent.com/yamahei21python/uzu/main/docs/KAGGLE_SLOTS.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/lanes/clean-suite-08b.json
- https://raw.githubusercontent.com/yamahei21python/uzu/main/lanes/balanced-h-08b.json
- https://raw.githubusercontent.com/yamahei21python/uzu/main/lanes/smoke-08b.json
- https://raw.githubusercontent.com/yamahei21python/uzu/main/experiments/qwen35-0.8b/config.json
- HF dataset: https://huggingface.co/datasets/yamahei21/uzu-quant-kit （configs / scripts）

## 前提（実測にもとづく所要時間）
- 0.8B 1 条件（calibration 32/128 + 変換 + eval 100chunks ×2 言語）: 40〜50 分（T4x2）
- balanced H: raw EN + raw JA + 3 比率 × (merge + 変換 + eval) で 2.5〜3 時間
- smoke（2 sample / 2 chunks）: 5〜10 分
- robustness sweep: 27 条件 = 18 時間相当

## やること
### A. レーン一覧と工数
現在の候補を表にする（所要、GPU/CPU、依存、優先度）:
1. clean-suite-08b（mix65 / mix35 / jaonly）
2. balanced-h-08b（50 / 65 / 35）
3. robustness sweep（27 条件）
4. 4B RTN eval（100 chunks）
5. emb8 / embedding 量子化の追試
6. CPU レーン（BPC、コーパス監査、guard）
7. clean student 後の QAD 10step（VRAM 次第）

### B. 週次の割り当て
- owner 枠 1（本実験） / owner 枠 2（検証、空き） / Muse 枠 2
- 1 週間（30h/週/アカウント）で何が終わるか
- 12h 制限を跨ぐジョブの分割方法（どの単位で切るか、resume の効き方）
- クォータの何 % を使うか、失敗時のバッファを何 % 残すか

### C. スイープの詰め方
27 条件を 3 カーネルに分ける具体案:
- 各カーネルの条件リスト（samples / seq / damping）
- 1 カーネルあたりの時間
- 途中で落ちたときの resume 単位
- 並列 2 本で回すときの組み合わせ

### D. 危険ポイント
- 同じ条件の二重実行
- クォータ切れで 12h ジョブが途中で死ぬ
- 同時 3 本以上立ててしまう
- Kaggle の dataset version 固定忘れ
それぞれの防止策を 1 行ずつ。

## 成果物
1. `research/kaggle_schedule.md` — A〜D、表中心
2. `research/kaggle_schedule.json` — レーン × 枠 × 時間 × 依存
3. `research/kaggle_kernel_plan.md` — スイープを 3 カーネルに分けた具体リスト

## 制約
- 所要時間は上記の実測前提を使い、勝手に短く見積もらない
- 不明なものは "unknown" と書く
- 日本語。20 分以内
- 書き込めない場合は数値を作らずエラー報告

## 完了報告
- 1 週間で終わるレーン数
- robustness sweep を何日で回すか
- 最初に着手すべき 2 レーン
