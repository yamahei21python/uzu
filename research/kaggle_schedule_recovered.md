# Kaggle スケジュール設計（アングラ回収・要検証）

> snapshot 503 のため本文のみ復元。`kaggle_schedule.json` / `kaggle_kernel_plan.md` は未取得。

Kaggle / アングラの実行スケジュール設計（クォータと枠の配分）を完了し、以下の成果物ファイルを作成しました。

### 成果物一覧
1. `research/kaggle_schedule.md` — A〜D の詳細計画（表中心、枠配分、Resume 仕様、危険防止策）
2. `research/kaggle_schedule.json` — レーン × 枠 × 時間 × 依存の構造化メタデータ
3. `research/kaggle_kernel_plan.md` — Robustness Sweep 27条件の 3 カーネル（各 9 条件・約 6h）分割リスト

---

## 完了報告

### 1. 1 週間で終わるレーン数
**全 7 レーン（候補すべて）が 1 週間で完了可能です。**
- **理由**:
  - 全 GPU 実験（clean-suite、balanced-h、robustness sweep 3シャード、4B RTN eval、emb8 追試、QAD 10step）の合計想定工数は **約 26.8〜27.8 時間**（QAD を約 1.0h と仮定）。
  - owner（本実験枠 1 本）と Muse（本実験枠 2 本）の合計クォータは 60 時間/週であり、計画利用時間は **約 45.5%** に収まります。
  - CPU レーン（BPC・監査・guard）は GPU クォータを消費せず並行処理できるため、すべてのレーンを安全バッファ（約 54.5% 残存）を残した状態で 1 週間以内に完走できます。

### 2. robustness sweep を何日で回すか
**約 1〜2 日（実稼働時間 約 12 時間）で回します。**
- **理由**:
  - 27 条件（18 時間相当）を damping 軸で 3 カーネル（各 9 条件・約 6.0 時間）に分割（`robustness-08b-shard{1,2,3}of3`）。
  - Muse の 2 枠を使い、**Round 1 で Shard 1 と Shard 2 を並列実行（6h）**、**Round 2 で Shard 3 を投入（6h）** することで、実時間 12 時間（約 1.5 暦日）で完走させます。

### 3. 最初に着手すべき 2 レーン
1. **`clean-suite-08b`（mix65 / mix35 / jaonly、owner 枠 1）**:
   - 優先度 P0（最優先）。混合コーパスの最適点（mixed50 との比較）を確定させ、後続のモデル変換や評価のベースラインを固めるため。
2. **`balanced-h-08b`（50 / 65 / 35、Muse 枠 1）**:
   - 優先度 P1。コーパス事前混合アプローチと並行して、言語バランス Hessian 加算（raw EN + raw JA）による多言語性能を検証するため。

---

## 主要ポイントの要約

### A. レーン一覧と工数
| レーン | 所要時間 | 環境 | 依存 | 優先度 |
|---|---|:---:|---|:---:|
| `clean-suite-08b` | 2.0〜2.5 時間 | T4x2 | smoke-08b 合格 | P0 |
| `balanced-h-08b` | 2.5〜3.0 時間 | T4x2 | smoke-08b 合格 | P1 |
| `robustness sweep` (27条件) | 18.0 時間 (6h × 3) | T4x2 | clean-suite-08b | P1 |
| `4B RTN eval` (100chunks) | 1.0〜1.5 時間 | T4x2 | 4B RTN 重み | P2 |
| `emb8 追試` | 0.7〜0.8 時間 | T4x2 | smoke-08b 合格 | P2 |
| `CPU レーン` (BPC/監査) | 15〜30 分 | CPU | 各 GPU の `ppl_*.json` | P1 |
| `clean student QAD 10step` | **unknown** (推定 0.5〜1h) | T4x2 (replicated) | clean student, QAD v3 | P3 |

### B. 週次枠割り当てと運用
- **owner 枠 1**: 単発・基盤本実験（clean-suite、4B RTN、emb8）
- **owner 枠 2**: **検証用に常時空ける**（手渡し前 smoke-08b のみ実行、本実験厳禁）
- **Muse 枠 1 & 2**: 重いバッチ本実験（sweep 3シャード、balanced-h）の並列実行
- **12h 制限・Resume**: 全ジョブを 6h 以内にシャード分割。`run_lane.py` の `result.json` チェックにより条件単位で自動スキップ・中間再開。

### D. 危険ポイントと防止策（各 1 行）
1. **同じ条件の二重実行**: spec の出力先 `lane` 名をシャードごとに一意にし、Issue #31 で実行担当・コミットハッシュ・投入 spec 名を一元管理する。
2. **クォータ切れで 12h ジョブが途中で死ぬ**: ジョブ投入前に Kaggle 画面で残クォータが予定工数の 1.5 倍以上あることを確認し、全バッチを 6h 単位のシャードに厳格分割する。
3. **同時 3 本以上立ててしまう**: `kaggle kernels list -m --sort-by dateRun` で RUNNING 数を事前確認し、owner は 1 本（1 枠空け維持）、Muse は 2 本厳守を徹底する。
4. **Kaggle の dataset version 固定忘れ**: ノートブック起動スクリプトで最新版自動取得を禁止し、データセット URL/タグのバージョン固定と `build_kit.py --check` による SHA256 検証をパイプライン先頭で必須化する。
