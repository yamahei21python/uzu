タスク: INT4 量子化ハイパラの根拠表（文献 + 実測の突き合わせ）

## 背景
uzu は RHT（Hadamard block 32）+ asymmetric INT4 group32、layout output_input を既定にしている。
この既定値（group 32、block 32、damping 1e-3、output_block を使う）は、根拠が弱いまま決まっている。
4B/9B に広げる前に、文献と自前実測を突き合わせて「変えるべきか / 据え置くか」を決めたい。

## 入力
- HF dataset: https://huggingface.co/datasets/yamahei21/uzu-quant-kit
  - `scripts/rht_int4_convert.py`, `scripts/yaqa_calibration.py`, `configs/*.json`
- 自前実測（0.8B base、dedup 済み）:
  - `experiments/qwen35-0.8b/clean-enonly-int4-g32/ppl_en.json`, `ppl_ja.json`
  - 値: BF16 EN 35.38 / JA 43.42、INT4 EN 39.39 / JA 62.52（100 chunks）
  - BPC: EN 1.2428 → 1.2803、JA 3.4484 → 3.7817
- GitHub: https://github.com/yamahei21python/uzu main（`research/bpc_fertility_audit.md`）

## やること

### A. 文献の整理
以下を含む主要手法から、**group size / block size / damping（正則化）/ 対称・非対称 / 出力側ブロックを使うか** の根拠を集める:
- GPTQ
- AWQ
- QuIP# / QuIP
- OmniQuant
- SmoothQuant（活性化側の話として）
- SpinQuant / QuaRot（回転ベース。RHT の位置づけ）
- 可能なら最近の（2025〜2026）実務的なレポート

各手法について:
- 推奨 group size と、その根拠（実験 or 理論）
- damping / Hessian 正則化の推奨値
- 対称 vs 非対称の使い分け
- 出力側（output_block）を入れるかどうか
- 報告されている PPL 劣化幅（モデルサイズ別）

**実在する文献だけを書く。URL と arXiv ID を添える。確認できないものは "未確認" と書く。**

### B. 自前実測との突き合わせ
- 0.8B の実測劣化（EN +3.0% / JA +9.7% BPC）は、文献の報告と比べて大きいか小さいか
- group32 は文献の推奨と一致するか
- damping 1e-3 は妥当か（条件数の実測が `calibration.json` にある。`input_condition` / `input_max_eig` を見る）
- `output_block` を使う条件は妥当か（`output_block: small`, limit 2048）

### C. 判断表
| パラメータ | 現状 | 文献の推奨 | 変更するか | 根拠 | 検証コスト |
|---|---|---|---|---|---|
| group size | 32 | | | | |
| block size (RHT) | 32 | | | | |
| damping | 1e-3 | | | | |
| symmetric | false | | | | |
| output_block | small / 2048 | | | | |
| 量子化対象 | Linear のみ + emb/readout | | | | |

「変える」と判断したものは、0.8B で確かめる具体コマンドを書く（`yaqa_calibration.py` のオプション名で）。

### D. 4B/9B への含意
- 4B/9B で group32 のままにするリスク
- embedding / lm_head の量子化（9B は untied）をどう扱うべきか
- 層別ビット配分に回すべき情報（Hessian trace の使い方）

## 成果物
1. `research/int4_hyperparameter_evidence.md`
2. `research/int4_hyperparameter_evidence.json` — 判断表の機械可読版
3. `research/int4_citations.txt` — URL / arXiv ID の一覧（本文と一致させる）

## 制約
- 数値・引用の捏造禁止。確認できないものは明記
- 有料論文の本文転載はしない（要旨と公開ベンチのみ）
- 日本語で書く
- 実行時間 20 分以内

## 完了報告
- 「変更する」と判断したパラメータ（最大 2 個）と根拠
- 0.8B 実測との差分が文献の想定内か
- 4B/9B に持ち込む際の注意 1 つ
