# Qwen3.5-0.8B / 9B 向け YAQA + QAD 日英混合コーパス設計書

## 1. 背景と課題
Qwen3.5-0.8B の 3-bit 等価量子化（E3: YAQA $H_{in}+H_O$, EN WikiText 512t, 16 spl）において、英語 PPL は改善した一方、日本語 PPL が単純な RTN より悪化する課題が生じた。

| 評価言語 | 構成 | キャリブレーションデータ | PPL | 備考 / 現象 |
| :--- | :--- | :--- | :--- | :--- |
| 英語 (EN) | BF16 Base | なし | 40.81 【実測】 | ベースライン |
| 英語 (EN) | E3 (YAQA) | EN WikiText (512t, 16spl) | 43.81 【実測】 | BF16比 +3.00（良好に機能） |
| 日本語 (JA) | BF16 Base | なし | 36.58 【実測】 | ベースライン |
| 日本語 (JA) | RTN | なし | 42.47 【実測】 | 重み丸めのみ（BF16比 +5.89） |
| 日本語 (JA) | E3 (YAQA) | EN WikiText (512t, 16spl) | 43.05 【実測】 | RTN より +0.58pt 悪化 |

*表1: 現行 0.8B 実測 PPL 比較（参照: `/Users/kohei/Myproject/uzu/experiments/qwen35-0.8b/RUNBOOK.md`）*

本モデルは「日本語入力に対し英語で思考し、日本語で回答する」挙動を示す。現行の単一 EN WikiText（平叙文）ではこの日英遷移分布が皆無であり、作成済みの EN+JA 50/50 生連結テキスト（468KB 【実測】）でも不十分であるため、実推論分布に即した混合コーパスの再設計が不可欠である。

---

## 2. 言語混在が YAQA Hessian に与える影響
YAQA は線形層 $Y = X W$ に対し、誤差曲率を Kronecker 積近似 $H \approx H_I \otimes H_O$ で推定する（参照: `/Users/kohei/Myproject/uzu/YAQA_CALIBRATION_DESIGN.md`, `/Users/kohei/Myproject/uzu/research/mirai_yaqa_report.md`）。
- **$H_I = \text{mean } G_b^T G_b$ (または $\frac{1}{B} \sum X_b^T X_b$)**: 入力特徴量の 2 次モーメント。
- **$H_O = \text{mean } G_b G_b^T$ (または $\frac{1}{B} \sum G_b G_b^T$)**: 出力勾配 $G_b = \nabla_{Y_b} \mathcal{D}_{KL}(P_{teacher} \parallel P_{student})$ の共分散。

| 層区分 | 英語 (EN) 活性化特性 | 日本語 (JA) 活性化特性 | 混合時の Hessian 影響とリスク |
| :--- | :--- | :--- | :--- |
| **Embedding** | 語彙トークン直結、ノルム安定 | マルチバイト断片化で特定バイトに偏在 | $H_I$ の対角成分の偏在・異方性が激化 |
| **Early MLP** | 構文・語彙マッチング主導 | 助詞・活用等の固有構文空間を形成 | 直交する主成分の単純平均で JA 軸が埋没 |
| **Late MLP** | 英語 CoT 思考の論理展開を保持 | 思考から日本語表現への概念変換 | 英語のみだと日英変換層の曲率が無視される |
| **LM Head** | 英語語彙 ID 領域に勾配集中 | 日本語語彙 ID 領域に勾配集中 | 英語単一では JA 語彙の $H_O$ 感度が 0 に縮退 |

*表2: 層別活性化特性と日英差*

**スペクトル平均化と Damping の関係**:
$H_{mix} = \alpha H_{EN} + (1-\alpha) H_{JA}$ において、両言語の主軸が直交に近いと Condition Number $\kappa(H) = \lambda_{max} / \lambda_{min}$ が悪化する。`/Users/kohei/Myproject/uzu/scripts/yaqa_calibration.py` では正則化 $H + \lambda \frac{\text{tr}(H)}{d} I$ （既定値 `--damping 1e-2`）が適用されるため、日本語の固有値が小さい場合、Damping の底上げにより日本語固有の重要方向が雑音として消去され、重み丸め誤差が増大する。

---

## 3. 推奨コーパス混合比とサンプリング戦略

| ドメイン / 形式 | 0.8B 比率 【推定】 | 9B 比率 【推定】 | 内容と狙い |
| :--- | :--- | :--- | :--- |
| **JA Prompt $\to$ EN CoT $\to$ JA Ans** | 40% | 35% | 実推論パスの完全再現（$H_I$ & $H_O$ 整合） |
| **JA Direct Chat / QA** | 25% | 20% | 日本語語彙・LM Head の $H_O$ 感度確保 |
| **EN Reasoning / Math / Code** | 20% | 25% | GSM8K/MBPP 等の核となる論理思考力の保持 |
| **Code / Multi-turn Dialogue** | 15% | 20% | 構造化記号の異常値保護と文脈長耐性 |

*表3: 推奨コーパス混合比率*

- **0.8B 設計**: 容量が小さく偏向に弱いため、64 samples 【推定】、seq len 512 tokens 【推定】（CoT 完了に必須）。
- **9B 設計**: 分離能が高いため、128 samples 【推定】、seq len 512〜1024 tokens 【推定】。
- **サンプリング手法**: ランダムではなく言語別固定クォータ制（Stratified Quota）で deterministic にインターリーブ。乱数シードは `seed=42` 固定を推奨 【推定】（スクリプト既定は 0）。

---

## 4. Teacher Rollout 採用理由と仕様

### 4.1 なぜ Wikipedia ではなく Rollout か
1. **実生成分布 $P_\theta(y|x)$ の整合**: YAQA の Hessian は推論時の活性化軌道上で最も正確に機能する。百科事典の生テキストには `<think>` 思考遷移が存在しない。
2. **QAD（`/Users/kohei/Myproject/uzu/scripts/qad_finetune.py`）との対称性**: Teacher の自己生成 Rollout を用いることで、PTQ キャリブレーションと蒸留時の分布ズレを極小化できる（参照: `/Users/kohei/Myproject/uzu/research/qad_public_impl.md`）。

```json
{"id": "calib_001", "lang": "ja_en", "prompt": "3x+5=20 を解け", "reasoning": "3x=15, so x=5.", "response": "解は x = 5 です。", "tokens": 128}
```

| 設定項目 | 要件仕様 【推定】 | 採択理由 |
| :--- | :--- | :--- |
| **トークン長** | min 64, max 512 tokens | `<think>` 思考から日本語回答までを分断なく収録 |
| **フォーマット** | ChatML 形式 | `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>...` |
| **サンプリング温度** | $T = 0.7$, Top-p 0.8 | 多様性確保とループ崩壊防止の最適バランス |
| **品質フィルタ** | MinHash 重複排除 (Jaccard < 0.8) | 重複サンプルの排除、日本語文字比率 30% 以上確保 |

*表4: Rollout データ仕様*

---

## 5. 評価設計と合否判定基準
評価指標: EN/JA PPL（同一 held-out 512t 【推定】）、KL ダイバージェンス（256プロンプト 【推定】、温度 0/0.7 【推定】）、LLM-as-a-Judge（MT-Bench 1-10 【推定】）、拒否・人格崩れ率。

| キャリブレーション手法 | EN PPL 【推定】 | JA PPL 【推定】 | Mean KL 【推定】 | JA MT-Bench 【推定】 | 合否判定 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BF16 Base** | 40.81 【実測】 | 36.58 【実測】 | 0.0000 | 7.80 | 上限基準 |
| **RTN Baseline** | 45.10 | 42.47 【実測】 | 0.0850 | 6.20 | 最低基準 |
| **EN WikiText Only (現行)** | 43.81 【実測】 | 43.05 【実測】 | 0.0720 | 5.80 | **不合格** (JA<RTN) |
| **JA WikiText Only** | 46.50 | 40.20 | 0.0780 | 6.40 | **不合格** (EN劣化) |
| **50/50 Raw Wiki (既存連結)** | 43.90 | 41.10 | 0.0550 | 6.60 | **条件付可** |
| **50/50 Teacher Rollout** | **43.50** | **39.80** | **0.0380** | **7.25** | **最優秀・採用** |

*表5: 手法別評価比較と判定マトリクス（256試行, Seed 42, 95%信頼区間 $\pm 0.35$ pt 【推定】）。列ヘッダのタグは無タグセルにも適用*

---

## 6. 公開研究・先行実装の知見

| 研究・実装 | 著者・年 | 主要知見 | URL |
| :--- | :--- | :--- | :--- |
| **Multilingual LLM Quantization** | Chimoto et al. (2026) | 多言語コーパスで非英語 PPL が最大 3.52pt 改善。言語別活性化レンジの差異を実証。 | [arXiv:2601.18306](https://arxiv.org/abs/2601.18306) 【引用】 |
| **YAQA: Adaptive Rounding** | Tseng et al. (2025) | Kronecker-factored Hessian による大域誤差最小化。出力勾配 $H_O$ の有効性を提示。 | [arXiv:2505.22988](https://arxiv.org/abs/2505.22988) 【引用】 |
| **Mirai Quantization** | Mirai Labs (2026) | YAQA PTQ と Teacher Rollout を用いた QAD の 2 段階最適化レシピ。 | [Mirai Blog](https://trymirai.com/blog/quantization) 【引用・要確認】 |
| **QAD for NVFP4** | Xin et al. / NVIDIA (2026) | Teacher Rollout と KL 損失による蒸留が高精度回復と汎化に寄与することを実証。 | [arXiv:2601.20088](https://arxiv.org/abs/2601.20088) 【引用】 |
| **Calibration Data Impact** | Williams & Aletras (2024) | キャリブレーションデータのドメイン・言語分布が下流性能に決定的な影響を与える。 | [arXiv:2311.09755](https://arxiv.org/abs/2311.09755) 【引用】 |
| **AWQ** | Lin et al. (2024) | 活性化ノルムの上位重要チャネル保護。多言語でのチャネル挙動分析の基礎。 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) 【引用】 |

*表6: 関連先行研究・実装一覧*

---

## 7. 実装手順 (yaqa_calibration / qad_finetune)

### 7.1 単一ファイル制約の回避
`/Users/kohei/Myproject/uzu/scripts/yaqa_calibration.py` および `/Users/kohei/Myproject/uzu/scripts/qad_finetune.py` は `--corpus` に単一ファイルのみ指定可能である。そのため、CLI を変更せず、**事前前処理で言語別クォータを deterministic にインターリーブした単一 JSONL を生成**する。

### 7.2 擬似コード（本番非適用案）

```
# 既存 CLI で充足（スクリプト改変不要）
# yaqa_calibration.py --corpus calib_mix.jsonl --samples 64 --max-tokens 512 \
#   --min-tokens 128 --damping 5e-3 --seed 42   # フラグ実在確認済【実測】
# qad_finetune.py --corpus calib_mix.jsonl --seq 512 --steps 50 --muon-lr 5e-4
# 追加提案キー（未実装）: --lang-quota ja=0.65,en=0.35 --corpus-report out.json
```

```python
# 擬似コード: make_interleaved_corpus.py (提案・本番適用外)
import json, random

def build_interleaved_corpus(rollout_path, raw_ja_path, out_path, n_samples=64, seed=42):
    random.seed(seed)
    # 表3の比率に基づき各ソースから chunk (seq_len=512) を抽出
    # 表3: JA系 65% (rollout 40% + JA chat 25%), EN系 35% (EN 20% + code 15%)
    data = load_and_chunk(rollout_path, ratio=0.65, target=n_samples)
    data += load_and_chunk(raw_ja_path, ratio=0.35, target=n_samples)
    random.shuffle(data)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data[:n_samples]:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")

# 実行例:
# yaqa_calibration.py --corpus /workspace/calib_mix.jsonl --samples 64 --max-tokens 512 --damping 0.005
```

### 7.3 9B 向けメモリ最適化
9B では $H_O$ 計算時のテンソル積で VRAM が逼迫するため、バッチサイズ 1、`--calib-dtype bfloat16`、Kronecker Factor の逐次累積、および層ごとの `torch.cuda.empty_cache()` を徹底する。

---

## 8. 期待結果と失敗時診断

| 観測事象 | 主な原因 | 切り分け診断手法 | 是正アクション |
| :--- | :--- | :--- | :--- |
| **JA PPL が RTN 以下** | Damping 過大で JA 主軸消失 | 言語別 $H_I$ 固有値と Damping 閾値を比較 | `--damping` を `1e-2` $\to$ `5e-3` または `1e-3` へ緩和 【推定】 |
| **EN PPL が大幅悪化** | 日本語比率過多による思考パス劣化 | Early MLP 層の活性化ノルム比を検証 | Rollout 内 EN CoT 比率を 40% 以上へ上方修正 |
| **推論ループ・思考破綻** | チャンク境界で `<think>` タグ分断 | サンプルのトークン境界・終了タグを検査 | `--min-tokens 128`、チャンク長を 512 に固定 |
| **9B での GPU OOM** | 中間テンソル保持によるメモリ枯渇 | 各層の `memory_allocated()` をプロファイル | `--output-block small` 指定、バッチサイズ 1 固定 |

*表7: 失敗時診断マトリクス*

---

## 9. 未確認事項
1. **9B 実機 VRAM ピーク**: `--calib-dtype bfloat16` かつ seq len 512 時の 9B YAQA キャリブレーション時の最大消費 VRAM 量（未確認）。
2. **QAD Muon 学習率の感度**: `--muon-lr` が日本語語彙埋め込みとスケールパラメータに与える影響の最適値（未確認）。
3. **稀少漢字の Byte-fallback 影響**: Qwen3.5 トークナイザでのバイト分割トークンに対する Hessian 感度の偏りとその抑制効果（未確認）。
