# 0.8B 追加測定 5本

対象: Qwen3.5-0.8B。目的は 4B/9B へ転移する判断材料の取得。全部やるが、B は条件付き。

## 1. 校正比率スイート（実行中）
- mix65 / mix35 / JA-only
- 出力: `yaqa-hin-hout-{mix65,mix35,jaonly}-int4-g32/ppl_{en,ja}.json`
- 判定: mixed50（EN 43.17 / JA 41.70）を基準に、EN/JA のバランス最良を選ぶ
- コスト: ローカル CPU 約1時間
- 状態: mix65 EN eval 中

## 2. QAD 50step（実行中）
- Kaggle T4 x2 / replicated / 50 step / seq 128
- 出力: `/tmp/qad_kaggle_out_v5/qad`
- 回収: loss、PPL EN/JA、uzu generate EN/JA
- 判定: YAQA E3（43.81 / 43.05）を EN/JA 両方で上回るか
- 状態: RUNNING

## 3. JAX vs uzu logits parity（調査中）
- 目的: QAD export の scales 軸入替 / signs 反転を検出
- 成果物: `research/uzu_logits_parity.md`
- 次の実装: 任意 token 列の raw logits を uzu から取得し、lalamo JAX と比較
- 判定: top-1 一致、KL、cosine。閾値はレポートに従う
- コスト: GPU 不要、実装 2〜4時間

## 4. tied embedding/readout 8bit vs 4bit（待機）
- 0.8B は tied なので emb と head は分離不可。tied 束だけ比較
- 条件: E3 と同じ linear YAQA + `--quantize-embedding --embedding-bits 8`
- 比較先: E3b（emb 4bit、EN 45.07 / JA 44.37）
- 出力: `yaqa-hin-hout-emb8-int4-g32/ppl_{en,ja}.json`
- 判定: 8bit で JA 44.37 が 43.5 以下まで戻るか
- コスト: 再校正 + 変換 + eval で約1時間
- 状態: ratio suite 完了待ち

## 5. QAD 200step（条件付き）
- 条件: QAD 50step が YAQA E3 を EN/JA 両方で上回る
- 内容: 同じカーネルを `--steps 200` に変更
- 判定: 50step からさらに改善するか。過学習/忘却も見る
- コスト: 50step の約4倍
- 状態: 50step 結果待ち

## 実行順
1. 比率スイート（進行中）
2. QAD 50step 回収（進行中）
3. emb8 ablation（比率スイート後）
4. logits parity（調査レポート到着後）
5. QAD 200step（50step 次第）

## ログ
`logs/2026-11-07/` に集約。各条件の `ppl_*.json` は `experiments/qwen35-0.8b/` 配下。
