# DFlash 再学習の仕組み（Huihui 9B 用）

目的: Huihui Qwen3.5-9B（abliterated + Claude 蒸留）用の DFlash draft を、
必要になったときに再学習・差し替えできる状態にしておく。

## 1. uzu 側の構成

```
model_dir/
  config.json
  model.safetensors
  tokenizer.json
  speculator/
    config.json
    model.safetensors
    shapes.json
```

- `speculator/` があれば uzu が自動ロード（`engine/language_model/mod.rs`）
- DFlash draft は target の embeddings / lm_head を共有（チェックポイントに含まれない）
- Weaver は任意。DFlash の候補から検証木を作る部品で、MTP 本体ではない

## 2. まず公式 speculator を流用

```bash
python scripts/attach_speculator.py \
  --model-dir /path/to/huihui-9b-uzu \
  --repo trymirai/Qwen3.5-9B-M
```

検証:
- target hidden_dim == draft model_dim (4096)
- target layers >= max(target_layer_ids)+1 (32)
- 9B-M の draft は target_layer_ids `[1,5,...,29]`

流用時の注意:
- draft は Qwen3.5-9B で学習済み。Huihui の abliteration / 蒸留で acceptance が落ちる可能性
- まず acceptance（tokens per forward pass）と tok/s を測る

## 3. Acceptance 計測

```bash
bash scripts/bench_mtp.sh /path/to/huihui-9b-uzu "富士山について3文で説明して"
```

- speculator あり/なしを切り替えて同じプロンプトを実行
- `tokens per forward pass`（acceptance の代理）と `generation speed` を比較
- 目安: acceptance < 2.5、または tok/s が改善しないなら再学習を検討

## 4. 再学習（SpecForge）

公式 DFlash の学習レシピは未公開。第三者実装 **SpecForge** を使う。

### 4.1 データ

1. Huihui 9B（BF16）でプロンプトから rollout を生成
2. 各トークン位置で target の隠れ状態を取得
   - DFlash は target_layer_ids `[1,5,9,13,17,21,25,29]`
   - 最終層の hidden も必要
3. DFlash の top-K marginals を保存
4. プロンプトは calibration / chat / 日本語 / 創作を混ぜる

論文設定:
- 5 層 draft、block size 16
- max seq 3072、1 系列あたり anchor 512
- AdamW、lr 6e-4、6 epochs、cosine
- loss decay γ=7
- 約 800K サンプル（本番）。まず 5〜10K で fine-tune 検証

### 4.2 初期値

- `trymirai/Qwen3.5-9B-M/speculator` の DFlash draft
- または `z-lab/Qwen3.5-9B-DFlash`
- 既存チェックポイントから optimizer / step を復元して継続できる（SpecForge）

### 4.3 計算

- 9B BF16 の hidden state 抽出: 18GB → T4 1枚不可
- Modal A100 40/80GB、または Kaggle CPU 32GB（遅い）
- 学習: DFlash は 5 層と小さいが、target の forward が支配的
- まず fine-tune 1〜2h（A100）で acceptance を確認

### 4.4 変換

SpecForge が出力した HF DFlash チェックポイントを lalamo で uzu 形式へ:

```bash
uv run lalamo speculator convert <HF_DFLASH_REPO> \
  --output-dir /path/to/huihui-9b-uzu/speculator \
  --overwrite
```

Weaver も使う場合:

```bash
uv run lalamo speculator convert <HF_DFLASH_REPO> \
  --weaver <HF_WEAVER_REPO> \
  --output-dir /path/to/huihui-9b-uzu/speculator \
  --overwrite
```

## 5. パイプライン

```
Huihui 9B BF16
   ↓ rollout + hidden states
SpecForge DFlash training
   ↓ HF DFlash checkpoint
lalamo speculator convert
   ↓ speculator/
uzu model_dir
   ↓
bench_mtp.sh で acceptance / tok/s
```

## 6. 未確認・リスク

- SpecForge の Qwen3.5-9B（GDN hybrid）対応は未確認。0.8B/4B の実績のみ
- SpecForge 出力が lalamo の DFlash loader と一致するか未確認
- 9B の hidden state 抽出で KV cache / GDN state の扱い
- ライセンス: SpecForge、z-lab、学習データ
- 公式 9B-M speculator の再配布条件

## 7. 判断基準

| 指標 | 流用可 | 再学習 |
|---|---|---|
| tokens per forward pass | >= 3.0 | < 2.5 |
| tok/s | 1.5x 以上 | 改善なし |
| 日本語/創作での破綻 | なし | あり |

まず公式 speculator を付けて計測し、基準を満たせば再学習しない。
EOF
