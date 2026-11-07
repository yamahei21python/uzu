  - [ ] 2B無検閲モデル uzu化（`TheCluster/Qwen3.5-2B-Uncensored-MLX-bf16` → lalamo変換）
    - 変換時間: ~15分
    - 推論速度: 50+ t/s（投機なし）
    - M3 16GBで軽く動く
    - 優先度: 高

  - [ ] 9B無検閲モデル試行（`TheCluster/MLX-mxfp4` または `nDimensional/safetensors`）
    - 変換時間: 30分〜2時間
    - 推論速度: 15〜20 t/s
    - 投機なし（専用speculator未所持）
    - 優先度: 中

- [ ] `shapes.json` + `registry.json` パッチの自動化スクリプト作成
    - Apple M3 追加の手動作業を省略化
    - リファクタ適用もまとめて

- [ ] `rht_wrapper.rs` 修正の再検証
    - `activation_quantization.is_some()` 削除 → M3でもRHT wrapper化
    - 今は1行改変で済んでるが、uzu本体にPR出せるか？

- [ ] `seg_dl.sh` のリファクタ適用
    - middle_refactor の改善点（引数バリデーション等）を反映
    - ただし実行中編集は禁止 → DL完了後

- [ ] GitHub Actions または cron で seg_dl 自動再起動
    - HF/CNハング時の自動検出・kill・再試行

- [ ] MLX `qwen3_5` サポート状況追跡
    - `mlx-lm` 0.32+ で対応予想？
    - Python 3.12 venv で最新版管理

- [ ] lalamo ModelSpec 追加手順のドキュメント化
    - qwen.py に1行追加 → 任意のHFモデルをuzu化
    - カスタムvendor/family/name設定のテンプレート

- [ ] 4B以下モデルでも投機(speculator)なしは正常
    - ドキュメントに明記: 4B以下はspeculatorファイル自体が存在しない設計

- [ ] HauhauCS GGUF → uzu化不可の理由を文書化
    - GGUF形式はlalamo非対応
    - safetensors/MLX形式のみ変換可能

- [ ] 中間層活用(speculator)の簡易解説メモ
    - DFlashは独立6層Transformer
    - ターゲットのlayer [1,5,9...29] hidden stateをKV cacheに注入
    - 外部モデルでも構造的には動くがaccept rate低下

- [ ] 帯域計算式の保存
    - M3(100GB/s) × η(0.78) ÷ 重み(GB) = 理論t/s
    - 9B: 15 t/s（素）、投機ONで60+ t/s
    - 投稿の92 t/sはM5(153GB/s)前提

- [ ] DEVICE.md 更新
    - Metalプローブ結果
    - shapes.json欠落の詳細
    - 修正手順（検証済み）

## 完了済み (Keep)

- ✅ uzu 9B-M ステージングDL完了 (5.19GB + spec 755MB)
- ✅ uzu 9B 素計測: 15.07 t/s (t/f=1.00)
- ✅ uzu 9B 投機ON計測: 61.96 t/s (t/f=11.12) → 長文コード生成で真価発揮
- ✅ llama.cpp Q4_K_M: 13.3 t/s（同一M3で比較）
- ✅ MLX OptiQ-4bit: 13.25 t/s（Python 3.12 venv + mlx-lm 0.32.2）
- ✅ キュー全完了（GGUF/MLX/lalamo/mlx-lm）
- ✅ lalamo環境構築（uv + Python 3.12）
- ✅ `seg_dl.sh` 動作実証（Range 206対応、7 MB/s安定）
- ✅ `rht_wrapper.rs` 修正 + 再ビルド成功（1分19秒）
