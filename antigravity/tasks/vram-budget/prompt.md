タスク: 4B / 9B 量子化パイプラインの VRAM 収支を実測ベースで解明し、T4 16GB で動かす具体案を出す

## 背景
uzu は Qwen3.5 系を RHT (Hadamard32) + asymmetric INT4 group32 に変換し、YAQA（Hessian 校正）と QAD（蒸留微調整）で品質を詰めている。
0.8B は完走する。しかし:
- 4B の YAQA calibration が Kaggle T4x2 (16GB × 2) で OOM。`--layer-filter` で layer 0-3 だけに絞っても SIGKILL される
- 9B の RTN eval（BF16 18GB + INT4 5GB）と QAD が VRAM 不足で不可
- 4B の INT4 RTN 変換・評価は成功している（2.40GB モデル、T4 で動く）

何がどれだけメモリを食っているのか、どの枝を切れば T4 で通るのかが分かっていない。勘ではなく収支で決めたい。

## 入力（すべて公開）
1. HF dataset: https://huggingface.co/datasets/yamahei21/uzu-quant-kit
   - `configs/qwen35-0.8b.json`, `configs/qwen35-4b.json`, `configs/qwen35-9b.json`
   - `scripts/yaqa_calibration.py`, `scripts/rht_int4_convert.py`, `scripts/eval_quantized.py`, `scripts/qad_finetune.py`
   - `data/*` にコーパス
2. GitHub: https://github.com/yamahei21python/uzu （main）
3. lalamo: https://github.com/trymirai/lalamo commit e401e17761f2fcd337b0cf20218d67d656e6ecba

## やること
### A. メモリ収支モデル
`configs/*.json` と `yaqa_calibration.py` を読み、以下を層ごとに積み上げる:
- BF16 重み（パラメータ数 × 2 bytes）
- Hessian アキュムレータ（input_block / output_block の dtype とサイズ。何サンプル分を保持するか）
- JAX のコンパイル済み実行時バッファ、XLA workspace、CUDA context（実測で埋める）
- 変換時のピーク（元モデル + 量子化結果 + 一時バッファ）

出力は表: 「4B / 9B × (BF16 load, calibration, convert, eval, QAD) × 推定ピーク GB」。
推定は式と係数を明記する。**不明な項は "unknown" と書き、実測で埋める**。

### B. sandbox 実測
GPU は無い前提でよい。numpy / jax CPU で同じ shape の配列を確保し、RSS / `resource.getrusage` / `tracemalloc` / `/proc/meminfo` で実測する。
- 4B と 9B の layer 0 の Hessian アキュムレータを実 shape で確保し、実際のバイト数を測る
- `--layer-filter` が効いているのに OOM する理由を、コードを読んで特定する（フィルタ後も全体をロードしている、など）
- 可能なら `psutil` でプロセスのピーク RSS を測る

### C. 選択肢の比較
以下を、実装コスト / 想定削減量 / 副作用 で並べる:
1. 層ストリーミング（1 層ずつロードして H を蓄積）
2. CPU offload（重みを CPU に置き、forward のみ GPU）
3. Hessian を float16 / block-diagonal に落とす
4. 勾配チェックポイント（QAD）
5. 8bit 化 or group size 拡大で作業領域を削る
6. 4B は Modal A100、9B は eval だけ CPU（時間と引き換え）
7. BF16 を最初から INT4 でロードして eval する（baseline を妥協する）

それぞれ「T4 16GB に収まるか（Yes/No/条件付き）」を明記する。

### D. 推奨
「まずこれを試す」を 1〜2 個に絞り、**そのままコピペで走るコマンド**を書く。owner の環境は Kaggle T4x2 と Mac M3 16GB。実装が必要な場合は、どのファイルのどの関数を変えるかを行番号付きで示す。

## 成果物
1. `research/vram_budget_4b_9b.md` — 収支表、実測、比較、推奨、再現コマンド
2. `scripts/memory_model.py` — 収支計算（引数でモデルサイズ・dtype・group size を変えられる）
3. `research/vram_measurements.txt` — sandbox の実測ログ全文
4. `research/vram_budget.json` — 機械可読の数値

## 制約
- 数値を捏造しない。推定と実測を必ず区別する
- 秘密情報・認証は使わない
- 重い DL は不要（config とコードだけで足りる）
- 日本語で書く
- 実行時間 20 分以内

## 完了報告
- 4B calibration の推定ピーク GB と、OOM の主因 1 つ
- T4 で通る可能性が最も高い案 1 つと、その根拠
- 作成ファイル一覧
