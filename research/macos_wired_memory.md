# macOS GPU Wired Memory 調査レポート: Apple M3 16GB における 9B INT4 推論

## 目的と背景
Apple M3 (統合メモリ 16GB) において、Metal を用いた推論エンジン uzu で 9B INT4 モデル（推論見積り 7.2〜12GB）を実行するための GPU wired memory 設定とリスクを整理する。現在すでに **6.5GB の swap 使用【実測】** が確認されており、メモリ枯渇リスクが極めて高い。

---

## 1. iogpu.wired_limit_mb の概要と仕様

### 1.1 意味とアーキテクチャ
Apple Silicon では CPU/GPU が物理 DRAM を共有する。Metal が確保する GPU バッファはスワップやページアウトができない「Wired Memory」となる。`iogpu.wired_limit_mb` は GPU が固定確保できる上限（MB 単位）を定めるカーネルパラメータである。

### 1.2 基本仕様とバージョン差
- **既定値**: `0`【引用】。無制限ではなく、物理 RAM 容量からカーネルが自動計算（約 67%〜75%【引用】）。
- **16GB M3 既定上限**: 約 **10.7GB (10,950MB〜11,468MB)【引用】**。
- **再起動要否**: **不要（即時反映）【引用】**。ただし再起動で既定値（0）に復帰。永続化には `/Library/LaunchDaemons/` の plist が必要（`/etc/sysctl.conf` は macOS Big Sur 以降無視【引用】）。

| macOS バージョン | 有効な sysctl キー | 単位 |
| :--- | :--- | :--- |
| **macOS 14 Sonoma / 15 Sequoia** | `iogpu.wired_limit_mb` | MB【引用】 |
| **macOS 13 Ventura (後期)** | `iogpu.wired_limit_mb` | MB【引用】 |
| **macOS 12 Monterey / 13 (初期)** | `debug.iogpu.wired_mem_limit` / `wired_limit` | Bytes / MB【引用】 |

### 1.3 設定コマンドと上限目安 (16GB RAM)
```bash
# 確認 / 設定 (12GB) / 既定値 (0) へのリセット
sysctl iogpu.wired_limit_mb
sudo sysctl iogpu.wired_limit_mb=12288
sudo sysctl iogpu.wired_limit_mb=0
```

| 設定値 (`wired_limit_mb`) | GPU Wired 上限【推定】 | OS・他プロセス残枠【推定】 | 評価 |
| :--- | :--- | :--- | :--- |
| `0` (既定) | 約 10.7 GB | 約 5.3 GB | **安全**（標準動作） |
| `12288` (12GB) | 12.0 GB | 約 4.0 GB | **注意**（他アプリ全終了必須） |
| `13312` (13GB) | 13.0 GB | 約 3.0 GB | **高リスク**（スラッシング・不安定化） |
| `14336` (14GB) | 14.0 GB | 約 2.0 GB | **極めて危険**（OS クラッシュ・Panic） |

---

## 2. 16GB M3 で Wired Limit を上げた場合のリスク

### 2.1 システムリスクの整理
OS 基本機能（Kernel, WindowServer 等）はアイドル時でも **2.5〜3.5GB【推定】** を消費する。

| リスク要因 | 現象とメカニズム | 深刻度 |
| :--- | :--- | :--- |
| **System Instability** | Wired 領域拡大により CPU 側物理 RAM が枯渇。UI 凍結、レインボーカーソル化が発生【推定】。 | 高 |
| **Swap 激増・スラッシング** | 現在 **swap 6.5GB【実測】** の環境下で 12GB を占有すると、SSD への I/O 集中で極度の性能低下【推定】。 | 最高 |
| **Jetsam OOM Killer** | Wired メモリはカーネルが回収不可【引用】。推論以外の周辺プロセス（Shell, Finder 等）が強制終了【引用】。 | 高 |
| **Kernel Panic** | `IOGPUFamily`（`completeMemory() prepare count underflow` 等）でドライバが破綻し OS 再起動【引用】。 | 最高 |

### 2.2 Apple 公式・開発コミュニティの見解
- **Apple 公式定義**: `recommendedMaxWorkingSetSize` は「性能に悪影響を与えずに割り当て可能な近似値」と定義【引用: [Apple Developer](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)】。
- **コミュニティ報告**: 物理 RAM の限界値（13GB+）を指定して推論を行うと、Jetsam が対処できず `IOGPUMemory` クラッシュや即死 OOM を引き起こす報告が多数存在【引用: [mlx #3346](https://github.com/ml-explore/mlx/issues/3346), [exo #1972](https://github.com/exo-explore/exo/issues/1972)】。

---

## 3. 推奨設定値と 9B INT4 (12GB) の現実性評価

### 3.1 9B INT4 推論時のメモリ内訳試算
9B パラメータモデル（Gemma-2-9B 等）の推論メモリ見積り（7.2〜12GB）の内訳：

| 構成要素 | 計算基準 | メモリ量【推定】 |
| :--- | :--- | :--- |
| **ベースモデル重み** | 9B × 0.5 バイト (4-bit) + 量子化メタデータ | 約 4.8 〜 5.2 GB |
| **Embedding / LM Head** | FP16 (Vocab 256k) または INT4 | 約 0.4 〜 1.2 GB |
| **KV キャッシュ (2K / 8K)** | FP16, GQA 適用時 | 2K: 約 0.4 GB / 8K: 約 1.8 GB |
| **Metal Scratch / Activation** | 実行時テンソルバッファ | 約 1.0 〜 2.0 GB |
| **合計所要量** | **短文: 約 7.2GB / 長文(8K+): 約 9.5〜12.0GB** | **7.2 〜 12.0 GB** |

### 3.2 設定値の現実性判定
- **12GB 設定 (`12288`)**:
  - **判定**: **条件付きで可能だが極めて危険【推定】**。
  - **条件**: 他アプリを全終了し、事前再起動等で既存 **swap 6.5GB【実測】** を完全に解消していること。
  - **問題点**: 残存 RAM 約 4GB【推定】では OS 動作が限界で、わずかな負荷増でスラッシングに陥る。
- **13GB 設定 (`13312`)**:
  - **判定**: **非現実的・採用不可【推定】**。
  - **理由**: 残存 RAM が 3GB【推定】を割り込み、WindowServer クラッシュや Kernel Panic を誘発する。
- **総合結論**: 12GB への上限引き上げは非常手段に留め、**代替案によって総消費を 10GB 未満（既定 10.7GB 枠内）に抑制する方針を強く推奨**する。

---

## 4. 代替案: Wired Limit を上げずに収める方法

パラメータを変更せず、既定の **~10.7GB【引用】** 内に収める最適化手法：

| 最適化手法 | 具体策 | 削減効果【推定】 | 影響度 |
| :--- | :--- | :--- | :--- |
| **KV 量子化** | KV キャッシュを FP16 → INT8 または INT4 化 | **1.0 〜 2.5 GB** | 精度低下は極小 |
| **Context 短縮** | 最大コンテキスト長を 4,096 トークン以下に制限 | **1.0 〜 2.0 GB** | 長文入力制限 |
| **Embedding INT4** | 語彙埋め込み・出力層を 4-bit 量子化 | **0.5 〜 1.0 GB** | 精度影響小 |
| **Mixed Precision** | 重要層のみ 5/6-bit、他は 4-bit (Q4_K_M 等) | **0.3 〜 0.6 GB** | 高品質維持 |
| **mmap / ゼロコピー** | `MTLResourceStorageModeShared` で重み二重持ち防止 | **0.8 〜 1.5 GB** | 速度向上 |
| **Metal キャッシュ制限** | 一時バッファの再利用キャッシュ上限設定 | **0.5 〜 1.0 GB** | 速度影響微小 |

---

## 5. 計測・監視方法

### 5.1 主要ツールと監視項目
| ツール | コマンド / コード | 監視対象と判断基準 |
| :--- | :--- | :--- |
| **sysctl** | `sysctl iogpu.wired_limit_mb` | カーネル設定値の確認（0 は既定） |
| **Metal API** | Swift スクリプト（下記） | ランタイム認識上限 `recommendedMaxWorkingSetSize` |
| **memory_pressure** | `memory_pressure` | `System-wide memory free %`（Warn/Critical 監視） |
| **vm_stat** | `vm_stat 1` | `Pages wired down`, `swapins/swapouts` の急増 |
| **footprint** | `sudo footprint <PID>` | 推論プロセスの Wired / Dirty メモリ実測値 |
| **swapusage** | `sysctl vm.swapusage` | スワップ使用量（`used` の増減） |

### 5.2 実行用コマンド・スクリプト
```bash
# 1. Metal 推奨最大ワーキングセットサイズ (GB) の確認
swift -e 'import Metal; let dev = MTLCreateSystemDefaultDevice()!; print("GPU:", dev.name, "| Max Working Set:", Double(dev.recommendedMaxWorkingSetSize)/(1024*1024*1024), "GB")'

# 2. スワップとシステムメモリ圧迫度の監視
sysctl vm.swapusage
memory_pressure

# 3. 推論プロセスのメモリフットプリント詳細
sudo footprint <PID>
```

---

## 6. 安全な検証・実験手順

```
[事前準備] 不要アプリ終了・再起動(Swap解消)
   ↓
[ベースライン計測] sysctl, recommendedMaxWorkingSetSize, swap 記録
   ↓
[一時設定] sudo sysctl iogpu.wired_limit_mb=12288
   ↓
[ロード検証] uzu 起動 (Context=2048) → footprint / memory_pressure 確認
   ↓
[生成・負荷試験] トークン生成 → vm_stat / スワップ / スタッター監視
   ↓
[ロールバック] sudo sysctl iogpu.wired_limit_mb=0
```

1. **環境クリーンアップ**: 他アプリを全終了。既存 **swap 6.5GB【実測】** を解消するため Mac を**再起動**。
2. **ベースライン記録**: `sysctl vm.swapusage`（used=0MB 確認）および Swift で既定上限（約 10.7GB【引用】）を記録。
3. **設定変更**: `sudo sysctl iogpu.wired_limit_mb=12288` を実行。Swift で上限上昇を確認。
4. **ロードテスト**: uzu で 9B INT4 を短文コンテキスト（2048）でロード。`footprint` で実メモリ計測。
5. **推論負荷テスト**: 生成を実行し、`memory_pressure`（正常維持か）と `vm_stat` の `swapouts` を監視。
6. **復元**: 検証終了後、必ず `sudo sysctl iogpu.wired_limit_mb=0` で既定値に戻す。

---

## 7. 未確認事項

1. **実機 M3 16GB での厳密な既定バイト数**: macOS 14/15 での正確な `recommendedMaxWorkingSetSize`【未確認】。
2. **uzu の実装仕様**: KV キャッシュの INT8/INT4 対応、Embedding INT4、および mmap ゼロコピー適用の有無【未確認】。
3. **既存 swap 6.5GB の要因**: スワップを消費している常駐プロセスの特定と再起動なしでの解放可否【未確認】。
4. **連続推論時の Metal バッファ挙動**: キャッシュ肥大化による長期リークの有無【未確認】。
5. **sysctl 反映タイミング**: 実行中プロセスへ即時反映されるか、プロセス再起動が必要かの実機差【未確認】。

---

## 引用・参照文献
- Apple Developer: [MTLDevice recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- ModelPiper: [iogpu.wired_limit_mb on Mac: Raising the Metal Memory Ceiling](https://modelpiper.com/blog/iogpu-wired-limit-mb-mac)
- Local AI Master: [Mac Out of Memory on Local LLMs: The Metal VRAM Fix](https://localaimaster.com/blog/mac-memory-pressure-local-llm)
- GitHub Devnote: [Override macOS Metal "VRAM" Cap on Apple Silicon](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap)
- GitHub ml-explore/mlx: [Kernel panic in IOGPUMemory #3346](https://github.com/ml-explore/mlx/issues/3346)
- GitHub exo-explore/exo: [Kernel Panic Bug Report #1972](https://github.com/exo-explore/exo/issues/1972)
- Reddit r/LocalLLaMA: [M1/M2/M3: increase VRAM allocation with sudo sysctl iogpu.wired_limit_mb](https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/)
