# 公式 uzu モデル（Mirai M）の構成を解析

2026-11-08 / 対象: `~/.cache/mirai/models/mirai/alibaba-qwen3.5-{0.8b,9b}-mirai-mirai-m-4`

## 1. 量子化の仕様

| 項目 | 0.8B-M | 9B-M |
|---|---|---|
| ファイルサイズ | 0.409 GB | 4.833 GB |
| bits | **全て 4bit** | **全て 4bit** |
| group size | **32** | **32** |
| RHT | block 32 / input_output | 同 |
| scales の dtype | BF16 | BF16 |
| zero points | U8（4bit を 2 個詰め） | 同 |
| embedding | 量子化（tied、単一） | **input/output を別々に量子化**（untied） |
| 8bit / 6bit 配分 | **無し** | **無し** |
| speculator | 同梱なし | **同梱あり（755MB, BF16）** |

**公式は「uniform 4bit group32 + embedding 量子化」だけ**。我々の down-out8 / wide8 は公式を超える試み。

## 2. scales が意外に大きい（サイズの 11%）

9B-M の内訳（実ファイルから集計）:

| 種別 | サイズ |
|---|---:|
| 量子化重み（U8） | 4,402 MB |
| **scales（BF16）** | **534 MB（11%）** |
| その他（F32/I32） | 12.5 MB |

- 埋め込みの scales だけで **127MB**（input/output 各 248320×128×2B）
- 層の scales は group 32 で `params/32 × 2B`

### 含意: group 64 にすると −5.5%

9B の次元（12352 / 24576 / 8192 / 4096 / 9216）はすべて 64 で割り切れる。
`group_size=64` にすると scales が半分 → **約 267MB 減（4.83 → 4.57GB）**、TPS **+5.8%** 相当。
公式が 32 を選んでいるのは品質優先と推測される（未検証）。

## 3. speculator（9B のみ同梱）

| ファイル | サイズ |
|---|---:|
| `speculator/model.safetensors` | **755 MB（BF16）** |
| `speculator/config.json` | 17 KB |
| `speculator/shapes.json` | 1.4 KB |

- 9B は 4.83GB + 0.755GB = **5.59GB**。M3 16GB に余裕
- 0.8B には同梱なし（小型で既に速いため）
- 投機は**帯域の壁を越える唯一の手段**（`research/uzu_engine_tps.md`）

## 4. 我々の変換との差分

| 項目 | 公式 | 我々（従来） | 対応 |
|---|---|---|---|
| embedding 量子化 | する | **しない**（既定 off） | **必ず付ける**（速度 +63%） |
| 8bit 配分 | しない | wide8 で品質改善 | 公式超え。4B で要検証 |
| group | 32 | 32 | 4B/9B は 64 の余地（−5.5%） |
| speculator | 9B に同梱 | なし | 欲しければ draft 学習（SpecForge） |

## 5. 速度の実測（0.8B）

| モデル | サイズ | Generate |
|---|---:|---:|
| 公式 0.8B-M | 0.521GB | 142.9 t/s |
| 我々（RTN + emb4） | 0.521GB | 143.0 t/s |

**同一サイズ・同一速度**。差は embedding の量子化だけ。

## 6. そこから得られる設計指針

1. **embedding/readout は必ず量子化**する。0.8B では 34% を占め、速度差が +63%
2. **4bit group32 を基準**にする（公式と同じ）
3. **group 64 は 4B/9B の高速化オプション**（サイズ −5.5%、品質影響は未検証）
4. **8bit 配分は公式にない追加レバー**。品質は上がるがサイズ増 → 速度 −4〜12%
5. **speculator は速度の次元を変える**。9B は公式同梱の 755MB があるが、Huihui 用は自前で学習が必要
