# lalamo × uzu：RHT + YAQA + INT4 Group32 量子化・相互運用調査まとめ

調査対象：lalamo / uzu / YAQA　｜　基準コミット：lalamo e401e177・uzu efa1d7d2

## 1. 調査の目的

今回の調査では、Huihui Qwen3.5 9Bを対象に、次の構成を実現できるか検証した。

- **RHT（Randomized Hadamard Transform）**による重みの非コヒーレンス化
- YAQAによるモデル出力品質を考慮した適応的量子化
- INT4 Group32による重み圧縮
- lalamoで変換・保存したモデルをuzuで読み込んで推論する構成

最終的に確認したいのは、単に量子化できるかではなく、量子化後のモデルをuzuで正しく復元し、元のモデルに近い出力品質と実用的な推論速度を得られるかという点である。

## 2. 結論

条件付きで実現可能

現時点では、lalamo側にRHT・INT4・YAQAの丸め処理が存在し、uzu側にもRHT付き量子化Linearを読み込む仕組みが確認できた。

ただし、両者の保存形式や演算の完全な互換性はまだ実証されていない。

### 調査項目

| 調査項目 | 判定 | 根拠・残課題 |
| --- | --- | --- |
| lalamoのRHT | 実装確認 | Hadamard変換と符号処理を確認 |
| lalamoのINT4 Group32 | 実装確認 | 非対称INT4の設定が可能 |
| lalamoのYAQA丸め | 実装確認 | YAQA round処理が存在 |
| YAQA Calibration | 未実装相当 | Preconditioner推定パイプラインを確認できず |
| lalamoの保存・再ロード | 設計上対応 | Specをメタデータから復元する設計。Hybrid実機テストなし |
| uzuのRHT対応 | 実装確認 | RHTLinearWrapperを確認 |
| uzuのINT4 Group32 | 部分確認 | 意味上の対応は有力。バイト配置未検証 |
| lalamo→uzuのHybridSpec | 変換必須 | JSON構造が異なる |

Qwen3.5 9B全体の変換・推論については、未検証。全Linearのshapeとロード互換性を確認する必要がある。

## 3. lalamo側の調査結果

### 3.1 対象コミットとuzuとの互換性

調査対象は次のコミット。

- lalamo：e401e17761f2fcd337b0cf20218d67d656e6ecba
- uzu：efa1d7d226721d3b89b8b9298a32a8d3cf761d83

lalamoでは、コミット 5d7e33b6 のTransformer Layer RefactorでTransformer構造が変更され、residual_configが導入されている。

今回の対象である e401e177 時点のTransformerConfigには、residual_configが存在しない。

そのため、uzuが同設定に未対応であることを考えると、古いlalamoコミットを使う方針には合理性がある。

ただし、residual_configがないことだけでは互換性を保証できない。HybridMatrixやモデル構造全体のロードテストが必要。

### 3.2 Qwen3.5のLinear構造

確認した主なファイル。

- model_import/model_configs/huggingface/qwen3_5.py
- model_import/common.py
- model_import/loaders/common.py
- model_import/loaders/huggingface.py

Qwen3.5のLinearには、主に次の種類がある。

| 構成 | 主な対象 |
| --- | --- |
| Attention | Q/K/V/G projection、output projection |
| DeltaNet | input/output projection |
| MLP | up/gate/down projection |
| MoE variant | Expert Linear |

Embedding、RMSNorm、DeltaNet convolutionなどは今回のLinear量子化対象から除外する。

重要なのは、HFの重み名を直接指定して変換するのではなく、lalamoのモデル構造からLinearを走査して処理すること。

また、MoEモデルではExpert Linearが3次元WeightMatrixになる可能性があるため、Dense 9Bモデルと同じ処理を無条件に適用してはいけない。

### 3.3 Linearへの量子化Spec適用

確認した構造。

- modules/linear.py
- utils/surgery.py
- weight_matrix.py
- compressed/hybrid.py
- compressed/int.py
- compressed/incoherence.py

lalamoにはLinearのWeightMatrixを置換するための構造操作機能がある。

実装方針は以下。

1. モデル内のLinearを列挙する。
2. Linearの既存WeightMatrixを取得する。
3. 必要に応じてdecompressする。
4. HybridSpecを使って圧縮する。
5. Linearのweightsを新しいHybridMatrixに置換する。

Embeddingなどを誤って量子化しないため、WeightMatrix全体ではなくLinearだけを対象にする。

なお、HybridSpec.compressには適切な乱数キーを渡す必要がある。APIの実際の呼び出し条件を確認したうえでテストすること。

## 4. RHT + INT4 Group32の仕様

### 4.1 RHT

lalamoの compressed/incoherence.py にはHadamardSpecがある。

設定可能なblock sizeは以下。

- 32
- 64
- 128

今回の想定は block_size=32。

INPUT_OUTPUTモードでは、入力次元と出力次元がblock sizeで割り切れる必要がある。

したがって、両軸にRHTを適用する場合、Linearの入力・出力次元がともに32の倍数であることを確認しなければならない。

### 4.2 INT4 Group32

想定設定。

```python
IntSpec(
    bits=4,
    group_size=32,
    is_symmetric=False,
    layout=Layout.OUTPUT_INPUT,
)
```

これは非対称INT4 Group32である。

OUTPUT_INPUTの場合、group_sizeは入力軸に適用される。

したがって、少なくとも入力次元が32で割り切れる必要がある。

### 4.3 Qwen3.5 9Bでの注意点

モデル全体に適用する前に、全Linearのshapeを列挙するべきである。

確認する内容。

- 入力次元が32の倍数か
- 出力次元が32の倍数か
- 3次元Expert Linearが存在するか
- RHT対象外にすべき特殊Linearがあるか

現時点ではQwen3.5 9Bの全Linearについて、shape制約を満たすことは確認できていない。

## 5. YAQAの調査結果

今回の調査で、特に重要だった部分。

### 5.1 lalamoにYAQA処理は存在する

主なファイル。

- preconditioner.py
- compressed/utils/yaqa.py
- compressed/int.py
- tests/unit/compressed/test_yaqa.py

IntSpec.compress()にPreconditionerが渡されると、yaqa_round_blockwise()が呼ばれる。

つまり、YAQAの適応的丸め処理自体はlalamoに実装されている。

### 5.2 しかしCalibrationが不足している

PreconditionerDict.init_for_model()はIdentity Preconditionerを作成する処理であり、実際のCalibrationによる推定ではない。

また、テストの正定値行列生成もランダム行列から作った人工データであり、実モデルのCalibrationではない。

したがって、現状のlalamoをそのまま使っても、モデルの出力品質を考慮したYAQA Calibrationが完了するわけではない。

### 5.3 YAQAで収集すべき情報

YAQAでは、モデル全体の出力KLを考慮した層ごとのHessianをKronecker factorizationで近似する。

概念的には、

H ≈ H_O ⊗ H_I

- H_I：入力側の情報。活性化の二次モーメントに近い。
- H_O：出力側の曲率情報。出力分布に対する勾配・Fisher由来の情報。

ここで注意したいのは、単純にLinearの出力活性化共分散を収集すればYAQAになるわけではないこと。

YAQA論文と公式実装に沿ってHessian情報を収集する必要がある。

参考資料。

- Model-Preserving Adaptive Rounding（YAQA論文）
- PMLR掲載論文
- YAQA公式実装

### 5.4 メモリ面の懸念

4096×4096のFP32行列は約64MiB。

三角格納でも約32MiBとなる。

入力側・出力側のPreconditionerを全層分保持する場合、さらに中間コピーやFP32変換が発生する可能性がある。

MacBook M3 16GBで9Bモデル全体のCalibrationを実施する場合、メモリが厳しくなる可能性が高い。

そのため、層単位の処理やGPUクラウドの利用を検討する必要がある。

## 6. uzu側との相互運用調査

### 6.1 HybridSpecのJSON構造が異なる

ここは明確な互換性問題。

2026-11-06時点で、lalamoの `HybridSpec` をuzu形式へ平坦化する変換スクリプトを作成し、safetensorsのmetadataのみを書き換える単体テストに成功した。テンソルデータには触れない。

| 項目 | lalamo | uzu |
| --- | --- | --- |
| 量子化Spec | quantization_spec | quantization_spec |
| Adapter | adapter_spec | adapter_spec: Option<Unsupported> |
| RHT block size | incoherence_spec.block_size | incoherence_block_size |
| RHT mode | incoherence_spec.processing_mode | incoherence_processing_mode |

同じ概念を扱っていても、JSONの階層が異なる。

そのため、lalamoが出力したHybridSpecをそのままuzuに渡すことはできない。

uzu用のSpec変換処理が必要。

今回Adapterを使わないのであれば、adapter_spec=nullにする方針が妥当。

### 6.2 uzuのRHTロード処理

確認したファイル。

- encodable_block/linear/rht_wrapper.rs
- encodable_block/weight_matrix.rs
- config/weight_matrix/hybrid_spec.rs

uzuの RHTLinearWrapper::load_inner() は、次の処理を行う。

1. Hybrid Specを読み込む。
2. RHT対応構成かを判定する。
3. 量子化Specを読み込む。
4. input_signsを読み込む。
5. 必要ならoutput_signsも読み込む。
6. LinearMatmulを構築する。
7. 入力にRHTを適用する。

lalamo側にも符号乗算とHadamard変換が存在するため、基本的な演算構造は対応しているように見える。

ただし、以下は未検証。

- 正規化係数
- 符号の適用順序
- 入出力軸の対応
- 転置の扱い
- FP32/BF16の丸め差

したがって、現時点で「RHT演算が完全一致する」とは断言できない。

### 6.3 INT4 Group32のデータ形式

uzu側にはU4、ScaleZeroPoint、GroupOutputの扱いがある。

lalamoの非対称INT4 Group32と意味上の対応は有力。

しかし、実際のデータ互換性はまだ確認できていない。

特に以下が重要。

- 4bit値の上位/下位nibbleの順序
- UInt8への格納方法
- Group32の適用軸
- Scaleのshapeと配置
- Zero Pointのshapeと配置
- Zero Pointのpacking方式

ここが一致しない場合、uzu側で読み込めても重みが正しく復元されない可能性がある。

## 7. Git clone時の異常について

別途、lalamoのclone時にuzuのリポジトリが取得される問題も確認した。

観測結果。

- gh repo view trymirai/lalamo は正常。
- GitHub APIも trymirai/lalamo を正常に返す。
- lalamoのtarballを直接取得すると正しいソースが入っている。
- clone後のoriginが trymirai/uzu になっていた。

最有力候補はGitのURL書き換え設定。

```sh
git config --show-origin --get-regexp '^url\..*\.insteadof$'
```

この設定を確認すれば、GitHub URLがuzuに書き換えられているか判定できる。

ただし、実際の設定値は未取得なので、原因確定には至っていない。

## 8. 推奨する実装・検証手順

モデル全体をいきなり変換するのは危険。小さいLinearから順番に互換性を確定させるべき。

### 推奨ロードマップ

1. Step 1：lalamo単体の量子化 — **完了**
   小さなLinear（128×128）でRHT + INT4 Group32のHybridMatrixを生成。
   RMSE 0.041220、相対誤差 0.006817。

2. Step 2：保存・再ロード — **完了**
   export → ShapeDtypeMatrixへの再ロードに成功。
   再ロード後と量子化直後の出力差は 0.000000e+00。

3. Step 3：uzu向けSpec変換 — **完了**
   `incoherence_spec` を `incoherence_block_size` / `incoherence_processing_mode` へ平坦化するスクリプトを作成。
   safetensors metadataのみ書き換え、テンソルデータ無変更。

4. Step 4：INT4データ形式の照合 — **一部完了**
   lalamoのexport shapeがuzuの期待shapeと一致することを確認。
   - weights: `(rows, columns/2) uint8`
   - scales: `(groups, rows) float32`（GroupOutput）
   - zero_points: `(groups, rows/2) uint8`
   nibble順序も、lalamoのlow-nibble-firstとuzuの `read_packed<4>` が一致。
   実データでのuzu復元照合は未実施。

5. Step 5：uzuでモデル全体をロード — **完了**
   Qwen3.5-0.8B（BF16 / RHT+INT4）の両方が uzu CLI でロード・生成に成功。
   INT4 モデルは HybridSpec 96件を uzu 形式へ書き換えた状態で正常に動作。

   | モデル | サイズ | 実測メモリ | prefill | generation |
   |---|---|---|---|---|
   | BF16 | 1.4 GB | 1.49 GB | 34.3 t/s | 8.5 t/s |
   | RHT+INT4 G32 | 764 MB | 0.84 GB | 279.0 t/s | 12.5 t/s |

   INT4 の方がメモリ約44%減、速度も向上。数値的な一致は次の Step で確認する。

6. Step 6：出力比較 — **進行中**
   BF16 と RHT+INT4 に同一プロンプトを与え、生成品質を比較中。

7. Step 7：YAQA Calibration — 未実施

8. Step 8：Qwen3.5 9B全体の変換 — 未実施

### 実測したexport構造（128×128 Linear）

| テンソル | shape | dtype |
|---|---|---|
| `quantized.weights` | `(128, 64)` | uint8 |
| `quantized.scales` | `(4, 128)` | float32 |
| `quantized.zero_points` | `(4, 64)` | uint8 |
| `incoherence_signs.input_signs` | `(128,)` | int32 |
| `incoherence_signs.output_signs` | `(128,)` | int32 |

uzu側の `QuantParams`（GroupOutput、group_size=32）が期待するshapeと一致する。

### lalamo config.json → uzu config の正規化

uzu の config 構造体は `#[uzu_config]` マクロで生成され、次が強制される。

- `#[serde(deny_unknown_fields)]` — 未知フィールドは即エラー
- 全フィールドに `deserialize_with = "required"` — `Option<T>` でもキー省略不可（`null` は可）

lalamo の config はこの2点で不一致だった。

| フィールド | lalamo | uzu | 対処 |
|---|---|---|---|
| `token_codec_config.response_format` | あり | 未定義 | 削除 |
| `token_codec_config.output_parser_regex` | 無し | 必須 | `null` 追加 |
| `token_codec_config.end_of_thinking_tag` | 無し | 必須 | `null` 追加 |

症状は `data did not match any variant of untagged enum AnyModelConfig` で、重み・HybridSpec の検証より前に落ちる。
BF16 / INT4 の両方が同一エラーになるのはこのため。

正規化は `scripts/normalize_lalamo_config.py`（単体）と `scripts/rht_int4_convert.py`（変換時に自動適用）で行う。

### 最初に実施すべきテスト

小さなLinearを使って、以下を測定する。

| 比較対象 | 確認内容 |
| --- | --- |
| 元のBF16 | 基準出力 |
| lalamo RHT + INT4 | 量子化誤差 |
| lalamo保存・再ロード | 保存形式の再現性 |
| uzuロード後 | 相互運用性 |
| lalamoとuzuの差 | 実装間の演算一致 |

出力誤差は例えば次で評価できる。

Relative Error = (||y_uzu - y_lalamo||_2) / (||y_lalamo||_2 + epsilon)

まずは同じ量子化重み・同じ入力で比較する。ここで差が大きければ、Qwen3.5全体へ進む前に原因を特定できる。

## 9. 残っている重要課題

優先度順に整理すると次のとおり。

| 優先度 | 課題 | 解決しない場合の影響 |
| --- | --- | --- |
| 最優先 | INT4のpacked bytes照合 | uzuで重みを正しく復元できない |
| 最優先 | HybridSpecの変換 | uzuがSpecを読み込めない |
| 最優先 | RHT演算の数値一致 | Linear出力が一致しない |
| 高 | YAQA Calibration実装 | YAQAの品質改善を得られない |
| 高 | 全Linearのshape検査 | 量子化時に一部の層で失敗 |
| 高 | HybridMatrix保存・再ロード | モデル配布・再利用に支障 |
| 中 | M3 16GBのメモリ検証 | 9Bモデルの変換が完了しない |
| 中 | clone設定異常の原因特定 | 調査環境の再現性に問題 |

## 11. 0.8B 実機検証の結果と方針

Qwen3.5-0.8B（BF16 / RHT+INT4 Group32）でモデル全体の互換性を確認した。

### 確認済み

| 項目 | 結果 |
|---|---|
| BF16 → uzu ロード・生成 | 成功 |
| RHT+INT4 → uzu ロード・生成 | 成功（HybridSpec 96件を uzu 形式へ正規化） |
| 全96 Linear の逆量子化誤差 | 一律約 8.1%（無相関ノイズでなく、理論値と一致） |
| RHT 往復（合成行列） | 再現誤差 2.7e-15 |
| HybridSpec metadata 平坦化 | 成功、テンソル内容不変 |
| lalamo logits BF16 vs INT4 | Top-1 0.780 / Top-5 0.719 / KL 0.364 / cosine 0.966（59位置） |

初回の相対誤差 ~1.39（無相関と同等）は、Python の RHT 逆変換で出力軸 Hadamard を列方向に掛けていた実装バグが原因だった。
`scripts/verify_rht_roundtrip.py` を追加し、往復再現が壊れたら即失敗するようにした。

### 量子化方式の比較（lalamo、59位置）

| モデル | Top-1 | Top-5 | KL | cosine | サイズ |
|---|---|---|---|---|---|
| BF16 | 基準 | 基準 | 0 | 1 | 1.4GB |
| RTN（RHT + INT4） | 0.780 | 0.719 | 0.364 | 0.966 | 800MB |
| YAQA H_in（E2-old） | 0.797 | 0.776 | 0.157 | 0.974 | 800MB |
| E3（H_in + H_O, emb BF16） | 0.780 | 0.759 | **0.157** | 0.974 | 800MB |
| E3b（E3 + emb INT4） | 0.746 | 0.759 | 0.193 | 0.970 | **439MB** |
| 公式 Mirai 0.8B-M | **0.881** | **0.793** | **0.108** | **0.983** | **419MB** |

E2-old の条件: token sum、64 token、実サンプル6、H_in のみ、damping 1e-3。
Hessian 重み付き再構成誤差は全96層で改善（RTN 比の中央値 0.244x）。

### 公式 0.8B-M との比較

- ファイル 419MB、tensor 617、HybridSpec 96
- **embedding も INT4**（layout input_output、RHT output のみ、scales BF16）
- 線形96層は spec / shape / dtype が完全一致
- scale 比（公式 / 我々の YAQA）は 0.99996
- 4-bit code ヒストグラムは公式とほぼ一致
- code 一致率は RTN 0.706 / YAQA 0.620
  → 公式は YAQA 系の分布を持ち、QAD で個別コードを変更した可能性
- 公式 vs E3b の KL は 0.277、E3（emb BF16）は 0.247
  → embedding 量子化だけでは公式に近づかない。**QAD の寄与が大きい**
- E3b は公式と同じ 419〜439MB を達成するが、BF16 比 KL は 0.193
  → サイズ優先なら QAD なしでは品質を犠牲にする

### 修正版 calibration と E3

調査レポート（`research/mirai_yaqa_report.md`）の指摘を反映:

- 損失: per-sample token mean（公式と同じ）
- calibration: 連続 512 token チャンク、16 samples
- H_O: out_dim <= 2048 の48層で取得
- damping: `H/=mean(diag)` → `1e-2` から Cholesky 成功まで増加

**H_O を捨てた `--output-block none` は YAQA ではなく LDLQ 相当**。
E3（H_in + H_O small）を変換中。

### 実機短文テスト（参考）

- 両モデルとも `temperature=0.8` では生成が不安定（0.8B の弱さ）
- INT4 は BF16 と比べてメモリ約44%減（0.84 GB 対 1.49 GB）
- 速度は条件依存（この M3 では 43→6〜14 tok/s に低下、kernel 経路の影響は未調査）

## 10. 最終評価と方針

今回の調査から、以下の方針を推奨する。

lalamoを量子化・変換側、uzuを推論側として分離する構成は、技術的に検討を進める価値がある。

ただし、まだQwen3.5 9B全体を変換できると結論づける段階ではない。

特に次の3点を先に解決するべき。

1. lalamoとuzuのINT4 Group32のバイト形式を完全に一致させる。
2. HybridSpecとRHT符号データの保存形式をuzuが読める形に変換する。
3. 小さなLinearで両実装の出力が数値的に一致することを実証する。

その後にYAQA Calibrationを実装し、品質を比較する。

現段階の最大のリスクはYAQAのアルゴリズムではなく、量子化済み重みの相互運用性が未実証であること。

ここを先に潰すのが、最も手戻りの少ない進め方だと思う。
