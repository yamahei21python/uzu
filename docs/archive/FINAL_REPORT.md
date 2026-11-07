# Weaver／DFlash 自作調査 結果（確定版 v3）

- 日付: 2026-11-06（v3: v2 の再レビューを反映。Weaver が任意と確定したことに伴う工数表の再構成、判断軸の誤読の訂正、保存容量・版対応など新規リスクの追記。v3.1: 採用時に Claude が実値確認した2点――対象モデルの hidden 次元 4096 と、uzu タグ 0.6.17 = コミット 8962d3e の対応――を反映）
- 位置づけ: 本書が調査結果の正式版。経緯文書（REPORT.md と各アングラ元レポート）は履歴として残すが、判断には本書を使う
- 調査方法: Antigravity（アングラ）への4本の委託（第一弾の暫定調査）＋ Claude による出典のスポットチェック ＋ 外部レビュー3件と、それを受けた Claude の再検証。検証主体が混ざらないよう、各所に（Claude確認）（アングラ）（レビュー経由）の印を付けた。v3 で加えた数値は（Claude再計算）と記す。v3 では外部ソースを再開封しておらず、追加したのは文書内の整合確認と計算のみ（v3.1 の2点は Claude が新規に確認）
- 役割分担: 実機確認（コマンドの実挙動、受理率の実測）は本人が実施する。本書は紙上調査の確定分まで

## 結論サマリ

1. **Weaver／DFlash の解説自体は本物。** 論文・チェックポイントは出典どおり。速度値も出典どおりだが、Apple Silicon 値は比較条件が異なり、投機単体の倍率としては読めない（§1）。
2. **「NGramSpeculator を公式CLIで週末に自作」は現行版では成立しない。** 該当コマンドも uzu 側の読み込みも削除済み（§2）。
3. **DFlash は作れる見込みが高い。** 公式の学習コードは非公開だが、第三者の SpecForge に学習実装と Qwen3.5 系レシピがあり、論文に学習設定の開示もある（§3）。残る実証は「9B の GDN ハイブリッドで隠れ状態取得が通るか」と「SpecForge 出力が convert で取り込めるか」の2点。Weaver の学習コードの有無は未確認だが、Weaver は任意なので、自作の要否は「Argmax 方式に対する上乗せ効果」で決まる（§2、§7）。
4. **流用できる完成品も2系統ある。** z-lab 公式の Qwen3.5-9B 用 DFlash と、Mirai の uzu 用パッケージに同梱された DFlash＋Weaver の束（§3）。どちらも対象は公式 Qwen3.5-9B で、無検閲モデル用ではない。
5. **uzu は DFlash 単体（Weaver なし）の speculator を設計上受理する**（§2、ソース確認済み）。Weaver は任意の上乗せであり、前提ではない。確認したソースは uzu のタグ 0.6.17 そのものである（§7-9、解決済み）。
6. **データセットは揃っている**が、スターターミックス1,000件はスモークテスト規模。本番規模は別途必要（§4）。隠れ状態をオフライン保存する方式は、10万件級で TB 級の容量になる見込みで、規模を上げる段階ではオンライン方式の検討が要る（§4）。
7. **工数**: Weaver を作る構成は 36〜97h、Weaver を使わない構成（A0／A3）を含めると 16〜97h。A0（学習ゼロ）だけは下限 16h が週末枠（16〜20h）に入るが、中央値は約24h。束づくり全体は週末には収まらない。Modal が縮めるのは計算待ちだけで、開発時間は縮まない（§6）。
8. **進め方**: 手がかりなしに自作へ進まず、まず既存と同等の束を再現して工程を証明してから自分用モデルへ進む（計画書 FINAL_PLAN.md）。

## 1. Weaver／DFlash の事実確認

他AIの主張10件はアングラ判定で全件 VERIFIED。ただし検証主体は分けて読むこと: 大半は Claude も自分の環境で直接開いて確認したが、uzu ブログ経由の主張と Apple Silicon の実測値はアングラ記録に依拠する。

- Weaver は Mirai Labs の軽量 AR アダプタ（56.7M パラメータ）。DFlash が出す各位置の top-K 周辺分布（K=512）に残差ロジットを予測して条件付き依存を復元し、提案木を作る。論文「Trees from Marginals」arXiv:2607.06763（2026-07-07投稿）（Claude確認）
- DFlash は隠れ状態条件付けのブロック拡散ドラフター。論文 arXiv:2602.06036（Chen／Liang／Liu、2026-02-05投稿）（Claude確認）
- 速度値（混ぜて読まないこと）: 論文要旨は自己回帰比 4.08倍。HF カードの表（Qwen3.6-27B、B200、BF16）は 392.8 tok/s・4.37倍・受理長 +77%。Apple Silicon 値（uzu、Qwen3.5-9B、Mirai-M で base M5 92.1 tok/s 等）は（アングラ）で、比較対象が「uzu＋投機」対「llama.cpp の素のデコード」のため、投機単体の倍率ではない（ベースライン不一致）。**92.1 tok/s は投機ありの値であり、uzu 素の自己回帰（投機なし）の tok/s は本書の時点で未測定**（§7-11）
- チェックポイントの実態（Claude確認）: 単体 Weaver は `trymirai/weaver` の Qwen3.6-27B 向け（qwen36_27b_weaver.pth）のみ。一方、**Qwen3.5-9B 向けは Mirai の uzu 用パッケージ `trymirai/Qwen3.5-9B-M`（4bit）／`-L`（8bit）に、DFlash＋Weaver の束として同梱**されている（speculator/config.json に draft_config と weaver_config の両方、candidate_pool 512、shapes.json 同梱、speculator 重み約755MB）。「9B向け Weaver は無い」は誤りで、「束の形では存在する」が正しい
- 対象モデル `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` は実在（Claude確認）: Jackrong 版（DeepSeek-V4-Pro 蒸留）を Heretic で無検閲化（拒否 100/100 → 3/100、KL 0.0422）

## 2. 公式スタック（lalamo／uzu）の実態

- lalamo v0.17.0（コミット 66e0b2a、Claude も照合）: speculator 系コマンドは `convert` のみ。他AIが挙げた `collect-traces`／`train` は2025年秋のプロトタイプに実在したが、2026-05 の PR #198 で削除済み。現行 lalamo に学習機能はない
- uzu 0.6.17: 投機は DFlash-TfM に一本化。NGram 読み込み（ngram_speculator.rs）は2026-07 の PR #640 で削除
- **Weaver は任意（Claude確認、uzu ソース）**: `crates/uzu-engine/src/speculators/dflash_tfm.rs`（コミット 8962d3e）では、Weaver は `Option` として保持され、config に weaver_config がある場合だけ構築される。木の構築方法は列挙型で `Argmax` と `Weaver` の2値であり、shapes.json で Argmax を選べば **DFlash 単体（Weaver 重みなし）で動作する設計**になっている（Weaver 方式を選んで重みが無い場合のみエラー）。実機での起動確認は再現フェーズ R1 で行う。なお、このコミット 8962d3e は uzu のタグ 0.6.17 そのものであることを git のタグ参照で確認済み（Claude確認、v3.1）
- 落とし穴2件: (a) `lalamo speculator convert` は shapes.json を出力しないが uzu は必須とするため、手動作成（または生成器）が要る。(b) 対象モデルは lalamo の登録一覧に無いため、convert にそのまま渡すと "not found"（終了コード2）になり、カスタム ModelSpec の登録が要る（アーキテクチャは HFQwen35Config に適合）
- 他AIの挙げた NGram の「1.2〜1.8倍」は公式測定のない推定値

## 3. 使える部品の一覧（流用と学習の材料）

### 3.1 完成品チェックポイント
| 部品 | 中身 | 対象 | 備考 |
|---|---|---|---|
| z-lab/Qwen3.5-9B-DFlash（Claude確認） | DFlash ドラフト単体 | Qwen/Qwen3.5-9B（公式） | Z-Lab と Modal の共同再学習版（40k系列長＋sliding-window）。参考値: 最大5.01倍（HF カード、SGLang、B200、同時実行1、block=16、Math500）。受理長は課題別に約5.5〜7.9（論文 Table 11: Math500 7.3／HumanEval 7.9／MT-Bench 5.5。HF カードの同時実行1・block=16 でも 5.7〜7.9 の範囲）。いずれも GPU＋SGLang の値で、uzu の受理長と同じ定義かは未確認 |
| trymirai/Qwen3.5-9B-M／L（Claude確認） | 本体（量子化）＋DFlash＋Weaver の束 | 公式 Qwen3.5-9B | uzu 用完成品。shapes.json まで同梱された唯一の「正解の束」。ドラフトは対象層8層を使う構成 |
| trymirai/weaver（Claude確認） | Weaver 単体 | Qwen3.6-27B | 27B 系を使う場合のみ |

### 3.2 学習の実装（第三者）
- **SpecForge（sgl-project）（Claude確認）**: ツリーに dflash 関連73件。Qwen3.5-4B／Qwen3.5-35B-A3B 等の学習 config、オンライン／オフライン両方式のレシピ、AMD 向け資料（Qwen3.5-4B DFlash の loss／受理率グラフ）を含む
- 学習済みの実例（Claude確認）: HF `Moonlight556/Qwen3.5-0.8B-DFlash`（dflash.py と training_state.pt＝学習再開用の状態を含む）。Qwen3.5-0.8B（GDNハイブリッド）で SpecForge 系の学習が回った実績
- **vLLM speculators（Red Hat）（Claude確認）**: dflash 関連47件（ユーザーガイド、DFlash2／D-PACE の研究文書）。DFlash 学習に対応。ただし（レビュー経由・未検証）Qwen3.6 ハイブリッドで隠れ状態抽出時に KV キャッシュ系エラーが出た報告と、Red Hat レシピの再現性問題の報告がある。ハイブリッド系では SpecForge の方が実績があるというのがレビューの見立て
- AMD Primus の Qwen3.5-4B レシピ、NVIDIA NeMo Automodel の SpecForge 移植は（レビュー経由）（Claude は原典未開封。ただし SpecForge 内の AMD 資料の存在は確認済み）

### 3.3 公式側の開示状況
- z-lab/dflash は全14ファイル・学習系0件（Claude確認）。著者は Issue #1（2026-01）で学習レシピの公開を約束したが、2026-11 現在も未公開（Issue #163／#174 も未回答）
- 論文の付録A.1 に学習設定が開示されている（Claude確認・原文）: 6エポック、AdamW、学習率 6e-4、cosine（warmup 0.04）、最大系列長 3072、1系列あたりアンカー 512、損失減衰 γ=7（block 16）。データは約800Kサンプル（Nemotron Post-Training V2＋CodeAlpaca）で、応答は対象モデル自身に再生成。学習はオンライン／オフライン両方式の定義あり
- Weaver 側: trymirai/sglang の reproduction は推論・ベンチのみで学習コードなし（アングラ確認）。論文には構造と学習率スケジュール等の記述はあるがスクリプトは添付されていない


## 4. データセット

- 無検閲／roleplay 系の質問集・対話集は英語・日本語とも実在し、検証済み17件の一覧は `fetched/dataset/DATASETS.csv` にある。要点: JailbreakHub（15,140件、MIT、プロンプトのみ）、ShareGPT_Vicuna_unfiltered（約53K会話）、QuixiAI/dolphin（1M＋）、Aratako 日本語 NSFW roleplay（約15.3K、日）、dolly-ja（15,015件、日）、gsm8k／humaneval（MIT）
- スターターミックス（初回1,000件）: 日本語55%／英語45%（Aratako 300＋dolly-ja 250＋JailbreakHub 200＋ShareGPT 150＋gsm8k/humaneval 100）。**これはスモークテスト用**であり、本番学習の規模ではない（DFlash 論文は約800K）。Weaver 学習に必要な規模自体は未確認
- 収集の中身: テキストの trace だけでは足りない。プロンプトごとに対象モデルの生成文・隠れ状態（ドラフトが条件に使う層）と DFlash の top-K marginals を揃える必要があり、それがデータ収集ハーネスの仕事。プロンプトの抽出はすべて「プロンプトのみ」で、応答は対象モデル本人に生成させる（他モデルの応答では対象のトークン癖とずれる）。応答生成の条件（thinking の有無、温度）は受理率に効くため、データの版情報として記録する（計画書 S1）
- **隠れ状態の保存容量（Claude再計算）**: オフライン方式で隠れ状態を保存する場合、1トークンあたり「対象層数 × hidden 次元 × 2 byte（BF16）」。hidden 次元は **4096 で確定**（対象 heretic 版の config.json の text_config で実値を確認済み。Claude確認、v3.1）。8層で約64KB／トークン、5層で約40KB／トークン。平均1,000トークンの系列では、1万件≒0.7TB（8層）、10万件≒6.5TB、80万件≒52TB。平均3,072トークンなら約3倍（10万件で約20TB）。したがって1万件級まではオフラインで足りるが、10万件以上ではオンライン方式（学習中に対象モデルで再計算する。保存は不要だが、各エポックで対象モデルの順伝播が要る）が現実的になる。方式の選択は計画書 S1／S2
- 注意（レビュー指摘、一部未検証）: JailbreakHub の prompt は短い質問ではなく長い DAN 系テンプレート中心とみられ、そのまま混ぜると分布がずれるおそれ。使用前に長さ分布を確認する。ライセンス: Aratako 版は CC-BY-NC-SA-4.0（商用不可＋継承）で Claude 3.5 由来の合成データ。ShareGPT の「Apache-2.0」はリポジトリ表記にすぎず元データの由来は別問題。個人利用では実害は小さいが、公開・配布する成果物に使う場合は再確認が要る

## 5. 言語に関する論点（確定版）

- 英語だけの trace でも speculator は動作する（検証器が正しさを保証するため出力は壊れず、落ちるのは日本語での受理率＝速度だけ）。NGram が最も言語依存が強く、AR アダプタは句読点・コード・定型表現の分だけ転移する
- プロンプトだけを日本語訳し、生成は対象モデル自身にさせる方式は有効（学ぶのは訳文ではなく対象モデル本人の日本語）。翻訳に PLaMo を使う案は、guardrail（content_filter 400 の実績）で無検閲系ほど弾かれやすい点が注意。代替は対象の無検閲モデル自身による翻訳。英語原文も混ぜる二言語ミックスが前提
- 容量: Weaver 型 AR アダプタはパラメータ固定のため、学習データ量でファイル容量も推論速度も増えない（増えるのは学習時間と trace 保存容量）。NGram は表が育つ方式でファイルは増えるが、参照はハッシュ引きで推論速度はほぼ不変

## 6. 工数の見積もり（再計算済み）

項目別レンジはアングラ便4の値（根拠つき推定、実測ではない）。合計は項目の単純和で Claude が再計算した（アングラ印刷の合計には 2〜3h のずれがあった）。v3 では、Weaver が任意と確定した（§2）ため、Weaver を作らないルート A0／A3 を追加した（Claude再計算）。

| # | 項目 | レンジ |
|---|---|---|
| 1 | データ収集ハーネス | 6〜10h |
| 2 | Weaver 風アダプタの学習コード新規作成 | 14〜22h |
| 3-流用 | DFlash 流用時のロード・形状確認 | 0〜2h |
| 3a | DFlash 学習・自作（学習コードから） | 20〜35h |
| 3b | DFlash 学習・SpecForge 利用（設定＋データ準備＋9B動作確認＋形式確認） | 6〜14h（Claude暫定） |
| 4 | エクスポート・shapes.json 等のグルー（ModelSpec 登録を含むと仮定） | 4〜8h |
| 5 | uzu 統合: DFlash-TfM 経路なら配備のみ／単体 AR なら Rust＋Metal 新規実装 | 4〜8h ／ 20〜35h |
| 6 | デバッグ・受理率調整（A系／B） | 8〜14h ／ 12〜20h |

| ルート | 構成（項目の和） | 合計（中央値） |
|---|---|---|
| A0 | 学習ゼロ流用: z-lab DFlash または Mirai 束を Argmax 方式で載せる（3-流用+4+5A+6A） | 16〜32h（約24h） |
| A3 | SpecForge で DFlash のみ訓練、Weaver なし（1+3b+4+5A+6A） | 28〜54h（約41h） |
| A1 | z-lab DFlash 流用＋Weaver のみ自作学習＋uzu 既設経路（1+2+3-流用+4+5A+6A） | 36〜64h（約50h） |
| A1+ | z-lab を初期値に SpecForge で対象データ微調整 | A1 と A2′ の間（微調整の効きが未検証のため数値なし） |
| A2′ | SpecForge で DFlash を訓練＋Weaver 自作（1+2+3b+4+5A+6A） | 42〜76h（約59h） |
| A2（旧） | DFlash も学習コードから自作（1+2+3a+4+5A+6A） | 56〜97h（約77h） |
| B | 単体 AR アダプタ＋uzu 改造（1+2+4+5B+6B） | 56〜95h（約76h） |

- A0／A3 は時間だけを見れば安いが、受理率が判断軸（計画書）に届くかは別問題で、実測するまで有利とは言えない
- 「55〜80時間」との関係: A0／A3 は範囲の下に外れる。A1 は上限 64h だけが 55h 以上で、大半は下側。A2′ は範囲内。A2（旧）／B は中央値が範囲内で、上限は 80h を超えうる
- **週末（16〜20h）との関係**: A0 だけは下限 16h が枠に入るが、中央値は約24h。A3 以降は下限でも超える。束づくり全体は週末には収まらない。週末で現実的なのは判断用の測定（紙上判定・R0 の測定・A0 の測定）までで、これは項目表に無い Claude の見立て
- Modal の効果: 縮むのは計算待ちだけ。1,000件規模の trace 生成＋Weaver 学習はローカル約8.4時間 → Modal A100 で約18分・約$0.76（Starter の月$30無料枠内）（Claude確認）。開発時間は縮まない（連携で +1〜2h）。本番規模では計算費も桁が上がる（10万件で約$76 が線形延長の目安。**この値は trace 生成＋Weaver 学習側のみで、DFlash 本体の学習計算は含まない**。含めた概算は計画書のコスト構造）。価値は「当日中に試行錯誤を回せる」リードタイム短縮

## 7. 未確認リスト（現時点の残件）

1. **Weaver の学習コードが公開されているか**（SpecForge 等にも Weaver 学習があるか未確認。自作範囲を決める残件）。ただし、作る価値があるかは Argmax 方式に対する上乗せ効果で決まり、計画書 R0b で実測する
2. ~~uzu が DFlash 単体の speculator を受理するか~~ → **解決済み（§2）**: ソース上は Argmax 方式で動作する設計。実機確認は R1 で行う。残る細目: (a) ドラフト層数（5層／8層等の任意値）を uzu がそのまま受理するか、(b) `lalamo speculator convert` が Weaver なしの speculator（weaver_config の無い config）を出力できるか、(c) shapes.json での Argmax 指定の表記
3. SpecForge の隠れ状態取得が 9B の GDN ハイブリッドで通るか（実績は 0.8B／4B）
4. SpecForge の出力が `lalamo speculator convert` でそのまま取り込めるか
5. z-lab 公式 DFlash の受理率が、蒸留版／heretic版へどこまで転移するか（A1／A1+ の成否）
6. 本番学習に必要なデータ規模
7. uzu／lalamo で speculator を入力言語ごとに切り替えられるか
8. 検証主体がアングラ／レビューに留まる項目: Apple Silicon 実測値、uzu ブログ本文、削除 PR の原文、AMD Primus／NeMo の原典、vLLM 側の不具合報告
9. ~~確認したソースのコミット（uzu 8962d3e）と、実機で使う uzu 0.6.17 の対応~~ → **解決済み（v3.1、Claude確認）**: git のタグ参照で `refs/tags/0.6.17` = コミット 8962d3e7a796 を確認。残るのは実機バイナリがその版でビルドされているかの現場確認のみで、R0a の開始時に版表示で照合すれば足りる
10. **隠れ状態の保存容量と方式**: hidden 次元は 4096 で確定した（§4）ため容量の数値はそのまま使える。残るのは、オフライン／オンラインの境界を実測で決めることだけ
11. **uzu 素の自己回帰 tok/s（投機なし）と、uzu の受理長の定義**: SGLang のブロック方式の受理長と同じ定義か未確認。倍率の基準になるため R0 で最初に測る

## 8. 出典

- アングラ元レポート4本（`~/workspace/weaver-investigation/fetched/` に回収・sha256 検証済み）: factcheck、ngram-feasibility、dataset（＋DATASETS.csv）、cost（55〜80h 内訳・Modal）
- Claude が直接確認した一次情報: arXiv:2607.06763、arXiv:2602.06036（HTML 版の付録A.1 を含む）、HF trymirai/weaver、HF trymirai/Qwen3.5-9B-M の speculator/config.json と shapes.json、HF z-lab/Qwen3.5-9B-DFlash のカード、HF Moonlight556/Qwen3.5-0.8B-DFlash、HF xtools-at/…-heretic（config.json の text_config: hidden 4096／32層／vocab 248,320 を含む）、GitHub trymirai/lalamo（コミット一致）、GitHub trymirai/uzu の dflash_tfm.rs（コミット 8962d3e、Weaver の Option 化と Argmax 方式）と git タグ参照（0.6.17 = 8962d3e）、GitHub z-lab/dflash のツリー（14件）、GitHub sgl-project/SpecForge のツリー（dflash 関連73件）、GitHub vllm-project/speculators のツリー（同47件）、SpecForge の qwen3.5-4b-dflash.json、Modal 公式料金ページ
- 外部レビュー3件（2026-11-06、ユーザー経由）。指摘のうち決定的なものは上記のとおり Claude が再検証し、未検証のものはその旨を明記した
- v3 の追加（Claude再計算）: 工数ルートの和（A0／A3 を含む全ルート）、隠れ状態の保存容量。いずれも文書内の数値と明示した仮定からの計算で、外部ソースは使っていない
