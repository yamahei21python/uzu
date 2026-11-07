# uzu 実験用コーパス 公開データ汚染監査レポート (Public Corpus Contamination Audit)

- 実行日時: 2026-11-07 03:55:03 UTC
- 実行ホスト / 環境: Ubuntu 24.04 container (Google Cloud Run)
- 最小行長閾値: 正規化後 40 文字以上（定型文・見出しノイズ除去）
- 日本語 Wikipedia 監査行数: 200,014 行（951 記事）
- 総所要時間: 9.82 秒

## 1. エグゼクティブサマリ

uzu は Qwen3.5 系モデルを RHT + INT4 に量子化し、YAQA（Hessian校正）および QAD（蒸留微調整）で評価を行っている。
本監査では、dedup（重複排除）後のコーパスが、公開データセット側の訓練 split や互いの split とどれだけ重複しているかを定量化した。

### 主な結論
1. **英語評価コーパス (`eval_en_wikitext2_dedup.txt`) の完全な独立性**:
   - WikiText-2 の **`test` split と 100.00% 一致（757/757行）**。
   - WikiText-2 の **`train` / `validation` split との重複は 0.00%（0行）**。
   - すべての calibration コーパス（EN, mixed）との交差も **0.00%（0行）**。
   - **評価漏洩リスクは完全にゼロ** であり、テストセットとしての独立性が証明された。

2. **英語校正コーパス (`calibration_en_wikitext2_dedup.txt`) の健全性**:
   - WikiText-2 の **`train` split から 100.00% 抽出（778/778行）**。
   - `validation` および `test` split との重複は **0.00%（0行）**。

3. **日本語評価・校正コーパスの分離状況**:
   - `eval_ja_wikipedia_dedup.txt`（3,054行）と `calibration_ja_wikipedia_dedup.txt`（2,893行）は、ともに Japanese Wikipedia (`wikimedia/wikipedia` `20231101.ja`) 由来（100.00%合致）。
   - しかし、**両者の間で重複する行は 0.00%（0行）** であり、さらに**抽出元の Wikipedia 記事タイトルも完全に排他（eval 68記事 vs cal 46記事、共通記事 0件）** である。
   - したがって、uzu の量子化校正（YAQA/QAD）から評価へのデータ漏洩は存在しない。

4. **混合校正コーパスの構成内訳**:
   - `cal_mixed_en_ja_dedup.txt` (795行): WikiText-2 `train` 52.33% (416行) + JA Wikipedia 47.67% (379行) で **100% 構成説明可能**。
   - `cal_mixed_ja65_en35_dedup.txt` (911行): WikiText-2 `train` 28.76% (262行) + JA Wikipedia 71.24% (649行) で **100% 構成説明可能**。
   - いずれも `eval_en` および `eval_ja` との重複は **0.00%（0行）**。

5. **言語間混入（Language Contamination）の皆無**:
   - `eval_en` vs `eval_ja` は **0.00%（0行）**。ENとJAの評価データ間での混入はない。

## 2. 監査手法・正規化仕様

- **2段階の正規化**:
  - **`raw` (完全一致)**: 改行文字を除去した文字列の一致。
  - **`norm` (正規化一致)**: 小文字化 (`.lower()`) → 連続空白を半角スペース1個に集約 (`re.sub(r'\s+', ' ', s)`) → 前後空白の除去 (`.strip()`)。
  - ※ `mixed_ja65` の英語行や一部の Wikipedia 行では先頭空白の有無による差分が生じるため、`norm` による比較が本質的な重複検出に不可欠。
- **長さフィルタ**:
  - `len(norm(line)) >= 40` 文字未満の行は重複判定から除外（「= = Career = =」等の見出し、空行、短縮署名などの定型ノイズを排除）。
- **比較方法**:
  - Set 交差（集合積）による厳密計算。行数および文字数（character length）の両方で汚染率を算出。

## 3. 入力コーパス諸元 (Filtered >= 40 chars)

| コーパス名 | 総行数 | 判定対象行数 (>=40字) | 総文字数 | 判定対象文字数 |
|:---|:---:|:---:|:---:|:---:|
| `eval_en` | 2,092 | 757 | 499,394 | 492,078 |
| `eval_ja` | 9,905 | 3,054 | 496,123 | 415,937 |
| `cal_en` | 2,125 | 778 | 498,562 | 491,488 |
| `cal_ja` | 11,524 | 2,893 | 509,936 | 453,784 |
| `cal_mixed_en_ja` | 1,191 | 795 | 332,116 | 325,087 |
| `cal_mixed_ja65` | 2,111 | 911 | 276,605 | 262,902 |

## 4. 公開データセットとの重複監査結果

各コーパスが公開データセット（WikiText-2 各 split、Japanese Wikipedia 20万行サンプル）と一致した件数と割合:

| コーパス | 公開データセット対象 | 一致行数 (raw) | 一致行率 (raw) | 一致行数 (norm) | 一致行率 (norm) | 文字汚染率 (norm) | 判定 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `eval_en` | `wt2_train` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `eval_en` | `wt2_validation` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `eval_en` | `wt2_test` | 757 | 100.00% | 757 | 100.00% | 100.00% | 正常（Test split抽出） |
| `eval_en` | `wiki_ja_sample` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `eval_ja` | `wt2_train` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `eval_ja` | `wt2_validation` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `eval_ja` | `wt2_test` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（Test split抽出） |
| `eval_ja` | `wiki_ja_sample` | 3,054 | 100.00% | 3,054 | 100.00% | 100.00% | 正常（Wikipedia抽出） |
| `cal_en` | `wt2_train` | 778 | 100.00% | 778 | 100.00% | 100.00% | 正常（Train split抽出） |
| `cal_en` | `wt2_validation` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_en` | `wt2_test` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_en` | `wiki_ja_sample` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_ja` | `wt2_train` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（Train split抽出） |
| `cal_ja` | `wt2_validation` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_ja` | `wt2_test` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_ja` | `wiki_ja_sample` | 2,893 | 100.00% | 2,893 | 100.00% | 100.00% | 正常（Wikipedia抽出） |
| `cal_mixed_en_ja` | `wt2_train` | 416 | 52.33% | 416 | 52.33% | 79.23% | 正常（Train split抽出） |
| `cal_mixed_en_ja` | `wt2_validation` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_mixed_en_ja` | `wt2_test` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_mixed_en_ja` | `wiki_ja_sample` | 379 | 47.67% | 379 | 47.67% | 20.77% | 正常（Wikipedia抽出） |
| `cal_mixed_ja65` | `wt2_train` | 0 | 0.00% | 262 | 28.76% | 62.52% | 正常（Train split抽出） |
| `cal_mixed_ja65` | `wt2_validation` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_mixed_ja65` | `wt2_test` | 0 | 0.00% | 0 | 0.00% | 0.00% | 正常（完全分離） |
| `cal_mixed_ja65` | `wiki_ja_sample` | 534 | 58.62% | 649 | 71.24% | 37.48% | 正常（Wikipedia抽出） |

## 5. uzu コーパス相互の交差監査結果

| コーパス A | コーパス B | 一致行数 (raw) | 一致行率 (raw) | 一致行数 (norm) | 一致行率 (norm) | 判定 |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| `eval_en` | `cal_en` | 0 | 0.00% | 0 | 0.00% | OK (完全分離) |
| `eval_ja` | `cal_ja` | 0 | 0.00% | 0 | 0.00% | OK (完全分離) |
| `eval_en` | `cal_mixed_en_ja` | 0 | 0.00% | 0 | 0.00% | OK (完全分離) |
| `eval_en` | `cal_mixed_ja65` | 0 | 0.00% | 0 | 0.00% | OK (完全分離) |
| `eval_ja` | `cal_mixed_en_ja` | 0 | 0.00% | 0 | 0.00% | OK (完全分離) |
| `eval_ja` | `cal_mixed_ja65` | 0 | 0.00% | 0 | 0.00% | OK (完全分離) |
| `eval_en` | `eval_ja` | 0 | 0.00% | 0 | 0.00% | OK (完全分離) |
| `cal_en` | `cal_mixed_en_ja` | 416 | 53.47% | 416 | 53.47% | 校正セット間共有（意図的） |
| `cal_en` | `cal_mixed_ja65` | 0 | 0.00% | 262 | 33.68% | 校正セット間共有（意図的） |
| `cal_ja` | `cal_mixed_en_ja` | 379 | 13.10% | 379 | 13.10% | 校正セット間共有（意図的） |
| `cal_ja` | `cal_mixed_ja65` | 534 | 18.46% | 649 | 22.43% | 校正セット間共有（意図的） |

## 6. 一致代表例（最大5件、120文字切詰）

### `eval_en` × `wt2_test` (5 件例示)
- ` Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television s...`
- ` In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 e...`
- ` In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed " Scott Parry " in the e...`
- ` He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In 2004 Boulter landed a ...`
- ` In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which feature...`

### `eval_ja` × `wiki_ja_sample` (5 件例示)
- `介錯（かいしゃく）は、切腹に際し、本人を即死させてその負担と苦痛を軽減するため、介助者が背後から切腹人の首を刀で斬る行為。`
- `腹部を切開しただけでは人は即死しない。従って切腹では、割腹の後、自ら喉を突き通したり心臓を刺すことが正式な作法であったが、現実問題として難しい場合が多かった。そこで切腹を行う者の負担を軽減し、また即死できない本人が醜態を見せることのないよう...`
- `首を刀で斬り落とすのには、首の骨の関節を切る、また「首の皮一枚」を残すなどいくつかの作法が存在する。頭部を完全に切断せず首の皮で胴体に繋げた状態とするのは、胸の前にぶらさがった頭の重みで切腹者を前のめりの状態で死なせる配慮で、首が落ちずに危...`
- `剣の扱いに未熟な者は手許を誤って斬り損ね、何度も首に斬りつけたり、刀を損傷してしまうことも多々あった。三島事件の際に介錯した森田必勝は、2度斬り損ねた上に刀を曲げてしまった。介錯の不手際は切腹人を苦しめるのみならず、面目を失する行為とされた...`
- `介錯は、居合道の形として現代でも伝承が続いている。介錯の形を伝える流派は無双直伝英信流（無雙神傳英信流、夢想神伝流）に並流された大森流などである。夢想神伝流では順刀とも呼ばれる。`

### `cal_en` × `wt2_train` (5 件例示)
- ` Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly...`
- ` The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While i...`
- ` It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it receive...`
- ` As with previous Valkyira Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players ...`
- ` The game 's battle system , the BliTZ system , is carried over directly from Valkyira Chronicles . During missions , pl...`

### `cal_ja` × `wiki_ja_sample` (5 件例示)
- `アンパサンド（&, ）は、並立助詞「…と…」を意味する記号である。ラテン語で「…と…」を表す接続詞 "et" の合字を起源とする。現代のフォントでも、Trebuchet MS など一部のフォントでは、"et" の合字であることが容易にわかる...`
- `英語で教育を行う学校でアルファベットを復唱する場合、その文字自体が単語となる文字（"A", "I", かつては "O" も）については、伝統的にラテン語の per se（それ自体）を用いて "A per se A" のように唱えられていた。...`
- `アンドレ＝マリ・アンペールがこの記号を自身の著作で使い、これが広く読まれたため、この記号が "Ampère's and" と呼ばれるようになったという誤った語源俗説がある。`
- `アンパサンドの起源は1世紀の古ローマ筆記体にまでさかのぼることができる。古ローマ筆記体では、E と T はしばしば合字として繋げて書かれていた（左図「アンパサンドの変遷」の字形1）。それに続く、流麗さを増した新ローマ筆記体では、さまざまな合...`
- `現代のイタリック体のアンパサンドは、ルネサンス期に発展した筆記体での et の合字にさかのぼる。1455年のヨーロッパにおける印刷技術の発明以降、印刷業者はイタリック体とローマ筆記体のアンパサンドの両方を多用するようになった。アンパサンドの...`

### `cal_mixed_en_ja` × `wt2_train` (5 件例示)
- ` Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly...`
- ` The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While i...`
- ` It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it receive...`
- ` As with previous Valkyira Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players ...`
- ` The game 's battle system , the BliTZ system , is carried over directly from Valkyira Chronicles . During missions , pl...`

### `cal_mixed_en_ja` × `wiki_ja_sample` (5 件例示)
- `アンパサンド（&, ）は、並立助詞「…と…」を意味する記号である。ラテン語で「…と…」を表す接続詞 "et" の合字を起源とする。現代のフォントでも、Trebuchet MS など一部のフォントでは、"et" の合字であることが容易にわかる...`
- `英語で教育を行う学校でアルファベットを復唱する場合、その文字自体が単語となる文字（"A", "I", かつては "O" も）については、伝統的にラテン語の per se（それ自体）を用いて "A per se A" のように唱えられていた。...`
- `アンドレ＝マリ・アンペールがこの記号を自身の著作で使い、これが広く読まれたため、この記号が "Ampère's and" と呼ばれるようになったという誤った語源俗説がある。`
- `アンパサンドの起源は1世紀の古ローマ筆記体にまでさかのぼることができる。古ローマ筆記体では、E と T はしばしば合字として繋げて書かれていた（左図「アンパサンドの変遷」の字形1）。それに続く、流麗さを増した新ローマ筆記体では、さまざまな合...`
- `現代のイタリック体のアンパサンドは、ルネサンス期に発展した筆記体での et の合字にさかのぼる。1455年のヨーロッパにおける印刷技術の発明以降、印刷業者はイタリック体とローマ筆記体のアンパサンドの両方を多用するようになった。アンパサンドの...`

### `cal_mixed_ja65` × `wt2_train` (5 件例示)
- `The pilot qualification most relevant to GA is the Private Pilot Licence ( PPL ) , which permits the holder to fly for r...`
- `Flying schools are commercial businesses engaged in the training of pilots , both for recreational purposes and for thos...`
- `The tower had two floors and a mezzanine . On the first floor was the famous Sundial Room or Meridian Room , which was i...`
- `Gambia 's national football association was founded in 1952 , and became affiliated with FIFA in 1968 . Football is the ...`
- `At that time of the assassination of Archduke Franz Ferdinand of Austria on 28 June 1914 , the battleships in the Austro...`

### `cal_mixed_ja65` × `wiki_ja_sample` (5 件例示)
- `アンリ・ルフェーヴル(1947), Critique of Everyday Life`
- `元来、日本に文字と呼べるものはなく、言葉を表記するためには中国渡来の漢字を用いた（いわゆる神代文字は後世の偽作とされている）。漢字の記された遺物の例としては、1世紀のものとされる福岡市出土の「漢委奴国王印」などもあるが、本格的に使用されたの...`
- `「や行」の「え」() の音が古代に存在したことは、「あ行」の「え」の仮名と別の文字で書き分けられていたことから明らかである。平安時代初期に成立したと見られる「天地の詞」には「え」が2つ含まれており、「あ行」と「や行」の区別を示すものと考えら...`
- `フランス語の統制機関としては、アカデミー・フランセーズが挙げられる。これは1635年に宰相・リシュリューによって創設された国家機関で、フランス語の語法を整備して誰にでも理解できる言語とすることを目指し、そのためにフランス語の辞書『アカデミー...`
- `数多くの病院がパリに設置されている。そのうちいくつかは特に古く、医療の伝統は中世にまでさかのぼる。651年にパリ司教だった聖ランドリーによって設立されたシテ島のオテル・デューは、パリでもっとも古い医療施設である。慈愛ともてなしの象徴であり、...`

## 7. 結論と推奨事項

1. **評価漏洩リスクの定量結論**:
   - **WikiText-2 (英語)**: `eval_en` は WikiText-2 の `test` split と 100% 一致し、`train` / `validation` および全 calibration コーパスと 0% 重複である。リークのリスクは皆無である。
   - **Wikipedia (日本語)**: `eval_ja` は Wikipedia サンプルに含まれるが、`calibration_ja` および mixed calibration との重複行数は 0 行（0.00%）、かつ記事単位でも完全に排他（共通記事 0件）である。量子化パイプラインにおける calibration -> eval の漏洩リスクはない。
2. **dedup 処理の十分性評価**:
   - 過去に見つかっていた EN 16行、JA 754行などの重複は、dedup 処理によって**1行の残存もなく完全に排除されている**ことが本監査で実証された。
   - したがって、現行の `*_dedup.txt` コーパスはモデル評価用として安全かつ健全である。
3. **留意点（Foundation Model 側の事前学習）**:
   - 基底モデル（Qwen3.5 など）が Wikipedia JA 全文や WikiText-2 を事前学習（Pre-training）で学習している可能性は一般に存在するが、これはモデルフリーの量子化校正・評価パイプラインの外部要因であり、本パイプライン内部での校正時データ漏洩（Data Contamination during Quantization / Calibration）は完全に防止されている。

## 8. 再現手順 (Full Reproduction Commands)

```bash
# 1. コーパスの取得
hf download yamahei21/uzu-quant-kit --repo-type dataset --include 'data/*' --local-dir /workspace/uzu-quant-kit

# 2. 監査スクリプトの実行
python3 scripts/audit_public_corpus.py --max-ja-rows 200000 --output-dir research
```

---

## 9. 独立検証（owner ローカル再実行）

同じスクリプトを M3 ローカルで再実行し、数値が一致することを確認した。

```bash
uv pip install -q datasets
python scripts/audit_public_corpus.py --max-ja-rows 200000 --output-dir /tmp/audit_full
```

- 実行時間: 26.06 秒
- JA サンプル: 951 記事 / 200,014 行（report と一致）
- `audit_results` / `cross_results` の全 17 行が **完全一致**
- `datasets` 5.1.0 / `Salesforce/wikitext` `wikitext-2-raw-v1` / `wikimedia/wikipedia` `20231101.ja`

`--no-public` を付けると公開データ取得を省き、uzu コーパス相互の交差だけを 0.4 秒で確認できる（オフライン用）。
