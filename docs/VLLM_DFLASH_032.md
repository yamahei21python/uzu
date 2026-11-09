# vLLM 0.32.0 の DFlash 受理ゼロ（原因確定）

27B で観測した「提案 3,704 / 受理 0」は**モデルや分布のずれではなく、vLLM 0.32.0 の
回帰**。一次情報で確定した（2026-11-09）。

## 結論

| 項目 | 内容 |
|---|---|
| 原因 | PR #60692（0.32.0 に merge、`36f397281`）が DFlash の draft を FULL CUDA graph で replay する際、**kernel block table を更新しない** |
| 症状 | draft が古い block table を読む → 提案が target と噛み合わず受理 0。出力テキスト自体は正しい（検証は走る） |
| 修正 | PR #64059（2026-11-04 merge、main / `0.32.1rc1.dev505`）。**コード 8 行** |
| リリース | **どのリリースにも入っていない**。`v0.32.0` / `v0.32.1rc0` / `v0.33.0rc3` / `v0.33.0` を commit 比較で確認（2026-11-09）。入るのは main / nightly のみ |
| ワークアラウンド | `cudagraph_mode=PIECEWISE` または `--enforce-eager`（DFlash は piecewise を持たないので draft が eager になる） |

## 実測（Issue #63587 / PR #64059、B200）

| 構成 | 受理率 | req/s |
|---|---:|---:|
| 0.31.0 | 0.531 | 14.6 |
| 0.32.0（既定 FULL） | **0.004** | 3.7 |
| 0.32.0 + `PIECEWISE` | 0.382 | 11.6 |
| 0.32.0 + `--enforce-eager` | 0.389 | — |
| 0.32.1rc1 + #64059（FULL のまま） | **0.552** | **20.6** |

PIECEWISE は診断には十分だが、**恒久対策にはならない**（スループットが修正版の約半分）。

## うちの run への当てはめ

条件一致:

- vLLM 0.32.0、`method=dflash`（DFlash2）
- 既定の `cudagraph_mode = FULL_AND_PIECEWISE` → ログに `Capturing dflash2 CUDA graphs (FULL)`
- 起動時に `Speculative decoding (method=dflash) is enabled but no KV cache group
  could be identified as the draft model's.` — Issue #63587 にも同じ行が出ている
  （ただし**これは直接原因ではない**）

統計的な裏付け: 受理率が仮に 5% あっても 3,704 提案で 0 受理になる確率は約 5×10⁻¹¹。
分布のずれでは説明できない。

## 決定実験（1 本だけ）

```bash
vllm serve <27B target> \
  --speculative-config '{"method":"dflash","model":"<dflash2 draft>","num_speculative_tokens":8}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --no-enable-prefix-caching
```

- `PIECEWISE` を選ぶ理由: `--enforce-eager` と同じ診断になり、target のグラフは残る
- `--no-enable-prefix-caching` を入れる理由: Issue #58894（prefix cache ヒット後に受理が
  恒久 0 になる別件、**未修正**）を交絡から外す

予測: 受理率 **25〜45%**（B200 実測 0.382）。

| 結果 | 判断 |
|---|---|
| 受理率 20% 以上 | 原因確定（#60692）。**パッチ適用 or PIECEWISE で運用** |
| 0 のまま | 別原因。draft の checkpoint 不一致 / #58894 / 27B 固有を疑う |

## 恒久対策（優先順）

1. **#64059 の 10 行を当てる**（`vllm/v1/worker/gpu/spec_decode/{dflash,autoregressive,}speculator.py`）
   - 0.32.0 のイメージに `git apply` できる見込み。FULL グラフを維持できて最速
2. 0.32.1 / main のビルドに上げる（**0.33.0 でも直らない**。commit 比較で確認済み。nightly 待ち）
3. 当面 `PIECEWISE` で運用（速度は半分）

## 注意

- PR #64059 の split 条件は **FlashAttention v4 hd256（Blackwell）で kernel block が分割される
  場合**。Hopper / Ada では分割が起きないため、**同じ修正では直らない可能性がある**。
  その場合はまず `PIECEWISE` で切り分ける（PIECEWISE で直れば #60692 系、直らなければ別件）
- `--speculative-config '{"enforce_eager": true}'` は**効かない**。runner が target の
  cudagraph mode を draft に渡すため（Issue #63587 に明記）
- MTP（31.3 t/s）はこの問題の影響を受けない。サービスは MTP で維持してよい
