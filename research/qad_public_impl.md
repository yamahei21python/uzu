# QAD (Quantization-Aware Distillation) 公開実装調査・0.8B 最小再現計画書

PTQ (YAQA) 後の軽量 QAD 手法を分析し、JAX/lalamo（4-bit asym Group32 + RHT）環境で 0.8B モデルを再現するための実装比較・最小再現計画書である。

**数値区分**: 【引用】出典あり /【推定】計算根拠を併記 /【実測】本環境で測定。以下、明示なき数値は【推定】。実測値は未取得。

---

## 1. 背景と前提仕様

- **Mirai 手法**: PTQ 後、BF16 教師 rollout で軽量 QAD（4-bit zp / bf16 scale）。重みは **Muon + Deterministic STE**、Scale/ZP は **AdamW** で最適化。
- **lalamo 現状**: `rounding.py`（方向性 STE）と YAQA 反復はあるが、学習ループ・Muon・AdamW・蒸留損失は未実装。

| 項目 | 仕様・設定値 | 補足 |
| :--- | :--- | :--- |
| **対象モデル** | Qwen3.5-0.8B（Mirai 公表。Qwen2.5 に 0.8B は無い） | パラメータ数 約 0.8B |
| **量子化形式** | 4-bit Asymmetric Integer (Group32), zp 4-bit, scale bf16 | 符号なし整数 $[0, 15]$、32要素単位で Scale/ZP |
| **外れ値抑制** | RHT (Block-diagonal Hadamard) | ブロック幅 32 単位で直交回転、活性化外れ値を分散 |
| **フレームワーク** | JAX / Optax | JIT コンパイル・関数型自動微分 |

---

## 2. 公開 QAD / QAT 実装の比較

| 手法名 | リポジトリ / ライセンス | 最適化対象 | 損失 (Loss) | Logits/特徴量 | STE 種別 | Scale/ZP 学習 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LLM-QAT** | [facebookresearch/LLM-QAT][1] (NOASSERTION/独自) | W, Act, KV | KL Div | Rollout Logits 全域 KD | Standard STE | なし (PTQ 固定) |
| **BitDistiller** | [OpenBitSys/BitDistiller][2] (MIT) | W, Clip幅 | CAKLD + CE | 高確信度トークン重み付け | Tailored Clip | あり (Clip幅調整) |
| **EfficientQAT** | [OpenGVLab/EfficientQAT][3] (MIT) | P1:W, P2:s/z | P1:MSE, P2:CE/KD | P1:隠れ層, P2:Logits | Round STE | あり (P2 で E2E) |
| **OmniQuant** | [OpenGVLab/OmniQuant][4] (MIT) | LWC, LET | Block MSE | 出力隠れ層 (Logits 非使用) | Diff Clamp | あり (LWC 逆算) |
| **QTIP** | [Cornell-RelaxML/qtip][5] (GPL-3.0) | Trellis, Codebook | 2次歪み / MSE | Trellis 探索 / 隠れ層一致 | 代理 STE | あり (LUT / Scale) |

**示唆**: LLM-QAT の Rollout 蒸留を骨格とし温度 $T=1.0$ の Forward KL / Top-K 蒸留を採用。EfficientQAT 同様、重みと Scale/ZP の更新を分離して振動を防ぐ。

---

## 3. Muon の JAX 実装と代替策

- **存在・API**: Optax 公式の `optax.contrib.muon` [6] で利用可能（Experimental）。`ndim == 2`（行列重み）を Muon、`ndim != 2` を AdamW で自動更新。
- **原理**: 勾配モメンタム $M$ を多項式反復で極分解 $\text{Polar}(M) = M(M^T M)^{-1/2}$ 近似。特異値を 1 に正規化しランク崩壊を防ぐ。

| オプティマイザ | 適用パラメータ | メモリ | 収束性・特異値保持 | JAX 安定性 | 選択基準 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Muon (`optax.contrib.muon`)** | 2D 潜在重み | 低 ($M$ のみ) | 最良 (直交化でランク維持) | 実験的 | **第 1 推奨**。2D 行列の QAD に最適 |
| **AdamW (`optax.adamw`)** | Scale, Zero-point | 高 ($M, V$) | 良 (座標別適応学習) | 検証済み | **Scale / ZP に必須**。対角要素に適合 |
| **SGD + Momentum + RMS** | 2D 潜在重み | 最小 ($M$ のみ) | 中 (LR 調整がシビア) | 検証済み | Muon が XLA エラー時の退避策 |
| **AdamW 全面適用** | 全パラメータ | 最大 ($M, V$) | 標準 (スペクトル偏りあり) | 検証済み | アルゴリズム検証用ベースライン |

---

## 4. Deterministic STE と Stochastic Rounding

- **Stochastic Rounding (SR)**: $q_{SR}(x) = \lfloor x \rfloor + \text{Bernoulli}(x - \lfloor x \rfloor)$。無偏だが順伝播ごとに高周波ノイズが混入し、QAD の微小な KL 蒸留信号（$\Delta \sim 0.04$）が埋没して収束が遅延する。
- **Deterministic STE**: $\hat{w} = \text{round}(w), \frac{\partial \hat{w}}{\partial w} = 1$。有偏だが分散 0 で再現性が高く蒸留に適する。

### lalamo「範囲外へ戻す勾配のみ許可」の数式
4-bit asymmetric 範囲 $[0, 15]$ において、上流勾配を $g = \frac{\partial \mathcal{L}}{\partial \hat{w}}$、更新量を $\Delta w = -\eta \tilde{g}$ とする。lalamo の方向性 STE マスク $M(w, g)$ は以下で定義される：
$$\tilde{g} = g \cdot M(w, g)$$
$$M(w, g) = \begin{cases}
1 & \text{if } 0 \le w \le 15 \quad (\text{正常範囲内：通常透過}) \\
1 & \text{if } w > 15 \;\land\; g > 0 \quad (\Delta w < 0 \text{ で範囲内へ引き戻す}) \\
0 & \text{if } w > 15 \;\land\; g \le 0 \quad (\Delta w \ge 0 \text{ でさらに発散するため遮断}) \\
1 & \text{if } w < 0 \;\land\; g < 0 \quad (\Delta w > 0 \text{ で範囲内へ引き戻す}) \\
0 & \text{if } w < 0 \;\land\; g \ge 0 \quad (\Delta w \le 0 \text{ でさらに発散するため遮断})
\end{cases}$$

| 方式 | 範囲内 ($0 \le w \le 15$) | 範囲外・引き戻し勾配 | 範囲外・発散勾配 | 発生する現象 |
| :--- | :--- | :--- | :--- | :--- |
| **Naive STE** | 透過 ($g$) | 透過 ($g$) | 透過 ($g$) | 重みが無限遠へ漂流 (Weight Drift) |
| **Clipped STE** | 透過 ($g$) | **遮断 (0)** | **遮断 (0)** | 境界外でパラメータ死滅 (Dead Zone) |
| **lalamo Restoring STE** | **透過 ($g$)** | **透過 ($g$)** | **遮断 (0)** | **死滅を防ぎ、重み暴走を完全防止** |

---

## 5. 0.8B 最小 QAD 再現計画

### 5.1 三層計画（CPU/M3 と GPU の切り分け）
| 層 | モデル | 学習対象 | データ / step | 環境 | 目的 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T0 スモーク** | Qwen3.5-0.8B | W + s + z | 1k tok × 20 step | M3 / CPU | ループ・勾配・保存の疎通 |
| **T1 省メモリ** | Qwen3.5-0.8B | s / z のみ（W 凍結） | 512 × batch 8 × 100 step | M3 16GB | scale / zp 学習の検証（約 4GB） |
| **T2 本番** | Qwen3.5-0.8B | W + s + z | 512 × 16 × 300 step（2.46M tok） | GPU 24〜48GB | Mirai 再現 |

### 5.2 ハイパーパラメータ【推定】
| 項目 | 値 | 根拠 |
| :--- | :--- | :--- |
| データ | Mirai は内部 rollout（非公開）。公開代理: `teknium/OpenHermes-2.5`（PTQ 校正と非重複 3,000 件） | ドメイン感度が高い。ライセンス要確認 |
| rollout 長 | 512 tok | 計算量と長文崩れの折衷 |
| batch | micro 2 × accum 8 = 16 | 13GB 級メモリに収める |
| step | 300（2.46M tok） | 0.8B 軽量 KD の経験則。理論根拠なし |
| lr: W | 2.0e-4（Muon）, cosine, warmup 30 | Muon LLM 事例の下限 |
| lr: s / z | 1.0e-4（AdamW） | scale の桁変化を避ける |
| 損失 | Forward KL（T=1.0）。全語彙が重ければ top-k 64 | KL = CE(t,s) − H(t) で実装可 |
| 評価 | teacher KL, PPL, MMLU-Pro。公表 KL 0.041 は **2B** の値【引用】 | 0.8B の目標 KL は別途設定 |

### 5.3 メモリ【推定】（N=0.8B, fp32 master, seq 512, micro 2, remat）
| 内訳 | 計算 | GB |
| :--- | :--- | :--- |
| latent W fp32 | 0.8G × 4B | 3.2 |
| 勾配 fp32 | 同 | 3.2 |
| Muon モメンタム fp32 | 2D 重み ~95% | 3.0 |
| int4 重み | 0.8G × 0.5B | 0.4 |
| teacher bf16 | 0.8G × 2B | 1.6 |
| 活性化（remat） | 層数 × 数 MB | 1.0〜1.5 |
| XLA workspace | — | 0.5〜1.0 |
| **合計** | | **12.9〜14.4** |

**結論**: M3 16GB は OS 常駐を引くと実質 12〜13GB。T2 は OOM 危険で **24GB 以上 GPU 必須【推定】**。T0/T1 は M3 可。元稿の 5.92GB は fp32 master・勾配・Muon 状態を二重計上しておらず誤り。

### 5.4 時間【推定】
| 環境 | 実効 tok/s | 2.46M tok | 判定 |
| :--- | :--- | :--- | :--- |
| M3 GPU | 30〜60 | 11〜23 h | T2 非現実的 |
| M3 CPU | 5〜15 | 45〜137 h | T0 のみ |
| RTX 4090 | 1,500〜3,000 | 14〜27 min | T2 推奨 |
| A100 40GB | 2,000〜4,000 | 10〜20 min | T2 推奨 |

根拠: student 6ND + teacher 2ND ≈ 6.4 GFLOP/token、実効利用率 10〜30%。**実測ではない**。

---

## 6. 量子化重み・Scale・ZP 同時学習の注意点

| 項目 | 具体的メカニズム | 放置した場合の悪影響 | 回避・対策プラクティス |
| :--- | :--- | :--- | :--- |
| **STE 勾配干渉** | $\hat{W} = (Q - z)s$ で全変数に勾配連動 | パラメータ同士が干渉し振動 | 重みは Muon、Scale/ZP は AdamW とオプティマイザを完全分離 |
| **Scale Collapse** | $s \to 0$ で $\frac{W}{s} \to \infty$ となり全要素飽和 | 動的表現幅の消失・勾配 NaN | スケール下限値をクリップ ($s \ge 10^{-5}$)、または Softplus 化 |
| **Zero-point ドリフト** | $z$ が連続値として浮動移動 | 推論時 Int4 解凍値との不一致 | ZP にも STE 丸めを適用、または一定間隔で整数近傍へ射影 |
| **RHT 再適用タイミング** | QAD 中に RHT を再計算 | 量子化座標系が回転し崩壊 | **RHT は PTQ 前に 1 度のみ適用し QAD 中は完全固定（凍結）** |
| **RHT と勾配の整合** | RHT を毎 step 量子化直前に掛ける場合、dequant 後に逆 RHT を必ず通す | 潜在 W と量子化座標が混在し学習が破綻 | RHT を前処理で 1 度だけ適用して固定するのが安全 |

---

## 7. 実装候補擬似コード (JAX / Optax)

```python
import jax, jax.numpy as jnp, optax

G = 32

# 1) 素の量子化（custom_vjp の fwd から呼ぶ。再帰禁止）
def _qdq(w, s, z):
    # w:(out,in), s/z:(out,in//G)
    wg = w.reshape(w.shape[0], -1, G)
    q = jnp.clip(jnp.round(wg / s[..., None] + z[..., None]), 0.0, 15.0)
    return ((q - z[..., None]) * s[..., None]).reshape(w.shape)

@jax.custom_vjp
def qdq_ste(w, s, z):
    return _qdq(w, s, z)

def _fwd(w, s, z):
    return _qdq(w, s, z), (w, s, z)

def _bwd(res, g):
    w, s, z = res
    wg = w.reshape(w.shape[0], -1, G)
    gg = g.reshape(w.shape[0], -1, G)
    q = wg / s[..., None] + z[..., None]      # 丸め前座標
    inside  = (q >= 0.0) & (q <= 15.0)
    restore = ((q > 15.0) & (gg > 0.0)) | ((q < 0.0) & (gg < 0.0))
    gw = jnp.where(inside | restore, gg, 0.0).reshape(w.shape)  # dq/dw=1 とみなす
    qc = jnp.clip(jnp.round(q), 0.0, 15.0)
    gs = jnp.sum(gg * (qc - z[..., None]), axis=-1)  # d/dequant の直接項
    gz = jnp.sum(gg * (-s[..., None]), axis=-1)
    return gw, gs, gz
qdq_ste.defvjp(_fwd, _bwd)

# 2) 明示ラベル。scale は 2D なので muon 既定分割では誤って Muon に入る
def make_tx(params, lr_w=2e-4, lr_s=1e-4):
    def lab(path, leaf):
        key = getattr(path[-1], "key", str(path[-1]))
        if leaf.ndim == 2 and key == "weight":
            return "muon"
        return "adam_s" if key == "scale" else "adam_z"
    labels = jax.tree_util.tree_map_with_path(lab, params)
    return optax.multi_transform(
        {"muon": optax.contrib.muon(lr_w, ns_steps=5, beta=0.95),
         "adam_s": optax.adamw(lr_s),
         "adam_z": optax.adamw(lr_s)},
        labels,
    )

# 3) 蒸留 step。KL = CE(t,s) - H(t) なので CE で最適化
@jax.jit
def qad_step(params, opt_state, teacher, tokens, tx):
    def loss_fn(p):
        qw = jax.tree.map(qdq_ste, p["weight"], p["scale"], p["zp"])
        s_logits = student_forward(qw, p, tokens)
        t_logits = jax.lax.stop_gradient(teacher_forward(teacher, tokens))
        logp_t = jax.nn.log_softmax(t_logits, -1)
        logp_s = jax.nn.log_softmax(s_logits, -1)
        return -jnp.mean(jnp.sum(jnp.exp(logp_t) * logp_s, -1))
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    p = optax.apply_updates(params, updates)
    p["scale"] = jnp.maximum(p["scale"], 1e-5)      # scale collapse 防止
    return p, opt_state, loss
```

補足: `stop_gradient(w + (wq - w))` では方向判定ができないため `custom_vjp` 必須。`optax.contrib.muon` は experimental。バージョンを pin し、XLA エラー時は SGD momentum + AdamW に退避。

---

## 8. 未確認事項および技術リスク

| リスク要因 | 発生確率 | 影響度 | 兆候・課題 | 回避策・実証計画 |
| :--- | :--- | :--- | :--- | :--- |
| **擬似量子化と実カーネルの乖離** | 中 | 高 | Float 逆量子化と Int4 ハードウェアカーネルの計算順序・桁落ち差 | QAD 直後に lalamo の `compress` を適用し、実機 (uzu/Metal) で KL を計測 |
| **Rollout のドメイン偏向** | 高 | 高 | SFT 単一ドメイン蒸留によるコード・推論性能の劣化 | Agentログ 45% + SFT 30% + 推論/対話 25% のデータ混合を実施 |
| **XLA の動的形状リコンパイル** | 高 | 中 | バッチごとのシーケンス長差異によるコンパイル停止 | `max_length=512` に固定パディングし Padding Mask を徹底 |
| **Muon Newton-Schulz 発散** | 低 | 高 | 初期イテレーションでの特異値過大による NaN 発生 | グローバルノルムクリップ (`optax.clip_by_global_norm(1.0)`) を併用 |
| **Qwen3.5-0.8B の入手性** | 中 | 高 | ブログは Qwen3.5-0.8B と明記。HF 公開状況・重み形式は未確認 | HF で存在確認。無ければ 0.6B 代理で T0/T1 |
| **optax muon API 互換** | 中 | 中 | 署名・既定分割 (ndim==2→Muon) が版で変わる | requirements に optax を pin、初期化 smoke test |
| **スループット未実測** | 高 | 中 | 5.4 の tok/s は FLOP 概算のみ | 20 step の実測で係数を置換 |
| **Mirai ブログ URL 到達性** | 中 | 中 | 引用 [8] の内容・KL 0.041 を一次確認していない | 原典を再取得し数値を照合 |
| **データライセンス** | 中 | 中 | OpenHermes-2.5 の再配布条件未確認 | dataset card を確認し社内規程と突合 |

---

## 9. 引用文献・リポジトリ一覧

[1] **LLM-QAT**: Liu et al., *LLM-QAT*, ACL Findings 2024. https://github.com/facebookresearch/LLM-QAT  
[2] **BitDistiller**: Du et al., *BitDistiller*, ACL 2024. https://github.com/OpenBitSys/BitDistiller  
[3] **EfficientQAT**: Chen et al., *EfficientQAT*, ACL 2025. https://github.com/OpenGVLab/EfficientQAT  
[4] **OmniQuant**: Shao et al., *OmniQuant*, ICLR 2024. https://github.com/OpenGVLab/OmniQuant  
[5] **QTIP**: Tseng et al., *QTIP*, NeurIPS 2024. https://github.com/Cornell-RelaxML/qtip  
[6] **Optax Muon**: Google DeepMind, *Optax Contrib Documentation*. https://optax.readthedocs.io/en/latest/api/contrib.html  
[7] **OpenHermes-2.5**: Teknium, Hugging Face Hub. https://huggingface.co/datasets/teknium/OpenHermes-2.5  
[8] **Mirai Quantization**: Mirai Labs, *Mirai Quantization Blog*, June 2026. https://trymirai.com/blog/quantization（2026-11-07 到達確認。KL 0.041 は 2B の公表値）
