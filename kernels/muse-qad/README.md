# Muse 用 QAD カーネル（自動生成）

`scripts/make_kaggle_kernel.py --kind qad` の生成物。手で編集しない。

## 前提

- データセット: `yamahei21python/uzu-bf16-08b-lalamo`（teacher の BF16 lalamo）
- HF: `kentasatomuse/qad-best-student`（`student/` と較正データ）

## 中身

1. kit を **commit 固定**で取得（`build_kit --check` で検証）
2. student を HF から取得
3. `qad_finetune.py`
   - `adamw` / `--parallel tensor` / `--seq 32` / `--steps 50`
   - `--freeze-scales` / `--skip-checkpoint` / `--master-dtype float32`
   - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
4. 評価（**float32**）: before = student, after = run
5. `=== FINAL ===` に JSON を出力

## 再生成

```bash
python scripts/make_kaggle_kernel.py --kind qad \
  --spec lanes/asym8-wide8-d1e-1-08b.json \
  --kernel-id kentasatomuse/uzu-qad-muse \
  --student-repo kentasatomuse/qad-best-student \
  --dataset yamahei21python/uzu-bf16-08b-lalamo \
  --out kernels/muse-qad --seq 32 --steps 50
```

## 注意

- `--parallel` は `tensor` か `replicated` のみ（`data`/`fsdp` は kit が拒否する）
- `--optimizer muon` は `tensor` と併用不可（同じく kit が拒否）
- OOM する場合は `--seq 16`。条件が変わるので結果は別物として扱う
- 旧ハーネス（v4）に **freeze-scales が入っていない場合**、このカーネルでは入っているぶん
  メモリが軽い。14MiB の差ならそこで通る可能性が高い
