#!/usr/bin/env python3
"""Minimal Quantization-Aware Distillation (QAD) for lalamo HybridMatrix models.

Pipeline:
  1. Load a BF16 teacher and an RHT+INT4 PTQ student (lalamo format).
  2. Switch student HybridMatrix nodes to CompressionImplementation.TRAINING.
  3. Teacher-forced forward on a corpus; KL(teacher || student) loss.
  4. Optimize:
       - master_weights: Muon (optax.contrib.muon)
       - scales / zero points: AdamW
     with deterministic STE from lalamo's rounding.
  5. Switch back to INFERENCE, save, rewrite metadata for uzu.

This is a first-stage QAD: the loss uses teacher-forced corpus text. For the
full Mirai recipe, replace the corpus with teacher-generated rollouts.

Usage:
  JAX_PLATFORM_NAME=cpu python scripts/qad_finetune.py \
    --teacher-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
    --student-dir /tmp/qwen35_0.8b_yaqa_hin_hout_lalamo_eval \
    --corpus experiments/qwen35-0.8b/data/calibration_en_wikitext2_dedup.txt \
    --eval-corpus experiments/qwen35-0.8b/data/eval_en_wikitext2.txt \
    --eval-corpus experiments/qwen35-0.8b/data/eval_ja_wikipedia.txt \
    --output-dir /tmp/qwen35_0.8b_qad_lalamo \
    --steps 50 --seq 128 --lr 1e-4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--eval-corpus", type=Path, action="append", default=[], help="evaluation corpus to keep disjoint")
    parser.add_argument("--allow-overlap", action="store_true", help="allow training/eval overlap for analysis only")
    parser.add_argument("--lalamo-path", default=os.environ.get("LALAMO_PATH", "/tmp/lalamo-direct"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument(
        "--freeze-scales",
        action="store_true",
        help="train master_weights only; leave scales and zero-points fixed",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="linearly ramp the learning rate over this many steps",
    )
    parser.add_argument(
        "--optimizer",
        choices=("muon", "adamw"),
        default="muon",
        help="muon: master=Muon + scales=AdamW; adamw: all trainables=AdamW",
    )
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.95, help="short QAD (<100 steps) uses 0.95, not 0.999")
    parser.add_argument("--scale-min", type=float, default=1e-6, help="lower clamp for scale params; 0 disables")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="global norm clip; 0 disables")
    parser.add_argument("--scale-lr", type=float, default=1e-6, help="AdamW lr for scales/zero-points")
    parser.add_argument("--early-stop-patience", type=int, default=2, help="stop after N non-improving val evals")
    parser.add_argument("--student-provenance", type=Path, default=None, help="default: <student-dir>/provenance.json")
    parser.add_argument("--allow-unverified-student", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-chunks", type=int, default=2, help="fixed chunks used for comparable validation loss")
    parser.add_argument("--val-every", type=int, default=5, help="evaluate fixed validation chunks every N steps")
    parser.add_argument("--save-every", type=int, default=10, help="checkpoint interval; 0 = only final")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="default: <output-dir>/checkpoints")
    parser.add_argument("--resume", default=None, help="'auto' or a checkpoint directory")
    parser.add_argument("--keep-checkpoints", type=int, default=2, help="retain the newest N checkpoints")
    parser.add_argument("--parallel", choices=("replicated", "tensor", "fsdp", "data"), default="replicated")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.lalamo_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from corpus_guard import assert_disjoint

    assert_disjoint(args.corpus, args.eval_corpus, allow_overlap=args.allow_overlap)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from lalamo.compressed.hybrid import HybridMatrix
    from lalamo.model import Model
    from lalamo_forward import apply_attention_env, describe
    from lalamo.module import Keychain
    from lalamo.modules import DecoderForwardPassConfig, ForwardPassMode
    from lalamo.utils.sharding import ShardingConfig
    from lalamo.utils.surgery import map_nodes_of_type_with_path
    from lalamo.weight_matrix import CompressionImplementation, GradientEstimator

    from normalize_lalamo_config import normalize
    from rewrite_lalamo_hybrid_spec import rewrite

    if args.parallel == "tensor":
        sharding = ShardingConfig.tensor_parallel()
    elif args.parallel == "fsdp":
        sharding = ShardingConfig.fully_sharded_data_parallel()
    elif args.parallel == "data":
        sharding = ShardingConfig.data_parallel()
    else:
        sharding = ShardingConfig.replicated()
    keychain = Keychain.init(seed=args.seed, sharding_config=sharding)
    inference_config = apply_attention_env(
        DecoderForwardPassConfig.for_inference(mode=ForwardPassMode.MULTI_TOKEN)
    )
    training_config = apply_attention_env(
        DecoderForwardPassConfig.for_training(
            gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING
        )
    )
    print(f"attention: {describe()}", flush=True)

    def tokenizer_hash(directory: Path) -> str:
        return hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest()

    for label, directory in (("teacher", args.teacher_dir), ("student", args.student_dir)):
        config = json.loads((directory / "config.json").read_text())
        if "response_format" not in config.get("token_codec_config", {}):
            raise SystemExit(
                f"{label}: {directory} is uzu-normalized; QAD requires raw lalamo config"
            )
    if tokenizer_hash(args.teacher_dir) != tokenizer_hash(args.student_dir):
        raise SystemExit("teacher/student tokenizer.json mismatch")

    provenance_path = args.student_provenance or (args.student_dir / "provenance.json")
    if not provenance_path.exists():
        if not args.allow_unverified_student:
            raise SystemExit(
                f"student provenance missing: {provenance_path}. "
                "Rebuild the student with a clean calibration or pass --allow-unverified-student for analysis."
            )
        provenance = {"overlap_checked": False, "kind": "unknown"}
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not provenance.get("overlap_checked") and not args.allow_unverified_student:
            raise SystemExit(
                f"student provenance says overlap_checked=false: {provenance_path}"
            )
    print(f"student provenance: kind={provenance.get('kind')} overlap_checked={provenance.get('overlap_checked')}")

    print("loading teacher...")
    teacher = Model.load(args.teacher_dir, sharding, dtype=jnp.bfloat16)
    print("loading student...")
    student = Model.load(args.student_dir, sharding, dtype=jnp.bfloat16)

    def to_training(path, matrix: HybridMatrix) -> HybridMatrix:
        return matrix.switch_implementation(CompressionImplementation.TRAINING)

    student = map_nodes_of_type_with_path(HybridMatrix, to_training, student)
    print("student switched to TRAINING implementation")

    def logits_for(model, ids, config):
        token_ids = jnp.asarray([ids], dtype=jnp.int32)
        positions = jnp.arange(len(ids), dtype=jnp.int32)[None, :]
        result = model.decoder(
            token_ids=token_ids,
            token_positions=positions,
            state=None,
            return_updated_state=False,
            lengths_without_padding=None,
            forward_pass_config=config,
            keychain=keychain.split(1)[0],
        )
        return result.logits[0]

    texts = [line.strip() for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    token_ids = teacher.token_codec.tokenizer.encode("\n\n".join(texts), add_special_tokens=True).ids
    chunks = [token_ids[start : start + args.seq] for start in range(0, len(token_ids), args.seq)]
    chunks = [chunk for chunk in chunks if len(chunk) >= 16]
    if not chunks:
        raise SystemExit("no usable calibration chunks")
    print(f"chunks: {len(chunks)} x {args.seq} tokens")
    val_ids = chunks[-args.val_chunks :] if args.val_chunks > 0 else []

    def collect_params(model):
        params = {}
        trainable = ("master_weights",) if args.freeze_scales else ("master_weights", "scales", "master_zero_points")
        for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
            key = jax.tree_util.keystr(path)
            if any(name in key for name in trainable) and isinstance(leaf, jax.Array):
                params[key] = leaf
        return params

    def inject_params(model, params):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: params.get(jax.tree_util.keystr(path), leaf),
            model,
        )

    params = collect_params(student)
    print(f"trainable arrays: {len(params)}")
    total_elements = sum(int(np.prod(value.shape)) for value in params.values())
    print(f"trainable elements: {total_elements / 1e6:.1f}M")

    def is_scale_key(key: str) -> bool:
        return key.endswith("scales") or key.endswith("master_zero_points")

    if args.optimizer == "adamw":
        labels = {
            key: ("scale" if is_scale_key(key) else "weight")
            for key in params
        }
        optimizer = optax.multi_transform(
            {
                "weight": optax.adamw(args.lr, b1=args.adam_b1, b2=args.adam_b2, weight_decay=0.0),
                "scale": optax.adamw(args.scale_lr, b1=args.adam_b1, b2=args.adam_b2, weight_decay=0.0),
            },
            labels,
        )
        print(f"optimizer: AdamW weight_lr={args.lr} scale_lr={args.scale_lr}", flush=True)
    else:
        labels = {
            key: (
                "muon"
                if key.endswith("master_weights") and value.ndim >= 2
                else "scale"
                if is_scale_key(key)
                else "adam"
            )
            for key, value in params.items()
        }
        optimizer = optax.multi_transform(
            {
                "muon": optax.contrib.muon(args.muon_lr),
                # weight_decay=0.0 is required: optax.adamw defaults to 1e-4 and would
                # shrink scales/zero-points towards zero every step (scale collapse).
                "adam": optax.adamw(args.lr, b1=args.adam_b1, b2=args.adam_b2, weight_decay=0.0),
                "scale": optax.adamw(args.scale_lr, b1=args.adam_b1, b2=args.adam_b2, weight_decay=0.0),
            },
            labels,
        )
        print(f"optimizer: Muon(master) + AdamW(scale_lr={args.scale_lr})", flush=True)

    if args.grad_clip > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip), optimizer)
        print(f"grad clip: global norm {args.grad_clip}", flush=True)

    scale_min = float(args.scale_min)

    def clamp_scales(model_params):
        if scale_min <= 0.0:
            return model_params
        return {
            key: (jnp.maximum(value, scale_min) if key.endswith("scales") else value)
            for key, value in model_params.items()
        }

    params = clamp_scales(params)
    opt_state = optimizer.init(params)
    print(
        f"trainable: {'master_weights only' if args.freeze_scales else 'master_weights + scales + zero_points'}"
        f"  warmup_steps={args.warmup_steps}",
        flush=True,
    )

    def loss_fn(current_params, ids):
        model = inject_params(student, current_params)
        teacher_logits = jax.lax.stop_gradient(logits_for(teacher, ids, inference_config))
        student_logits = logits_for(model, ids, training_config)
        temperature = args.temperature
        teacher_log_probs = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)
        student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
        teacher_probs = jnp.exp(teacher_log_probs)
        kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1).mean()
        return kl * (temperature**2)

    @jax.jit
    def train_step(current_params, opt_state, ids, lr_factor):
        loss, grads = jax.value_and_grad(loss_fn)(current_params, ids)
        if args.warmup_steps > 0:
            grads = jax.tree_util.tree_map(lambda grad: grad * lr_factor, grads)
        grad_norm = optax.global_norm(grads)
        updates, new_state = optimizer.update(grads, opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        new_params = clamp_scales(new_params)
        return new_params, new_state, loss, grad_norm

    def validation_loss(current_params):
        if not val_ids:
            return None
        values = [float(loss_fn(current_params, ids)) for ids in val_ids]
        return sum(values) / len(values)

    if args.dry_run:
        print("dry-run: no training")
        return

    import orbax.checkpoint as ocp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or (args.output_dir / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_meta = {
        "teacher_dir": str(args.teacher_dir),
        "student_dir": str(args.student_dir),
        "eval_corpora": [str(path) for path in args.eval_corpus],
        "overlap_checked": bool(args.eval_corpus),
        "seq": args.seq,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "scale_lr": args.scale_lr,
        "muon_lr": args.muon_lr,
        "student_provenance_sha256": (
            hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            if provenance_path.exists()
            else None
        ),
        "adam_b1": args.adam_b1,
        "adam_b2": args.adam_b2,
        "scale_min": args.scale_min,
        "freeze_scales": bool(args.freeze_scales),
        "warmup_steps": args.warmup_steps,
    }

    def save_checkpoint(step: int, current_params, current_state, current_rng) -> None:
        path = checkpoint_dir / f"step_{step:05d}"
        tmp_path = checkpoint_dir / f".{path.name}.tmp"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        checkpointer.save(
            tmp_path,
            {"params": current_params, "opt_state": current_state, "step": step},
            force=True,
        )
        with open(tmp_path / "rng.pkl", "wb") as file:
            pickle.dump(current_rng.bit_generator.state, file)
        with open(tmp_path / "meta.json", "w", encoding="utf-8") as file:
            json.dump(checkpoint_meta, file, indent=2, sort_keys=True)
        (tmp_path / "COMPLETE").write_text("ok\n", encoding="utf-8")
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp_path, path)
        keep = max(args.keep_checkpoints, 1)
        for old in sorted(checkpoint_dir.glob("step_*"))[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
        print(f"checkpoint -> {path}", flush=True)

    rng = np.random.default_rng(args.seed)
    start_step = 0
    if args.resume:
        if args.resume == "auto":
            candidates = sorted(
                path for path in checkpoint_dir.glob("step_*") if (path / "COMPLETE").exists()
            )
            resume_path = candidates[-1] if candidates else None
        else:
            resume_path = Path(args.resume)
        if resume_path is None:
            print("resume: no complete checkpoint found; starting from 0", flush=True)
        elif not resume_path.exists():
            raise SystemExit(f"checkpoint not found: {resume_path}")
        else:
            meta_file = resume_path / "meta.json"
            if not meta_file.exists():
                raise RuntimeError(f"checkpoint missing meta.json: {resume_path}")
            saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            for key, expected in checkpoint_meta.items():
                if saved_meta.get(key) != expected:
                    raise RuntimeError(
                        f"checkpoint meta mismatch for {key}: saved={saved_meta.get(key)!r} current={expected!r}"
                    )
            restored = checkpointer.restore(resume_path)
            params = restored["params"]
            opt_state = restored["opt_state"]
            start_step = int(restored["step"])
            rng_file = resume_path / "rng.pkl"
            if not rng_file.exists():
                raise RuntimeError(f"checkpoint missing rng.pkl: {resume_path}")
            with open(rng_file, "rb") as file:
                rng.bit_generator.state = pickle.load(file)
            print(f"resumed from {resume_path} at step {start_step}", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    progress_path = args.output_dir / "progress.jsonl"
    if not args.resume:
        progress_path.write_text("", encoding="utf-8")

    def progress(record: dict) -> None:
        record = {"run_id": run_id, **record}
        with open(progress_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def scale_stats(current_params):
        values = [
            np.asarray(value, dtype=np.float32).ravel()
            for key, value in current_params.items()
            if key.endswith("scales")
        ]
        if not values:
            return 0.0, 0.0
        merged = np.concatenate(values)
        return float(merged.min()), float(merged.max())

    initial_scale_min, initial_scale_max = scale_stats(params)
    started_at = time.time()
    initial_val = validation_loss(params)
    best_val = initial_val
    bad_val_evals = 0
    if initial_val is not None:
        progress({"step": 0, "val_loss": initial_val, "elapsed_s": 0.0})
        print(f"step 0/{args.steps}  val_kl={initial_val:.6f}", flush=True)

    for step in range(start_step, args.steps):
        ids = chunks[int(rng.integers(0, len(chunks)))]
        step_started = time.time()
        lr_factor = 1.0
        if args.warmup_steps > 0:
            lr_factor = min(1.0, (step + 1) / args.warmup_steps)
        params, opt_state, loss, grad_norm = train_step(params, opt_state, ids, lr_factor)
        jax.block_until_ready(params)
        now = time.time()
        loss_value = float(loss)
        grad_value = float(grad_norm)
        scale_lo, scale_hi = scale_stats(params)
        bad = not (
            np.isfinite(loss_value)
            and np.isfinite(grad_value)
            and np.isfinite(scale_lo)
            and np.isfinite(scale_hi)
        )
        collapsed = initial_scale_max > scale_min * 10 and scale_hi <= scale_min * 10
        if bad or collapsed:
            reason = "non-finite loss/grad/scale" if bad else "scale collapse"
            progress(
                {
                    "step": step + 1,
                    "status": "failed",
                    "reason": reason,
                    "loss": loss_value,
                    "grad_norm": grad_value,
                    "scale_min": scale_lo,
                    "scale_max": scale_hi,
                    "elapsed_s": now - started_at,
                }
            )
            raise RuntimeError(f"QAD stopped at step {step + 1}: {reason}")
        val_loss = None
        if val_ids and ((step + 1) % args.val_every == 0 or step + 1 == args.steps):
            val_loss = validation_loss(params)
            if val_loss is not None:
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    bad_val_evals = 0
                else:
                    bad_val_evals += 1
                    if args.early_stop_patience > 0 and bad_val_evals >= args.early_stop_patience:
                        progress(
                            {
                                "step": step + 1,
                                "status": "early_stop",
                                "val_loss": val_loss,
                                "best_val_loss": best_val,
                                "elapsed_s": now - started_at,
                            }
                        )
                        raise RuntimeError(
                            f"QAD early stop at step {step + 1}: val loss did not improve "
                            f"(best={best_val:.6f}, current={val_loss:.6f})"
                        )
        record = {
            "run_id": run_id,
            "step": step + 1,
            "loss": loss_value,
            "val_loss": val_loss,
            "grad_norm": grad_value,
            "scale_min": scale_lo,
            "scale_max": scale_hi,
            "step_s": now - step_started,
            "elapsed_s": now - started_at,
        }
        with open(progress_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        val_text = "" if val_loss is None else f"  val_kl={val_loss:.6f}"
        print(
            f"step {step + 1}/{args.steps}  kl={loss_value:.6f}{val_text}  "
            f"grad={grad_value:.3g}  scale=[{scale_lo:.3g},{scale_hi:.3g}]  "
            f"step_s={now - step_started:.2f}  elapsed_s={now - started_at:.1f}",
            flush=True,
        )
        if args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(step + 1, params, opt_state, rng)

    if args.steps > 0 and (not args.save_every or args.steps % args.save_every != 0):
        save_checkpoint(args.steps, params, opt_state, rng)

    trained = inject_params(student, params)
    trained = map_nodes_of_type_with_path(
        HybridMatrix,
        lambda path, matrix: matrix.switch_implementation(CompressionImplementation.INFERENCE),
        trained,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trained.save(args.output_dir)
    rewrite(args.output_dir / "model.safetensors")
    normalize(args.output_dir)
    print(f"saved QAD model -> {args.output_dir}")


if __name__ == "__main__":
    main()
