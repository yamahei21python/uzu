#!/usr/bin/env python3
"""Sketch B YAQA calibration for lalamo language models.

Universal part of the pipeline (Layer 2 in YAQA_CALIBRATION_DESIGN.md):

  forward  -> pseudo target sampling from the model's own logits
  backward -> per-sample weight gradient G_b for every selected Linear
  H_I = mean_b G_b^T G_b
  H_O = mean_b G_b G_b^T
  damping + PSD checks -> preconditioners.npz

Per-model work is limited to selecting LinearSite entries.

Usage:
  JAX_PLATFORM_NAME=cpu python scripts/yaqa_calibration.py \
    --model-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
    --corpus experiments/qwen35-0.8b/data/calibration.txt \
    --output experiments/qwen35-0.8b/yaqa-hin/calibration.npz \
    --samples 32 --max-tokens 128 --output-block none
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


@dataclass(frozen=True)
class LinearSite:
    path: tuple[str, ...]
    linear: object
    out_dim: int
    in_dim: int
    components: tuple[int, ...]

    @property
    def key(self) -> str:
        return "/".join(self.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=None, help="saved lalamo model directory")
    parser.add_argument("--spec-name", default="Qwen3.5-0.8B", help="built-in lalamo spec for direct HF import")
    parser.add_argument("--hf-repo", default=None, help="load directly from this Hugging Face repo (no save)")
    parser.add_argument("--hf-dir", default=None, help="load directly from a local HF snapshot directory")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lalamo-path", default=os.environ.get("LALAMO_PATH", "/tmp/lalamo-direct"))
    parser.add_argument(
        "--parallel",
        choices=("replicated", "tensor", "fsdp", "data"),
        default="replicated",
        help="JAX device sharding: tensor/fsdp use all visible GPUs as one memory pool",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=32, help="drop shorter trailing chunks")
    parser.add_argument("--layer-filter", default="", help="only paths containing this substring")
    parser.add_argument("--output-block", choices=("none", "small", "all"), default="none")
    parser.add_argument("--output-block-limit", type=int, default=2048)
    parser.add_argument("--damping", type=float, default=1e-2, help="initial sigma_reg after mean-diagonal normalization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calib-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="model dtype during calibration; bfloat16 halves memory for 4B/9B",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.lalamo_path)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from lalamo.model import Model
    from lalamo.module import Keychain
    from lalamo.modules import DecoderForwardPassConfig, ForwardPassMode
    from lalamo.modules.linear import Linear
    from lalamo.utils.sharding import ShardingConfig
    from lalamo.utils.surgery import select_nodes_of_type
    from tokenizers import Tokenizer

    if args.parallel == "tensor":
        sharding = ShardingConfig.tensor_parallel()
    elif args.parallel == "fsdp":
        sharding = ShardingConfig.fully_sharded_data_parallel()
    elif args.parallel == "data":
        sharding = ShardingConfig.data_parallel()
    else:
        sharding = ShardingConfig.replicated()
    keychain = Keychain.init(seed=args.seed, sharding_config=sharding)
    forward_pass_config = DecoderForwardPassConfig.for_inference(mode=ForwardPassMode.MULTI_TOKEN)

    dtype = jnp.dtype(args.calib_dtype)
    if args.hf_repo or args.hf_dir:
        from dataclasses import replace as dataclass_replace
        from pathlib import Path as _Path

        from lalamo.model_import.common import import_model
        from lalamo.model_import.model_specs.qwen import QWEN_MODELS

        spec = next((item for item in QWEN_MODELS if item.name == args.spec_name), None)
        if spec is None:
            raise SystemExit(f"lalamo spec {args.spec_name!r} not found")
        if args.hf_dir:
            from lalamo.model_import.origins import LocalOrigin

            weight_files = tuple(sorted(path.name for path in _Path(args.hf_dir).glob("*.safetensors")))
            if not weight_files:
                raise SystemExit(f"no safetensors in {args.hf_dir}")
            origin = LocalOrigin(root=args.hf_dir, weight_files=weight_files)
        else:
            from lalamo.model_import.origins import HuggingFaceOrigin

            origin = HuggingFaceOrigin(repo=args.hf_repo)
        print(f"importing {args.spec_name} from {origin.description} ({args.calib_dtype})...")
        results = import_model(dataclass_replace(spec, origin=origin), sharding_config=sharding, dtype=dtype)
        model = results.model
        tokenizer = model.token_codec.tokenizer
    else:
        if args.model_dir is None:
            raise SystemExit("provide --model-dir, --hf-repo, or --hf-dir")
        print(f"loading model ({args.calib_dtype})...")
        model = Model.load(args.model_dir, sharding, dtype=dtype)
        tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    print("loaded")

    def module_at(root, attribute_path):
        node = root
        for part in attribute_path:
            if part.startswith("["):
                node = node[int(part[1:-1])]
            else:
                node = getattr(node, part.lstrip("."))
        return node

    layer_filters = [item for item in args.layer_filter.split(",") if item]
    sites: list[LinearSite] = []
    for path, linear in select_nodes_of_type(Linear, model):
        joined = "/".join(path)
        if layer_filters and not any(item in joined for item in layer_filters):
            continue
        shape = tuple(int(x) for x in linear.weights.shape)
        if len(shape) < 2:
            continue
        components = shape[:-2]
        out_dim, in_dim = shape[-2], shape[-1]
        sites.append(LinearSite(path=path, linear=linear, out_dim=out_dim, in_dim=in_dim, components=components))
    if not sites:
        raise SystemExit("no LinearSite selected")
    print(f"sites: {len(sites)}")

    def logits_for(m, ids):
        token_ids = jnp.asarray([ids], dtype=jnp.int32)
        positions = jnp.arange(len(ids), dtype=jnp.int32)[None, :]
        result = m.decoder(
            token_ids=token_ids,
            token_positions=positions,
            state=None,
            return_updated_state=False,
            lengths_without_padding=None,
            forward_pass_config=forward_pass_config,
            keychain=keychain.split(1)[0],
        )
        return result.logits[0]

    def cross_entropy(logits, targets):
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        # Official get_hess_llama.py uses F.cross_entropy(..., reduction="mean")
        # over all tokens of the batch. Per-sample mean keeps the same scale
        # independently of sequence length.
        return -jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=-1))

    base_arrays = tuple(site.linear.weights.weights for site in sites)

    def model_with_weights(arrays):
        updated = model
        for site, array in zip(sites, arrays):
            current = module_at(updated, site.path)
            new_weights = eqx.tree_at(lambda wm: wm.weights, current.weights, array)
            new_linear = eqx.tree_at(lambda linear: linear.weights, current, new_weights)
            updated = eqx.tree_at(lambda root, path=site.path: module_at(root, path), updated, new_linear)
        return updated

    current_ids: list[int] = []
    current_targets = None

    def per_sample_loss(arrays):
        m = model_with_weights(arrays)
        logits = logits_for(m, current_ids)
        return cross_entropy(logits, current_targets)

    texts = [line.strip() for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        raise SystemExit("empty calibration corpus")

    def build_chunks() -> list[list[int]]:
        # Concatenate documents and cut into full-context windows, matching the
        # official calibration flow (continuous sequences instead of single lines).
        joined = "\n\n".join(texts)
        ids = tokenizer.encode(joined, add_special_tokens=True).ids
        chunks = [ids[start : start + args.max_tokens] for start in range(0, len(ids), args.max_tokens)]
        chunks = [chunk for chunk in chunks if len(chunk) >= args.min_tokens]
        if not chunks:
            raise SystemExit("no usable calibration chunks")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(chunks)
        return chunks

    chunks = build_chunks()
    if len(chunks) < args.samples:
        print(f"warning: only {len(chunks)} calibration chunks available, using all")
        args.samples = len(chunks)

    accumulators = []
    for site in sites:
        components = site.components or ()
        input_shape = (*components, site.in_dim, site.in_dim)
        accumulators.append(
            {
                "input": np.zeros(input_shape, dtype=np.float32),
                "output": np.zeros((*components, site.out_dim, site.out_dim), dtype=np.float32)
                if args.output_block == "all"
                or (args.output_block == "small" and site.out_dim <= args.output_block_limit)
                else None,
                "samples": 0,
            }
        )

    grad_fn = jax.grad(per_sample_loss)
    rng = np.random.default_rng(args.seed)
    sample_index = 0
    processed = 0
    while processed < args.samples:
        ids = chunks[sample_index]
        sample_index += 1
        current_ids = ids
        logits = logits_for(model, ids)
        current_targets = jax.random.categorical(
            jax.random.key(args.seed + processed), logits, axis=-1
        )
        grads = grad_fn(base_arrays)
        for site, accumulator, gradient in zip(sites, accumulators, grads):
            G = np.asarray(gradient, dtype=np.float64)
            if G.ndim == 2:
                accumulator["input"] += G.T @ G
                if accumulator["output"] is not None:
                    accumulator["output"] += G @ G.T
            else:
                # MoE-style leading component axes: one Hessian per component.
                for component in np.ndindex(*site.components):
                    g = G[component]
                    accumulator["input"][component] += g.T @ g
                    if accumulator["output"] is not None:
                        accumulator["output"][component] += g @ g.T
            accumulator["samples"] += 1
        processed += 1
        if processed % 4 == 0:
            print(f"  sample {processed}/{args.samples}")

    records = []
    arrays: dict[str, np.ndarray] = {}

    def regularize(matrix: np.ndarray, name: str):
        # Official pipeline: H /= mean(diag), then add sigma_reg (1e-2) until
        # Cholesky succeeds. Fall back to eigenvalue clipping if needed.
        scale = float(np.mean(np.diagonal(matrix, axis1=-2, axis2=-1)))
        normalized = matrix / max(scale, 1e-30)
        sigma = args.damping
        for _ in range(8):
            candidate = normalized + np.eye(normalized.shape[-1]) * sigma
            try:
                np.linalg.cholesky(candidate)
                return candidate.astype(np.float32), sigma
            except np.linalg.LinAlgError:
                sigma *= 10.0
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        eigenvalues = np.clip(eigenvalues, 1e-6, None)
        rebuilt = (eigenvectors * eigenvalues[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
        print(f"warning: {name} needed eigenvalue clipping")
        return rebuilt.astype(np.float32), None

    for site, accumulator in zip(sites, accumulators):
        count = max(accumulator["samples"], 1)
        H_I = accumulator["input"] / count
        H_O = None if accumulator["output"] is None else accumulator["output"] / count
        H_I, damping_input = regularize(H_I, f"{site.key}/input_block")
        record = {
            "path": site.key,
            "in_dim": site.in_dim,
            "out_dim": site.out_dim,
            "components": list(site.components),
            "samples": accumulator["samples"],
            "input_trace": float(np.trace(H_I, axis1=-2, axis2=-1).sum()),
            "input_min_eig": None,
            "input_max_eig": None,
            "input_condition": None,
            "input_damping": damping_input,
            "output_block": H_O is not None,
        }
        try:
            eigenvalues = np.linalg.eigvalsh(H_I)
            record["input_min_eig"] = float(eigenvalues.min())
            record["input_max_eig"] = float(eigenvalues.max())
            record["input_condition"] = float(eigenvalues.max() / max(eigenvalues.min(), 1e-30))
        except np.linalg.LinAlgError:
            record["input_condition"] = float("inf")
        arrays[f"{site.key}/input_block"] = H_I
        if H_O is not None:
            H_O, damping_output = regularize(H_O, f"{site.key}/output_block")
            arrays[f"{site.key}/output_block"] = H_O
            record["output_damping"] = damping_output
            try:
                eigenvalues = np.linalg.eigvalsh(H_O)
                record["output_min_eig"] = float(eigenvalues.min())
                record["output_max_eig"] = float(eigenvalues.max())
                record["output_condition"] = float(eigenvalues.max() / max(eigenvalues.min(), 1e-30))
            except np.linalg.LinAlgError:
                record["output_condition"] = float("inf")
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(f"saved {args.output} ({len(arrays)} arrays)")

    summary = {
        "model_dir": str(args.model_dir),
        "corpus": str(args.corpus),
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "output_block": args.output_block,
        "damping": args.damping,
        "layers": records,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved {args.json_out}")
    for record in records:
        print(
            f"  {record['path']} in={record['in_dim']} out={record['out_dim']} "
            f"cond={record['input_condition']:.3e} out_block={record['output_block']}"
        )


if __name__ == "__main__":
    main()
