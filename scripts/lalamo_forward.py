"""Force the lalamo attention implementation when the GPU cannot run Pallas.

lalamo's inference config defaults to the Pallas-Triton flash attention kernel
(``MixerForwardPassConfig.for_inference``). Pallas-Triton needs a Triton backend
that T4 (compute capability 7.5) does not provide, so the first forward fails
during XLA compilation even with ``--xla_gpu_enable_triton_gemm=false``.

``UZU_ATTENTION_IMPL=xla`` switches the mixer config to the plain XLA path.
``pallas`` forces the default back. Anything else (including unset) keeps the
config as-is.
"""

from __future__ import annotations

import os
import subprocess

ENV_VAR = "UZU_ATTENTION_IMPL"
# BF16 tensor cores start at Ampere. On Turing the XLA dot lowers to
# ALG_DOT_BF16_BF16_F32, which cuBLAS does not implement, so a bf16 forward
# dies inside the attention kernel instead of at startup.
BF16_MIN_CAPABILITY = 8.0


def gpu_capabilities() -> list[float]:
    """Return the compute capability of every visible GPU (empty off-GPU)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    caps = []
    if result.returncode != 0:
        return caps
    for line in (result.stdout or "").splitlines():
        try:
            caps.append(float(line.strip()))
        except ValueError:
            continue
    return caps


def bf16_block_reason(dtype: str, capabilities: list[float]) -> str | None:
    """Return why this dtype cannot run here, or None when it can."""
    if dtype != "bfloat16" or not capabilities:
        return None
    old = [capability for capability in capabilities if capability < BF16_MIN_CAPABILITY]
    if not old:
        return None
    return (
        f"GPU compute capability {old[0]:.1f} has no BF16 matmul "
        "(ALG_DOT_BF16_BF16_F32 is unimplemented); use float32. "
        "UZU_ATTENTION_IMPL=xla does not help: it only avoids Pallas-Triton."
    )



def requested() -> str | None:
    """Explicit env wins; otherwise pick xla automatically on pre-Ampere.

    Pallas-Triton needs cc >= 8.0. Scripts that do not go through run_lane
    (Modal jobs, make_qad_dataset -> yaqa_calibration) used to fall back to
    Pallas and die at the first forward, so the device decides for them.
    """
    value = os.environ.get(ENV_VAR, "").strip().lower()
    if value in ("xla", "pallas"):
        return value
    capabilities = gpu_capabilities()
    if any(capability < BF16_MIN_CAPABILITY for capability in capabilities):
        return "xla"
    return None


def apply_attention_env(config):
    """Return config with the attention implementation overridden, if requested."""
    mode = requested()
    if mode is None:
        return config

    from dataclasses import replace

    from lalamo.modules import AttentionImplementation

    target = AttentionImplementation.XLA if mode == "xla" else AttentionImplementation.PALLAS
    transformer = config.transformer_forward_pass_config
    mixer = transformer.mixer_forward_pass_config
    if getattr(mixer, "attention_implementation", None) == target:
        return config
    return replace(
        config,
        transformer_forward_pass_config=replace(
            transformer,
            mixer_forward_pass_config=replace(mixer, attention_implementation=target),
        ),
    )


def describe() -> str:
    mode = requested()
    return f"attention={mode or 'default'}"
