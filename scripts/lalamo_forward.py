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

ENV_VAR = "UZU_ATTENTION_IMPL"


def requested() -> str | None:
    value = os.environ.get(ENV_VAR, "").strip().lower()
    return value if value in ("xla", "pallas") else None


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
