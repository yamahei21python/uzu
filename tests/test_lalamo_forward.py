from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lalamo_forward  # noqa: E402


class FakeAttentionImplementation(Enum):
    PALLAS = "pallas"
    XLA = "xla"


@dataclass
class FakeMixer:
    attention_implementation: FakeAttentionImplementation = FakeAttentionImplementation.PALLAS


@dataclass
class FakeTransformer:
    mixer_forward_pass_config: FakeMixer = field(default_factory=FakeMixer)


@dataclass
class FakeDecoder:
    transformer_forward_pass_config: FakeTransformer = field(default_factory=FakeTransformer)


@pytest.fixture
def fake_lalamo(monkeypatch):
    modules = types.ModuleType("lalamo.modules")
    modules.AttentionImplementation = FakeAttentionImplementation
    lalamo = types.ModuleType("lalamo")
    lalamo.modules = modules
    monkeypatch.setitem(sys.modules, "lalamo", lalamo)
    monkeypatch.setitem(sys.modules, "lalamo.modules", modules)
    return modules


def mixer_impl(config: FakeDecoder) -> FakeAttentionImplementation:
    return config.transformer_forward_pass_config.mixer_forward_pass_config.attention_implementation


def test_unset_keeps_the_default(monkeypatch, fake_lalamo) -> None:
    monkeypatch.delenv("UZU_ATTENTION_IMPL", raising=False)
    config = FakeDecoder()
    assert lalamo_forward.apply_attention_env(config) is config
    assert mixer_impl(config) is FakeAttentionImplementation.PALLAS


def test_xla_overrides_the_mixer(monkeypatch, fake_lalamo) -> None:
    monkeypatch.setenv("UZU_ATTENTION_IMPL", "xla")
    config = lalamo_forward.apply_attention_env(FakeDecoder())
    assert mixer_impl(config) is FakeAttentionImplementation.XLA


def test_pallas_is_restorable(monkeypatch, fake_lalamo) -> None:
    monkeypatch.setenv("UZU_ATTENTION_IMPL", "pallas")
    config = FakeDecoder()
    config.transformer_forward_pass_config.mixer_forward_pass_config.attention_implementation = (
        FakeAttentionImplementation.XLA
    )
    assert mixer_impl(lalamo_forward.apply_attention_env(config)) is FakeAttentionImplementation.PALLAS


def test_unknown_value_is_ignored(monkeypatch, fake_lalamo) -> None:
    monkeypatch.setenv("UZU_ATTENTION_IMPL", "cuda")
    config = FakeDecoder()
    assert lalamo_forward.apply_attention_env(config) is config


def test_describe_reports_the_mode(monkeypatch) -> None:
    monkeypatch.setenv("UZU_ATTENTION_IMPL", "xla")
    assert lalamo_forward.describe() == "attention=xla"


def test_turing_blocks_bf16() -> None:
    reason = lalamo_forward.bf16_block_reason("bfloat16", [7.5, 7.5])
    assert reason is not None
    assert "float32" in reason


def test_ampere_allows_bf16() -> None:
    assert lalamo_forward.bf16_block_reason("bfloat16", [8.0, 8.0]) is None
    assert lalamo_forward.bf16_block_reason("bfloat16", [8.9]) is None


def test_float32_is_never_blocked() -> None:
    assert lalamo_forward.bf16_block_reason("float32", [7.5]) is None


def test_no_gpu_means_no_block() -> None:
    # CPU has no nvidia-smi; bf16 there is a different (working) path
    assert lalamo_forward.bf16_block_reason("bfloat16", []) is None


def test_gpu_capabilities_parses_nvidia_smi(monkeypatch) -> None:
    class Result:
        stdout = "7.5\n7.5\n"
        returncode = 0

    monkeypatch.setattr(lalamo_forward.subprocess, "run", lambda *a, **k: Result())
    assert lalamo_forward.gpu_capabilities() == [7.5, 7.5]


def test_gpu_capabilities_tolerates_missing_nvidia_smi(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(lalamo_forward.subprocess, "run", boom)
    assert lalamo_forward.gpu_capabilities() == []


def test_pre_ampere_auto_selects_xla(monkeypatch) -> None:
    monkeypatch.delenv("UZU_ATTENTION_IMPL", raising=False)
    monkeypatch.setattr(lalamo_forward, "gpu_capabilities", lambda: [7.5, 7.5])
    assert lalamo_forward.requested() == "xla"


def test_ampere_keeps_default(monkeypatch) -> None:
    monkeypatch.delenv("UZU_ATTENTION_IMPL", raising=False)
    monkeypatch.setattr(lalamo_forward, "gpu_capabilities", lambda: [8.0, 8.0])
    assert lalamo_forward.requested() is None


def test_explicit_pallas_wins_over_auto(monkeypatch) -> None:
    monkeypatch.setenv("UZU_ATTENTION_IMPL", "pallas")
    monkeypatch.setattr(lalamo_forward, "gpu_capabilities", lambda: [7.5])
    assert lalamo_forward.requested() == "pallas"


@dataclass
class FakeEmbeddingConfig:
    activation_dtype: object = "placeholder"


@dataclass
class FakeFullConfig:
    embedding_forward_pass_config: FakeEmbeddingConfig = field(default_factory=FakeEmbeddingConfig)


def test_force_f32_activations_on_pre_ampere(monkeypatch) -> None:
    import jax.numpy as jnp

    monkeypatch.setattr(lalamo_forward, "gpu_capabilities", lambda: [7.5])
    config = FakeFullConfig(FakeEmbeddingConfig(jnp.bfloat16))
    patched = lalamo_forward.force_float32_activations(config)
    assert patched.embedding_forward_pass_config.activation_dtype == jnp.float32


def test_force_f32_activations_keeps_ampere_dtype(monkeypatch) -> None:
    import jax.numpy as jnp

    monkeypatch.setattr(lalamo_forward, "gpu_capabilities", lambda: [8.0])
    config = FakeFullConfig(FakeEmbeddingConfig(jnp.bfloat16))
    assert lalamo_forward.force_float32_activations(config) is config


def test_force_f32_activations_without_gpu_is_a_noop(monkeypatch) -> None:
    monkeypatch.setattr(lalamo_forward, "gpu_capabilities", lambda: [])
    config = FakeFullConfig(FakeEmbeddingConfig("placeholder"))
    assert lalamo_forward.force_float32_activations(config) is config


def test_enter_mesh_activates_the_mesh(monkeypatch) -> None:
    import types

    calls: list[object] = []

    class Manager:
        def __enter__(self) -> "Manager":
            calls.append("enter")
            return self

    class Mesh:
        empty = False

    class Sharding:
        mesh = Mesh()

    fake_jax = types.SimpleNamespace(set_mesh=lambda mesh: (calls.append(mesh), Manager())[1])
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    lalamo_forward.enter_mesh(Sharding())
    assert isinstance(calls[0], Mesh)
    assert "enter" in calls


def test_enter_mesh_without_mesh_is_a_noop() -> None:
    class Sharding:
        mesh = None

    assert lalamo_forward.enter_mesh(Sharding()) is None


def test_enter_mesh_skips_an_empty_mesh() -> None:
    class Mesh:
        empty = True

    class Sharding:
        mesh = Mesh()

    assert lalamo_forward.enter_mesh(Sharding()) is None
