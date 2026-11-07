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
