"""Testes para sanitização de texto."""

from __future__ import annotations

import pytest

from ml_advisor.utils.text import sanitize_text


def test_basico():
    assert sanitize_text("  olá mundo  ", max_length=100) == "olá mundo"


def test_remove_null_bytes():
    assert sanitize_text("a\x00b", max_length=10) == "ab"


def test_colapsa_espacos():
    assert sanitize_text("a    b\t\tc", max_length=10) == "a b c"


def test_colapsa_newlines():
    result = sanitize_text("a\n\n\n\n\nb", max_length=10)
    assert result == "a\n\nb"


def test_preserva_paragrafo():
    result = sanitize_text("a\n\nb", max_length=10)
    assert result == "a\n\nb"


def test_vazio_levanta():
    with pytest.raises(ValueError, match="vazio"):
        sanitize_text("   \n\n   ", max_length=10)


def test_excede_tamanho_levanta():
    with pytest.raises(ValueError, match="máximo"):
        sanitize_text("a" * 11, max_length=10)


def test_nao_str_levanta():
    with pytest.raises(TypeError, match="str"):
        sanitize_text(123, max_length=10)  # type: ignore[arg-type]
