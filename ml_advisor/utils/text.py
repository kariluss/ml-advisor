"""Sanitização de texto livre fornecido pelo usuário."""

from __future__ import annotations
import re


def sanitize_text(
    text: str,
    *,
    max_length: int,
    field_name: str = "text",
) -> str:
    """Limpa e valida texto livre do usuário.

    - Remove null bytes (``\\x00``).
    - Colapsa múltiplos espaços/tabs em um só.
    - Colapsa 3+ quebras de linha em 2 (preserva parágrafos).
    - Remove espaços no início/fim.
    - Valida tamanho máximo.

    Raises:
        TypeError: Se ``text`` não for ``str``.
        ValueError: Se ficar vazio após limpeza ou exceder ``max_length``.
    """
    if not isinstance(text, str):
        raise TypeError(f"{field_name} deve ser str, recebido {type(text).__name__}")

    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError(f"{field_name} está vazio após limpeza")

    if len(cleaned) > max_length:
        raise ValueError(
            f"{field_name} tem {len(cleaned)} caracteres, " f"máximo permitido é {max_length}"
        )

    return cleaned
