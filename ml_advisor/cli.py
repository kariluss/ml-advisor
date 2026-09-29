"""Interface de linha de comando do ml_advisor."""

from __future__ import annotations

import argparse
import json
import sys

from ml_advisor.settings import (
    DEFAULT_PRESET,
    PRESETS,
    SettingsError,
    load_settings,
)


def _cmd_config_show(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(config_path=args.config)
    except SettingsError as exc:
        print(f"✗ Erro de configuração: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(settings.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("Configuração efetiva do ml-advisor:\n")
        for line in settings.describe().splitlines():
            print(f"  {line}")
    return 0


def _cmd_config_presets(_args: argparse.Namespace) -> int:
    print("Presets disponíveis:\n")
    for name, values in PRESETS.items():
        marker = "  (default)" if name == DEFAULT_PRESET else ""
        print(f"  [{name}]{marker}")
        for key, value in values.items():
            print(f"      {key}: {value}")
        print()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-advisor",
        description="Ferramentas de linha de comando do ML Advisor.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config", help="Gerenciar configuração.")
    config_sub = config.add_subparsers(dest="action", required=True)

    show = config_sub.add_parser("show", help="Mostra a configuração efetiva.")
    show.add_argument("--json", action="store_true", help="Saída em JSON.")
    show.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho do ml_advisor.toml.",
    )
    show.set_defaults(func=_cmd_config_show)

    presets = config_sub.add_parser("presets", help="Lista presets disponíveis.")
    presets.set_defaults(func=_cmd_config_presets)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
