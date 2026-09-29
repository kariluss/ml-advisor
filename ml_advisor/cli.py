"""Interface de linha de comando do ml_advisor."""

from __future__ import annotations

import argparse
import json
import sys

from ml_advisor.settings import (
    AVAILABLE_PROVIDERS,
    DEFAULT_PRESET,
    DEFAULT_PROVIDER,
    PLANNED_PROVIDERS,
    PRESETS,
    SettingsError,
    list_presets,
    load_settings,
)


def _cmd_config_show(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(
            provider=args.provider,
            config_path=args.config,
        )
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


def _cmd_config_presets(args: argparse.Namespace) -> int:
    try:
        presets = list_presets(args.provider)
    except SettingsError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    if args.provider is None:
        for provider, values in presets.items():
            marker = " (implementado)" if provider in AVAILABLE_PROVIDERS else " (planejado)"
            print(f"[{provider}]{marker}")
            for name, pvalues in values.items():
                default_marker = " *" if name == DEFAULT_PRESET else ""
                print(f"    {name}{default_marker}: {pvalues['model']}")
            print()
    else:
        for name, pvalues in presets.items():
            default_marker = " *" if name == DEFAULT_PRESET else ""
            print(f"  {name}{default_marker}: {pvalues['model']}")
    return 0


def _cmd_config_providers(_args: argparse.Namespace) -> int:
    print(f"Default: {DEFAULT_PROVIDER}\n")
    print("Implementados:")
    for provider in sorted(AVAILABLE_PROVIDERS):
        print(f"  ✓ {provider}")
    print("\nPlanejados (ainda não funcionam):")
    for provider in sorted(PLANNED_PROVIDERS):
        print(f"  · {provider}")
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
    show.add_argument("--provider", default=None, help="Provider a consultar.")
    show.add_argument(
        "--config",
        default=None,
        help="Caminho do ml_advisor.toml.",
    )
    show.set_defaults(func=_cmd_config_show)

    presets = config_sub.add_parser("presets", help="Lista presets.")
    presets.add_argument("--provider", default=None, help="Filtrar por provider.")
    presets.set_defaults(func=_cmd_config_presets)

    providers = config_sub.add_parser("providers", help="Lista providers disponíveis e planejados.")
    providers.set_defaults(func=_cmd_config_providers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
