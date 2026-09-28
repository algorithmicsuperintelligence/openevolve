"""
Command-line interface for OpenEvolve
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from openevolve import OpenEvolve
from openevolve.config import Config, load_config

logger = logging.getLogger(__name__)

#: OrcaRouter management commands. They are handled before normal argument
#: parsing so the evolution CLI keeps its positional-argument shape.
ORCAROUTER_COMMANDS = ("connect", "status", "logout", "models")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")

    parser.add_argument("initial_program", help="Path to the initial program file")

    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    parser.add_argument(
        "--provider",
        help="LLM provider: openai (default), claude_code, copilot_cli, "
        "orcarouter (API key), orcarouter_oauth (browser sign-in)",
        default=None,
    )

    parser.add_argument(
        "--model",
        help="Model name for the selected provider. For the OrcaRouter providers use an id "
        "from `openevolve-run.py models` (vendor/model namespace preserved).",
        default=None,
    )

    parser.add_argument(
        "--orcarouter-oob",
        action="store_true",
        help="OrcaRouter sign-in: use the out-of-band (pasted code) flow instead of the "
        "loopback browser redirect. Required on hosts that cannot receive a redirect.",
    )

    return parser.parse_args()


def parse_orcarouter_command(argv: List[str]) -> Optional[List[str]]:
    """Return the arguments for an OrcaRouter command, or None.

    `openevolve-run.py connect orcarouter` is handled here so the two
    credential choices stay discoverable from the command line.
    """
    args = list(argv)
    if args and args[0] in ORCAROUTER_COMMANDS:
        return args
    if len(args) > 1 and args[0] == "orcarouter" and args[1] in ORCAROUTER_COMMANDS:
        return [args[1], *args[2:]]
    return None


def _catalog_rows(result, capability: str = "chat", modality: str = "text") -> List[str]:
    rows = []
    for option in result.options(capability, modality):
        ctx = option.get("context_length")
        bits = [option["id"]]
        if ctx:
            bits.append(f"ctx={ctx}")
        modalities = ",".join(option.get("input_modalities") or []) or "text"
        bits.append(f"in={modalities}")
        if option.get("reasoning_efforts"):
            bits.append("reasoning=" + "/".join(option["reasoning_efforts"]))
        if option.get("verified"):
            bits.append("verified-fallback")
        rows.append("  " + "  ".join(bits))
    return rows


def run_orcarouter_command(argv: List[str]) -> int:
    """Handle `connect`, `status`, `logout` and `models` for OrcaRouter."""
    parser = argparse.ArgumentParser(prog="openevolve-run.py", description="OrcaRouter setup")
    parser.add_argument("command", choices=ORCAROUTER_COMMANDS)
    parser.add_argument(
        "provider",
        nargs="?",
        default="orcarouter",
        choices=["orcarouter", "orcarouter_oauth"],
        help="orcarouter = paste an API key (default); orcarouter_oauth = browser sign-in",
    )
    parser.add_argument("--oob", action="store_true", help="Use the out-of-band code flow")
    parser.add_argument(
        "--capability",
        default="chat",
        choices=["chat", "embedding", "image", "video", "rerank"],
        help="Capability filter for `models`",
    )
    parser.add_argument(
        "--modality",
        default="text",
        choices=["text", "image", "audio", "video"],
        help="Required input modality for `models` (chat only)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--api-base", default=None, help="Override the inference base URL")
    command_args = parser.parse_args(argv)

    from openevolve.llm.orcarouter import (
        PROVIDER_API_KEY,
        PROVIDER_OAUTH,
        OrcaRouterLLM,
        logout,
        orcarouter_credential_status,
    )
    from openevolve.llm.orcarouter_auth import KEY_DASHBOARD_URL, OrcaAuthError, OrcaCredentialStore
    from openevolve.llm.orcarouter_catalog import OrcaCatalogClient

    import json

    store = OrcaCredentialStore()
    provider_id = PROVIDER_OAUTH if command_args.provider == PROVIDER_OAUTH else PROVIDER_API_KEY

    if command_args.command == "logout":
        removed = logout(store)
        print(
            "Removed the stored OrcaRouter credential."
            if removed
            else "No stored OrcaRouter credential to remove."
        )
        return 0

    if command_args.command == "status":
        status = orcarouter_credential_status(provider_id, store)
        if command_args.json:
            print(json.dumps(status, indent=2))
            return 0
        print(f"OrcaRouter credential status")
        print(f"  authenticated : {status['authenticated']}")
        print(f"  secret        : {status['secret_masked'] or '(none)'}")
        print(f"  auth source   : {status['auth_source'] or '(none)'}")
        print(f"  generation    : {status['generation']}")
        print(f"  granted scope : {status['scope'] or '(unknown)'}")
        print(f"  inference base: {status['api_base']}")
        if status["needs_reauth"]:
            print(f"  ACTION        : credential rejected — sign in again or paste a new key")
        print(f"  key dashboard : {KEY_DASHBOARD_URL}")
        return 0

    if command_args.command == "models":
        status = orcarouter_credential_status(provider_id, store)
        api_key = None
        if status["authenticated"]:
            stored = store.load()
            api_key = stored.api_key if stored else None
        catalog = OrcaCatalogClient(api_key=api_key)
        result = catalog.discover(capability=command_args.capability)
        payload = {
            **result.status(),
            "capability": command_args.capability,
            "modality": command_args.modality,
            "models": result.options(command_args.capability, command_args.modality),
        }
        if command_args.json:
            print(json.dumps(payload, indent=2))
            return 0
        print(
            f"OrcaRouter models ({result.source}"
            + (", degraded" if result.degraded else "")
            + f") for capability="
            f"{command_args.capability} modality={command_args.modality}"
        )
        rows = _catalog_rows(result, command_args.capability, command_args.modality)
        print("\n".join(rows) if rows else "  (no compatible models advertised)")
        if result.degraded:
            print(
                f"  note: live discovery failed ({result.error}); "
                "showing a verified fallback catalog"
            )
        return 0

    # connect
    try:
        if provider_id == PROVIDER_OAUTH:
            stored = store.load()
            if stored is not None and not stored.needs_reauth:
                print(
                    "An OrcaRouter credential is already stored "
                    f"({stored.masked}, source={stored.source}). Reusing it; "
                    "run `logout` first to force a new sign-in."
                )
            else:
                from openevolve.llm.orcarouter_auth import PkceCredentialProvider

                credential = PkceCredentialProvider(store=store, oob=command_args.oob).acquire()
                print(f"Signed in. Stored credential {credential.masked}.")
            return 0

        from openevolve.llm.orcarouter_auth import ApiKeyCredentialProvider

        key = os.environ.get("ORCAROUTER_API_KEY")
        if not key and sys.stdin.isatty():
            key = input(f"Paste your OrcaRouter API key (sk-orca-…): ").strip()
        credential = ApiKeyCredentialProvider(key, store=store).acquire()
        print(f"Stored OrcaRouter API key {credential.masked}.")
        print(f"Manage or revoke keys at {KEY_DASHBOARD_URL}")
        return 0
    except OrcaAuthError as exc:
        print(f"Error: {exc}")
        return 1


def _resolve_orcarouter_models(requested: Optional[str], api_base: Optional[str], provider_id: str):
    """Turn an OrcaRouter model request into configs from the live catalog.

    Returns ``(configs, error)``. Exactly one of the two is set.
    """
    from openevolve.llm.orcarouter_auth import (
        OrcaAuthError,
        OrcaCredentialStore,
        resolve_api_base,
    )
    from openevolve.llm.orcarouter_catalog import (
        CAPABILITY_CHAT,
        OrcaCatalogClient,
        filter_for_entry_point,
    )
    from openevolve.llm.orcarouter import (
        PROVIDER_OAUTH,
        effective_api_base,
        resolve_credential_for_discovery,
    )
    from openevolve.config import LLMModelConfig

    store = OrcaCredentialStore()
    stored = store.load()
    usable = resolve_credential_for_discovery(store)
    api_key = None
    if usable is not None:
        api_key = usable.api_key
    elif stored is None or stored.needs_reauth:
        api_key = os.environ.get("ORCAROUTER_API_KEY")
    base = effective_api_base(api_base) or resolve_api_base()

    if provider_id == PROVIDER_OAUTH and (stored is None or stored.needs_reauth):
        return None, (
            "No OrcaRouter account credential is stored. Run "
            "`openevolve-run.py connect orcarouter orcarouter_oauth` first, or set "
            "ORCAROUTER_API_KEY."
        )
    if api_key is None:
        return None, (
            "No OrcaRouter credential available. Run "
            "`openevolve-run.py connect orcarouter` or set ORCAROUTER_API_KEY."
        )

    catalog = OrcaCatalogClient(api_key=api_key, api_base=base)
    result = catalog.discover(capability=CAPABILITY_CHAT)
    compatible = filter_for_entry_point(result.models, CAPABILITY_CHAT, "text")

    if requested:
        ids = [m.id for m in compatible]
        if requested not in ids:
            return None, (
                f"model '{requested}' is not offered by this OrcaRouter account for "
                f"text chat. Run `openevolve-run.py models` to see the catalog."
            )
        selected = [m for m in compatible if m.id == requested]
    else:
        selected = compatible

    if not selected:
        return None, (
            "the OrcaRouter catalog advertised no text-chat models for this account"
            + (f" ({result.error})" if result.error else "")
        )

    configs = []
    for model in selected:
        cfg = LLMModelConfig(name=model.id, weight=1.0)
        cfg.provider = provider_id
        if model.reasoning_efforts:
            cfg.reasoning_effort = "medium"
        configs.append(cfg)
    return configs, None


async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    # Check if files exist
    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    # Load base config from file or defaults
    config = load_config(args.config)

    # OrcaRouter selects its models from the live catalog instead of a free-form
    # string, so resolve the provider before rebuilding the model list.
    if args.provider:
        config.llm.provider = args.provider
        print(f"Using provider: {config.llm.provider}")
    if args.orcarouter_oob:
        config.llm.orcarouter_oob = True

    orcarouter_models = None
    if args.provider in ("orcarouter", "orcarouter_oauth"):
        orcarouter_models, failure = _resolve_orcarouter_models(
            args.model, args.api_base, args.provider
        )
        if failure:
            print(f"Error: {failure}")
            return 1
        if args.api_base:
            config.llm.api_base = args.api_base

    # Create config object with command-line overrides
    if args.api_base or args.primary_model or args.secondary_model or orcarouter_models:
        # Apply command-line overrides
        if args.api_base:
            config.llm.api_base = args.api_base
            print(f"Using API base: {config.llm.api_base}")

        if args.primary_model:
            config.llm.primary_model = args.primary_model
            print(f"Using primary model: {config.llm.primary_model}")

        if args.secondary_model:
            config.llm.secondary_model = args.secondary_model
            print(f"Using secondary model: {config.llm.secondary_model}")

        # Rebuild models list to apply CLI overrides
        if args.primary_model or args.secondary_model:
            config.llm.rebuild_models()
            print(f"Applied CLI model overrides - active models:")
            for i, model in enumerate(config.llm.models):
                print(f"  Model {i+1}: {model.name} (weight: {model.weight})")

        if orcarouter_models is not None:
            config.llm.models = orcarouter_models
            config.llm.evaluator_models = []
            if args.config is None:
                # No config file: share the OrcaRouter provider settings the way
                # LLMConfig.__post_init__ would for a loaded config.
                config.llm.update_model_params(
                    {
                        "provider": args.provider,
                        "api_base": config.llm.api_base,
                        "temperature": config.llm.temperature,
                        "top_p": config.llm.top_p,
                        "max_tokens": config.llm.max_tokens,
                        "timeout": config.llm.timeout,
                        "retries": config.llm.retries,
                        "retry_delay": config.llm.retry_delay,
                    }
                )
                config.llm.evaluator_models = config.llm.models.copy()
            print(f"OrcaRouter models ({len(config.llm.models)}):")
            for model in config.llm.models:
                print(f"  {model.name}")

    # Initialize OpenEvolve
    try:
        openevolve = OpenEvolve(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config=config,
            output_dir=args.output,
        )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            print(f"Loading checkpoint from {args.checkpoint}")
            openevolve.database.load(args.checkpoint)
            print(
                f"Checkpoint loaded successfully (iteration {openevolve.database.last_iteration})"
            )

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = await openevolve.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(openevolve.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        print(f"\nEvolution complete!")
        print(f"Best program metrics:")
        for name, value in best_program.metrics.items():
            # Handle mixed types: format numbers as floats, others as strings
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    orcarouter_argv = parse_orcarouter_command(sys.argv[1:])
    if orcarouter_argv is not None:
        return run_orcarouter_command(orcarouter_argv)
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
