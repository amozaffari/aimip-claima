"""Shell tab-completion script generators (`aimip completion bash|zsh`)."""

from __future__ import annotations

from .errors import ValidationError
from .setup_data import CANONICAL_MODELS

TOP_LEVEL_COMMANDS: tuple[str, ...] = (
    "models",
    "show",
    "cite",
    "plan",
    "run",
    "submit",
    "status",
    "watch",
    "logs",
    "ls",
    "cancel",
    "summary",
    "doctor",
    "approve",
    "ic",
    "collect",
    "lora",
    "setup",
    "fetch",
    "site",
    "chat",
    "mcp",
    "ui",
    "completion",
)

PLAN_FLAGS: tuple[str, ...] = (
    "-m",
    "--model",
    "--start",
    "--steps",
    "--length",
    "--end",
    "--members",
    "--seed",
    "--ic",
    "--ocean-ic",
    "--lora",
    "--forcing-set",
    "--name",
    "--out",
    "--config",
    "--monthly",
    "--no-monthly",
    "--prediction",
    "--no-prediction",
    "--local",
    "--registry",
    "--playground",
    "--data",
    "--partition",
    "--qos",
    "--account",
    "--gres",
    "--cpus-per-task",
    "--mem",
    "--time",
    "--job-name",
    "--site-config",
    "--json",
)

COMMON_RUN_FLAGS: tuple[str, ...] = (
    *PLAN_FLAGS,
    "--no-fetch",
    "--dry-run",
    "--submit",
    "--approve",
    "--print-sbatch-path",
)


def render_completion(shell: str) -> str:
    """Render a standalone `bash` or `zsh` completion script for `aimip`."""
    norm = (shell or "").strip().lower()
    cmds = " ".join(TOP_LEVEL_COMMANDS)
    models = " ".join(CANONICAL_MODELS)
    plan_flags = " ".join(PLAN_FLAGS)
    run_flags = " ".join(COMMON_RUN_FLAGS)

    if norm == "bash":
        return f"""# bash completion for aimip
_aimip_completions() {{
    local cur prev cmd subcmd
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    cmd="${{COMP_WORDS[1]:-}}"
    subcmd="${{COMP_WORDS[2]:-}}"

    if [[ ${{COMP_CWORD}} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "{cmds} --version --help" -- "${{cur}}") )
        return 0
    fi

    case "${{prev}}" in
        -m|--model)
            COMPREPLY=( $(compgen -W "{models}" -- "${{cur}}") )
            return 0
            ;;
        --task)
            COMPREPLY=( $(compgen -W "amip slab-ocean coupled scenario" -- "${{cur}}") )
            return 0
            ;;
        --family)
            COMPREPLY=( $(compgen -W "fme e2s-dlesym arches emu" -- "${{cur}}") )
            return 0
            ;;
        --status)
            COMPREPLY=( $(compgen -W "certified runs experimental" -- "${{cur}}") )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "table json" -- "${{cur}}") )
            return 0
            ;;
    esac

    case "${{cmd}}" in
        show)
            if [[ ${{COMP_CWORD}} -eq 2 && "${{cur}}" != -* ]]; then
                COMPREPLY=( $(compgen -W "{models}" -- "${{cur}}") )
                return 0
            fi
            COMPREPLY=( $(compgen -W "--examples --json --registry --playground --data" -- "${{cur}}") )
            return 0
            ;;
        cite)
            if [[ ${{COMP_CWORD}} -eq 2 && "${{cur}}" != -* ]]; then
                COMPREPLY=( $(compgen -W "{models}" -- "${{cur}}") )
                return 0
            fi
            COMPREPLY=( $(compgen -W "--json --registry --playground --data" -- "${{cur}}") )
            return 0
            ;;
        models)
            COMPREPLY=( $(compgen -W "--task --family --status --installed --format --json --registry --playground --data" -- "${{cur}}") )
            return 0
            ;;
        plan)
            if [[ ${{COMP_CWORD}} -eq 2 && "${{cur}}" != -* ]]; then
                COMPREPLY=( $(compgen -W "{models}" -- "${{cur}}") )
                return 0
            fi
            COMPREPLY=( $(compgen -W "{plan_flags}" -- "${{cur}}") )
            return 0
            ;;
        run)
            if [[ ${{COMP_CWORD}} -eq 2 && "${{cur}}" != -* ]]; then
                COMPREPLY=( $(compgen -W "{models}" -- "${{cur}}") )
                return 0
            fi
            COMPREPLY=( $(compgen -W "{run_flags}" -- "${{cur}}") )
            return 0
            ;;
        submit)
            COMPREPLY=( $(compgen -W "--expect-digest --playground --site-config --json" -- "${{cur}}") )
            return 0
            ;;
        status)
            COMPREPLY=( $(compgen -W "--sacct-json --site-config --json" -- "${{cur}}") )
            return 0
            ;;
        ls|cancel|summary)
            COMPREPLY=( $(compgen -W "--site-config --json" -- "${{cur}}") )
            return 0
            ;;
        doctor)
            COMPREPLY=( $(compgen -W "--site-config --json --registry --playground --data" -- "${{cur}}") )
            return 0
            ;;
        approve)
            COMPREPLY=( $(compgen -W "--state-dir --json" -- "${{cur}}") )
            return 0
            ;;
        collect)
            COMPREPLY=( $(compgen -W "--out --ocean-out -m --model --to --site-config --json --registry --playground --data" -- "${{cur}}") )
            return 0
            ;;
        setup)
            COMPREPLY=( $(compgen -W "--models --dest --lock --playground --only-download --verify-only --write-lock --search-dir --force-rebuild --accept-derived --json" -- "${{cur}}") )
            return 0
            ;;
        fetch)
            COMPREPLY=( $(compgen -W "-m --model --dest --lock --playground --accept-derived --json" -- "${{cur}}") )
            return 0
            ;;
        mcp)
            COMPREPLY=( $(compgen -W "--state-dir --help" -- "${{cur}}") )
            return 0
            ;;
        chat)
            COMPREPLY=( $(compgen -W "--help" -- "${{cur}}") )
            return 0
            ;;
        ic)
            if [[ ${{COMP_CWORD}} -eq 2 ]]; then
                COMPREPLY=( $(compgen -W "list spin" -- "${{cur}}") )
                return 0
            fi
            ;;
        lora)
            if [[ ${{COMP_CWORD}} -eq 2 ]]; then
                COMPREPLY=( $(compgen -W "info scale" -- "${{cur}}") )
                return 0
            fi
            ;;
        site)
            if [[ ${{COMP_CWORD}} -eq 2 ]]; then
                COMPREPLY=( $(compgen -W "init show" -- "${{cur}}") )
                return 0
            fi
            ;;
        completion)
            COMPREPLY=( $(compgen -W "bash zsh --json" -- "${{cur}}") )
            return 0
            ;;
        logs)
            COMPREPLY=( $(compgen -W "-f --follow --max-bytes --interval --max-polls --json --site-config" -- "${{cur}}") )
            return 0
            ;;
        watch)
            COMPREPLY=( $(compgen -W "--interval --max-polls --timeout --json --site-config" -- "${{cur}}") )
            return 0
            ;;
        ui)
            COMPREPLY=( $(compgen -W "--host --port --allow-remote --help" -- "${{cur}}") )
            return 0
            ;;
    esac
}}
complete -F _aimip_completions aimip
"""

    if norm == "zsh":
        return f"""#compdef aimip
_aimip() {{
    local -a commands models plan_flags run_flags
    commands=({cmds})
    models=({models})
    plan_flags=({plan_flags})
    run_flags=({run_flags})

    if (( CURRENT == 2 )); then
        _describe 'command' commands
        return
    fi

    case "${{words[CURRENT-1]}}" in
        -m|--model)
            _describe 'model' models
            return
            ;;
        --task)
            compadd amip slab-ocean coupled scenario
            return
            ;;
        --family)
            compadd fme e2s-dlesym arches emu
            return
            ;;
        --status)
            compadd certified runs experimental
            return
            ;;
        --format)
            compadd table json
            return
            ;;
    esac

    case "${{words[2]}}" in
        show)
            if (( CURRENT == 3 )) && [[ "${{words[CURRENT]}}" != -* ]]; then
                _describe 'model' models
                return
            fi
            compadd -- --examples --json --registry --playground --data
            ;;
        cite)
            if (( CURRENT == 3 )) && [[ "${{words[CURRENT]}}" != -* ]]; then
                _describe 'model' models
                return
            fi
            compadd -- --json --registry --playground --data
            ;;
        models)
            compadd -- --task --family --status --installed --format --json --registry --playground --data
            ;;
        plan)
            if (( CURRENT == 3 )) && [[ "${{words[CURRENT]}}" != -* ]]; then
                _describe 'model' models
                return
            fi
            compadd -- ${{plan_flags[@]}}
            ;;
        run)
            if (( CURRENT == 3 )) && [[ "${{words[CURRENT]}}" != -* ]]; then
                _describe 'model' models
                return
            fi
            compadd -- ${{run_flags[@]}}
            ;;
        submit)
            compadd -- --expect-digest --playground --site-config --json
            ;;
        status)
            compadd -- --sacct-json --site-config --json
            ;;
        ls|cancel|summary)
            compadd -- --site-config --json
            ;;
        doctor)
            compadd -- --site-config --json --registry --playground --data
            ;;
        approve)
            compadd -- --state-dir --json
            ;;
        collect)
            compadd -- --out --ocean-out -m --model --to --site-config --json --registry --playground --data
            ;;
        setup)
            compadd -- --models --dest --lock --playground --only-download --verify-only --write-lock --search-dir --force-rebuild --accept-derived --json
            ;;
        fetch)
            compadd -- -m --model --dest --lock --playground --accept-derived --json
            ;;
        mcp)
            compadd -- --state-dir --help
            ;;
        chat)
            compadd -- --help
            ;;
        ic)
            (( CURRENT == 3 )) && compadd list spin
            ;;
        lora)
            (( CURRENT == 3 )) && compadd info scale
            ;;
        site)
            (( CURRENT == 3 )) && compadd init show
            ;;
        completion)
            (( CURRENT == 3 )) && compadd bash zsh
            ;;
        logs)
            compadd -- -f --follow --max-bytes --interval --max-polls --json --site-config
            ;;
        watch)
            compadd -- --interval --max-polls --timeout --json --site-config
            ;;
        ui)
            compadd -- --host --port --allow-remote --help
            ;;
    esac
}}
compdef _aimip aimip
"""

    raise ValidationError(
        f"Unsupported shell {shell!r} for completion.",
        field="shell",
        allowed=["bash", "zsh"],
        next_command="aimip completion bash",
    )
