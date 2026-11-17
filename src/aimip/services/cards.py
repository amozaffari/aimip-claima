"""TTY and plain non-TTY formatters for plan cards, model cards, `doctor`, `ls`, `watch`, and `--config` files."""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

from ..errors import ValidationError
from ..validators import validate_safe_path
from .approval import format_bytes
from .types import ResolvedPlan, RunSpec, coerce_bool, coerce_optional_int

_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
_ANSI_ESC2_RE = re.compile(r"\x1b[@-Z\\-_]")

ALLOWED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "start",
        "steps",
        "length",
        "end",
        "members",
        "seed",
        "ic",
        "ocean_ic",
        "lora",
        "forcing_set",
        "name",
        "out",
        "monthly",
        "prediction",
        "mode",
        "resources",
        "partition",
        "qos",
        "account",
        "gres",
        "cpus_per_task",
        "mem",
        "time",
        "job_name",
    }
)

ALLOWED_RESOURCE_KEYS: frozenset[str] = frozenset(
    # "qos": MN5 places jobs by QOS (acc_debug, acc_ehpc), not by partition
    {"partition", "qos", "account", "gres", "cpus_per_task", "mem", "time", "job_name"}
)


def sanitize_terminal_text(raw: str) -> str:
    """Strip ANSI escape sequences and terminal control bytes (preserving `\\n` and `\\t`)."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _ANSI_OSC_RE.sub("", text)
    text = _ANSI_CSI_RE.sub("", text)
    text = _ANSI_ESC2_RE.sub("", text)
    cleaned: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch in ("\n", "\t") or (32 <= code < 127) or code >= 160:
            cleaned.append(ch)
    return "".join(cleaned)


def load_experiment_yaml(path: str) -> dict[str, Any]:
    """Load and validate an `experiment.yaml` file (`--config`)."""
    validated_path = validate_safe_path(
        os.path.abspath(path) if not path.startswith("/") else path,
        label="--config file",
    )
    if os.path.islink(validated_path) or not os.path.isfile(validated_path):
        raise ValidationError(
            f"Experiment config file not found or symlinked: {validated_path}",
            field="config",
        )
    try:
        with open(validated_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(
            f"Failed to parse experiment config {validated_path}: {exc}",
            field="config",
        ) from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError(
            f"Experiment config {validated_path} must be a YAML mapping.",
            field="config",
        )
    unknown = sorted(set(raw.keys()) - ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown key(s) in {validated_path}: {', '.join(unknown)}.",
            field=unknown[0],
            allowed=sorted(ALLOWED_CONFIG_KEYS),
        )
    res_block = raw.get("resources")
    if res_block is not None:
        if not isinstance(res_block, dict):
            raise ValidationError(
                f"'resources' in {validated_path} must be a mapping.",
                field="resources",
            )
        bad_res = sorted(set(res_block.keys()) - ALLOWED_RESOURCE_KEYS)
        if bad_res:
            raise ValidationError(
                f"Unknown resource key(s) in {validated_path}: {', '.join(bad_res)}.",
                field=f"resources.{bad_res[0]}",
                allowed=sorted(ALLOWED_RESOURCE_KEYS),
            )
    return dict(raw)


def merge_config_and_cli(
    config_path: str | None,
    cli_values: dict[str, Any],
) -> RunSpec:
    """Merge optional `--config experiment.yaml` with explicit CLI flags and record `cli_overrides`."""
    cfg = load_experiment_yaml(config_path) if config_path else {}
    merged_res: dict[str, Any] = dict(cfg.get("resources") or {})
    for rk in ALLOWED_RESOURCE_KEYS:
        if rk in cfg and cfg[rk] is not None:
            merged_res[rk] = cfg[rk]

    overrides: list[str] = []

    def _pick(key: str, default: Any = None) -> Any:
        cli_val = cli_values.get(key)
        has_cfg = key in cfg and cfg[key] is not None
        if cli_val is not None:
            if config_path and (has_cfg or key in cfg):
                if cfg.get(key) != cli_val:
                    overrides.append(f"{key}={cli_val!r}")
            elif config_path:
                overrides.append(f"{key}={cli_val!r}")
            return cli_val
        if has_cfg:
            return cfg[key]
        return default

    # Horizon override rule: an explicit CLI `--steps`, `--length`, or `--end` clears horizon keys from YAML.
    cli_horizons = [k for k in ("steps", "length", "end") if cli_values.get(k) is not None]
    if cli_horizons:
        for hk in ("steps", "length", "end"):
            if hk not in cli_horizons and hk in cfg and cfg[hk] is not None:
                cfg = {k: v for k, v in cfg.items() if k != hk}

    # Output override rule: an explicit CLI `--name` clears YAML `out` and vice-versa.
    if cli_values.get("name") is not None and "out" in cfg:
        cfg = {k: v for k, v in cfg.items() if k != "out"}
    elif cli_values.get("out") is not None and "name" in cfg:
        cfg = {k: v for k, v in cfg.items() if k != "name"}

    cli_res = cli_values.get("resources") or {}
    for rk, rv in cli_res.items():
        if rv is not None:
            if config_path and merged_res.get(rk) != rv:
                overrides.append(f"resources.{rk}={rv!r}")
            merged_res[rk] = rv

    model_val = _pick("model")
    start_val = _pick("start")
    if not model_val:
        raise ValidationError(
            "Missing required model name (pass `<model>` / `-m <model>` or set `model:` in --config).",
            field="model",
            next_command="aimip models",
        )
    if not start_val:
        raise ValidationError(
            "Missing required start timestamp (pass `--start YYYY-MM-DD` or set `start:` in --config).",
            field="start",
            next_command=f"aimip show {model_val}",
        )

    steps_raw = _pick("steps")
    length_raw = _pick("length")
    end_raw = _pick("end")
    members_raw = _pick("members")
    seed_raw = _pick("seed")
    ic_raw = _pick("ic")
    ocean_ic_raw = _pick("ocean_ic")
    lora_raw = _pick("lora")
    forcing_raw = _pick("forcing_set")
    name_raw = _pick("name")
    out_raw = _pick("out")
    monthly_raw = _pick("monthly", True)
    pred_raw = _pick("prediction", False)
    mode_raw = _pick("mode", "slurm")

    if merged_res.get("cpus_per_task") is not None:
        merged_res["cpus_per_task"] = coerce_optional_int(
            merged_res["cpus_per_task"], "resources.cpus_per_task"
        )

    return RunSpec(
        model=str(model_val),
        start=str(start_val),
        steps=coerce_optional_int(steps_raw, "steps"),
        length=str(length_raw) if length_raw is not None else None,
        end=str(end_raw) if end_raw is not None else None,
        members=coerce_optional_int(members_raw, "members"),
        seed=coerce_optional_int(seed_raw, "seed"),
        ic=str(ic_raw) if ic_raw is not None else None,
        ocean_ic=str(ocean_ic_raw) if ocean_ic_raw is not None else None,
        lora=str(lora_raw) if lora_raw is not None else None,
        forcing_set=str(forcing_raw) if forcing_raw is not None else None,
        name=str(name_raw) if name_raw is not None else None,
        out=str(out_raw) if out_raw is not None else None,
        monthly=coerce_bool(monthly_raw, "monthly", default=True),
        prediction=coerce_bool(pred_raw, "prediction", default=False),
        resources=merged_res,
        mode=str(mode_raw),
        config_file=validate_safe_path(
            os.path.abspath(config_path) if not config_path.startswith("/") else config_path,
            label="--config file",
        )
        if config_path
        else None,
        cli_overrides=tuple(overrides),
    )


def render_plan_card(plan: ResolvedPlan, *, is_tty: bool = False) -> str:
    """Render a `ResolvedPlan` as either a rich TTY card or a deterministic plain-ASCII non-TTY card."""
    use_ansi = bool(is_tty and not os.environ.get("NO_COLOR"))
    b_on = "\x1b[1m" if use_ansi else ""
    b_off = "\x1b[0m" if use_ansi else ""
    dim_on = "\x1b[2m" if use_ansi else ""

    n_cached = len(plan.assets) - len(plan.missing_assets)
    n_missing = len(plan.missing_assets)
    if n_missing == 0:
        inputs_line = f"{n_cached} cached, 0 to fetch (0 B)"
    else:
        if all(a.prospective for a in plan.missing_assets):
            src_hint = "ARCO-ERA5" if plan.family == "e2s-dlesym" else "staging source"
        else:
            src_hint = "Hugging Face / Zenodo"
        inputs_line = (
            f"{n_cached} cached, {n_missing} to fetch (~{format_bytes(plan.missing_bytes)} from {src_hint})"
        )

    res = plan.resources
    basis = res.get("estimate_basis") or "1x A100-80GB baseline"
    sched = plan.runtime_identity.get("scheduler", "slurm")
    if sched == "slurm" and plan.spec.mode != "local":
        res_line = (
            f"slurm partition={res.get('partition') or '<none>'}"
            + (f" qos={res.get('qos')}" if res.get("qos") else "")
            + f" account={res.get('account') or '<none>'} "
            f"gres={res.get('gres') or '<none>'} wall={res.get('time')} "
            f"(~{res.get('gpu_hours_estimate', 0):.3f} GPU-h; basis: {basis})"
        )
        if res.get("walltime_capped"):
            res_line += (
                f" [CAPPED from {res.get('walltime_uncapped')}: job may stop before end; continue from restart.nc]"
            )
    else:
        res_line = (
            f"local execution (~{res.get('gpu_hours_estimate', 0):.3f} GPU-h, est. {res.get('time')}; basis: {basis})"
        )

    if plan.family == "arches":
        period_line = (
            f"outputs 2015-03 .. {plan.end_time[:7]}  ({plan.steps} steps / monthly means, "
            f"start={plan.start_time[:10]}, calendar={plan.calendar})"
        )
        ic_line = "2015-02 IPSL scenario state (previous 2015-01)"
    else:
        period_line = (
            f"{plan.start_time} -> {plan.end_time}  "
            f"({plan.steps} steps, dt={plan.timestep_hours}h, calendar={plan.calendar})"
        )
        ic_line = f"{plan.ic_source or 'shipped IC'} @ {plan.start_time}"
        if not plan.ic_checked:
            ic_line += " (initial condition: not checked until the files are fetched)"

    title_target = plan.run_name or os.path.basename(plan.output_dir)
    digest_tag = f"prospective digest {plan.short_digest}" if plan.requires_staging else f"digest {plan.short_digest}"
    header = (
        f"{b_on}PLAN {title_target}{b_off}  {dim_on}[{digest_tag}]{b_off}"
        if use_ansi
        else f"PLAN {title_target}  [{digest_tag}]"
    )

    out_note = "  [exists]" if os.path.exists(plan.output_dir) else ""
    if plan.spec.monthly and plan.spec.prediction:
        outputs_line = "monthly means (monthly_mean_predictions.nc) + full step predictions (--prediction: large output)"
    elif plan.spec.prediction:
        outputs_line = "full step predictions only (--prediction: large output; monthly means disabled)"
    elif plan.spec.monthly:
        outputs_line = "monthly means (monthly_mean_predictions.nc)"
    else:
        outputs_line = "restart state only (restart.nc)"
    digest_line = (
        f"  digest:      {plan.short_digest}  (approval prefix: {plan.approval_prefix}; "
        "prospective prepare stage — final submit digest after staging)"
        if plan.requires_staging
        else f"  digest:      {plan.short_digest}  (approval prefix: {plan.approval_prefix})"
    )
    lines = [
        header,
        f"  model:       {plan.model} ({plan.family}, status={plan.status_tier}, task={plan.task})",
        f"  period:      {period_line}",
        f"  ic:          {ic_line}",
    ]
    if plan.ic_provenance:
        lines.append(f"  ic_note:     {plan.ic_provenance}")
    lines.extend(
        [
            f"  ensemble:    {plan.members} member(s), seed={plan.seed}  ({plan.ensemble_note or plan.ensemble_strategy})",
            f"  forcing:     {plan.forcing_set or '(none)'}",
            f"  inputs:      {inputs_line}",
            f"  outputs:     {outputs_line}",
            f"  resources:   {res_line}",
            f"  output:      {plan.output_dir}{out_note}",
        ]
    )
    if plan.limitations:
        lines.append(f"  limits:      {plan.limitations}")
    lines.append(digest_line)
    if plan.lora_checkpoint:
        lines.append(f"  lora:        {plan.lora_checkpoint}")
    if plan.cli_overrides:
        lines.append(f"  overrides:   {', '.join(plan.cli_overrides)}")
    return "\n".join(lines)


def render_show_card(card: dict[str, Any], *, include_examples: bool = False) -> str:
    """Format `services.show(model)` for terminal output."""
    fsets = ", ".join(
        f"{fs['name']} ({fs['window'][0][:10]}..{fs['window'][1][:10]})"
        for fs in card.get("forcing_sets", [])
    )
    step_label = "1mo step" if card.get("family") == "arches" else f"{card['timestep_hours']}h step"
    ver_note = f", verified on {card['verified_on']}" if card.get("verified_on") else ""
    lines = [
        f"{card['name']} — {card['title']}",
        f"  status:      {card['status']}{ver_note}   (task: {card['task']}, family: {card['family']})",
        f"  step/cal:    {step_label}, calendar={card['calendar']}, stochastic={card['stochastic']}",
        f"  grid:        {card['grid']}",
        f"  forcing:     {fsets or '(none)'}",
        *(
            [f"  ic:          {card['ic_provenance']}"]
            if card.get("ic_provenance")
            else []
        ),
        f"  outputs:     {', '.join(card.get('outputs', [])) or '(not recorded)'}",
        f"  installed:   {'yes' if card.get('installed') else 'no'} ({card.get('cached_files', 0)}/{card.get('total_files', 0)} files, ~{format_bytes(card.get('total_bytes', 0))})",
        f"  cost:        {card['cost_note']}",
        f"  licence:     {card['licence']}",
        f"  limits:      {card['limitations']}",
    ]
    if card.get("upstream_url"):
        lines.append(f"  upstream:    {card['upstream_url']}")
    if include_examples or card.get("examples"):
        examples = card.get("examples") or []
        if include_examples:
            lines.append("  examples:")
            for ex in examples:
                lines.append(f"    {ex}")
        elif examples:
            lines.append(f"  example:     {examples[0]}")
    return "\n".join(lines)


def render_doctor_report(report: dict[str, Any]) -> str:
    """Format `services.doctor()` check table with PASS / INFO / FAIL badges and fix hints."""
    lines = []
    for chk in report.get("checks", []):
        st = str(chk.get("status") or ("pass" if chk.get("ok") else "fail")).lower()
        if st in ("info", "warn"):
            badge = "INFO"
        elif chk.get("ok"):
            badge = "PASS"
        else:
            badge = "FAIL"
        lines.append(f"{badge:<6} {chk['name']:<22} {chk['detail']}")
        if not chk.get("ok") and chk.get("fix_hint"):
            lines.append(f"       fix: {chk['fix_hint']}")
    overall = "ok" if report.get("ok") else "issues detected"
    lines.append(f"Overall: {overall} ({report.get('passed', 0)}/{report.get('total', 0)} passed)")
    return "\n".join(lines)
