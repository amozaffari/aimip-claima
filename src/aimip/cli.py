"""Command-line interface for AIMIP (`python -m aimip`)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
import re
import sys
from typing import Any

from . import __version__
from .errors import EXIT_CODES, AimipError, ConfigError, ValidationError, enrich_error


class _AimipArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises `ValidationError` instead of printing unframed usage text on `--json`."""

    def error(self, message: str) -> None:
        field_match = re.search(r"argument (?:-[A-Za-z0-9]/)?--([A-Za-z0-9_-]+)", message)
        field = field_match.group(1).replace("-", "_") if field_match else None
        raise ValidationError(message, field=field)
from .registry import (
    ModelSpec,
    get_model,
    get_model_ic_provenance,
    resolve_playground_dir,
)


def _emit_json_ok(result: Any) -> int:
    """Emit the versioned AIMIP v1 JSON envelope on stdout."""
    print(json.dumps({"aimip": "1", "ok": True, "result": result}, indent=2))
    return 0


def _wants_json(args: argparse.Namespace | None, raw_argv: Sequence[str]) -> bool:
    """Return True when `--json` or `--format json` was requested."""
    if args is not None and (
        bool(getattr(args, "json", False)) or getattr(args, "format", None) == "json"
    ):
        return True
    if "--json" in raw_argv or "--format=json" in raw_argv:
        return True
    for idx, tok in enumerate(raw_argv[:-1]):
        if tok == "--format" and raw_argv[idx + 1] == "json":
            return True
    return False


def _add_registry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to registry.json (defaults to $AIMIP_REGISTRY, $AIMIP_DATA/registry.json, or packaged registry.default.json).",
    )
    parser.add_argument(
        "--playground",
        default=None,
        help="Override $AIMIP_PLAYGROUND root directory for ${PLAYGROUND} expansion.",
    )
    parser.add_argument(
        "--data",
        dest="data_dir",
        default=None,
        help="Override $AIMIP_DATA root directory for ${DATA} expansion.",
    )


def _add_slurm_args(parser: argparse.ArgumentParser) -> None:
    slurm_grp = parser.add_argument_group("SLURM / site batch options")
    slurm_grp.add_argument(
        "--site-config",
        dest="site_config",
        default=None,
        help="Path to site.json (defaults to $AIMIP_SITE_CONFIG or $AIMIP_HOME/site.json).",
    )
    slurm_grp.add_argument(
        "--partition",
        default=None,
        help="SLURM partition override (defaults to site.json sbatch.partition).",
    )
    slurm_grp.add_argument(
        "--qos",
        default=None,
        help=(
            "SLURM QOS override (defaults to site.json sbatch.qos; 'none' omits --qos). On MN5 the QOS places the "
            "job: acc_ehpc (up to 3 days) or acc_debug (up to 2 h, one job at a time)."
        ),
    )
    slurm_grp.add_argument(
        "--account",
        default=None,
        help="SLURM account override (defaults to site.json sbatch.account; '' or 'none' omits --account).",
    )
    slurm_grp.add_argument(
        "--gres",
        default=None,
        help="SLURM GPU generic resource override (defaults to site.json sbatch.gres; '' or 'none' omits --gres).",
    )
    slurm_grp.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help="SLURM CPUs per task override (defaults to site.json sbatch.cpus_per_task).",
    )
    slurm_grp.add_argument(
        "--mem",
        default=None,
        help="SLURM memory allocation override (defaults to site.json sbatch.mem); 'none' writes no --mem line "
        "(MN5 refuses --mem and gives memory per core).",
    )
    slurm_grp.add_argument(
        "--time",
        default=None,
        help="Override SLURM walltime HH:MM:SS (default: auto-estimated, capped by site.json max_time).",
    )
    slurm_grp.add_argument(
        "--job-name",
        default=None,
        help="Optional SLURM job name token.",
    )


def _add_plan_and_run_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model_pos",
        nargs="?",
        default=None,
        metavar="model",
        help="Model name in registry (or pass `-m/--model` or `--config`).",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_opt",
        default=None,
        help="Model name in registry.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to declarative experiment.yaml (CLI flags override file keys).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start ISO timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of forward steps.")
    parser.add_argument(
        "--length",
        default=None,
        help="Simulation horizon with unit (`20y`, `6m`, `30d`, `12h`), converted with model calendar.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Target end ISO timestamp, converted with model calendar and step size.",
    )
    parser.add_argument(
        "--ic",
        default=None,
        help="Custom/spun IC netCDF, <prev>/restart.nc, or <prev-run-dir> for coupled runs.",
    )
    parser.add_argument(
        "--ocean-ic",
        dest="ocean_ic",
        default=None,
        help="Companion ocean IC netCDF for coupled models (`fme.coupled.inference`).",
    )
    parser.add_argument("--members", type=int, default=None, help="Number of ensemble members K.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for fme.core.set_seed / rollout (default: 0).",
    )
    parser.add_argument("--lora", default=None, help="Replace model checkpoint with LoRA checkpoint.")
    parser.add_argument(
        "--forcing-set",
        default=None,
        help="Optional explicit forcing set name (default: auto-selected by [start, end] window).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Short run name resolved under site `run_root` (`<run_root>/<name>`).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output run dir under an allowed run root (default: $AIMIP_DEFAULT_OUT_ROOT/<model>-<start>-<hash>).",
    )
    parser.add_argument(
        "--monthly",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable monthly_mean_predictions.nc output (default: --monthly).",
    )
    parser.add_argument(
        "--prediction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable step-by-step autoregressive_predictions.nc output (default: --no-prediction).",
    )


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a plain-text aligned table without external formatting dependencies."""
    if not rows:
        widths = [len(h) for h in headers]
        return "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + "\n" + "  ".join(
            "-" * w for w in widths
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    def _fmt_row(cells: Sequence[str]) -> str:
        return "  ".join(
            cells[i].ljust(widths[i]) if i < len(cells) - 1 else cells[i]
            for i in range(len(cells))
        )

    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def _spec_from_cli_args(args: argparse.Namespace) -> Any:
    """Build a `RunSpec` by merging optional `--config` with CLI flags."""
    from .services import merge_config_and_cli

    if args.model_pos and args.model_opt and args.model_pos != args.model_opt:
        raise ValidationError(
            f"Conflicting model arguments: positional {args.model_pos!r} vs -m/--model {args.model_opt!r}.",
            field="model",
        )
    chosen_model = args.model_opt or args.model_pos
    cli_mode = "local" if getattr(args, "local", False) else None
    cli_values: dict[str, Any] = {
        "model": chosen_model,
        "start": args.start,
        "steps": args.steps,
        "length": args.length,
        "end": args.end,
        "members": args.members,
        "seed": args.seed,
        "ic": args.ic,
        "ocean_ic": args.ocean_ic,
        "lora": args.lora,
        "forcing_set": args.forcing_set,
        "name": args.name,
        "out": args.out,
        "monthly": args.monthly,
        "prediction": args.prediction,
        "mode": cli_mode,
        "resources": {
            "partition": args.partition,
            "qos": args.qos,
            "account": args.account,
            "gres": args.gres,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time,
            "job_name": args.job_name,
        },
    }
    return merge_config_and_cli(args.config, cli_values)


def _cmd_models(args: argparse.Namespace) -> int:
    from .services import models as svc_models

    entries = svc_models(
        task=args.task,
        family=args.family,
        status=args.status,
        installed_only=args.installed,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
    )

    if args.json or args.format == "json":
        return _emit_json_ok(entries)

    rows: list[list[str]] = []
    for item in entries:
        fam = item["family"]
        runtime = {"e2s-dlesym": "e2s 0.19", "arches": "arches"}.get(
            fam, item["fme_version"]
        )
        dt = "1mo" if fam == "arches" else f"{item['timestep_hours']}h"
        base_kind = "stochastic" if item["stochastic"] else "deterministic"
        kind = (
            f"coupled/{base_kind}"
            if (item["is_coupled"] or fam == "e2s-dlesym")
            else base_kind
        )
        sets_str = ", ".join(
            f"{fs['name']}[{fs['window'][0][:10]}..{fs['window'][1][:10]}]"
            for fs in item["forcing_sets"]
        )
        ic_ts = item["available_ic_timestamps"]
        if fam == "e2s-dlesym":
            ic_summary = "ERA5 at --start (ARCO-ERA5)"
        elif fam == "arches":
            ic_summary = "2015 scenario state"
        elif not ic_ts:
            ic_summary = f"unstaged ({len(item['ic_files'])} file)"
        elif len(ic_ts) <= 2:
            ic_summary = ", ".join(ic_ts)
        else:
            ic_summary = f"{len(ic_ts)} ({ic_ts[0]} .. {ic_ts[-1]})"

        rows.append(
            [
                item["name"],
                item["status"],
                item["task"],
                runtime,
                kind,
                dt,
                sets_str,
                ic_summary,
                item["notes"],
            ]
        )

    print(
        _format_table(
            ["MODEL", "STATUS", "TASK", "RUNTIME", "TYPE", "DT", "FORCING_SETS", "IC_TIMES", "NOTES"],
            rows,
        )
    )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from .services import render_show_card, show as svc_show

    card = svc_show(
        args.model,
        include_examples=args.examples,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
    )
    if args.json:
        return _emit_json_ok(card)
    print(render_show_card(card, include_examples=args.examples))
    return 0


def _cmd_cite(args: argparse.Namespace) -> int:
    from .services import cite as svc_cite

    doc = svc_cite(
        args.model,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
    )
    if args.json:
        return _emit_json_ok(doc)
    print(f"# {doc['model']} — {doc['title']} (licence: {doc['licence']})")
    print(doc["bibtex"])
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from .services import plan as svc_plan, render_plan_card

    spec = _spec_from_cli_args(args)
    resolved = svc_plan(
        spec,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
        site_config_path=args.site_config,
    )
    if args.json:
        return _emit_json_ok(resolved.to_dict())
    print(render_plan_card(resolved, is_tty=sys.stdout.isatty()))
    return 0


def _resolve_mode(args: argparse.Namespace) -> str:
    if getattr(args, "local", False):
        return "local"
    if getattr(args, "submit", False):
        return "submit"
    return "dry-run"


def _build_slurm_and_site(args: argparse.Namespace) -> tuple[Any, Any]:
    from .runner import SlurmOptions
    from .site_config import load_site_config

    site_cfg = load_site_config(getattr(args, "site_config", None))
    slurm_opts = SlurmOptions.from_site(
        site_cfg,
        partition=args.partition,
        qos=args.qos,
        account=args.account,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
        job_name=args.job_name,
    )
    return slurm_opts, site_cfg


def _cmd_ic_list(args: argparse.Namespace) -> int:
    from .ic import list_model_ics, list_model_ocean_ics

    spec = get_model(
        args.model,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
    )
    infos = list_model_ics(spec)
    ocean_infos = list_model_ocean_ics(spec) if spec.is_coupled else []
    provenance = get_model_ic_provenance(spec.name, spec.notes)

    if args.json:
        payload: dict[str, Any] = {
            "model": spec.name,
            "ic_provenance": provenance,
            "ic_files": [info.to_dict() for info in infos],
        }
        if spec.is_coupled:
            payload["ocean_ic_files"] = [info.to_dict() for info in ocean_infos]
        return _emit_json_ok(payload)

    rows: list[list[str]] = []
    if spec.is_coupled and len(ocean_infos) == len(infos):
        for info, o_info in zip(infos, ocean_infos):
            dims_str = "/".join(info.spatial_dims) if info.spatial_dims else "-"
            for idx, ts in enumerate(info.timestamps):
                rows.append(
                    [ts, info.calendar, dims_str, str(idx), info.path, o_info.path]
                )
        print(
            _format_table(
                ["TIMESTAMP", "CALENDAR", "DIMS", "INDEX", "ATMOS_FILE", "OCEAN_FILE"],
                rows,
            )
        )
    else:
        for info in infos:
            dims_str = "/".join(info.spatial_dims) if info.spatial_dims else "-"
            for idx, ts in enumerate(info.timestamps):
                rows.append([ts, info.calendar, dims_str, str(idx), info.path])
        print(_format_table(["TIMESTAMP", "CALENDAR", "DIMS", "INDEX", "FILE"], rows))

    if provenance:
        print(f"\nProvenance: {provenance}")
    return 0


def _cmd_ic_spin(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    from .ic import collect_spin, prepare_spin
    from .runner import resolve_model_for_prepare
    from .time_utils import compute_spin_steps, select_forcing_set
    from .validators import validate_iso_timestamp

    playground_dir = resolve_playground_dir(args.playground)

    if getattr(args, "collect", False):
        if not args.out:
            raise ConfigError("`aimip ic spin --collect` requires `--out <ic.nc>`.")
        ref_ic: str | None = None
        ref_ocean_ic: str | None = None
        if args.model:
            spec = get_model(
                args.model,
                registry_path=args.registry,
                playground_dir=args.playground,
                data_dir=args.data_dir,
            )
            if spec.ic_files and os.path.isfile(spec.ic_files[0]):
                ref_ic = spec.ic_files[0]
            if spec.ocean_ic_files and os.path.isfile(spec.ocean_ic_files[0]):
                ref_ocean_ic = spec.ocean_ic_files[0]
        resolved_spin = args.spin_dir or f"{args.out}.spin"
        merged_info = collect_spin(
            resolved_spin,
            args.out,
            out_ocean_ic_path=args.ocean_out,
            target_time=args.to_time,
            shipped_ic_reference=ref_ic,
            shipped_ocean_ic_reference=ref_ocean_ic,
            playground_dir=playground_dir,
        )
        if args.json:
            return _emit_json_ok(merged_info.to_dict())
        dims_str = "/".join(merged_info.spatial_dims) if merged_info.spatial_dims else "-"
        print(
            f"Merged {merged_info.n_samples} spin restart(s) from {resolved_spin} into {merged_info.path} "
            f"(time={merged_info.timestamps[0]}, calendar={merged_info.calendar}, "
            f"dims=sample/{dims_str}, vars={len(merged_info.variable_names)})."
        )
        return 0

    if not args.model or not args.from_times or not args.to_time:
        raise ConfigError(
            "`aimip ic spin` requires `-m/--model`, `--from T1,T2,...`, and `--to T` "
            "(unless `--collect` is specified)."
        )

    from_list = [tok.strip() for tok in args.from_times.split(",") if tok.strip()]
    mode = _resolve_mode(args)
    slurm_opts, site_cfg = _build_slurm_and_site(args)
    auto_fetch = not bool(getattr(args, "no_fetch", False))

    def _spin_precheck(candidate_spec: ModelSpec) -> None:
        norm_to = validate_iso_timestamp(args.to_time, label="--to")
        for idx, ft in enumerate(from_list):
            norm_from = validate_iso_timestamp(ft, label=f"--from[{idx}]")
            spin_steps = compute_spin_steps(
                norm_from,
                norm_to,
                candidate_spec.timestep_hours,
                calendar=candidate_spec.calendar,
            )
            select_forcing_set(
                norm_from,
                spin_steps,
                candidate_spec.timestep_hours,
                candidate_spec.forcing_sets,
                calendar=candidate_spec.calendar,
                model_name=candidate_spec.name,
                preferred_set_name=args.forcing_set,
            )

    spec = resolve_model_for_prepare(
        args.model,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
        auto_fetch=auto_fetch,
        precheck_fn=_spin_precheck,
    )

    target_ic_hint: str | None = None
    effective_spin_dir = args.out
    if args.out and args.out.endswith(".nc"):
        target_ic_hint = args.out
        effective_spin_dir = f"{args.out}.spin"
    elif args.spin_dir:
        effective_spin_dir = args.spin_dir

    manifest = prepare_spin(
        model=spec,
        from_times=from_list,
        to_time=args.to_time,
        spin_dir=effective_spin_dir,
        mode=mode,
        seed=0 if args.seed is None else args.seed,
        lora_path=args.lora,
        forcing_set_name=args.forcing_set,
        slurm_opts=slurm_opts,
        site_config=site_cfg,
        playground_dir=playground_dir,
        data_dir=args.data_dir,
        auto_fetch=auto_fetch,
        argv=raw_argv,
    )

    if args.print_sbatch_path:
        print(manifest["sbatch_path"])
        return 0

    if args.json:
        return _emit_json_ok(manifest)

    spin_dir = manifest["spin_dir"]
    ic_out_hint = target_ic_hint or f"{spin_dir}/ic.nc"
    print(
        f"Prepared {manifest['member_count']} spin run(s) in {spin_dir} "
        f"(mode={mode}, target={manifest['target_time']})."
    )
    print(f"Sbatch:   {manifest['sbatch_path']}")
    if mode != "local":
        print(f"Next:     aimip submit {spin_dir}")
    print(f"Collect:  aimip collect {spin_dir} --out {ic_out_hint}")
    if target_ic_hint:
        print(f"          (or: aimip ic spin --collect --out {target_ic_hint})")
    else:
        print(f"          (or: aimip ic spin --collect --spin-dir {spin_dir} --out {ic_out_hint})")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from .ic import collect_spin

    from .services import resolve_run_target

    playground_dir = resolve_playground_dir(args.playground)
    resolved_spin_dir, _ = resolve_run_target(
        args.spin_dir,
        site_config_path=getattr(args, "site_config", None),
        playground_dir=playground_dir,
    )
    ref_ic: str | None = None
    ref_ocean_ic: str | None = None
    if args.model:
        spec = get_model(
            args.model,
            registry_path=args.registry,
            playground_dir=args.playground,
            data_dir=args.data_dir,
        )
        if spec.ic_files and os.path.isfile(spec.ic_files[0]):
            ref_ic = spec.ic_files[0]
        if spec.ocean_ic_files and os.path.isfile(spec.ocean_ic_files[0]):
            ref_ocean_ic = spec.ocean_ic_files[0]

    merged_info = collect_spin(
        resolved_spin_dir,
        args.out,
        out_ocean_ic_path=args.ocean_out,
        target_time=args.to_time,
        shipped_ic_reference=ref_ic,
        shipped_ocean_ic_reference=ref_ocean_ic,
        playground_dir=playground_dir,
    )
    if args.json:
        return _emit_json_ok(merged_info.to_dict())
    dims_str = "/".join(merged_info.spatial_dims) if merged_info.spatial_dims else "-"
    print(
        f"Merged {merged_info.n_samples} spin restart(s) from {resolved_spin_dir} into {merged_info.path} "
        f"(time={merged_info.timestamps[0]}, calendar={merged_info.calendar}, "
        f"dims=sample/{dims_str}, vars={len(merged_info.variable_names)})."
    )
    return 0


def _cmd_run(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    from .services import execute_tty_plan_flow, prepare as svc_prepare, render_plan_card

    mode = _resolve_mode(args)
    spec = _spec_from_cli_args(args)
    auto_fetch = not bool(getattr(args, "no_fetch", False))

    if getattr(args, "approve", False):
        tty_path = os.environ.get("AIMIP_TTY_PATH") or "/dev/tty"
        attempt = execute_tty_plan_flow(
            spec,
            allow_fetch=auto_fetch,
            tty_path=tty_path,
            registry_path=args.registry,
            playground_dir=args.playground,
            data_dir=args.data_dir,
            site_config_path=args.site_config,
            argv=list(raw_argv),
        )
        res = attempt.to_dict()
        if args.json:
            return _emit_json_ok(res)
        if attempt.host_command:
            print(f"Prepared for host submission: {attempt.host_command}")
            return 0
        if attempt.scheduler == "slurm":
            print(
                f"submitted job {attempt.attempt_id}   (then: aimip status {attempt.run_dir}  or: aimip watch {attempt.run_name})"
            )
        else:
            print(
                f"started locally, pid {attempt.attempt_id}   (log: {attempt.run_dir}/local-run.log)"
            )
        return 0

    prepared = svc_prepare(
        spec,
        allow_fetch=auto_fetch,
        execute_local=(mode == "local" and not args.print_sbatch_path),
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
        site_config_path=args.site_config,
        manifest_mode=mode,
        argv=list(raw_argv),
    )

    if args.print_sbatch_path:
        print(prepared.sbatch_path)
        return 0

    manifest = dict(prepared.manifest)
    if args.json:
        manifest.setdefault("digest", prepared.plan.digest)
        manifest.setdefault("short_digest", prepared.plan.short_digest)
        manifest.setdefault("run_name", prepared.plan.run_name)
        manifest.setdefault("plan_path", prepared.plan_path)
        manifest.setdefault("cli_overrides", list(prepared.plan.cli_overrides))
        return _emit_json_ok(manifest)

    out_dir = prepared.run_dir
    fs_name = (manifest.get("forcing_set") or {}).get("name") or prepared.plan.forcing_set or "(none)"
    print(render_plan_card(prepared.plan, is_tty=sys.stdout.isatty()))
    print()
    verb_word = "Completed local" if mode == "local" else "Prepared"
    print(
        f"{verb_word} {prepared.plan.model} run in {out_dir} "
        f"(mode={mode}, forcing_set={fs_name}, start={prepared.plan.start_time}, "
        f"end={prepared.plan.end_time}, steps={prepared.plan.steps}, "
        f"members={prepared.plan.members}, seed={prepared.plan.seed})."
    )
    print(f"Config:   {prepared.config_path}")
    print(f"Manifest: {prepared.manifest_path}")
    print(f"Plan:     {prepared.plan_path}")
    print(f"Sbatch:   {prepared.sbatch_path}")
    if mode != "local":
        name_hint = f"   (or: aimip submit {prepared.plan.run_name})" if spec.name else ""
        print(f"Next:     aimip submit {out_dir}{name_hint}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    from .services import submit as svc_submit

    attempt = svc_submit(
        args.out,
        expected_digest=getattr(args, "expect_digest", None),
        playground_dir=args.playground,
        site_config_path=args.site_config,
    )
    res = attempt.to_dict()
    if args.json:
        return _emit_json_ok(res)
    if attempt.host_command:
        print(f"Prepared for host submission: {attempt.host_command}")
        return 0
    if attempt.scheduler == "slurm":
        print(
            f"submitted job {attempt.attempt_id}   (then: aimip status {attempt.run_dir}  or: aimip watch {attempt.run_name})"
        )
    else:
        print(
            f"started locally, pid {attempt.attempt_id}   (log: {attempt.run_dir}/local-run.log)"
        )
    return 0


def _cmd_verify_submit(args: argparse.Namespace) -> int:
    from .services import resolve_run_target, verify_prepared_run_for_submit

    run_dir, run_name = resolve_run_target(
        args.out,
        site_config_path=args.site_config,
        playground_dir=args.playground,
    )
    trusted_dir, saved_digest, saved_mode = verify_prepared_run_for_submit(
        run_dir,
        require_plan=True,
        expected_digest=args.expect_digest,
        script_override_path=args.script_path,
        playground_dir=args.playground,
        site_config_path=args.site_config,
    )
    payload = {
        "run_dir": trusted_dir,
        "run_name": run_name,
        "digest": saved_digest,
        "mode": saved_mode or "slurm",
    }
    if args.json:
        return _emit_json_ok(payload)
    print(f"{saved_digest or ''}|{saved_mode or 'slurm'}|{trusted_dir}")
    return 0


def _cmd_exec_fme(args: argparse.Namespace) -> int:
    from .runner import exec_fme
    from .submit import check_trusted_run_dir

    check_trusted_run_dir(args.out)
    res = exec_fme(args.out)
    if getattr(args, "json", False):
        return _emit_json_ok(res)
    return 0


def _cmd_lora_info(args: argparse.Namespace) -> int:
    from .lora import inspect_lora

    info = inspect_lora(args.ckpt)
    if args.json:
        return _emit_json_ok(info.to_dict())

    print(f"Checkpoint:          {info.checkpoint_path}")
    print(f"Loader note:         {info.loader_note}")
    print(f"lora_rank:           {info.lora_rank}")
    print(f"lora_alpha:          {info.lora_alpha} (effective: {info.effective_lora_alpha})")
    print(f"spectral_lora_rank:  {info.spectral_lora_rank}")
    print(
        f"spectral_lora_alpha: {info.spectral_lora_alpha} "
        f"(effective: {info.effective_spectral_lora_alpha})"
    )

    if info.tensors:
        rows = [
            [
                t.module_path,
                t.name,
                t.role,
                "x".join(str(d) for d in t.shape),
                t.dtype,
                f"{t.frobenius_norm:.6g}",
            ]
            for t in info.tensors
        ]
        print()
        print(
            _format_table(
                ["MODULE", "TENSOR", "ROLE", "SHAPE", "DTYPE", "FROBENIUS_NORM"],
                rows,
            )
        )
    else:
        print("\nNo LoRAConv2d (lora_down/lora_up) tensors present in checkpoint.")

    if info.spectral_tensors:
        s_rows = [
            [
                t.module_path,
                t.name,
                t.role,
                "x".join(str(d) for d in t.shape),
                t.dtype,
                f"{t.frobenius_norm:.6g}",
            ]
            for t in info.spectral_tensors
        ]
        print()
        print(
            _format_table(
                ["MODULE", "SPECTRAL_TENSOR", "ROLE", "SHAPE", "DTYPE", "FROBENIUS_NORM"],
                s_rows,
            )
        )
    return 0


def _cmd_lora_scale(args: argparse.Namespace) -> int:
    from .lora import scale_lora

    result = scale_lora(args.ckpt, args.alpha, args.out)
    if args.json:
        return _emit_json_ok(result)

    sf = result.get("scale_factor")
    sf_str = f"{sf:.8g}" if sf is not None else "n/a"
    print(f"Input checkpoint:  {result['input_checkpoint']} (sha256={result['input_sha256']})")
    print(f"Output checkpoint: {result['output_checkpoint']} (sha256={result['output_sha256']})")
    print(
        f"Alpha scaling:     mode={result['scaling_mode']}, "
        f"alpha_old={result['alpha_old']} -> alpha_new={result['alpha_new']} "
        f"(scale_factor={sf_str}, weights_modified={result['weights_modified']})"
    )
    if result.get("spectral_lora_rank", 0) > 0:
        print(
            f"Spectral scaling:  spectral_alpha_old={result['spectral_alpha_old']} -> "
            f"spectral_alpha_new={result['spectral_alpha_new']} "
            f"(rank={result['spectral_lora_rank']})"
        )
    print(f"Sidecar metadata:  {result['sidecar_path']}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from .services import status as svc_status

    st = svc_status(
        args.out,
        sacct_json_path=args.sacct_json,
        site_config_path=args.site_config,
    )
    if args.json:
        return _emit_json_ok(st)

    print(f"Run directory: {st['out_dir']}")
    print(f"Model:         {st['model']} (forcing_set={st['forcing_set'] or 'unknown'})")
    print(f"State:         {st['state']}")
    if st.get("short_digest"):
        print(f"Digest:        {st['short_digest']}")
    if "spin_members_completed" in st:
        print(f"Spin members:  {st['spin_members_completed']}/{st['members']} completed")
    print(f"Mode:          {st['mode']}")
    print(
        f"SLURM job ID:  {st['slurm_job_id'] or 'none'} "
        f"(sacct_available={st['sacct_available']}, source={st.get('sacct_source') or 'none'})"
    )
    if st.get("local_pid") is not None:
        print(f"Local PID:     {st['local_pid']}")
    if st["sacct"] is not None:
        s = st["sacct"]
        print(f"SLURM sacct:   state={s['state']} elapsed={s['elapsed']} exit={s['exit_code']}")
    if st["rc"] is not None:
        print(f"Exit code:     {st['rc']} (wall_seconds={st['wall_seconds']}, finished={st['finished']})")
    present = [k for k, v in st["artifacts"].items() if v]
    print(f"Artifacts:     {', '.join(present) if present else 'none'}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    from .services import watch as svc_watch

    stream = sys.stderr if args.json else sys.stdout

    def _on_tick(tick: dict[str, Any]) -> None:
        jid = tick.get("job_id") or "-"
        el = tick.get("elapsed") or "00:00:00"
        print(
            f"[{tick['poll']}] {tick.get('run_name') or tick.get('run_dir')}  "
            f"model={tick.get('model')}  state={tick['state']}  job={jid}  elapsed={el}",
            file=stream,
        )

    final = svc_watch(
        args.out,
        poll_interval=args.interval,
        max_polls=args.max_polls,
        timeout_seconds=args.timeout,
        site_config_path=args.site_config,
        on_tick=_on_tick,
    )
    is_terminal = bool(final.get("terminal"))
    st_upper = (str(final.get("state") or "").split() or [""])[0].rstrip("+").upper()
    rc = 0
    if is_terminal:
        if st_upper != "COMPLETED":
            rc = EXIT_CODES["execution_error"]
    elif args.max_polls is None and (final.get("slurm_job_id") or final.get("local_pid")):
        print("still running, watch timed out", file=stream)
        rc = EXIT_CODES["deadline"]
    if args.json:
        _emit_json_ok(final)
    return rc


def _cmd_logs(args: argparse.Namespace) -> int:
    from .services import logs as svc_logs

    if args.json:
        res = svc_logs(
            args.out,
            follow=args.follow,
            max_bytes=args.max_bytes,
            poll_interval=args.interval,
            max_polls=args.max_polls,
            site_config_path=args.site_config,
        )
        return _emit_json_ok(res)

    printed = False

    def _on_chunk(chunk: str) -> None:
        nonlocal printed
        printed = True
        sys.stdout.write(chunk)
        if not chunk.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    res = svc_logs(
        args.out,
        follow=args.follow,
        max_bytes=args.max_bytes,
        poll_interval=args.interval,
        max_polls=args.max_polls,
        site_config_path=args.site_config,
        on_chunk=_on_chunk,
    )
    if not printed:
        print(f"(no log output yet in {res['run_dir']})")
    return 0


def _cmd_ls(args: argparse.Namespace) -> int:
    from .services import ls as svc_ls

    items = svc_ls(args.run_root, site_config_path=args.site_config)
    if args.json:
        return _emit_json_ok(items)
    if not items:
        print("No runs found.")
        return 0
    rows = [
        [
            str(it["name"]),
            str(it.get("model") or "-"),
            str(it.get("state") or "unknown"),
            str(it.get("job_id") or "-"),
            str(it.get("start_time") or "-"),
            str(it.get("steps") if it.get("steps") is not None else "-"),
            str(it.get("updated_at") or "-"),
        ]
        for it in items
    ]
    print(
        _format_table(
            ["NAME", "MODEL", "STATE", "JOB_ID", "START", "STEPS", "UPDATED"],
            rows,
        )
    )
    return 0


def _cmd_cancel(args: argparse.Namespace) -> int:
    from .services import cancel as svc_cancel

    res = svc_cancel(args.out, site_config_path=args.site_config)
    if args.json:
        return _emit_json_ok(res)
    print(
        f"Cancelled {res['scheduler']} attempt {res['attempt_id']} for {res['run_name']} ({res['run_dir']})."
    )
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    from .services import summary as svc_summary

    res = svc_summary(args.out, site_config_path=args.site_config)
    summary = res.to_dict()
    if args.json:
        return _emit_json_ok(summary)

    print(
        f"Summary for {summary['out_dir']} "
        f"({summary['n_samples']} member(s), {summary['n_months']} month(s), cos(lat) global mean):"
    )
    var_names = list(summary["variables"].keys())
    headers = ["MONTH", "COUNTS"] + [
        f"{v} ({summary['variables'][v]['units']})" for v in var_names
    ]
    rows: list[list[str]] = []
    for m_idx in range(summary["n_months"]):
        first_row = summary["variables"][var_names[0]]["series"][m_idx]
        cnt_str = "-" if first_row["counts"] is None else str(first_row["counts"])
        cells = [first_row["month"], cnt_str]
        for v in var_names:
            val = summary["variables"][v]["series"][m_idx]["ensemble_mean"]
            cells.append("nan" if val is None else f"{val:.4f}")
        rows.append(cells)

    print(_format_table(headers, rows))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .services import doctor as svc_doctor, render_doctor_report

    report = svc_doctor(
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
        site_config_path=args.site_config,
    )
    rc = 0 if report["ok"] else 1
    if args.json:
        _emit_json_ok(report)
        return rc
    print(render_doctor_report(report))
    return rc


def _cmd_approve(args: argparse.Namespace) -> int:
    from .services import approve_state_proposal

    tty_path = os.environ.get("AIMIP_TTY_PATH") or "/dev/tty"
    auth, proposal = approve_state_proposal(
        args.digest_prefix,
        state_dir=args.state_dir,
        tty_path=tty_path,
    )
    payload = {
        "approved": True,
        "action": auth.action,
        "digest": auth.digest,
        "short_digest": auth.digest[:12],
        "user": auth.user,
        "expires_at": auth.expires_at,
        "allow_download": auth.allow_download,
        "proposal_action": proposal.get("action"),
    }
    if args.json:
        return _emit_json_ok(payload)
    print(
        f"Approved {auth.action} for digest {auth.digest[:12]} "
        f"(user={auth.user}, valid for 10 minutes)."
    )
    return 0


def _cmd_completion(args: argparse.Namespace) -> int:
    from .completion import render_completion

    script = render_completion(args.shell)
    if args.json:
        return _emit_json_ok({"shell": args.shell, "script": script})
    sys.stdout.write(script)
    if not script.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    from .setup_data import run_setup

    result = run_setup(
        models=args.models,
        dest_dir=args.dest,
        playground_dir=args.playground,
        lock_path=args.lock,
        only_download=args.only_download,
        verify_only=args.verify_only,
        write_lock=args.write_lock,
        search_dirs=args.search_dirs or (),
        force_rebuild=args.force_rebuild,
        accept_derived=args.accept_derived,
    )
    if args.json:
        return _emit_json_ok(result)

    mode = result["mode"]
    if mode == "write-lock":
        print(
            f"Wrote lock {result['lock_path']} from {result['dest_dir']} "
            f"(hashed={result['hashed_files']}, baseline={result['preserved_baseline_files']}, "
            f"total={result['total_entries']})."
        )
        return 0
    if mode == "verify-only":
        print(
            f"Verified {result['verified_files']} file(s) in {result['dest_dir']} "
            f"against {result['lock_path']} for models={','.join(result['models'])}."
        )
        return 0
    if mode == "only-download":
        print(
            f"Downloaded {result['downloaded_files']} file(s) (skipped {result['skipped_existing_files']} existing) "
            f"into {result['dest_dir']} and verified {result['verified_files']} file(s) against {result['lock_path']}."
        )
        return 0

    print(
        f"Setup complete in {result['dest_dir']} for models={','.join(result['models'])} "
        f"(downloaded={result['downloaded_files']}, skipped={result['skipped_existing_files']}, "
        f"built={result['built_files']}, verified={result['verified_files']})."
    )
    print(f"Registry:  {result['registry_path']}")
    print(f"Lock:      {result['lock_path']}")
    if result.get("derived_overrides_path"):
        print(
            f"Overrides: {result['derived_overrides_path']} "
            f"({len(result.get('derived_overrides', ()))} derived override(s))"
        )
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .setup_data import fetch_model

    result = fetch_model(
        args.model,
        dest_dir=args.dest,
        playground_dir=args.playground,
        lock_path=args.lock,
        accept_derived=args.accept_derived,
        progress_stream=sys.stderr,
    )
    if args.json:
        return _emit_json_ok(result)

    print(
        f"Fetched model(s) {','.join(result['models'])} into {result['dest_dir']} "
        f"(downloaded={result['downloaded_files']}, bytes={result['downloaded_bytes']}, "
        f"skipped={result['skipped_existing_files']}, built={result['built_files']}, "
        f"verified={result['verified_files']})."
    )
    print(f"Registry:  {result['registry_path']}")
    print(f"Lock:      {result['lock_path']}")
    pin_computed = result.get("pin_after_first_fetch") or {}
    if pin_computed:
        for rel_p, info in sorted(pin_computed.items()):
            print(
                f"Pin-after-first-fetch: {rel_p} size={info['size']} sha256={info['sha256']}"
            )
    return 0


def _cmd_site_init(args: argparse.Namespace) -> int:
    from .site_config import init_site_config

    target_path, cfg = init_site_config(
        preset=args.preset,
        out_path=args.out,
        scheduler=args.scheduler,
        runtime=args.runtime,
        partition=args.partition,
        account=args.account,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        max_time=args.max_time,
        run_root=args.run_root,
        overwrite=args.overwrite,
        qos=args.qos,
        trusted_group_parents=args.trusted_group_parents,
    )
    payload = {"path": target_path, "site": cfg.to_dict()}
    if args.json:
        return _emit_json_ok(payload)

    print(
        f"Wrote site profile {target_path} "
        f"(preset={args.preset}, scheduler={cfg.scheduler}, runtime={cfg.runtime})."
    )
    if cfg.scheduler == "slurm":
        print(
            f"SLURM defaults: partition={cfg.partition or '<none>'}"
            + (f" qos={cfg.qos}" if cfg.qos else "")
            + f" account={cfg.account or '<none>'} "
            f"gres={cfg.gres or '<none>'} cpus={cfg.cpus_per_task} mem={cfg.mem or '<none>'} max_time={cfg.max_time}"
        )
    print(f"Run root:       {cfg.run_root}")
    return 0


def _cmd_site_show(args: argparse.Namespace) -> int:
    from .site_config import load_site_config, resolve_site_config_path

    path, _ = resolve_site_config_path(args.site_config, require_target=False)
    cfg = load_site_config(args.site_config)
    payload = {
        "path": path if (path and os.path.isfile(path)) else None,
        "site": cfg.to_dict(),
    }
    if args.json:
        return _emit_json_ok(payload)

    print(f"Site config:    {payload['path'] or 'built-in default'}")
    print(f"Name:           {cfg.name}")
    print(f"Scheduler:      {cfg.scheduler}")
    print(f"Runtime:        {cfg.runtime}")
    if cfg.scheduler == "slurm":
        print(
            f"SLURM defaults: partition={cfg.partition or '<none>'}"
            + (f" qos={cfg.qos}" if cfg.qos else "")
            + f" account={cfg.account or '<none>'} "
            f"gres={cfg.gres or '<none>'} cpus={cfg.cpus_per_task} mem={cfg.mem or '<none>'} max_time={cfg.max_time}"
        )
    print(f"Run root:       {cfg.run_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level `aimip` argument parser."""
    parser = _AimipArgumentParser(
        prog="aimip",
        description="AIMIP CLI for ACE/SamudrACE (`fme`), DLESyM, and ArchesClimate-SSP climate emulators.",
        epilog=(
            "Submission and interactive commands:\n"
            "  submit <run-dir>    Validate and submit <run-dir>/job.sbatch (Slurm or local background).\n"
            "  approve <prefix>    Approve a pending MCP run/cancel proposal on /dev/tty.\n"
            "  chat [args...]      Launch the interactive Gemini planning agent (API key or Vertex AI).\n"
            "  mcp [args...]       Start the stdio Model Context Protocol (MCP) server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aimip {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_AimipArgumentParser,
    )

    # 1. `aimip models`
    p_models = subparsers.add_parser(
        "models",
        help="List registered climate emulator models with optional task, family, and status filters.",
    )
    p_models.add_argument("--task", default=None, help="Filter by task (amip, slab-ocean, coupled, scenario).")
    p_models.add_argument("--family", default=None, help="Filter by family (fme, e2s-dlesym, arches, emu).")
    p_models.add_argument("--status", default=None, help="Filter by tier (certified, runs, experimental).")
    p_models.add_argument("--installed", action="store_true", help="Show only models whose files are staged locally.")
    p_models.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (table or json).",
    )
    p_models.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_models)

    # 1b. `aimip show <model>`
    p_show = subparsers.add_parser(
        "show",
        help="Display detailed model card (grid, step, calendar, forcing sets, licence, limits, examples).",
    )
    p_show.add_argument("model", help="Model name in registry.")
    p_show.add_argument("--examples", action="store_true", help="Print ready-to-copy CLI examples.")
    p_show.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_show)

    # 1c. `aimip cite <model>`
    p_cite = subparsers.add_parser(
        "cite",
        help="Print BibTeX entries and licence attribution for a model.",
    )
    p_cite.add_argument("model", help="Model name in registry.")
    p_cite.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_cite)

    # 1d. `aimip plan`
    p_plan = subparsers.add_parser(
        "plan",
        help="Resolve and validate a run specification without writing files or downloading data.",
    )
    _add_plan_and_run_spec_args(p_plan)
    p_plan.add_argument("--local", action="store_true", help="Plan for in-process local execution instead of Slurm.")
    p_plan.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_plan)
    _add_slurm_args(p_plan)

    # 2. `aimip ic {list,spin}`
    p_ic = subparsers.add_parser(
        "ic",
        help="Inspect shipped initial conditions or prepare multi-start spin runs.",
    )
    ic_sub = p_ic.add_subparsers(
        dest="ic_command",
        required=True,
        parser_class=_AimipArgumentParser,
    )

    p_ic_list = ic_sub.add_parser(
        "list",
        help="Decode and print timestamps in shipped IC netCDF files for a model.",
    )
    p_ic_list.add_argument("-m", "--model", required=True, help="Model name in registry.")
    p_ic_list.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_ic_list)

    p_ic_spin = ic_sub.add_parser(
        "spin",
        help="Prepare spin runs from shipped ICs T1,T2,... up to target T (or --collect restarts).",
    )
    p_ic_spin.add_argument("-m", "--model", default=None, help="Model name in registry.")
    p_ic_spin.add_argument(
        "--from",
        dest="from_times",
        default=None,
        help="Comma-separated shipped IC timestamps T1,T2,... (each Ti < T).",
    )
    p_ic_spin.add_argument(
        "--to",
        dest="to_time",
        default=None,
        help="Target ISO timestamp T to spin every member up to.",
    )
    p_ic_spin.add_argument(
        "--out",
        default=None,
        help="Spin directory (or target .nc file) under an allowed run root.",
    )
    p_ic_spin.add_argument(
        "--ocean-out",
        dest="ocean_out",
        default=None,
        help="Optional companion ocean IC output path when collecting a coupled spin (default: <out>_ocean.nc).",
    )
    p_ic_spin.add_argument(
        "--spin-dir",
        dest="spin_dir",
        default=None,
        help="Explicit spin directory when running `ic spin --collect --out <ic.nc>`.",
    )
    p_ic_spin.add_argument("--seed", type=int, default=None, help="Base RNG seed (default: 0).")
    p_ic_spin.add_argument("--lora", default=None, help="Optional LoRA checkpoint path.")
    p_ic_spin.add_argument(
        "--forcing-set",
        default=None,
        help="Optional explicit forcing set name when windows overlap.",
    )
    p_ic_spin.add_argument(
        "--no-fetch",
        action="store_true",
        help="Forbid automatic fetch-on-first-use when model files are missing.",
    )
    spin_mode = p_ic_spin.add_mutually_exclusive_group()
    spin_mode.add_argument("--dry-run", action="store_true", help="Prepare spin configs/sbatch only.")
    spin_mode.add_argument(
        "--submit",
        action="store_true",
        help="Prepare spin dirs and print `Next:     aimip submit <spin-dir>`.",
    )
    spin_mode.add_argument("--local", action="store_true", help="Run spin members locally in-process.")
    spin_mode.add_argument(
        "--collect",
        action="store_true",
        help="Merge completed <spin-dir>/0..K-1/restart.nc into `--out <ic.nc>`.",
    )
    p_ic_spin.add_argument(
        "--print-sbatch-path",
        action="store_true",
        help="Print only the generated job.sbatch path(s) on stdout.",
    )
    p_ic_spin.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_ic_spin)
    _add_slurm_args(p_ic_spin)

    # 3. `aimip collect <spin-dir> --out <ic.nc>`
    p_collect = subparsers.add_parser(
        "collect",
        help="Merge <spin-dir>/0..K-1/restart.nc into one multi-sample IC netCDF (--out ic.nc).",
    )
    p_collect.add_argument("spin_dir", help="Completed spin directory (<spin-dir>).")
    p_collect.add_argument(
        "--out",
        required=True,
        help="Output merged IC netCDF path under an allowed run root.",
    )
    p_collect.add_argument(
        "--ocean-out",
        dest="ocean_out",
        default=None,
        help="Optional companion ocean IC output path for coupled spins (default: <out>_ocean.nc).",
    )
    p_collect.add_argument("-m", "--model", default=None, help="Optional model name for IC metadata reference.")
    p_collect.add_argument(
        "--to",
        dest="to_time",
        default=None,
        help="Optional expected target ISO timestamp T verification override.",
    )
    p_collect.add_argument(
        "--site-config",
        dest="site_config",
        default=None,
        help="Optional site.json path for resolving bare spin names.",
    )
    p_collect.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_collect)

    # 4. `aimip run`
    p_run = subparsers.add_parser(
        "run",
        help="Prepare (or locally execute) an AIMIP model run.",
    )
    _add_plan_and_run_spec_args(p_run)
    p_run.add_argument(
        "--no-fetch",
        action="store_true",
        help="Forbid automatic fetch-on-first-use when model files are missing.",
    )
    run_mode = p_run.add_mutually_exclusive_group()
    run_mode.add_argument("--dry-run", action="store_true", help="Write config, sbatch, plan.json, and manifest only.")
    run_mode.add_argument(
        "--submit",
        action="store_true",
        help="Prepare files and print `aimip submit <out>` as the next host command.",
    )
    run_mode.add_argument("--local", action="store_true", help="Execute job immediately in-process.")
    run_mode.add_argument(
        "--approve",
        action="store_true",
        help="Prompt on /dev/tty to approve preparation/staging and final submission interactively.",
    )
    p_run.add_argument(
        "--print-sbatch-path",
        action="store_true",
        help="Print only <out>/job.sbatch on stdout.",
    )
    p_run.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_run)
    _add_slurm_args(p_run)

    # 4b. `aimip submit <out>`
    p_submit = subparsers.add_parser(
        "submit",
        help="Validate <out>/job.sbatch in a private copy, verify plan digest, and submit.",
    )
    p_submit.add_argument("out", help="Prepared run name or directory containing job.sbatch.")
    p_submit.add_argument(
        "--expect-digest",
        dest="expect_digest",
        default=None,
        help="Require plan.json and live re-plan to match this exact 64-hex digest.",
    )
    p_submit.add_argument(
        "--playground",
        default=None,
        help="Override $AIMIP_HOME / $AIMIP_PLAYGROUND install prefix.",
    )
    p_submit.add_argument(
        "--site-config",
        dest="site_config",
        default=None,
        help="Explicit site.json path (defaults to $AIMIP_SITE_CONFIG or $AIMIP_HOME/site.json).",
    )
    p_submit.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 4c. `aimip verify-submit <out>` (internal host-wrapper verification gate)
    p_vsub = subparsers.add_parser(
        "verify-submit",
        help="Verify plan.json, sealed artifact hashes, live re-plan digest, and private job.sbatch copy before host sbatch.",
    )
    p_vsub.add_argument("out", help="Prepared run name or directory containing plan.json.")
    p_vsub.add_argument(
        "--script-path",
        dest="script_path",
        default=None,
        help="Private copy of job.sbatch to verify against sealed hashes and plan resources.",
    )
    p_vsub.add_argument(
        "--expect-digest",
        dest="expect_digest",
        default=None,
        help="Optional 64-hex digest required to match plan.json and live re-plan.",
    )
    p_vsub.add_argument("--playground", default=None, help="Override $AIMIP_HOME / $AIMIP_PLAYGROUND.")
    p_vsub.add_argument("--site-config", dest="site_config", default=None, help="Explicit site.json path.")
    p_vsub.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 5. `aimip exec-fme <out>`
    p_exec = subparsers.add_parser(
        "exec-fme",
        help="Batch/job entrypoint: verify run directory, execute fme or emulator job, and update manifest.",
    )
    p_exec.add_argument("out", help="Prepared run output directory containing config.yaml.")
    p_exec.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 6. `aimip lora {info,scale}`
    p_lora = subparsers.add_parser(
        "lora",
        help="Inspect or dose-scale LoRA weights in an fme checkpoint.",
    )
    lora_sub = p_lora.add_subparsers(
        dest="lora_command",
        required=True,
        parser_class=_AimipArgumentParser,
    )

    p_lora_info = lora_sub.add_parser(
        "info",
        help="Inspect LoRA config and lora_down/lora_up tensor Frobenius norms.",
    )
    p_lora_info.add_argument("ckpt", help="Absolute path to fme checkpoint.")
    p_lora_info.add_argument("--json", action="store_true", help="Emit JSON output.")

    p_lora_scale = lora_sub.add_parser(
        "scale",
        help="Write a copy of CKPT with stepper.config LoRA alpha set to --alpha (weights untouched).",
    )
    p_lora_scale.add_argument("ckpt", help="Absolute path to input fme LoRA checkpoint.")
    p_lora_scale.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Target effective LoRA alpha (dose knob; 0 zeros out LoRA).",
    )
    p_lora_scale.add_argument(
        "--out",
        required=True,
        help="Absolute path under an allowed run root for new scaled checkpoint (refuses if exists).",
    )
    p_lora_scale.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 7. `aimip status <out>`
    p_status = subparsers.add_parser(
        "status",
        help="Report run state from aimip-manifest.json, sacct.txt / --sacct-json, or sacct.",
    )
    p_status.add_argument("out", help="Run name or output directory.")
    p_status.add_argument(
        "--sacct-json",
        dest="sacct_json",
        default=None,
        help="Read SLURM sacct JSON from FILE instead of invoking sacct.",
    )
    p_status.add_argument(
        "--site-config",
        dest="site_config",
        default=None,
        help="Explicit site.json path for resolving bare run names.",
    )
    p_status.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 7b. `aimip watch <out>`
    p_watch = subparsers.add_parser(
        "watch",
        help="Poll run status until terminal completion (never submits or alters runs).",
    )
    p_watch.add_argument("out", help="Run name or output directory.")
    p_watch.add_argument("--interval", type=float, default=5.0, help="Seconds between polls (default: 5.0).")
    p_watch.add_argument("--max-polls", dest="max_polls", type=int, default=None, help="Optional maximum poll count.")
    p_watch.add_argument("--timeout", type=float, default=3600.0, help="Maximum watch duration in seconds.")
    p_watch.add_argument("--site-config", dest="site_config", default=None, help="Explicit site.json path.")
    p_watch.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 7c. `aimip logs <out>`
    p_logs = subparsers.add_parser(
        "logs",
        help="Print sanitized Slurm or local run logs (bounded by --max-bytes).",
    )
    p_logs.add_argument("out", help="Run name or output directory.")
    p_logs.add_argument("--follow", "-f", action="store_true", help="Follow logs until the job finishes.")
    p_logs.add_argument("--max-bytes", dest="max_bytes", type=int, default=65536, help="Maximum tail bytes to read.")
    p_logs.add_argument("--interval", type=float, default=1.0, help="Follow poll interval in seconds (default: 1.0).")
    p_logs.add_argument("--max-polls", dest="max_polls", type=int, default=None, help="Optional max follow polls.")
    p_logs.add_argument("--site-config", dest="site_config", default=None, help="Explicit site.json path.")
    p_logs.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 7d. `aimip ls [run_root]`
    p_ls = subparsers.add_parser(
        "ls",
        help="List prepared, running, and completed runs in the configured run root.",
    )
    p_ls.add_argument("run_root", nargs="?", default=None, help="Optional run root directory override.")
    p_ls.add_argument("--site-config", dest="site_config", default=None, help="Explicit site.json path.")
    p_ls.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 7e. `aimip cancel <out>`
    p_cancel = subparsers.add_parser(
        "cancel",
        help="Cancel an active Slurm job (`scancel`) or detached local run (`SIGTERM`).",
    )
    p_cancel.add_argument("out", help="Run name or output directory.")
    p_cancel.add_argument("--site-config", dest="site_config", default=None, help="Explicit site.json path.")
    p_cancel.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 8. `aimip summary <out>`
    p_summary = subparsers.add_parser(
        "summary",
        help="Print cos(lat)-weighted global-mean monthly temperature / ocean diagnostic series.",
    )
    p_summary.add_argument("out", help="Run name or output directory.")
    p_summary.add_argument("--site-config", dest="site_config", default=None, help="Explicit site.json path.")
    p_summary.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 8b. `aimip doctor`
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Verify site configuration, runtime, GPU, scheduler tools, data directory, permissions, and version.",
    )
    p_doctor.add_argument("--site-config", dest="site_config", default=None, help="Explicit site.json path.")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_doctor)

    # 8c. `aimip approve <digest-prefix>`
    p_approve = subparsers.add_parser(
        "approve",
        help="Approve a pending MCP run or cancel proposal interactively on /dev/tty.",
    )
    p_approve.add_argument("digest_prefix", help="First 8+ hex characters of the proposed plan digest.")
    p_approve.add_argument("--state-dir", dest="state_dir", default=None, help="Override $AIMIP_STATE directory.")
    p_approve.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 8d. `aimip completion {bash,zsh}`
    p_comp = subparsers.add_parser(
        "completion",
        help="Generate shell tab-completion script for bash or zsh.",
    )
    p_comp.add_argument("shell", help="Target shell (bash or zsh).")
    p_comp.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 9. `aimip setup`
    p_setup = subparsers.add_parser(
        "setup",
        help="Download models & forcings from HF/Zenodo, run deterministic builders, and verify aimip.lock.",
    )
    p_setup.add_argument(
        "--models",
        default="all",
        help="Models to install: 'all' or comma-separated subset (e.g. ace2-era5,hiro-ace).",
    )
    p_setup.add_argument(
        "--dest",
        default=None,
        help="Destination data directory (defaults to $AIMIP_DATA).",
    )
    p_setup.add_argument(
        "--lock",
        default=None,
        help="Path to aimip.lock (defaults to shipped aimip.lock).",
    )
    p_setup.add_argument(
        "--playground",
        default=None,
        help="Fallback $AIMIP_PLAYGROUND root if $AIMIP_DATA is unset.",
    )
    setup_mode = p_setup.add_mutually_exclusive_group()
    setup_mode.add_argument(
        "--only-download",
        action="store_true",
        help="Download and verify raw HF/Zenodo artifacts only (skip derived builders).",
    )
    setup_mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files under --dest against aimip.lock without downloading or building.",
    )
    setup_mode.add_argument(
        "--write-lock",
        action="store_true",
        help="Generate or update aimip.lock by hashing an existing staged tree under --dest.",
    )
    p_setup.add_argument(
        "--search-dir",
        dest="search_dirs",
        action="append",
        default=None,
        help="Extra root directory to search when running --write-lock (repeatable).",
    )
    p_setup.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild derived netCDF files even if already present and matching aimip.lock.",
    )
    p_setup.add_argument(
        "--accept-derived",
        action="store_true",
        help=(
            "Opt-in: accept non-locked derived netCDF hashes (recording <dest>/derived-overrides.json) "
            "while still strictly verifying HF/Zenodo downloads (e.g. experimental/untested aarch64/arm64 hosts)."
        ),
    )
    p_setup.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 9b. `aimip fetch -m <model>`
    p_fetch = subparsers.add_parser(
        "fetch",
        help="Download, build (if derived), and lock-verify only the requested model(s) into $AIMIP_DATA.",
    )
    p_fetch.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model name (or comma-separated model names) to fetch and verify against aimip.lock.",
    )
    p_fetch.add_argument(
        "--dest",
        default=None,
        help="Destination data directory (defaults to $AIMIP_DATA).",
    )
    p_fetch.add_argument(
        "--lock",
        default=None,
        help="Path to aimip.lock (defaults to shipped aimip.lock).",
    )
    p_fetch.add_argument(
        "--playground",
        default=None,
        help="Fallback $AIMIP_PLAYGROUND root if $AIMIP_DATA is unset.",
    )
    p_fetch.add_argument(
        "--accept-derived",
        action="store_true",
        help="Opt-in: accept non-locked derived netCDF hashes while strictly verifying HF/Zenodo downloads.",
    )
    p_fetch.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 10. `aimip site {init,show}`
    p_site = subparsers.add_parser(
        "site",
        help="Initialize or inspect portable HPC site configuration ($AIMIP_HOME/site.json).",
    )
    site_sub = p_site.add_subparsers(
        dest="site_command",
        required=True,
        parser_class=_AimipArgumentParser,
    )

    p_site_init = site_sub.add_parser(
        "init",
        help="Write $AIMIP_HOME/site.json from a preset (levante, generic-slurm, mn5, local) and overrides.",
    )
    p_site_init.add_argument(
        "--preset",
        choices=("levante", "generic-slurm", "mn5", "local"),
        default="levante",
        help="Site preset (default: levante).",
    )
    p_site_init.add_argument(
        "--out",
        default=None,
        help="Output site.json path (defaults to $AIMIP_HOME/site.json).",
    )
    p_site_init.add_argument(
        "--scheduler",
        choices=("slurm", "none"),
        default=None,
        help="Override batch scheduler (slurm or none).",
    )
    p_site_init.add_argument(
        "--runtime",
        choices=("apptainer", "singularity", "docker", "podman", "native"),
        default=None,
        help="Override runtime (container runtime or native).",
    )
    p_site_init.add_argument(
        "--partition", default=None, help="Override SLURM partition ('none' omits it; needs --qos then)."
    )
    p_site_init.add_argument(
        "--qos", default=None, help="SLURM QOS written into every job script (MN5: acc_ehpc or acc_debug)."
    )
    p_site_init.add_argument(
        "--trusted-group-parent",
        dest="trusted_group_parents",
        action="append",
        default=None,
        help=(
            "Root-owned group directory (drwxrws---) accepted above run directories, e.g. /gpfs/scratch/bsc32 "
            "on MN5; repeatable. See docs/mn5.md."
        ),
    )
    p_site_init.add_argument(
        "--account", default=None, help="Override SLURM account ('' or 'none' omits --account)."
    )
    p_site_init.add_argument(
        "--gres", default=None, help="Override SLURM --gres ('' or 'none' omits --gres)."
    )
    p_site_init.add_argument(
        "--cpus-per-task", type=int, default=None, help="Override SLURM --cpus-per-task."
    )
    p_site_init.add_argument("--mem", default=None, help="Override SLURM --mem.")
    p_site_init.add_argument(
        "--max-time", default=None, help="Override default SLURM walltime cap HH:MM:SS."
    )
    p_site_init.add_argument(
        "--run-root", default=None, help="Override default run output root path or template."
    )
    p_site_init.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing site.json file."
    )
    p_site_init.add_argument("--json", action="store_true", help="Emit JSON output.")

    p_site_show = site_sub.add_parser(
        "show",
        help="Display effective site.json configuration.",
    )
    p_site_show.add_argument(
        "--site-config",
        default=None,
        help="Explicit site.json path (defaults to $AIMIP_SITE_CONFIG or $AIMIP_HOME/site.json).",
    )
    p_site_show.add_argument("--json", action="store_true", help="Emit JSON output.")

    subparsers.add_parser(
        "ui",
        help="Start the local AIMIP web UI on 127.0.0.1 with a per-session token.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning POSIX exit code."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv[:1] == ["chat"]:
        from .agent import main as agent_main

        return int(agent_main(effective_argv[1:]) or 0)
    if effective_argv[:1] == ["mcp"]:
        from .mcp_server import main as mcp_main

        return int(mcp_main(effective_argv[1:]) or 0)
    if effective_argv[:1] == ["ui"]:
        from .ui import main as ui_main

        return int(ui_main(effective_argv[1:]) or 0)

    parser = build_parser()
    args: argparse.Namespace | None = None
    recorded_argv = ["aimip", *effective_argv]

    try:
        args = parser.parse_args(effective_argv)
        if args.command == "models":
            return _cmd_models(args)
        if args.command == "show":
            return _cmd_show(args)
        if args.command == "cite":
            return _cmd_cite(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "ic":
            from .emu_catalog import EMU_MODELS

            if getattr(args, "model", None) in EMU_MODELS:
                raise ConfigError(
                    f"`aimip ic` works with the fme models only; {args.model} starts from "
                    + (
                        "the scenario's 2015 state."
                        if args.model == "archesclimate-ssp"
                        else "ERA5 at --start (fetched by `aimip run`)."
                    ),
                    field="model",
                )
            if args.ic_command == "list":
                return _cmd_ic_list(args)
            if args.ic_command == "spin":
                return _cmd_ic_spin(args, recorded_argv)
        if args.command == "collect":
            return _cmd_collect(args)
        if args.command == "run":
            return _cmd_run(args, recorded_argv)
        if args.command == "submit":
            return _cmd_submit(args)
        if args.command == "verify-submit":
            return _cmd_verify_submit(args)
        if args.command == "exec-fme":
            return _cmd_exec_fme(args)
        if args.command == "lora":
            if args.lora_command == "info":
                return _cmd_lora_info(args)
            if args.lora_command == "scale":
                return _cmd_lora_scale(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "watch":
            return _cmd_watch(args)
        if args.command == "logs":
            return _cmd_logs(args)
        if args.command == "ls":
            return _cmd_ls(args)
        if args.command == "cancel":
            return _cmd_cancel(args)
        if args.command == "summary":
            return _cmd_summary(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "approve":
            return _cmd_approve(args)
        if args.command == "completion":
            return _cmd_completion(args)
        if args.command == "setup":
            return _cmd_setup(args)
        if args.command == "fetch":
            return _cmd_fetch(args)
        if args.command == "site":
            if args.site_command == "init":
                return _cmd_site_init(args)
            if args.site_command == "show":
                return _cmd_site_show(args)
    except AimipError as exc:
        enrich_error(exc)
        print(f"aimip error: {exc}", file=sys.stderr)
        if exc.allowed:
            shown_allowed = ", ".join(str(x) for x in exc.allowed[:12])
            if len(exc.allowed) > 12:
                shown_allowed += ", ..."
            print(f"  allowed: {shown_allowed}", file=sys.stderr)
        if exc.next_command:
            print(f"  next:    {exc.next_command}", file=sys.stderr)
        if _wants_json(args, effective_argv):
            print(
                json.dumps(
                    {"aimip": "1", "ok": False, "error": exc.to_dict()},
                    indent=2,
                )
            )
        return exc.exit_code

    parser.error(f"Unhandled command {args.command!r}")
    return 2
