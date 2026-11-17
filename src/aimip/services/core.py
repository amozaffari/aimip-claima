"""Shared AIMIP application service layer (`src/aimip/services/`).

Used directly in-process by the CLI (`aimip`), the LangGraph assistant (`aimip chat`),
and the MCP server (`aimip mcp`). Neither the agent nor MCP shells out to `aimip`.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import time
from typing import Any, NoReturn

import yaml

from .. import __version__
from ..errors import (
    AimipError,
    ApprovalError,
    ConfigError,
    ExecutionError,
    MissingAssetsError,
    ValidationError,
    enrich_error,
)
from ..families import get_family_adapter
from ..registry import (
    ModelSpec,
    get_model,
    get_model_ic_provenance,
    load_registry,
    resolve_data_dir,
    resolve_playground_dir,
)
from ..runner import (
    SlurmOptions,
    get_fme_version,
    model_has_missing_files,
    render_sbatch_script,
    resolve_model_for_prepare,
)
from ..setup_data import load_lock_file, resolve_lock_path
from ..site_config import SiteConfig, expand_run_root_template, load_site_config
from ..status import (
    _discover_job_id,
    _parse_submitted_txt,
    get_run_status,
    refresh_sacct_batch,
    summarize_run,
)
from .. import submit as _submit_mod
from ..submit import (
    check_ancestry,
    check_safe_path,
    check_trusted_run_dir,
    inspect_local_process,
    is_trusted_group_parent,
    is_trusted_run_dir,
    sanitized_subprocess_env,
    trusted_group_parents,
    submit_run,
    write_in_trusted_dir,
)
from ..validators import (
    _resolve_candidate_realpath,
    validate_model_name,
    validate_run_path,
    validate_safe_path,
)
from .approval import (
    MemoryApprovalBroker,
    can_download_into_data_dir,
    current_user,
    format_bytes,
    read_sealed_prepared,
    record_sealed_prepared,
)
from .cards import render_plan_card, sanitize_terminal_text
from .metadata import card_for_model
from .types import (
    AssetRef,
    Attempt,
    Authorization,
    PreparedRun,
    ResolvedPlan,
    Result,
    RunSpec,
    compute_plan_digest,
)

_RUN_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
TERMINAL_RUN_STATES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "EXITED",
        "completed",
        "failed",
        "cancelled",
    }
)


def validate_run_name(name: str) -> str:
    """Validate a portable run name (`--name demo`)."""
    if not isinstance(name, str) or not _RUN_NAME_RE.match(name):
        raise ValidationError(
            f"Invalid run name {name!r}: use 1-64 characters from [A-Za-z0-9._-], starting with alphanumeric.",
            field="name",
        )
    return name


def _check_existing_ancestry(path: str, *, label: str, field: str) -> None:
    """Verify existing parent directories of *path* are safe (no foreign symlinks or non-sticky group/world write).

    Same rule as `submit.check_ancestry`, including its MN5 exception: group directories listed in `site.json`
    `trusted_group_parents` (root-owned, group- but not world-writable) are accepted.
    """
    euid = os.geteuid()
    trusted = trusted_group_parents()
    cur = path
    while True:
        parent = os.path.dirname(cur)
        if not parent or parent == cur or parent == "/":
            break
        cur = parent
        if not os.path.lexists(cur):
            continue
        try:
            ast = os.lstat(cur)
        except OSError as exc:
            raise ValidationError(
                f"Cannot inspect parent directory {cur} for {label}: {exc}",
                field=field,
            ) from exc
        if stat.S_ISLNK(ast.st_mode):
            if ast.st_uid not in (0, euid):
                raise ValidationError(
                    f"A symlink owned by another user is in the {label} path: {cur}",
                    field=field,
                )
            continue
        if (ast.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) and not (ast.st_mode & stat.S_ISVTX):
            if is_trusted_group_parent(cur, ast, trusted):
                continue  # MN5 group directory listed in site.json (see submit.trusted_group_parents)
            mode_str = stat.filemode(ast.st_mode)
            raise ValidationError(
                f"A parent directory is writable by others without sticky bit ({mode_str}): {cur}",
                field=field,
            )


def _canonicalize_run_path(
    path: str,
    *,
    label: str,
    playground_dir: str | None = None,
    for_output: bool = True,
    allow_leaf_symlink: bool = False,
) -> str:
    """Resolve and validate the canonical physical (`realpath`) run path, checking leaf symlinks and ancestry."""
    field = "out" if for_output else "run"
    safe = validate_safe_path(path, label=label)
    if os.path.islink(safe):
        if not allow_leaf_symlink:
            raise ValidationError(f"refusing a symlinked run directory: {safe}", field=field)
        try:
            lst = os.lstat(safe)
        except OSError as exc:
            raise ValidationError(f"Cannot inspect {label} symlink {safe}: {exc}", field=field) from exc
        if lst.st_uid not in (0, os.geteuid()):
            raise ValidationError(
                f"A symlink owned by another user is in the {label} path: {safe}",
                field=field,
            )

    resolved_pg = resolve_playground_dir(playground_dir)

    def _reject_inside_playground(candidate: str) -> None:
        if for_output and resolved_pg:
            pg = resolved_pg.rstrip("/")
            if candidate == pg or candidate.startswith(pg + "/"):
                raise ConfigError(
                    f"Refusing to write run outputs ({candidate}) inside "
                    f"$AIMIP_PLAYGROUND ({pg}); choose an --out directory outside the playground.",
                    field="out",
                )

    _reject_inside_playground(safe)
    if for_output:
        _check_existing_ancestry(safe, label=label, field=field)
        validate_run_path(safe, label=label)

    phys = _resolve_candidate_realpath(safe, label=label)
    phys = validate_safe_path(phys, label=label)
    if os.path.islink(phys):
        raise ValidationError(f"refusing a symlinked run directory: {phys}", field=field)

    if for_output:
        _check_existing_ancestry(phys, label=label, field=field)
        validate_run_path(phys, label=label)
        _reject_inside_playground(phys)
        return phys

    if os.path.lexists(os.path.join(phys, "plan.json")) or os.path.lexists(
        os.path.join(phys, "aimip-manifest.json")
    ):
        _check_existing_ancestry(safe, label=label, field=field)
        _check_existing_ancestry(phys, label=label, field=field)
        return phys
    return safe


def resolve_configured_run_root(*, site_config_path: str | None = None) -> str:
    """Resolve the active canonical writable `<run_root>` from `$AIMIP_DEFAULT_OUT_ROOT` or `site.json`."""
    env_root = os.environ.get("AIMIP_DEFAULT_OUT_ROOT")
    if env_root:
        validated = validate_run_path(env_root, label="AIMIP_DEFAULT_OUT_ROOT")
        return _canonicalize_run_path(
            validated,
            label="AIMIP_DEFAULT_OUT_ROOT",
            for_output=True,
            allow_leaf_symlink=True,
        )
    cfg = load_site_config(site_config_path)
    expanded = expand_run_root_template(cfg.run_root)
    if not expanded:
        raise ConfigError(
            f"Cannot expand site.json run_root {cfg.run_root!r}; run `aimip site init` to configure a valid run root.",
            field="run_root",
        )
    validated = validate_run_path(expanded, label="site run_root")
    return _canonicalize_run_path(
        validated,
        label="site run_root",
        for_output=True,
        allow_leaf_symlink=True,
    )


def resolve_run_target(
    run_or_name: str,
    *,
    site_config_path: str | None = None,
    playground_dir: str | None = None,
    for_output: bool = False,
) -> tuple[str, str]:
    """Accept either a bare run name (`demo` -> `<run_root>/demo`) or an absolute path.

    Returns `(canonical_validated_abs_path, display_name)`.
    """
    if not isinstance(run_or_name, str) or not run_or_name.strip():
        raise ValidationError("Run name or path must be a non-empty string.", field="run")
    raw = run_or_name.strip()

    if "/" not in raw:
        valid_name = validate_run_name(raw)
        root = resolve_configured_run_root(site_config_path=site_config_path)
        full = _canonicalize_run_path(
            os.path.join(root, valid_name),
            label="run output directory" if for_output else "run directory",
            playground_dir=playground_dir,
            for_output=for_output,
        )
        return full, valid_name

    canon = _canonicalize_run_path(
        raw,
        label="--out" if for_output else "run directory",
        playground_dir=playground_dir,
        for_output=for_output,
    )
    return canon, os.path.basename(canon.rstrip("/"))


def _build_synchronized_default_catalog(
    *,
    playground_dir: str | None,
    data_dir: str | None,
) -> dict[str, ModelSpec]:
    """Build canonical fallback `ModelSpec`s synchronized with active `SetupYearConfig` and `aimip.lock`.

    Guarantees that `plan()` before `fetch_model()` produces the exact same `required_files`,
    `checkpoint_sha256`, and `asset_locks` digest as `plan()` after `fetch_model()` writes
    `<AIMIP_DATA>/registry.json`.
    """
    from ..families import _load_lock_entries
    from ..registry import DATA_PLACEHOLDER, _parse_model_entry, build_default_registry_payload
    from ..setup_data import SetupYearConfig, build_artifact_catalog

    art_catalog = build_artifact_catalog(SetupYearConfig())
    lock_files = _load_lock_entries(data_dir=data_dir)
    payload = build_default_registry_payload(root_placeholder=DATA_PLACEHOLDER)
    placeholder_prefix = f"{DATA_PLACEHOLDER}/"

    out: dict[str, ModelSpec] = {}
    for raw_entry in payload["models"]:
        entry = dict(raw_entry)
        m_name = str(entry["name"])
        entry["required_files"] = [
            rel
            for rel, sp in art_catalog.items()
            if m_name in sp.models_output and not sp.pin_after_first_fetch
        ]
        ckpt_placeholder = str(entry.get("checkpoint") or "")
        if ckpt_placeholder.startswith(placeholder_prefix):
            ckpt_rel = ckpt_placeholder[len(placeholder_prefix) :]
            lock_item = lock_files.get(ckpt_rel)
            if isinstance(lock_item, dict) and lock_item.get("sha256"):
                entry["checkpoint_sha256"] = str(lock_item["sha256"])
        spec = _parse_model_entry(entry, playground_dir, data_dir)
        out[spec.name] = spec
    return out


def _load_catalog_for_discovery(
    *,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
) -> dict[str, ModelSpec]:
    """Load model catalog; when `<AIMIP_DATA>/registry.json` is unfetched or a partial subset and no explicit `--registry`
    was given, merge missing canonical entries synchronized with `SetupYearConfig` and `aimip.lock`."""
    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    primary = load_registry(
        registry_path=registry_path,
        playground_dir=resolved_pg,
        data_dir=resolved_data,
    )
    if registry_path is not None or os.environ.get("AIMIP_REGISTRY"):
        return primary

    from ..registry import PACKAGE_DEFAULT_REGISTRY_PATH, resolve_registry_path

    resolved_reg_path = resolve_registry_path(registry_path, data_dir=resolved_data)
    used_packaged_fallback = os.path.normpath(resolved_reg_path) == os.path.normpath(
        PACKAGE_DEFAULT_REGISTRY_PATH
    )
    # If an on-disk `$AIMIP_DATA/registry.json` is explicitly empty (`{"models": []}`), preserve empty catalog semantics.
    if not used_packaged_fallback and not primary:
        return primary

    if resolved_data:
        try:
            full_default = _build_synchronized_default_catalog(
                playground_dir=resolved_pg,
                data_dir=resolved_data,
            )
            if used_packaged_fallback:
                return full_default
            merged = dict(full_default)
            merged.update(primary)
            return merged
        except AimipError:
            return primary
    return primary


def models(
    *,
    task: str | None = None,
    family: str | None = None,
    status: str | None = None,
    installed_only: bool = False,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """List registered models with optional `--task`, `--family`, `--status`, and `--installed` filters."""
    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    catalog = _load_catalog_for_discovery(
        registry_path=registry_path,
        playground_dir=resolved_pg,
        data_dir=resolved_data,
    )

    allowed_tasks = sorted({card_for_model(m)["task"] for m in catalog} | {"amip", "slab-ocean", "coupled", "scenario"})
    allowed_families = sorted({spec.family for spec in catalog.values()} | {"fme", "e2s-dlesym", "arches", "emu"})
    allowed_statuses = ["certified", "runs", "experimental"]

    if task is not None and task not in allowed_tasks:
        raise ValidationError(
            f"Unknown --task filter {task!r}.",
            field="task",
            allowed=allowed_tasks,
            next_command="aimip models",
        )
    if family is not None and family not in allowed_families:
        raise ValidationError(
            f"Unknown --family filter {family!r}.",
            field="family",
            allowed=allowed_families,
            next_command="aimip models",
        )
    if status is not None and status not in allowed_statuses:
        raise ValidationError(
            f"Unknown --status filter {status!r}.",
            field="status",
            allowed=allowed_statuses,
            next_command="aimip models",
        )

    from ..ic import inspect_ic_file

    lock_files: dict[str, Any] = {}
    try:
        lp = resolve_lock_path(dest_dir=resolved_data, must_exist=False)
        if lp and os.path.isfile(lp):
            lock_files = load_lock_file(lp).get("files") or {}
    except Exception:
        lock_files = {}

    rows: list[dict[str, Any]] = []
    for spec in catalog.values():
        card = card_for_model(spec.name)
        fam_match = (
            family is None
            or spec.family == family
            or (family == "emu" and spec.family in ("e2s-dlesym", "arches"))
        )
        if not fam_match:
            continue
        if task is not None and card["task"] != task:
            continue
        if status is not None and card["status"] != status:
            continue

        missing = model_has_missing_files(spec, data_dir=resolved_data)
        is_installed = len(missing) == 0 and os.path.isfile(spec.checkpoint)
        if installed_only and not is_installed:
            continue

        ic_times: list[str] = []
        for ic_p in spec.ic_files:
            if os.path.isfile(ic_p):
                try:
                    info = inspect_ic_file(ic_p)
                    for t in info.timestamps:
                        if t not in ic_times:
                            ic_times.append(t)
                except Exception:
                    pass
        for fs in spec.forcing_sets:
            for ic_p in fs.ic_files:
                if os.path.isfile(ic_p):
                    try:
                        info = inspect_ic_file(ic_p)
                        for t in info.timestamps:
                            if t not in ic_times:
                                ic_times.append(t)
                    except Exception:
                        pass

        total_bytes = sum(
            int((lock_files.get(rel) or {}).get("size") or 0)
            for rel in spec.required_files
        )
        rows.append(
            {
                "name": spec.name,
                "title": card["title"],
                "family": spec.family,
                "task": card["task"],
                "status": card["status"],
                "licence": card["licence"],
                "fme_version": spec.fme_version,
                "fme_module": spec.fme_module,
                "is_coupled": spec.is_coupled,
                "calendar": spec.calendar,
                "timestep_hours": spec.timestep_hours,
                "stochastic": spec.stochastic,
                "default_n_ensemble_per_ic": spec.default_n_ensemble_per_ic,
                "checkpoint": spec.checkpoint,
                "checkpoint_exists": os.path.isfile(spec.checkpoint),
                "installed": is_installed,
                "missing_files_count": len(missing),
                "total_bytes": total_bytes,
                "forcing_sets": [fs.to_dict() for fs in spec.forcing_sets],
                "ic_files": list(spec.ic_files),
                "ocean_ic_files": list(spec.ocean_ic_files),
                "ic_provenance": get_model_ic_provenance(spec.name),
                "available_ic_timestamps": ic_times,
                "notes": spec.notes,
            }
        )
    return rows


def show(
    model_name: str,
    *,
    include_examples: bool = False,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Return the complete declarative model card for *model_name*."""
    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    catalog = _load_catalog_for_discovery(
        registry_path=registry_path,
        playground_dir=resolved_pg,
        data_dir=resolved_data,
    )
    validated = validate_model_name(model_name, allowed=set(catalog.keys()))
    spec = catalog[validated]
    card = card_for_model(spec.name)
    adapter = get_family_adapter(spec.family)

    lock_files: dict[str, Any] = {}
    try:
        lp = resolve_lock_path(dest_dir=resolved_data, must_exist=False)
        if lp and os.path.isfile(lp):
            lock_files = load_lock_file(lp).get("files") or {}
    except Exception:
        lock_files = {}

    missing = model_has_missing_files(spec, data_dir=resolved_data)
    total_files = len(spec.required_files) + (1 if spec.family == "arches" else 0)
    cached_files = max(0, total_files - len(missing))
    total_bytes = sum(
        int((lock_files.get(rel) or {}).get("size") or 0) for rel in spec.required_files
    )

    result: dict[str, Any] = {
        "name": spec.name,
        "title": card["title"],
        "family": spec.family,
        "adapter": adapter.family_id,
        "task": card["task"],
        "status": card["status"],
        "status_definition": card.get("status_definition", ""),
        "verified_on": card.get("verified_on"),
        "licence": card["licence"],
        "upstream_url": card.get("upstream_url"),
        "data_urls": card.get("data_urls", []),
        "grid": card["grid"],
        "calendar": spec.calendar,
        "timestep_hours": spec.timestep_hours,
        "stochastic": spec.stochastic,
        "is_coupled": spec.is_coupled,
        "fme_version": spec.fme_version,
        "forcing_sets": [fs.to_dict() for fs in spec.forcing_sets],
        "ic_provenance": get_model_ic_provenance(spec.name),
        "outputs": card["outputs"],
        "cost_note": card["cost_note"],
        "limitations": card["limitations"],
        "notes": spec.notes,
        "installed": len(missing) == 0 and os.path.isfile(spec.checkpoint),
        "cached_files": cached_files,
        "total_files": total_files,
        "missing_files": missing,
        "total_bytes": total_bytes,
        "examples": card["examples"] if include_examples else card["examples"][:1],
    }
    return result


def cite(
    model_name: str,
    *,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Return verified BibTeX entries (or upstream URL when no published paper is confirmed) for *model_name*."""
    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    catalog = _load_catalog_for_discovery(
        registry_path=registry_path,
        playground_dir=resolved_pg,
        data_dir=resolved_data,
    )
    validated = validate_model_name(model_name, allowed=set(catalog.keys()))
    card = card_for_model(validated)
    entries = list(card["bibtex"])
    upstream_url = card.get("upstream_url")
    data_urls = list(card.get("data_urls") or ())
    citation_note = card.get("citation_note")
    if entries:
        body_text = "\n\n".join(entries)
        if citation_note:
            note_lines = [f"% Note: {citation_note}"]
            if upstream_url:
                note_lines.append(f"% Upstream card: {upstream_url}")
            bibtex_text = "\n".join(note_lines) + "\n" + body_text
        else:
            bibtex_text = body_text
    elif upstream_url:
        bibtex_text = f"No published reference confirmed; cite upstream URL: {upstream_url}"
    else:
        bibtex_text = "No published reference confirmed for this model."
    return {
        "model": validated,
        "title": card["title"],
        "licence": card["licence"],
        "confirmed": bool(entries),
        "upstream_url": upstream_url,
        "data_urls": data_urls,
        "note": citation_note,
        "bibtex": bibtex_text,
        "entries": entries,
    }


def _build_slurm_options(spec: RunSpec, site: SiteConfig, *, require_valid_slurm: bool) -> SlurmOptions:
    res = spec.resources or {}
    effective_site = (
        dataclasses.replace(site, scheduler="none")
        if not require_valid_slurm
        else site
    )
    return SlurmOptions.from_site(
        effective_site,
        partition=str(res["partition"]) if res.get("partition") is not None else None,
        account=str(res["account"]) if res.get("account") is not None else None,
        gres=str(res["gres"]) if res.get("gres") is not None else None,
        cpus_per_task=int(res["cpus_per_task"]) if res.get("cpus_per_task") is not None else None,
        mem=str(res["mem"]) if res.get("mem") is not None else None,
        time=str(res["time"]) if res.get("time") is not None else None,
        job_name=str(res["job_name"]) if res.get("job_name") is not None else None,
        qos=str(res["qos"]) if res.get("qos") is not None else None,
    )


def plan(
    spec: RunSpec,
    *,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    site_config_path: str | None = None,
) -> ResolvedPlan:
    """Resolve and validate *spec* into a `ResolvedPlan` with ZERO side effects and ZERO downloads."""
    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    catalog = _load_catalog_for_discovery(
        registry_path=registry_path,
        playground_dir=resolved_pg,
        data_dir=resolved_data,
    )
    try:
        validated_name = validate_model_name(spec.model, allowed=set(catalog.keys()))
    except ValidationError as exc:
        raise enrich_error(exc, model=spec.model, allowed_models=sorted(catalog.keys())) from exc

    model_spec = catalog[validated_name]
    adapter = get_family_adapter(model_spec.family)
    site = load_site_config(site_config_path)

    if spec.mode not in ("slurm", "local"):
        raise ValidationError(
            f"Unsupported run mode {spec.mode!r}.",
            field="mode",
            allowed=["slurm", "local"],
        )
    if spec.name and spec.out:
        raise ValidationError(
            "Options --name and --out are mutually exclusive; pass only one.",
            field="out",
        )

    slurm_opts = _build_slurm_options(spec, site, require_valid_slurm=(spec.mode != "local"))

    try:
        validated = adapter.validate(model_spec, spec, data_dir=resolved_data)
    except AimipError as exc:
        raise enrich_error(exc, model=model_spec.name) from exc

    if slurm_opts.time:
        validated["requested_walltime"] = slurm_opts.time
    est = adapter.estimate(model_spec, validated, site)

    run_name: str | None = None
    if spec.name:
        output_dir, run_name = resolve_run_target(
            spec.name,
            site_config_path=site_config_path,
            playground_dir=resolved_pg,
            for_output=True,
        )
    elif spec.out:
        output_dir, run_name = resolve_run_target(
            spec.out,
            site_config_path=site_config_path,
            playground_dir=resolved_pg,
            for_output=True,
        )
    else:
        rroot = resolve_configured_run_root(site_config_path=site_config_path).rstrip("/")
        start_slug = str(validated["norm_start"]).replace(":", "")
        det_seed = (
            f"{model_spec.name}|{validated['norm_start']}|{validated['steps']}|"
            f"{validated['members']}|{validated['seed']}|{spec.ic}|{spec.ocean_ic}|"
            f"{validated.get('lora_checkpoint')}|{spec.forcing_set}"
        )
        short_hash = hashlib.sha256(det_seed.encode("utf-8")).hexdigest()[:8]
        output_dir = _canonicalize_run_path(
            f"{rroot}/{model_spec.name}-{start_slug}-{short_hash}",
            label="default --out directory",
            playground_dir=resolved_pg,
            for_output=True,
        )
        run_name = os.path.basename(output_dir.rstrip("/"))

    assets = tuple(
        adapter.stage_list(
            model_spec,
            {**validated, "output_dir": output_dir},
            data_dir=resolved_data,
        )
    )
    missing_assets = tuple(a for a in assets if not a.cached)
    missing_bytes = sum(int(a.size_bytes or 0) for a in missing_assets)
    staged_dyn_ic = next(
        (
            a
            for a in assets
            if a.role in ("dynamic_era5_ic", "dynamic_ic", "staged_ic")
            and not a.prospective
            and a.sha256
        ),
        None,
    )
    eff_ic_source = (
        staged_dyn_ic.path if staged_dyn_ic is not None else validated.get("ic_source")
    )
    eff_ic_sha256 = (
        staged_dyn_ic.sha256 if staged_dyn_ic is not None else validated.get("ic_sha256")
    )

    job_name = slurm_opts.job_name or f"aimip-{model_spec.name}"[:64]
    resources_dict: dict[str, Any] = {
        "scheduler": "none" if spec.mode == "local" else site.scheduler,
        "job_name": job_name,
        "partition": slurm_opts.partition,
        "account": slurm_opts.account,
        "gres": slurm_opts.gres,
        "cpus_per_task": slurm_opts.cpus_per_task,
        "mem": slurm_opts.mem,
        "time": est["time"],
        "walltime_uncapped": est["walltime_uncapped"],
        "walltime_capped": est["walltime_capped"],
        "gpu_hours_estimate": est["gpu_hours_estimate"],
        "estimate_basis": est["estimate_basis"],
    }
    if slurm_opts.qos:
        # MN5: the QOS is part of what the person approves. Added only when set, so plan digests on sites without
        # a QOS stay exactly as they were.
        resources_dict["qos"] = slurm_opts.qos

    emu_py = (os.environ.get("AIMIP_EMU_PYTHON") or "").strip() if model_spec.family != "fme" else None
    fme_overlay = (
        (os.environ.get("AIMIP_FME_2026_10_OVERLAY") or "").strip()
        if (model_spec.family == "fme" and model_spec.fme_version == "2026.10.0")
        else None
    )
    arches_code_override = (
        (os.environ.get("AIMIP_ARCHES_CODE") or "").strip()
        if model_spec.family == "arches"
        else None
    )

    runtime_identity: dict[str, Any] = {
        "aimip_version": __version__,
        "site_name": site.name,
        "runtime": site.runtime,
        "scheduler": site.scheduler,
        "image_sha": os.environ.get("AIMIP_IMAGE_SHA") or "unknown",
        "fme_version": model_spec.fme_version if model_spec.family == "fme" else None,
        "installed_fme_version": get_fme_version() if model_spec.family == "fme" else None,
        "emu_python": emu_py or None,
        "fme_overlay": fme_overlay or None,
        "arches_code_override": arches_code_override or None,
    }

    fs_obj = validated.get("forcing_set")
    fs_name = fs_obj.name if hasattr(fs_obj, "name") else (str(fs_obj) if fs_obj else None)
    fs_dir = getattr(fs_obj, "dir", None) if fs_obj is not None else None
    fs_ocean_dir = getattr(fs_obj, "ocean_dir", None) if fs_obj is not None else None

    eff_calendar = str(validated.get("calendar") or model_spec.calendar)

    canonical_payload: dict[str, Any] = {
        "schema": "aimip-plan-v1",
        "model": model_spec.name,
        "family": model_spec.family,
        "adapter": adapter.family_id,
        "fme_module": model_spec.fme_module,
        "calendar": eff_calendar,
        "timestep_hours": model_spec.timestep_hours,
        "start_time": validated["norm_start"],
        "end_time": validated["end_time"],
        "steps": int(validated["steps"]),
        "members": int(validated["members"]),
        "seed": int(validated["seed"]),
        "ensemble_strategy": validated["ensemble_strategy"],
        "forcing_set": fs_name,
        "forcing_dir": fs_dir,
        "ocean_forcing_dir": fs_ocean_dir,
        "ic_source": eff_ic_source,
        "ic_sha256": eff_ic_sha256,
        "ocean_ic_source": validated.get("ocean_ic_source"),
        "ocean_ic_sha256": validated.get("ocean_ic_sha256"),
        "lora_checkpoint": validated.get("lora_checkpoint"),
        "lora_sha256": validated.get("lora_sha256"),
        "monthly": bool(spec.monthly),
        "prediction": bool(spec.prediction),
        "mode": spec.mode,
        "output_dir": output_dir,
        "resources": {
            "scheduler": resources_dict["scheduler"],
            "job_name": resources_dict["job_name"],
            "partition": resources_dict["partition"],
            "account": resources_dict["account"],
            "gres": resources_dict["gres"],
            "cpus_per_task": resources_dict["cpus_per_task"],
            "mem": resources_dict["mem"],
            "time": resources_dict["time"],
            **({"qos": resources_dict["qos"]} if "qos" in resources_dict else {}),  # MN5; see above
        },
        "runtime_identity": {
            "runtime": runtime_identity["runtime"],
            "image_sha": runtime_identity["image_sha"],
            "fme_version": runtime_identity["fme_version"],
            "emu_python": runtime_identity["emu_python"],
            "fme_overlay": runtime_identity["fme_overlay"],
            "arches_code_override": runtime_identity["arches_code_override"],
        },
        "asset_locks": {
            f"{a.role}:{a.path}": a.sha256
            for a in assets
            if not a.prospective
        },
    }
    digest = compute_plan_digest(canonical_payload)
    card_meta = card_for_model(model_spec.name)
    ic_prov = (
        f"Custom IC override: {spec.ic}"
        if spec.ic
        else get_model_ic_provenance(model_spec.name, model_spec.notes)
    )

    return ResolvedPlan(
        spec=spec,
        model=model_spec.name,
        family=model_spec.family,
        status_tier=card_meta["status"],
        task=card_meta["task"],
        calendar=eff_calendar,
        timestep_hours=model_spec.timestep_hours,
        start_time=validated["norm_start"],
        end_time=validated["end_time"],
        steps=int(validated["steps"]),
        members=int(validated["members"]),
        seed=int(validated["seed"]),
        ensemble_strategy=validated["ensemble_strategy"],
        ensemble_note=validated["ensemble_note"],
        forcing_set=fs_name,
        ic_source=eff_ic_source,
        ocean_ic_source=validated.get("ocean_ic_source"),
        lora_checkpoint=validated.get("lora_checkpoint"),
        resources=resources_dict,
        runtime_identity=runtime_identity,
        assets=assets,
        missing_assets=missing_assets,
        missing_bytes=missing_bytes,
        output_dir=output_dir,
        run_name=run_name,
        cli_overrides=spec.cli_overrides,
        canonical_payload=canonical_payload,
        digest=digest,
        ic_checked=bool(validated.get("ic_checked", True)),
        ic_provenance=ic_prov,
        limitations=str(card_meta.get("limitations") or ""),
    )


_SEALED_EXECUTION_ARTIFACTS: tuple[str, ...] = (
    "config.yaml",
    "emu-job.json",
    "aimip-manifest.json",
    "job.sbatch",
    "ic_era5.pt",
)
_HEX64_DIGEST_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def _is_safe_sealed_artifact_name(fname: str) -> bool:
    if fname in _SEALED_EXECUTION_ARTIFACTS:
        return True
    return (
        bool(fname)
        and "/" not in fname
        and "\\" not in fname
        and fname not in (".", "..", "plan.json", "experiment.yaml")
        and not fname.startswith(".")
    )


def _raise_no_fetch_missing_assets(spec: RunSpec, resolved: ResolvedPlan) -> NoReturn:
    """Raise `MissingAssetsError` when `--no-fetch` / `allow_fetch=False` forbids downloading missing assets."""
    non_prospective_missing = [a for a in resolved.missing_assets if not a.prospective]
    if non_prospective_missing:
        shown = ", ".join(a.rel_path for a in non_prospective_missing[:5])
        raise MissingAssetsError(
            f"Model {spec.model!r} is missing {len(non_prospective_missing)} required file(s) "
            f"(~{format_bytes(resolved.missing_bytes)}: {shown}) and --no-fetch forbids automatic downloads. "
            f"Run `aimip fetch -m {spec.model}` first or omit --no-fetch.",
            field="model",
            next_command=f"aimip fetch -m {spec.model}",
        )
    raise MissingAssetsError(
        f"Model {spec.model!r} needs to download initial state data (~{format_bytes(resolved.missing_bytes)}) "
        "during preparation, and --no-fetch forbids downloads.",
        field="model",
        next_command=f"aimip run {spec.model} --start {resolved.start_time} --steps {resolved.steps}",
    )


def prepare(
    spec: RunSpec,
    *,
    allow_fetch: bool | None = None,
    execute_local: bool = False,
    authorization: Authorization | None = None,
    require_authorization: bool = False,
    broker: MemoryApprovalBroker | None = None,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    site_config_path: str | None = None,
    manifest_mode: str | None = None,
    argv: list[str] | None = None,
) -> PreparedRun:
    """Materialize a run directory on disk (`config.yaml`, `aimip-manifest.json`, `job.sbatch`, `plan.json`, `experiment.yaml`).

    Side-effect-free with respect to job execution unless `execute_local=True` is explicitly requested
    on the unauthenticated CLI `--local` path. When `require_authorization=True` or `authorization` is
    passed, inline execution is always prohibited; prospective staging plans require an
    `Authorization(action='prepare')` (consumed here, never usable for `submit`), and execution must go
    through `submit()` with a final `action='submit'` authorization.
    """
    try:
        if execute_local and (require_authorization or authorization is not None):
            raise ApprovalError(
                "Authorization-controlled preparation cannot execute locally inline; "
                "prepare with `execute_local=False` and run execution through `submit()` with a final submit authorization.",
                field="execute_local",
            )

        initial_plan = plan(
            spec,
            registry_path=registry_path,
            playground_dir=playground_dir,
            data_dir=data_dir,
            site_config_path=site_config_path,
        )
        canonical_out = _canonicalize_run_path(
            initial_plan.output_dir,
            label="run output directory",
            playground_dir=playground_dir,
            for_output=True,
        )
        if canonical_out != initial_plan.output_dir:
            raise ApprovalError(
                "Plan changed, approve again (output path resolved to a different destination).",
                field="out",
            )

        if require_authorization or authorization is not None:
            if authorization is None:
                raise ApprovalError(
                    "Preparation requires explicit human approval of the plan digest.",
                    field="authorization",
                )
            now_check = float(broker._now_fn()) if broker is not None else time.time()
            if initial_plan.requires_staging:
                if broker is not None:
                    broker.consume(
                        authorization,
                        expected_action="prepare",
                        expected_digest=initial_plan.digest,
                    )
                else:
                    MemoryApprovalBroker.verify_authorization_fields(
                        authorization,
                        expected_action="prepare",
                        expected_digest=initial_plan.digest,
                        now_ts=now_check,
                    )
            elif authorization.action == "prepare":
                if broker is not None:
                    broker.consume(
                        authorization,
                        expected_action="prepare",
                        expected_digest=initial_plan.digest,
                    )
                else:
                    MemoryApprovalBroker.verify_authorization_fields(
                        authorization,
                        expected_action="prepare",
                        expected_digest=initial_plan.digest,
                        now_ts=now_check,
                    )
            else:
                if broker is not None and broker._tokens.get(authorization.nonce) != authorization:
                    raise ApprovalError(
                        "Authorization was already used, revoked, or not minted by this host broker.",
                        field="authorization",
                    )
                MemoryApprovalBroker.verify_authorization_fields(
                    authorization,
                    expected_action="submit",
                    expected_digest=initial_plan.digest,
                    now_ts=now_check,
                )

            if allow_fetch is False and initial_plan.missing_assets:
                _raise_no_fetch_missing_assets(spec, initial_plan)
            if initial_plan.missing_assets and not authorization.allow_download:
                raise ApprovalError(
                    "Plan requires downloading/staging missing assets, and separate download consent was not granted.",
                    field="allow_download",
                )
            effective_allow_fetch = bool(authorization.allow_download) and (allow_fetch is not False)
        else:
            effective_allow_fetch = bool(allow_fetch)
            if initial_plan.missing_assets and not effective_allow_fetch:
                _raise_no_fetch_missing_assets(spec, initial_plan)

        resolved_pg = resolve_playground_dir(playground_dir)
        resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
        model_spec = resolve_model_for_prepare(
            spec.model,
            registry_path=registry_path,
            playground_dir=resolved_pg,
            data_dir=resolved_data,
            auto_fetch=effective_allow_fetch,
        )
        adapter = get_family_adapter(model_spec.family)
        site = load_site_config(site_config_path)
        slurm_opts = _build_slurm_options(spec, site, require_valid_slurm=(spec.mode != "local"))
        validated = adapter.validate(model_spec, spec, data_dir=resolved_data)

        eff_manifest_mode = "dry-run" if spec.mode == "local" else (manifest_mode or "dry-run")
        prep_lock_fd: int | None = None
        if os.path.isdir(canonical_out) and not os.path.islink(canonical_out):
            prep_lock_fd = _submit_mod._acquire_run_submit_lock(canonical_out)
        try:
            manifest = adapter.build_config(
                model_spec,
                spec,
                validated,
                out_dir=canonical_out,
                allow_fetch=effective_allow_fetch,
                slurm_opts=slurm_opts,
                site_config=site,
                playground_dir=resolved_pg,
                data_dir=resolved_data,
                manifest_mode=eff_manifest_mode,
                argv=argv,
            )
            if prep_lock_fd is None and os.path.isdir(canonical_out) and not os.path.islink(canonical_out):
                prep_lock_fd = _submit_mod._acquire_run_submit_lock(canonical_out)

            # Re-resolve `plan(...)` after staging on the bound physical `canonical_out` path so `plan.json`
            # carries exact post-stage IC digests, `missing_assets == ()`, and the immutable physical destination.
            resolved_spec = RunSpec(
                **{
                    **spec.to_dict(),
                    "out": canonical_out,
                    "name": None,
                    "cli_overrides": spec.cli_overrides,
                }
            )
            final_plan = plan(
                resolved_spec,
                registry_path=registry_path,
                playground_dir=resolved_pg,
                data_dir=resolved_data,
                site_config_path=site_config_path,
            )
            if spec.name:
                final_plan = dataclasses.replace(final_plan, run_name=spec.name, spec=resolved_spec)

            from ..lora import sha256_file

            trusted_out = check_trusted_run_dir(canonical_out)

            if spec.mode == "local":
                manifest["mode"] = "local"
                write_in_trusted_dir(
                    trusted_out,
                    "aimip-manifest.json",
                    json.dumps(manifest, indent=2),
                )

            if execute_local and spec.mode == "local":
                if require_authorization or authorization is not None:
                    raise ApprovalError(
                        "Authorization-controlled preparation cannot execute locally inline.",
                        field="execute_local",
                    )
                manifest = adapter.execute(trusted_out)

            artifact_hashes: dict[str, str] = {}
            for fname in _SEALED_EXECUTION_ARTIFACTS:
                cand = os.path.join(trusted_out, fname)
                if os.path.isfile(cand) and not os.path.islink(cand):
                    artifact_hashes[fname] = sha256_file(cand)
            trusted_prefix = trusted_out.rstrip("/") + "/"
            for asset in final_plan.assets:
                if not asset.prospective and asset.path.startswith(trusted_prefix):
                    rel_leaf = asset.path[len(trusted_prefix) :]
                    if (
                        _is_safe_sealed_artifact_name(rel_leaf)
                        and rel_leaf not in artifact_hashes
                        and os.path.isfile(asset.path)
                        and not os.path.islink(asset.path)
                    ):
                        artifact_hashes[rel_leaf] = sha256_file(asset.path)

            record_sealed_prepared(
                digest=final_plan.digest,
                run_dir=trusted_out,
                artifact_hashes=artifact_hashes,
                site_config_path=site_config_path,
            )

            plan_doc = final_plan.to_dict()
            plan_doc["artifact_hashes"] = dict(artifact_hashes)
            plan_path = write_in_trusted_dir(
                trusted_out,
                "plan.json",
                json.dumps(plan_doc, indent=2),
            )
            write_in_trusted_dir(
                trusted_out,
                "experiment.yaml",
                yaml.safe_dump(spec.to_dict(), sort_keys=False),
            )

            return PreparedRun(
                plan=final_plan,
                run_dir=trusted_out,
                plan_path=plan_path,
                manifest_path=os.path.join(trusted_out, "aimip-manifest.json"),
                config_path=os.path.join(trusted_out, "config.yaml"),
                sbatch_path=os.path.join(trusted_out, "job.sbatch"),
                manifest=manifest,
                artifact_hashes=dict(artifact_hashes),
            )
        finally:
            if prep_lock_fd is not None:
                try:
                    fcntl.flock(prep_lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(prep_lock_fd)
    except BaseException:
        if broker is not None:
            broker.burn_token(authorization)
        raise


def verify_prepared_run_for_submit(
    run_dir: str,
    *,
    require_plan: bool = False,
    expected_digest: str | None = None,
    in_memory_hashes: dict[str, str] | None = None,
    script_override_path: str | None = None,
    playground_dir: str | None = None,
    registry_path: str | None = None,
    data_dir: str | None = None,
    site_config_path: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Verify `plan.json` schema, sealed `artifact_hashes`, live re-plan digest, and exact `job.sbatch` rendering.

    Returns `(trusted_dir, saved_digest, saved_mode)`. Raises `ApprovalError` on any drift, missing external
    seal, missing integrity fields, or tampered execution artifacts.
    """
    from ..lora import sha256_file

    trusted_dir = check_trusted_run_dir(run_dir)
    plan_file = os.path.join(trusted_dir, "plan.json")
    has_manifest = os.path.lexists(os.path.join(trusted_dir, "aimip-manifest.json"))
    norm_expected = expected_digest.strip().lower() if expected_digest else None
    if norm_expected and not _HEX64_DIGEST_RE.match(norm_expected):
        raise ValidationError(f"Invalid --expect-digest {expected_digest!r}", field="expect_digest")

    if not os.path.lexists(plan_file) and not has_manifest and not require_plan and norm_expected is None:
        return trusted_dir, None, None

    if not os.path.isfile(plan_file) or os.path.islink(plan_file):
        raise ApprovalError(
            f"Run directory {trusted_dir} is missing a valid plan.json integrity record; prepare and approve again.",
            field="digest",
        )

    try:
        with open(plan_file, encoding="utf-8") as fh:
            saved_plan = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"Corrupted plan.json in {trusted_dir}: {exc}", field="digest") from exc

    if not isinstance(saved_plan, dict) or saved_plan.get("schema") != "aimip-plan-v1":
        raise ApprovalError(
            f"Invalid or incomplete plan.json schema in {trusted_dir}; prepare and approve again.",
            field="digest",
        )

    saved_digest = str(saved_plan.get("digest") or "").strip().lower()
    if not _HEX64_DIGEST_RE.match(saved_digest):
        raise ApprovalError(
            f"Invalid or missing 64-hex digest in {plan_file}; prepare and approve again.",
            field="digest",
        )
    if norm_expected is not None and saved_digest != norm_expected:
        raise ApprovalError(
            f"Plan changed, approve again (plan.json digest {saved_digest[:12]} != expected {norm_expected[:12]}).",
            field="digest",
        )

    raw_spec = saved_plan.get("spec")
    if not isinstance(raw_spec, dict) or not raw_spec.get("model") or not raw_spec.get("start"):
        raise ApprovalError(
            f"Plan record in {plan_file} is missing 'spec'; prepare and approve again.",
            field="digest",
        )

    recorded_hashes = saved_plan.get("artifact_hashes")
    if not isinstance(recorded_hashes, dict) or not recorded_hashes:
        raise ApprovalError(
            f"Plan record in {plan_file} is missing sealed 'artifact_hashes'; prepare and approve again.",
            field="digest",
        )
    if any(
        not _is_safe_sealed_artifact_name(fname)
        or not isinstance(hval, str)
        or not _HEX64_DIGEST_RE.match(hval)
        for fname, hval in recorded_hashes.items()
    ):
        raise ApprovalError(
            f"Malformed 'artifact_hashes' in {plan_file}; prepare and approve again.",
            field="digest",
        )

    required_present = ["job.sbatch", "aimip-manifest.json"]
    has_cfg = False
    for opt_name in ("config.yaml", "emu-job.json", "ic_era5.pt"):
        opt_p = os.path.join(trusted_dir, opt_name)
        if os.path.lexists(opt_p):
            required_present.append(opt_name)
            if opt_name in ("config.yaml", "emu-job.json"):
                has_cfg = True
    if not has_cfg:
        raise ApprovalError(
            f"Run directory {trusted_dir} has neither config.yaml nor emu-job.json.",
            field="digest",
        )
    for req_name in required_present:
        if req_name not in recorded_hashes:
            raise ApprovalError(
                f"Plan record in {plan_file} omits hash for {req_name!r}; prepare and approve again.",
                field="digest",
            )

    if in_memory_hashes is not None and dict(in_memory_hashes) != dict(recorded_hashes):
        raise ApprovalError(
            "Plan changed, approve again (plan.json artifact_hashes diverged from prepared snapshot).",
            field="digest",
        )

    sealed = read_sealed_prepared(
        saved_digest,
        run_dir=trusted_dir,
        site_config_path=site_config_path,
    )
    if sealed is None or not isinstance(sealed.get("artifact_hashes"), dict):
        raise ApprovalError(
            f"Missing authoritative prepared-run seal for digest {saved_digest[:12]}; prepare and approve again.",
            field="digest",
        )
    if sealed.get("run_dir") != trusted_dir or sealed["artifact_hashes"] != recorded_hashes:
        raise ApprovalError(
            "Plan changed, approve again (plan.json artifact_hashes diverged from sealed state snapshot).",
            field="digest",
        )

    live_hashes: dict[str, str] = {}
    for fname, expected_sha in recorded_hashes.items():
        cand = (
            validate_safe_path(script_override_path, label="--script-path")
            if (fname == "job.sbatch" and script_override_path)
            else os.path.join(trusted_dir, str(fname))
        )
        if not os.path.isfile(cand) or os.path.islink(cand):
            raise ApprovalError(
                f"Plan changed, approve again ({fname} is missing or symlinked).",
                field="digest",
            )
        actual_sha = sha256_file(cand)
        live_hashes[fname] = actual_sha
        if actual_sha != str(expected_sha):
            raise ApprovalError(
                f"Plan changed, approve again ({fname} was modified after preparation).",
                field="digest",
            )

    spec_for_replan = RunSpec.from_dict({**raw_spec, "out": trusted_dir, "name": None})
    rechecked = plan(
        spec_for_replan,
        registry_path=registry_path,
        playground_dir=playground_dir,
        data_dir=data_dir,
        site_config_path=site_config_path,
    )
    if rechecked.requires_staging:
        raise ApprovalError(
            "Run directory still has unstaged prospective inputs; complete preparation before submitting.",
            field="digest",
        )
    if rechecked.digest != saved_digest:
        raise ApprovalError(
            "Plan changed, approve again (current inputs/resources/lock no longer match prepared plan digest).",
            field="digest",
        )

    # Independently verify `job.sbatch` and `aimip-manifest.json` against the canonical `rechecked` plan
    # so rewriting `artifact_hashes` in place without `$AIMIP_STATE` is ALSO rejected!
    resolved_pg = resolve_playground_dir(playground_dir)
    site = load_site_config(site_config_path)
    slurm_opts = _build_slurm_options(
        spec_for_replan,
        site,
        require_valid_slurm=(spec_for_replan.mode != "local"),
    )
    expected_sbatch = render_sbatch_script(
        model_name=rechecked.model,
        out_dir=trusted_dir,
        playground_dir=resolved_pg,
        slurm_opts=slurm_opts,
        default_walltime=str(rechecked.resources["time"]),
    )
    sbatch_to_check = (
        validate_safe_path(script_override_path, label="--script-path")
        if script_override_path
        else os.path.join(trusted_dir, "job.sbatch")
    )
    with open(sbatch_to_check, encoding="utf-8") as sfh:
        actual_sbatch = sfh.read()
    if actual_sbatch != expected_sbatch:
        raise ApprovalError(
            "Plan changed, approve again (job.sbatch directives do not match the approved plan resources).",
            field="digest",
        )

    manifest_file = os.path.join(trusted_dir, "aimip-manifest.json")
    try:
        with open(manifest_file, encoding="utf-8") as mfh:
            manifest_doc = json.load(mfh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"Corrupted aimip-manifest.json in {trusted_dir}: {exc}", field="digest") from exc
    if (
        not isinstance(manifest_doc, dict)
        or manifest_doc.get("out_dir") != trusted_dir
        or (manifest_doc.get("registry_entry") or {}).get("name") != rechecked.model
    ):
        raise ApprovalError(
            "Plan changed, approve again (aimip-manifest.json does not match the approved plan).",
            field="digest",
        )
    if (
        "config_sha256" in manifest_doc
        and "config.yaml" in live_hashes
        and manifest_doc["config_sha256"] != live_hashes["config.yaml"]
    ):
        raise ApprovalError(
            "Plan changed, approve again (config.yaml does not match aimip-manifest.json config_sha256).",
            field="digest",
        )

    return trusted_dir, saved_digest, spec_for_replan.mode


def submit(
    run: str | PreparedRun,
    *,
    authorization: Authorization | None = None,
    require_authorization: bool = False,
    expected_digest: str | None = None,
    broker: MemoryApprovalBroker | None = None,
    playground_dir: str | None = None,
    registry_path: str | None = None,
    data_dir: str | None = None,
    site_config_path: str | None = None,
) -> Attempt:
    """Validate, re-plan to detect drift, and submit a prepared run directory."""
    try:
        in_mem_hashes: dict[str, str] | None = None
        if isinstance(run, PreparedRun):
            run_dir = run.run_dir
            run_name = run.plan.run_name or os.path.basename(run_dir.rstrip("/"))
            in_mem_hashes = dict(run.artifact_hashes) if run.artifact_hashes else None
        else:
            run_dir, run_name = resolve_run_target(run, site_config_path=site_config_path)

        need_plan = bool(
            isinstance(run, PreparedRun)
            or require_authorization
            or authorization is not None
            or expected_digest is not None
        )
        trusted_dir, saved_digest, saved_mode = verify_prepared_run_for_submit(
            run_dir,
            require_plan=need_plan,
            expected_digest=expected_digest,
            in_memory_hashes=in_mem_hashes,
            playground_dir=playground_dir,
            registry_path=registry_path,
            data_dir=data_dir,
            site_config_path=site_config_path,
        )

        if require_authorization or authorization is not None:
            if authorization is None:
                raise ApprovalError(
                    "Submission requires explicit human approval of the plan digest.",
                    field="authorization",
                )
            if not saved_digest:
                raise ApprovalError(
                    f"Run directory {trusted_dir} has no plan.json digest to verify against human approval.",
                    field="digest",
                )
            now_check = float(broker._now_fn()) if broker is not None else time.time()
            if broker is not None and broker._tokens.get(authorization.nonce) != authorization:
                raise ApprovalError(
                    "Authorization was already used, revoked, or not minted by this host broker.",
                    field="authorization",
                )
            MemoryApprovalBroker.verify_authorization_fields(
                authorization,
                expected_action="submit",
                expected_digest=saved_digest,
                now_ts=now_check,
            )

        site_cfg = load_site_config(site_config_path)
        sched_override = "none" if saved_mode == "local" else None
        effective_sched = sched_override or site_cfg.scheduler
        in_container = os.environ.get("AIMIP_IMAGE_SHA", "native") not in ("native", "")

        # Container fallback: inside an isolated container, neither Slurm (`sbatch` absent) nor detached
        # local execution (`effective_sched == 'none'`, whose `job.sbatch` invokes the host wrapper) may
        # launch in-container — return the exact digest-bound host command instead.
        if in_container and (
            effective_sched == "none"
            or (effective_sched == "slurm" and shutil.which("sbatch") is None)
        ):
            if (require_authorization or authorization is not None) and broker is not None and authorization is not None:
                broker.consume(
                    authorization,
                    expected_action="submit",
                    expected_digest=str(saved_digest),
                )
            host_args = ["aimip", "submit"]
            if saved_digest:
                host_args.append(f"--expect-digest={saved_digest}")
            host_args.append(trusted_dir)
            host_cmd = shlex.join(host_args)
            return Attempt(
                run_dir=trusted_dir,
                run_name=run_name,
                digest=saved_digest,
                scheduler=effective_sched,
                attempt_id=None,
                state="PREPARED_AWAITING_HOST_SUBMIT",
                reconciled=False,
                host_command=host_cmd,
            )

        consumed_inside_lock = False

        def _verify_locked_private_copy(priv_script_path: str) -> None:
            nonlocal saved_digest, consumed_inside_lock
            _, locked_digest, _ = verify_prepared_run_for_submit(
                trusted_dir,
                require_plan=need_plan,
                expected_digest=saved_digest or expected_digest,
                in_memory_hashes=in_mem_hashes,
                script_override_path=priv_script_path,
                playground_dir=playground_dir,
                registry_path=registry_path,
                data_dir=data_dir,
                site_config_path=site_config_path,
            )
            saved_digest = locked_digest
            if require_authorization or authorization is not None:
                assert authorization is not None and saved_digest is not None
                if broker is not None:
                    broker.consume(
                        authorization,
                        expected_action="submit",
                        expected_digest=saved_digest,
                    )
                    consumed_inside_lock = True
                else:
                    MemoryApprovalBroker.verify_authorization_fields(
                        authorization,
                        expected_action="submit",
                        expected_digest=saved_digest,
                    )

        sub_res = submit_run(
            trusted_dir,
            playground_dir=playground_dir,
            site_config_path=site_config_path,
            scheduler_override=sched_override,
            verify_private_script_fn=_verify_locked_private_copy,
        )
        if (
            (require_authorization or authorization is not None)
            and broker is not None
            and authorization is not None
            and not consumed_inside_lock
        ):
            broker.consume(
                authorization,
                expected_action="submit",
                expected_digest=str(saved_digest),
            )
    except BaseException:
        if broker is not None:
            broker.burn_token(authorization)
        raise

    sched = str(sub_res["scheduler"])
    att_id = (
        str(sub_res["slurm_job_id"])
        if sched == "slurm"
        else str(sub_res["local_pid"])
    )
    return Attempt(
        run_dir=trusted_dir,
        run_name=run_name,
        digest=saved_digest,
        scheduler=sched,
        attempt_id=att_id,
        state="SUBMITTED" if sched == "slurm" else "RUNNING_LOCAL",
        reconciled=bool(sub_res.get("reconciled", False)),
    )


def execute_tty_plan_flow(
    spec: RunSpec,
    *,
    allow_fetch: bool = True,
    broker: MemoryApprovalBroker | None = None,
    tty_path: str = "/dev/tty",
    now_fn: Any = time.time,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    site_config_path: str | None = None,
    argv: list[str] | None = None,
) -> Attempt:
    """Shared `/dev/tty` human-approval execution flow used by CLI (`aimip run --approve`) and chat (`AgentTools`).

    - Staging models (`requires_staging == True`):
      Step 1 prompts on `/dev/tty` for `action='prepare'` (plus separate download/staging consent); that
      prepare-only `Authorization` is consumed by `prepare()` and can never submit.
      Step 2 displays the final card (with staged IC bytes hashed into the final digest) and prompts
      on `/dev/tty` for `action='submit'` before calling `submit()`.
    - Non-staging models (`requires_staging == False`):
      Collapses to one `/dev/tty` prompt only when `prospective digest == final digest`.
    """
    active_broker = broker if broker is not None else MemoryApprovalBroker(now_fn=now_fn)
    initial = plan(
        spec,
        registry_path=registry_path,
        playground_dir=playground_dir,
        data_dir=data_dir,
        site_config_path=site_config_path,
    )
    if not allow_fetch and initial.missing_assets:
        active_broker.revoke_all()
        _raise_no_fetch_missing_assets(spec, initial)

    if active_broker.pending is None:
        active_broker.register_proposal(
            action=initial.required_approval_action,
            digest=initial.digest,
            card_text=render_plan_card(initial, is_tty=False),
            card_data=initial.to_dict(),
            missing_bytes=initial.missing_bytes,
            missing_count=len(initial.missing_assets),
        )
    elif active_broker.pending.get("digest") != initial.digest:
        active_broker.revoke_all()
        raise ApprovalError(
            "Plan changed, approve again (current inputs/output/resources no longer match proposed digest).",
            field="digest",
        )

    step1_auth, _ = active_broker.approve_pending_on_tty(
        tty_path=tty_path,
        data_dir=data_dir,
        playground_dir=playground_dir,
    )
    prepared = prepare(
        spec,
        allow_fetch=allow_fetch,
        authorization=step1_auth,
        require_authorization=True,
        broker=active_broker,
        registry_path=registry_path,
        playground_dir=playground_dir,
        data_dir=data_dir,
        site_config_path=site_config_path,
        argv=argv,
    )

    if step1_auth.action == "prepare" or prepared.plan.digest != step1_auth.digest:
        if step1_auth.action != "prepare":
            active_broker.burn_token(step1_auth)
        active_broker.register_proposal(
            action="submit",
            digest=prepared.plan.digest,
            card_text=render_plan_card(prepared.plan, is_tty=False),
            card_data=prepared.plan.to_dict(),
            missing_bytes=0,
            missing_count=0,
        )
        final_auth, _ = active_broker.approve_pending_on_tty(
            tty_path=tty_path,
            data_dir=data_dir,
            playground_dir=playground_dir,
        )
        return submit(
            prepared,
            authorization=final_auth,
            require_authorization=True,
            broker=active_broker,
            playground_dir=playground_dir,
            registry_path=registry_path,
            data_dir=data_dir,
            site_config_path=site_config_path,
        )

    return submit(
        prepared,
        authorization=step1_auth,
        require_authorization=True,
        broker=active_broker,
        playground_dir=playground_dir,
        registry_path=registry_path,
        data_dir=data_dir,
        site_config_path=site_config_path,
    )


def _enrich_status_provenance(info: dict[str, Any], run_dir: str) -> dict[str, Any]:
    """Attach `plan.json` digest metadata, `display_state`, and `host_submit_command` when prepared."""
    plan_file = os.path.join(run_dir, "plan.json")
    if os.path.isfile(plan_file) and not os.path.islink(plan_file):
        try:
            with open(plan_file, encoding="utf-8") as fh:
                pdoc = json.load(fh)
            info["digest"] = pdoc.get("digest")
            info["short_digest"] = pdoc.get("short_digest")
        except (OSError, json.JSONDecodeError):
            pass

    raw_state = str(info.get("state") or "unknown")
    base_state = (raw_state.split() or [""])[0].rstrip("+").upper()
    is_terminal = bool(base_state in TERMINAL_RUN_STATES or raw_state.upper() in TERMINAL_RUN_STATES)
    job_id = info.get("slurm_job_id") or info.get("job_id")
    local_pid = info.get("local_pid")
    artifacts = info.get("artifacts") if isinstance(info.get("artifacts"), dict) else {}
    awaiting_submit = bool(
        not is_terminal
        and job_id is None
        and local_pid is None
        and info.get("finished") is None
        and (artifacts.get("job.sbatch") or info.get("digest"))
    )
    cancel_req = bool(info.get("cancel_requested") and not is_terminal)
    info["awaiting_submit"] = awaiting_submit
    info["cancel_requested"] = cancel_req
    if cancel_req:
        info["display_state"] = "cancel_requested"
    elif awaiting_submit:
        info["display_state"] = "prepared"
    else:
        info["display_state"] = raw_state

    if awaiting_submit:
        host_args = ["aimip", "submit"]
        saved_digest = info.get("digest")
        if isinstance(saved_digest, str) and saved_digest:
            host_args.append(f"--expect-digest={saved_digest}")
        host_args.append(run_dir)
        info["host_submit_command"] = shlex.join(host_args)
    else:
        info["host_submit_command"] = None

    sacct_block = info.get("sacct") if isinstance(info.get("sacct"), dict) else {}
    info.setdefault("elapsed", sacct_block.get("elapsed"))
    info.setdefault("exit_code", sacct_block.get("exit_code"))
    return info


def status(
    run: str,
    *,
    sacct_json_path: str | None = None,
    site_config_path: str | None = None,
    min_sacct_interval_seconds: float = 0.0,
) -> dict[str, Any]:
    """Inspect run or spin status by name or path."""
    run_dir, run_name = resolve_run_target(run, site_config_path=site_config_path)
    info = get_run_status(
        run_dir,
        sacct_json_path=sacct_json_path,
        min_sacct_interval_seconds=min_sacct_interval_seconds,
    )
    info["run_name"] = run_name
    return _enrich_status_provenance(info, run_dir)


def watch(
    run: str,
    *,
    poll_interval: float = 2.0,
    max_polls: int | None = None,
    timeout_seconds: float = 3600.0,
    site_config_path: str | None = None,
    on_tick: Callable[[dict[str, Any]], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll-only progress watcher (never submits jobs or triggers post-run analyses)."""
    started = time.monotonic()
    polls = 0
    history: list[dict[str, Any]] = []

    while True:
        polls += 1
        snap = status(run, site_config_path=site_config_path)
        state_val = str(snap.get("state") or "unknown")
        job_id = snap.get("slurm_job_id") or snap.get("job_id")
        sacct_block = snap.get("sacct") if isinstance(snap.get("sacct"), dict) else {}
        elapsed = sacct_block.get("elapsed") or snap.get("elapsed")
        tick = {
            "poll": polls,
            "run_name": snap.get("run_name"),
            "run_dir": snap.get("out_dir") or snap.get("spin_dir"),
            "model": snap.get("model"),
            "state": state_val,
            "job_id": job_id,
            "elapsed": elapsed,
        }
        history.append(tick)
        if on_tick is not None:
            on_tick(tick)

        base_state = (state_val.split() or [""])[0].rstrip("+")
        if base_state.upper() in TERMINAL_RUN_STATES or state_val.upper() in TERMINAL_RUN_STATES:
            snap["polls"] = polls
            snap["terminal"] = True
            return snap
        if (
            job_id is None
            and snap.get("local_pid") is None
            and base_state.upper() in ("UNKNOWN", "PREPARED", "DRY_RUN", "NOT_SUBMITTED")
            and max_polls is None
        ):
            # Unsubmitted directory will never advance on its own; return immediately rather than hanging.
            snap["polls"] = polls
            snap["terminal"] = False
            return snap
        if max_polls is not None and polls >= max_polls:
            snap["polls"] = polls
            snap["terminal"] = False
            return snap
        if (time.monotonic() - started) >= timeout_seconds:
            snap["polls"] = polls
            snap["terminal"] = False
            return snap
        sleep_fn(max(0.05, float(poll_interval)))


def logs(
    run: str,
    *,
    follow: bool = False,
    max_bytes: int = 65536,
    poll_interval: float = 1.0,
    max_polls: int | None = None,
    site_config_path: str | None = None,
    on_chunk: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Read bounded, terminal-sanitized job logs directly inside `<run_dir>` (refusing symlinks).

    Tracks per-file byte offsets when `follow=True` so growing logs keep streaming past `max_bytes`
    without losing lines or mixing `.out` / `.err` slices, and backs off live `sacct` refreshes to >= 15 s.
    """
    if max_bytes < 64 or max_bytes > 16 * 1024 * 1024:
        raise ValidationError("max_bytes must be between 64 and 16777216.", field="max_bytes")
    run_dir, run_name = resolve_run_target(run, site_config_path=site_config_path)
    trusted = check_trusted_run_dir(run_dir)
    offsets: dict[str, int] = {}

    def _discover_log_files() -> list[str]:
        found: list[str] = []
        for entry in sorted(os.listdir(trusted)):
            if entry == "local-run.log" or (
                entry.startswith("slurm-") and entry.endswith((".out", ".err"))
            ):
                full = os.path.join(trusted, entry)
                if os.path.islink(full):
                    raise ValidationError(f"Refusing symlinked log file: {full}", field="logs")
                if os.path.isfile(full):
                    found.append(full)
        return found

    def _read_initial_bounded(files: list[str]) -> tuple[str, bool]:
        chunks: list[str] = []
        remaining = max_bytes
        truncated = False
        for fp in files:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(fp, flags)
            try:
                size = os.fstat(fd).st_size
                offsets[fp] = size
                if remaining <= 0:
                    if size > 0:
                        truncated = True
                    continue
                if size > remaining:
                    os.lseek(fd, max(0, size - remaining), os.SEEK_SET)
                    truncated = True
                raw = os.read(fd, remaining + 1)
                if len(raw) > remaining:
                    raw = raw[-remaining:]
                    truncated = True
                text = sanitize_terminal_text(raw.decode("utf-8", errors="replace"))
                header = f"==> {os.path.basename(fp)} <==\n" if len(files) > 1 else ""
                chunks.append(header + text)
                remaining = max(0, remaining - len(raw))
            finally:
                os.close(fd)
        return "\n".join(chunks), truncated

    def _read_incremental(files: list[str]) -> tuple[str, bool]:
        deltas: list[str] = []
        inc_truncated = False
        for fp in files:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(fp, flags)
            try:
                size = os.fstat(fd).st_size
                prev = offsets.get(fp, 0)
                if size < prev:
                    prev = 0
                if size <= prev:
                    offsets[fp] = size
                    continue
                unread = size - prev
                if unread > max_bytes:
                    prev = size - max_bytes
                    inc_truncated = True
                os.lseek(fd, prev, os.SEEK_SET)
                raw = os.read(fd, size - prev)
                offsets[fp] = prev + len(raw)
                if raw:
                    text = sanitize_terminal_text(raw.decode("utf-8", errors="replace"))
                    if len(files) > 1:
                        text = f"==> {os.path.basename(fp)} <==\n{text}"
                    deltas.append(text)
            finally:
                os.close(fd)
        return "".join(deltas), inc_truncated

    files = _discover_log_files()
    content, truncated = _read_initial_bounded(files)
    if on_chunk is not None and content:
        on_chunk(content)

    if follow:
        polls = 0
        while True:
            polls += 1
            snap = status(
                trusted,
                site_config_path=site_config_path,
                min_sacct_interval_seconds=0.0 if polls == 1 else 15.0,
            )
            st = str(snap.get("state") or "")
            files = _discover_log_files()
            delta_text, inc_trunc = _read_incremental(files)
            if inc_trunc:
                truncated = True
            if delta_text:
                if on_chunk is not None:
                    on_chunk(delta_text)
                combined = content + delta_text if content else delta_text
                if len(combined) > max_bytes:
                    content = combined[-max_bytes:]
                    truncated = True
                else:
                    content = combined
            base_st = (st.split() or [""])[0].rstrip("+").upper()
            if base_st in TERMINAL_RUN_STATES or st.upper() in TERMINAL_RUN_STATES:
                break
            if max_polls is not None and polls >= max_polls:
                break
            sleep_fn(max(0.05, float(poll_interval)))

    return {
        "run_name": run_name,
        "run_dir": trusted,
        "log_files": [os.path.basename(p) for p in files],
        "truncated": truncated,
        "max_bytes": max_bytes,
        "content": content,
    }


def ls(
    run_root: str | None = None,
    *,
    site_config_path: str | None = None,
    min_sacct_interval_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """List prepared, running, and completed runs under *run_root* (or the configured site `run_root`)."""
    root = (
        validate_safe_path(run_root, label="run_root")
        if run_root
        else resolve_configured_run_root(site_config_path=site_config_path)
    )
    if not os.path.isdir(root) or os.path.islink(root):
        return []

    candidates: list[tuple[str, str]] = []
    batch_jobs: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(root)):
        if entry.startswith("."):
            continue
        cand = os.path.join(root, entry)
        if os.path.islink(cand) or not os.path.isdir(cand):
            continue
        if not is_trusted_run_dir(cand):
            continue
        has_marker = any(
            os.path.isfile(os.path.join(cand, m))
            for m in ("aimip-manifest.json", "spin-manifest.json", "plan.json")
        )
        if not has_marker:
            continue
        candidates.append((entry, cand))

        manifest_file = os.path.join(cand, "aimip-manifest.json")
        manifest_doc: dict[str, Any] = {}
        if os.path.isfile(manifest_file) and not os.path.islink(manifest_file):
            try:
                with open(manifest_file, encoding="utf-8") as mfh:
                    loaded = json.load(mfh)
                if isinstance(loaded, dict):
                    manifest_doc = loaded
            except (OSError, json.JSONDecodeError):
                manifest_doc = {}
        if "finished" not in manifest_doc or "rc" not in manifest_doc:
            jid = _discover_job_id(cand, manifest_doc, _parse_submitted_txt(cand))
            if jid:
                batch_jobs.append((cand, jid))

    if batch_jobs:
        refresh_sacct_batch(
            batch_jobs,
            min_interval_seconds=max(0.0, float(min_sacct_interval_seconds)),
        )
    per_run_interval = max(float(min_sacct_interval_seconds), 5.0 if batch_jobs else 0.0)

    items: list[dict[str, Any]] = []
    for entry, cand in candidates:
        try:
            st_info = _enrich_status_provenance(
                get_run_status(cand, min_sacct_interval_seconds=per_run_interval),
                cand,
            )
        except Exception:
            continue
        mtime = os.path.getmtime(cand)
        items.append(
            {
                "name": entry,
                "run_dir": cand,
                "kind": st_info.get("kind", "run"),
                "model": st_info.get("model"),
                "start_time": st_info.get("start_time") or st_info.get("target_time"),
                "end_time": st_info.get("end_time"),
                "steps": st_info.get("steps"),
                "members": st_info.get("members"),
                "state": st_info.get("state"),
                "display_state": st_info.get("display_state"),
                "awaiting_submit": st_info.get("awaiting_submit", False),
                "cancel_requested": st_info.get("cancel_requested", False),
                "host_submit_command": st_info.get("host_submit_command"),
                "job_id": st_info.get("slurm_job_id") or st_info.get("job_id"),
                "local_pid": st_info.get("local_pid"),
                "rc": st_info.get("rc"),
                "elapsed": st_info.get("elapsed"),
                "exit_code": st_info.get("exit_code"),
                "sacct_source": st_info.get("sacct_source"),
                "sacct_age_seconds": st_info.get("sacct_age_seconds"),
                "updated_at": datetime.datetime.fromtimestamp(
                    mtime, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    items.sort(key=lambda r: (r["updated_at"], r["name"]), reverse=True)
    return items


def cancel_digest_for(run_dir: str, attempt_id: str) -> str:
    """Compute the deterministic 64-hex approval digest for cancelling `(run_dir, attempt_id)`."""
    return compute_plan_digest(
        {"schema": "aimip-cancel-v1", "run_dir": run_dir, "attempt_id": str(attempt_id)}
    )


def cancel(
    run: str,
    *,
    authorization: Authorization | None = None,
    require_authorization: bool = False,
    broker: MemoryApprovalBroker | None = None,
    site_config_path: str | None = None,
) -> dict[str, Any]:
    """Cancel an active Slurm job (`scancel`) or detached local process (`SIGTERM`), refusing terminal runs."""
    run_dir, run_name = resolve_run_target(run, site_config_path=site_config_path)
    trusted = check_trusted_run_dir(run_dir)
    snap = get_run_status(trusted)
    job_id = snap.get("slurm_job_id") or snap.get("job_id")
    local_pid = snap.get("local_pid")

    if not job_id and not local_pid:
        raise ConfigError(
            f"Run {run_name!r} ({trusted}) has no recorded Slurm job ID or local PID to cancel.",
            field="run",
        )

    cur_state = str(snap.get("state") or "unknown").strip()
    base_cur = (cur_state.split() or [""])[0].rstrip("+").upper()
    if base_cur in TERMINAL_RUN_STATES or cur_state.upper() in TERMINAL_RUN_STATES:
        raise ConfigError(
            f"Run {run_name!r} ({trusted}) is already in terminal state {cur_state!r} and cannot be cancelled.",
            field="run",
        )

    attempt_id = str(job_id) if job_id else str(local_pid)
    expected_digest = cancel_digest_for(trusted, attempt_id)

    if require_authorization or authorization is not None:
        if authorization is None:
            raise ApprovalError(
                f"Cancelling {run_name!r} requires human approval (`aimip approve {expected_digest[:8]}`).",
                field="authorization",
                next_command=f"aimip approve {expected_digest[:8]}",
            )
        if broker is not None:
            broker.consume(
                authorization,
                expected_action="cancel",
                expected_digest=expected_digest,
            )
        else:
            MemoryApprovalBroker.verify_authorization_fields(
                authorization,
                expected_action="cancel",
                expected_digest=expected_digest,
            )

    if job_id:
        scancel_bin = shutil.which("scancel")
        if not scancel_bin:
            raise ExecutionError(
                f"scancel not found on PATH; run `aimip cancel {run_name}` on the cluster login node.",
                next_command=f"aimip cancel {run_name}",
            )
        proc = subprocess.run(
            [scancel_bin, str(job_id)],
            capture_output=True,
            text=True,
            env=sanitized_subprocess_env(),
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise ExecutionError(f"scancel {job_id} failed: {detail}" if detail else f"scancel {job_id} failed")
        write_in_trusted_dir(
            trusted,
            "cancel-requested.txt",
            f"job={job_id} requested_by={current_user()}",
        )
        return {
            "run_name": run_name,
            "run_dir": trusted,
            "scheduler": "slurm",
            "attempt_id": str(job_id),
            "cancelled": True,
            "status": "cancel_requested",
            "digest": expected_digest,
        }

    pid_int = int(local_pid)
    if pid_int <= 1:
        raise ValidationError(f"Refusing invalid local PID {pid_int}", field="local_pid")
    if os.environ.get("AIMIP_IMAGE_SHA", "native") not in ("", "native"):
        raise ExecutionError(
            f"Local PID {pid_int} runs in the host PID namespace; run `aimip cancel {run_name}` on the host.",
            next_command=f"aimip cancel {run_name}",
        )
    alive, pgid, reason = inspect_local_process(trusted, pid_int)
    if reason == "reused_pid":
        raise ExecutionError(
            f"Refusing to signal local PID {pid_int} for run {run_name!r}: process start identity changed (PID was reused)."
        )
    if not alive:
        raise ExecutionError(
            f"Local process {pid_int} for run {run_name!r} was not found (process already exited)."
        )
    caller_pgid = os.getpgrp()
    if reason is not None or pgid is None or pgid <= 1 or pgid == caller_pgid:
        raise ExecutionError(
            f"Refusing to cancel local PID {pid_int} for run {run_name!r}: "
            "cannot verify process start identity or dedicated session group (not cancelled)."
        )
    try:
        live_pgid = int(os.getpgid(pid_int))
    except ProcessLookupError as exc:
        raise ExecutionError(
            f"Local process {pid_int} for run {run_name!r} was not found (process already exited)."
        ) from exc
    except OSError as exc:
        raise ExecutionError(
            f"Refusing to cancel local PID {pid_int} for run {run_name!r}: "
            f"cannot verify live process group ({exc}; not cancelled)."
        ) from exc
    if live_pgid != pgid:
        raise ExecutionError(
            f"Refusing to cancel local PID {pid_int} for run {run_name!r}: "
            f"live process group {live_pgid} does not match recorded session group {pgid} (not cancelled)."
        )
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError as exc:
        raise ExecutionError(
            f"Local process {pid_int} for run {run_name!r} was not found (process already exited)."
        ) from exc
    except PermissionError as exc:
        raise ExecutionError(f"Permission denied signalling local process group {pgid}: {exc}") from exc

    write_in_trusted_dir(
        trusted,
        "cancel-requested.txt",
        f"pid={pid_int} pgid={pgid} requested_by={current_user()}",
    )

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        still_alive, _, still_reason = inspect_local_process(trusted, pid_int)
        pg_alive = False
        try:
            os.killpg(pgid, 0)
            pg_alive = True
        except ProcessLookupError:
            pg_alive = False
        except PermissionError:
            pg_alive = True
        if (not still_alive or still_reason == "reused_pid") and not pg_alive:
            break
        time.sleep(0.025)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    return {
        "run_name": run_name,
        "run_dir": trusted,
        "scheduler": "none",
        "attempt_id": str(pid_int),
        "cancelled": True,
        "status": "cancel_requested",
        "digest": expected_digest,
    }


def summary(
    run: str,
    *,
    site_config_path: str | None = None,
) -> Result:
    """Compute area-weighted monthly diagnostic series for *run* (accepting a run name or path)."""
    run_dir, _ = resolve_run_target(run, site_config_path=site_config_path)
    raw = summarize_run(run_dir)
    try:
        snap = get_run_status(run_dir, min_sacct_interval_seconds=15.0)
    except AimipError:
        snap = {}
    return Result(
        run_dir=run_dir,
        model=raw.get("model") or snap.get("model"),
        state=str(snap.get("state") or "UNKNOWN"),
        variables=dict(raw.get("variables") or {}),
        raw_summary=raw,
    )


def doctor(
    *,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    site_config_path: str | None = None,
) -> dict[str, Any]:
    """Check runtime, GPU, scheduler, data root, permissions, and image/package version without filesystem side effects."""
    checks: list[dict[str, Any]] = []

    def _add(
        name: str,
        ok: bool,
        detail: str,
        fix_hint: str | None = None,
        *,
        status_level: str | None = None,
    ) -> None:
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "status": status_level or ("pass" if ok else "fail"),
                "detail": detail,
                "fix_hint": None if ok else fix_hint,
            }
        )

    # 1. Site configuration
    site: SiteConfig | None = None
    try:
        site = load_site_config(site_config_path)
        site.validate(allow_change_me=False)
        _add(
            "site_config",
            True,
            f"site={site.name} scheduler={site.scheduler} runtime={site.runtime}"
            + (
                f" partition={site.partition or '<none>'}"
                + (f" qos={site.qos}" if site.qos else "")  # MN5: the QOS places the job
                + f" account={site.account or '<none>'}"
                if site.scheduler == "slurm"
                else ""
            ),
        )
    except AimipError as exc:
        site = SiteConfig(scheduler="none", runtime="native")
        _add(
            "site_config",
            False,
            exc.message,
            "Run `aimip site init --preset generic-slurm --partition <part> --account <acct>`.",
        )
    except OSError as exc:
        site = SiteConfig(scheduler="none", runtime="native")
        _add(
            "site_config",
            False,
            sanitize_terminal_text(f"Filesystem error reading site configuration: {exc}"),
            "Retry once the filesystem recovers, or check the `$AIMIP_SITE_CONFIG` / `$AIMIP_HOME` mount.",
        )

    # 2. Runtime check
    resolved_pg: str | None = None
    try:
        resolved_pg = resolve_playground_dir(playground_dir)
        if site.runtime == "native":
            fme_ver = get_fme_version()
            ov_dir = os.environ.get("AIMIP_FME_2026_10_OVERLAY") or (
                os.path.join(resolved_pg, "overlay-fme-2026.10.0") if resolved_pg else ""
            )
            emu_py = os.environ.get("AIMIP_EMU_PYTHON") or (
                os.path.join(resolved_pg, "venv-emu", "bin", "python") if resolved_pg else ""
            )
            ov_tag = "ready" if (ov_dir and os.path.isdir(ov_dir)) else "absent"
            emu_tag = "ready" if (emu_py and os.path.isfile(emu_py)) else "optional/absent"
            _add(
                "runtime",
                True,
                f"native Python ({os.sys.version.split()[0]}), fme={fme_ver}, overlay-2026.10={ov_tag}, emu={emu_tag}",
            )
        else:
            host_rt_ok = os.environ.get("AIMIP_HOST_RUNTIME_OK")
            host_rt_detail = os.environ.get("AIMIP_HOST_RUNTIME_DETAIL")
            rt_bin = shutil.which(site.runtime)
            img_sha = os.environ.get("AIMIP_IMAGE_SHA")
            in_container = bool(img_sha and img_sha != "native")
            if host_rt_ok == "1":
                _add(
                    "runtime",
                    True,
                    f"container runtime={site.runtime} ({host_rt_detail or 'verified by host wrapper'})",
                )
            elif host_rt_ok == "0":
                _add(
                    "runtime",
                    False,
                    host_rt_detail or f"container runtime {site.runtime!r} or image missing on host",
                    f"Load or install {site.runtime} on the login node, or re-run `install/install-container.sh`.",
                )
            elif rt_bin or in_container:
                _add(
                    "runtime",
                    True,
                    f"container runtime={site.runtime} ({rt_bin or 'inside container'})",
                )
            else:
                _add(
                    "runtime",
                    False,
                    f"container runtime binary {site.runtime!r} not found on PATH",
                    f"Install {site.runtime} or re-install in native mode with `./install/install-native.sh --prefix <dir>`.",
                )
    except (AimipError, OSError) as exc:
        _add(
            "runtime",
            False,
            sanitize_terminal_text(f"Filesystem error probing runtime prefix: {exc}"),
            "Retry once the filesystem recovers, or check the `$AIMIP_HOME` / `$AIMIP_PLAYGROUND` mount.",
        )

    # 3. GPU check (on a Slurm login node without a local GPU, report INFO rather than FAIL)
    login_gpu_info = (
        f"no GPU here — jobs run on GPU nodes via the scheduler (partition={site.partition})"
    )
    host_gpu_status = os.environ.get("AIMIP_HOST_GPU_STATUS")
    try:
        if host_gpu_status == "present":
            _add(
                "gpu",
                True,
                sanitize_terminal_text(os.environ.get("AIMIP_HOST_GPU_LINE") or "GPU available on host"),
            )
        elif host_gpu_status == "absent":
            if site.scheduler == "slurm":
                _add("gpu", True, login_gpu_info, status_level="info")
            else:
                _add(
                    "gpu",
                    False,
                    "No GPU detected on this host for local execution.",
                    "Run on a GPU node or configure a Slurm GPU partition.",
                )
        else:
            gpu_cmd = (site.gpu_check or "nvidia-smi").strip()
            gpu_bin = shutil.which(gpu_cmd.split()[0]) if gpu_cmd else None
            if not gpu_bin:
                if site.scheduler == "slurm":
                    _add("gpu", True, login_gpu_info, status_level="info")
                else:
                    _add(
                        "gpu",
                        False,
                        f"GPU check tool {gpu_cmd!r} not found on PATH",
                        "Run on a GPU node or load your CUDA/driver module.",
                    )
            else:
                gproc = subprocess.run(
                    [gpu_bin, "-L"] if os.path.basename(gpu_bin) == "nvidia-smi" else [gpu_bin],
                    capture_output=True,
                    text=True,
                    env=sanitized_subprocess_env(),
                    timeout=5,
                    check=False,
                )
                first_gpu = (gproc.stdout.strip().splitlines() or ["ok"])[0][:120]
                if gproc.returncode == 0:
                    _add("gpu", True, sanitize_terminal_text(first_gpu))
                elif site.scheduler == "slurm":
                    _add("gpu", True, login_gpu_info, status_level="info")
                else:
                    _add(
                        "gpu",
                        False,
                        f"{gpu_cmd} exited with {gproc.returncode}",
                        "Check NVIDIA driver status (`nvidia-smi`) or run on a GPU node.",
                    )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if site.scheduler == "slurm":
            _add("gpu", True, login_gpu_info, status_level="info")
        else:
            _add(
                "gpu",
                False,
                sanitize_terminal_text(f"GPU check failed: {exc}"),
                "Retry or check GPU driver health and `$PATH` mounts.",
            )

    # 4. Scheduler tools (never advise `--preset local` on a Slurm cluster!)
    try:
        if site.scheduler == "slurm":
            host_slurm_ok = os.environ.get("AIMIP_HOST_SLURM_OK")
            if host_slurm_ok == "1":
                _add("scheduler", True, "Slurm tools found on host (sbatch, squeue, sacct, scancel)")
            elif host_slurm_ok == "0":
                miss = os.environ.get("AIMIP_HOST_SLURM_MISSING") or "sbatch"
                _add(
                    "scheduler",
                    False,
                    f"Missing Slurm command(s) on host PATH: {miss}",
                    "Load the cluster Slurm module on the login node before running `aimip`.",
                )
            else:
                needed = ("sbatch", "squeue", "sacct", "scancel")
                missing_bins = [b for b in needed if shutil.which(b) is None]
                if not missing_bins:
                    _add("scheduler", True, "Slurm tools found (sbatch, squeue, sacct, scancel)")
                else:
                    _add(
                        "scheduler",
                        False,
                        f"Missing Slurm command(s) on PATH: {', '.join(missing_bins)}",
                        "Load the cluster Slurm module on the login node before running `aimip`.",
                    )
        else:
            bash_bin = shutil.which("bash")
            _add(
                "scheduler",
                bool(bash_bin),
                f"local background scheduler ({bash_bin or 'bash missing'})",
                "Install `/bin/bash` for local job execution.",
            )
    except OSError as exc:
        _add(
            "scheduler",
            False,
            sanitize_terminal_text(f"Filesystem error probing scheduler binaries: {exc}"),
            "Retry once the filesystem recovers, or check mounts in `$PATH`.",
        )

    # 5. Data root & lock file
    resolved_data: str | None = None
    try:
        resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
        if not resolved_data:
            _add(
                "data_root",
                False,
                "$AIMIP_DATA is not set and could not be derived",
                "Export AIMIP_DATA=/path/to/data or invoke `<prefix>/bin/aimip doctor`.",
            )
        elif os.path.islink(resolved_data):
            _add(
                "data_root",
                False,
                f"Data directory is a symlink: {resolved_data}",
                "Point AIMIP_DATA to a real directory owned by you or the install administrator.",
            )
        elif not os.path.isdir(resolved_data):
            _add(
                "data_root",
                False,
                f"Data directory does not exist yet: {resolved_data}",
                f"Create `{resolved_data}` or run `aimip fetch -m ace2-era5`.",
            )
        else:
            st_mode = os.stat(resolved_data).st_mode
            if st_mode & 0o022:
                _add(
                    "data_root",
                    False,
                    f"Data directory is group- or world-writable: {resolved_data}",
                    f"Run `chmod go-w {resolved_data}`.",
                )
            else:
                free_gib = shutil.disk_usage(resolved_data).free / (1024**3)
                cat_rows = models(
                    installed_only=True,
                    registry_path=registry_path,
                    playground_dir=resolved_pg,
                    data_dir=resolved_data,
                )
                _add(
                    "data_root",
                    True,
                    f"{resolved_data} ({free_gib:.1f} GiB free, {len(cat_rows)} model(s) ready)",
                )
    except AimipError as exc:
        _add(
            "data_root",
            False,
            exc.message,
            "Check `$AIMIP_DATA`, registry, and lock files, or retry if the mount was degraded.",
        )
    except OSError as exc:
        target_label = resolved_data or data_dir or os.environ.get("AIMIP_DATA") or "$AIMIP_DATA"
        _add(
            "data_root",
            False,
            sanitize_terminal_text(f"Filesystem error probing data root ({target_label}): {exc}"),
            "Retry once the parallel filesystem recovers, or check the `$AIMIP_DATA` mount.",
        )

    # 6. Run-root permissions & ancestry (read-only check; never creates directories!)
    try:
        rroot = resolve_configured_run_root(site_config_path=site_config_path)
        check_safe_path(rroot)
        if os.path.islink(rroot):
            _add(
                "permissions",
                False,
                f"Run root is a symlink: {rroot}",
                "Configure a non-symlink `run_root` in `site.json`.",
            )
        elif os.path.exists(rroot):
            check_ancestry(rroot)
            rst = os.lstat(rroot)
            if rst.st_mode & 0o022:
                _add(
                    "permissions",
                    False,
                    f"Run root is group/world-writable: {rroot}",
                    f"Run `chmod go-w {rroot}`.",
                )
            elif not os.access(rroot, os.W_OK):
                _add(
                    "permissions",
                    False,
                    f"Run root is not writable: {rroot}",
                    "Choose a writable `--run-root` via `aimip site init`.",
                )
            else:
                _add("permissions", True, f"run_root={rroot} (owner-only writable, safe ancestry)")
        else:
            parent_probe = rroot
            while parent_probe != "/" and not os.path.exists(parent_probe):
                parent_probe = os.path.dirname(parent_probe) or "/"
            check_ancestry(parent_probe if parent_probe != "/" else rroot)
            if not os.access(parent_probe, os.W_OK):
                _add(
                    "permissions",
                    False,
                    f"Parent directory {parent_probe} for run_root {rroot} is not writable.",
                    "Choose a writable `--run-root` via `aimip site init`.",
                )
            else:
                _add(
                    "permissions",
                    True,
                    f"run_root={rroot} (parent {parent_probe} writable, safe ancestry)",
                )
    except AimipError as exc:
        _add(
            "permissions",
            False,
            exc.message,
            "Set a private writable `run_root` without world-writable non-sticky parents "
            "(on MN5, list the root-owned group directory, e.g. /gpfs/scratch/<group>, in site.json "
            "`trusted_group_parents`; see docs/mn5.md).",
        )
    except OSError as exc:
        _add(
            "permissions",
            False,
            sanitize_terminal_text(f"Filesystem error probing run_root: {exc}"),
            "Retry once the parallel filesystem recovers, or check the `run_root` mount.",
        )

    # 7. Package / image version consistency
    try:
        version_file = os.path.join(resolved_pg, "VERSION") if resolved_pg else ""
        if version_file and os.path.isfile(version_file):
            with open(version_file, encoding="utf-8") as vf:
                recorded_ver = vf.read().strip()
            if recorded_ver == __version__:
                _add("image_version", True, f"aimip {__version__} matches {version_file}")
            else:
                _add(
                    "image_version",
                    False,
                    f"aimip package version ({__version__}) != prefix VERSION ({recorded_ver})",
                    "Re-run `install/install-container.sh` or `install/install-native.sh` to synchronize the installation.",
                )
        else:
            _add("image_version", True, f"aimip {__version__}")
    except OSError as exc:
        _add(
            "image_version",
            False,
            sanitize_terminal_text(f"Filesystem error reading VERSION file: {exc}"),
            "Retry once the filesystem recovers, or check the `$AIMIP_HOME` mount.",
        )

    passed = sum(1 for c in checks if c["ok"])
    return {
        "ok": passed == len(checks),
        "passed": passed,
        "total": len(checks),
        "checks": checks,
    }
