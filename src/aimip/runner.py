"""Config generation, SLURM batch script rendering, manifest tracking, and `exec-fme`.

Neither `torch` nor `fme` is imported at module import time:
- `prepare_run` without `--lora` runs completely without `torch` or `fme`.
- `prepare_run` with `--lora` lazily imports `torch` via `aimip.lora.inspect_lora`.
- `exec_fme` lazily imports `fme.core.set_seed` inside the container at job execution time.

fme 2026.5.1 runtime notes:
- Ensemble broadcast (`DataShapesNotUniform` when `n_ensemble_per_ic > 1`):
  1. `fme/ace/inference/inference.py:308-313` calls
     `get_initial_condition(..., n_ensemble=config.n_ensemble_per_ic)` (`inference.py:149`),
     invoking `BatchData.broadcast_ensemble(n_ensemble=K)`
     (`fme/ace/data_loading/batch_data.py:436-465`) which repeats the IC tensor to
     batch size `K`, duplicates `initial_time` to length `K`, and sets `n_ensemble = K`.
  2. `get_forcing_data` (`fme/ace/data_loading/getters.py:258-283`) iterates over
     `initial_time.values[:, 0]` (length `K`) to construct `ExplicitIndices` of length `K`.
  3. `InferenceDataset._get_batch_data` (`fme/ace/data_loading/inference.py:287-331`)
     collates `K` forcing samples via `BatchData.from_sample_tuples`, giving `forcing`
     batch size `K` with default `forcing.n_ensemble = 1`.
  4. `Stepper.predict_paired` (`fme/ace/stepper/single_module.py:1238-1241`, and
     `Stepper.predict` at `1163-1166`) checks `if forcing.n_ensemble == 1 and
     initial_condition.as_batch_data().n_ensemble > 1:` and calls
     `forcing.broadcast_ensemble(K)` a SECOND time, inflating forcing batch size to `K*K`
     (`torch.Size([4, 180, 360])` vs prognostic `torch.Size([2, 180, 360])` for `K=2`),
     raising `DataShapesNotUniform` in `Packer.pack` (`fme/core/packer.py:35`).
  5. For `model.stochastic` with `K > 1`, aimip replicates the IC `K` times along `"sample"`
     into `<out>/ic_members.nc` with `n_ensemble_per_ic: 1`. `NoiseConditionedModel.forward`
     (`fme/ace/registry/stochastic_sfno.py:110-127`, `isotropic_noise` at `19-45`)
     draws independent Gaussian noise across batch dimension `x.shape[0] == K` every step.
     When `K == 1`, the original IC path is used unchanged.
- Quiet git revision lookup (`fatal: not a git repository` stderr suppression):
  `DatasetMetadata.from_env()` (`fme/ace/inference/data_writer/dataset_metadata.py:5,19`)
  calls `get_job_env()` (`fme/core/training_history.py:108-112`), which calls
  `git_revision_short_hash()` (`fme/core/training_history.py:97-105`), executing
  `subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])` at line 100 without
  `stderr=subprocess.DEVNULL` (and catching only `CalledProcessError`, not `OSError`).
  `exec_fme` sets `cwd = validated_out`, sets `GIT_CEILING_DIRECTORIES` to its parent, and
  installs a quiet `git_revision_short_hash` wrapper with `stderr=subprocess.DEVNULL` that
  still returns `None` outside git repos so output metadata stays identical.

Launcher and submission contract (`container/aimip-wrapper.in`, `<prefix>/bin/aimip`, `aimip.submit`):
- Generated `job.sbatch` contains `# aimip-generated v1` on line 2 and its body calls ONLY
  `"<prefix>/bin/aimip" exec-fme "<run-dir>"`.
- Default `--out` when omitted: `$AIMIP_DEFAULT_OUT_ROOT/<model>-<start>-<short_hash>`.
"""

from __future__ import annotations

import dataclasses
import datetime
import fnmatch
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import yaml

from . import __version__
from .errors import ConfigError, MissingAssetsError, ValidationError
from .ic import (
    ICFileInfo,
    _default_companion_ocean_ic_path,
    find_shipped_coupled_ic_for_time,
    find_shipped_ic_for_time,
    inspect_ic_file,
    list_model_ics,
    replicate_ic_for_ensemble,
)
from .lora import inspect_lora, sha256_file
from .registry import (
    DEFAULT_REGISTRY_PATH,
    PACKAGE_DEFAULT_REGISTRY_PATH,
    ModelSpec,
    get_model,
    get_model_ic_provenance,
    load_registry,
    resolve_data_dir,
    resolve_playground_dir,
    resolve_registry_path,
)
from .site_config import (
    VALID_SCHEDULERS,
    SiteConfig,
    load_site_config,
    parse_slurm_walltime_seconds,
)
from .time_utils import parse_cftime, select_forcing_set
from .validators import (
    makedirs_no_group_other_write,
    strip_group_other_write,
    validate_int_range,
    validate_iso_timestamp,
    validate_model_name,
    validate_python_module,
    validate_run_path,
    validate_safe_path,
    validate_seed,
    validate_slurm_job_id,
    validate_slurm_mem,
    validate_slurm_time,
    validate_slurm_token,
    validate_visible_input_path,
)

# Reference ACE2-ERA5 throughput measured on one A100 GPU: 0.0644 GPU-h per simulated model year.
ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR = 0.0644
HOURS_PER_JULIAN_YEAR = 365.25 * 24.0
MIN_WALLTIME_SECONDS = 20 * 60
MAX_WALLTIME_SECONDS = 12 * 3600
WALLTIME_SAFETY_FACTOR = 3.0

DEFAULT_PLAYGROUND_FALLBACK = os.path.dirname(DEFAULT_REGISTRY_PATH) or "/opt/aimip"
DEFAULT_FME_2026_10_OVERLAY = "/opt/aimip/overlay-fme-2026.10.0"
_OVERLAY_EVICT_PREFIXES: tuple[str, ...] = ("fme", "torch_harmonics", "zarrs")

_SAFE_USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclasses.dataclass(frozen=True)
class SlurmOptions:
    """Validated batch resource parameters derived from `site.json` + CLI overrides."""

    partition: str = "gpu"
    account: str = ""
    gres: str = "gpu:a100_80:1"
    cpus_per_task: int = 32
    mem: str = "64G"
    time: str | None = None
    job_name: str | None = None
    scheduler: str = "slurm"
    # MN5 (BSC MareNostrum 5): the QOS (acc_debug, acc_ehpc, ...) places the job; empty means no `--qos` line.
    qos: str = ""

    def validate(self) -> SlurmOptions:
        """Validate all fields against shell and SLURM token allowlists."""
        if self.scheduler not in VALID_SCHEDULERS:
            raise ValidationError(
                f"Invalid scheduler {self.scheduler!r}: expected 'slurm' or 'none'."
            )
        if self.scheduler == "slurm":
            # A partition is required unless a QOS is given (MN5: the QOS selects the partition, so the job script
            # carries `--qos` and no `--partition`).
            if not self.partition and not self.qos:
                raise ConfigError(
                    "SLURM --partition must be non-empty when scheduler='slurm' "
                    "(or give --qos, as on MN5)."
                )
            if self.partition:
                if self.partition == "CHANGE_ME":
                    raise ConfigError(
                        "SLURM --partition is set to 'CHANGE_ME' in site.json; "
                        "pass --partition <name> or run `aimip site init`."
                    )
                validate_slurm_token(self.partition, "SLURM --partition")
            if self.qos:
                validate_slurm_token(self.qos, "SLURM --qos")

            if self.account:
                if self.account == "CHANGE_ME":
                    raise ConfigError(
                        "SLURM --account is set to 'CHANGE_ME' in site.json; "
                        "pass --account <acct> or run `aimip site init`."
                    )
                validate_slurm_token(self.account, "SLURM --account")

            if self.gres:
                validate_slurm_token(self.gres, "SLURM --gres")

            validate_int_range(
                self.cpus_per_task, "SLURM --cpus-per-task", min_value=1, max_value=256
            )
            if self.mem:  # empty: no --mem line (MN5 refuses --mem; memory comes per core)
                validate_slurm_mem(self.mem)

        if self.time is not None:
            validate_slurm_time(self.time)
        if self.job_name is not None:
            validate_slurm_token(self.job_name, "SLURM --job-name")
        return self

    @classmethod
    def from_site(
        cls,
        site: SiteConfig,
        *,
        partition: str | None = None,
        account: str | None = None,
        gres: str | None = None,
        cpus_per_task: int | None = None,
        mem: str | None = None,
        time: str | None = None,
        job_name: str | None = None,
        qos: str | None = None,
    ) -> SlurmOptions:
        """Merge CLI SLURM overrides over a loaded `SiteConfig`."""
        eff_partition = site.partition
        if partition is not None:
            eff_partition = "" if partition.strip().lower() == "none" else partition
        eff_qos = site.qos
        if qos is not None:
            eff_qos = "" if qos.strip().lower() == "none" else qos.strip()
        eff_account = site.account
        if account is not None:
            eff_account = "" if account.strip().lower() == "none" else account.strip()
        eff_gres = site.gres
        if gres is not None:
            eff_gres = "" if gres.strip().lower() == "none" else gres.strip()
        return cls(
            partition=eff_partition,
            account=eff_account,
            gres=eff_gres,
            cpus_per_task=(
                cpus_per_task if cpus_per_task is not None else site.cpus_per_task
            ),
            mem=(("" if mem.strip().lower() == "none" else mem.strip()) if mem is not None else site.mem),
            time=time,
            job_name=job_name,
            scheduler=site.scheduler,
            qos=eff_qos,
        ).validate()


@dataclasses.dataclass(frozen=True)
class ICResolution:
    """Resolved IC source, ensemble sizing, and optional `ic_members*.nc` replication plan."""

    source_ic: ICFileInfo
    start_indices: dict[str, Any] | None
    total_members: int
    n_ic_samples: int
    n_ensemble_per_ic: int
    replicate_repeats: int = 1
    replicate_select_single: bool = True
    replicate_sample_slice: tuple[int, int] | None = None
    ensemble_strategy: str = "single_member"
    source_ocean_ic: ICFileInfo | None = None


def model_has_missing_files(
    model: ModelSpec,
    *,
    playground_dir: str | None = None,
    data_dir: str | None = None,
) -> list[str]:
    """Return missing `model.required_files` relative paths under `$AIMIP_DATA` (empty for custom test registries)."""
    if not model.required_files:
        return []
    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    if not resolved_data:
        missing = list(model.required_files)
        if model.family == "arches":
            missing.append("code/ArchesClimate-SSP")
        return missing
    missing: list[str] = []
    for rel in model.required_files:
        if not os.path.isfile(os.path.join(resolved_data, rel)):
            missing.append(rel)
    if model.family == "arches":
        override = os.environ.get("AIMIP_ARCHES_CODE")
        code_dir = override if override else os.path.join(resolved_data, "code", "ArchesClimate-SSP")
        if not os.path.isdir(code_dir):
            missing.append("code/ArchesClimate-SSP")
    return missing


def _check_data_dir_writable_for_autofetch(
    model_name: str, resolved_data: str | None
) -> None:
    """Refuse auto-fetch cleanly when `$AIMIP_DATA` is read-only."""
    data_mode = os.environ.get("AIMIP_DATA_MODE", "").strip().lower()
    if data_mode == "ro" or (
        resolved_data
        and os.path.exists(resolved_data)
        and not os.access(resolved_data, os.W_OK)
    ):
        raise ConfigError(
            f"model {model_name} is not installed here: "
            f"ask the install owner to run `aimip fetch -m {model_name}`"
        )


def ensure_model_files_ready(
    model: ModelSpec,
    *,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    auto_fetch: bool = True,
) -> ModelSpec:
    """Fetch and lock-verify *model* when its canonical `required_files` are missing on disk."""
    missing = model_has_missing_files(
        model, playground_dir=playground_dir, data_dir=data_dir
    )
    if not missing:
        return model

    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    if not auto_fetch:
        shown = ", ".join(missing[:3])
        if len(missing) > 3:
            shown += f", ... ({len(missing)} missing)"
        raise MissingAssetsError(
            f"Model {model.name!r} is missing required data file(s) under "
            f"{resolved_data or '$AIMIP_DATA'} ({shown}) and --no-fetch was specified. "
            f"Run `aimip fetch -m {model.name}` first or omit --no-fetch.",
            field="model",
            next_command=f"aimip fetch -m {model.name}",
        )
    _check_data_dir_writable_for_autofetch(model.name, resolved_data)

    from .setup_data import fetch_model

    fetch_model(
        model.name,
        dest_dir=resolved_data,
        playground_dir=resolved_pg,
        progress_stream=sys.stderr,
    )
    # Keep the caller's model: reloading by name would pick the data registry and drop an explicit --registry /
    # $AIMIP_REGISTRY entry or a caller-supplied spec. Re-check that the fetch produced the files this spec needs.
    still_missing = model_has_missing_files(model, playground_dir=playground_dir, data_dir=data_dir)
    if still_missing:
        raise ConfigError(
            f"Fetched {model.name!r}, but files this model entry needs are still missing: "
            + ", ".join(still_missing[:3]) + (f", ... ({len(still_missing)} missing)" if len(still_missing) > 3 else "")
        )
    return model


def resolve_model_for_prepare(
    model_name: str,
    *,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    auto_fetch: bool = True,
    precheck_fn: Callable[[ModelSpec], None] | None = None,
) -> ModelSpec:
    """Resolve *model_name* for `aimip run` / `aimip ic spin`, auto-fetching on first use when needed.

    Handles both:
    - Fresh `$AIMIP_DATA` (falls back to `registry.default.json` and fetches missing files), AND
    - Partial `<AIMIP_DATA>/registry.json` (written after `aimip fetch -m <other>`), where a second
      canonical model is not yet listed in `<AIMIP_DATA>/registry.json` until fetched and lock-verified.
    Explicit `--registry` or `$AIMIP_REGISTRY` overrides are strictly respected without fallback.
    """
    pre_fetch_reg = resolve_registry_path(
        explicit_path=registry_path, data_dir=data_dir
    )
    catalog = load_registry(
        registry_path=registry_path,
        playground_dir=playground_dir,
        data_dir=data_dir,
    )
    validated_name = validate_model_name(model_name)
    explicit_reg_pinned = bool(registry_path or os.environ.get("AIMIP_REGISTRY"))
    used_packaged_fallback = not explicit_reg_pinned and os.path.normpath(
        pre_fetch_reg
    ) == os.path.normpath(PACKAGE_DEFAULT_REGISTRY_PATH)

    if validated_name in catalog:
        spec = catalog[validated_name]
        missing = model_has_missing_files(
            spec, playground_dir=playground_dir, data_dir=data_dir
        )
        if not missing:
            return spec
        if not used_packaged_fallback:
            if precheck_fn is not None:
                precheck_fn(spec)
            return ensure_model_files_ready(
                spec,
                playground_dir=playground_dir,
                data_dir=data_dir,
                auto_fetch=auto_fetch,
            )

    if explicit_reg_pinned or not os.path.isfile(PACKAGE_DEFAULT_REGISTRY_PATH):
        validate_model_name(model_name, allowed=set(catalog.keys()))
        raise ConfigError(f"Unknown model {model_name!r}.")

    default_catalog = load_registry(
        registry_path=PACKAGE_DEFAULT_REGISTRY_PATH,
        playground_dir=playground_dir,
        data_dir=data_dir,
    )
    if validated_name not in default_catalog:
        validate_model_name(model_name, allowed=set(default_catalog.keys()))
        raise ConfigError(f"Unknown model {model_name!r}.")

    default_spec = default_catalog[validated_name]
    if precheck_fn is not None:
        precheck_fn(default_spec)

    resolved_pg = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_pg)
    if not auto_fetch:
        raise MissingAssetsError(
            f"Model {default_spec.name!r} is not yet fetched/verified in "
            f"{resolved_data or '$AIMIP_DATA'} and --no-fetch was specified. "
            f"Run `aimip fetch -m {default_spec.name}` first or omit --no-fetch.",
            field="model",
            next_command=f"aimip fetch -m {default_spec.name}",
        )
    _check_data_dir_writable_for_autofetch(default_spec.name, resolved_data)

    from .setup_data import fetch_model

    fetch_model(
        default_spec.name,
        dest_dir=resolved_data,
        playground_dir=resolved_pg,
        progress_stream=sys.stderr,
    )
    return get_model(
        default_spec.name,
        registry_path=registry_path,
        playground_dir=playground_dir,
        data_dir=data_dir,
    )


def _format_walltime_hms(total_seconds: int) -> str:
    hours, rem = divmod(max(0, int(total_seconds)), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def compute_slurm_walltime_details(
    steps: int,
    timestep_hours: int = 6,
    n_trajectories: int = 1,
    max_walltime: str | None = None,
) -> tuple[str, str, bool]:
    """Return `(clamped_walltime, unclamped_walltime, walltime_capped)`."""
    validate_int_range(steps, "steps", min_value=1, max_value=10_000_000)
    validate_int_range(timestep_hours, "timestep_hours", min_value=1, max_value=24 * 366)
    validate_int_range(n_trajectories, "n_trajectories", min_value=1, max_value=100_000)

    cap_seconds = (
        parse_slurm_walltime_seconds(max_walltime)
        if max_walltime is not None
        else MAX_WALLTIME_SECONDS
    )
    cap_seconds = max(MIN_WALLTIME_SECONDS, cap_seconds)

    simulated_hours = float(steps) * float(timestep_hours) * float(n_trajectories)
    model_years = simulated_hours / HOURS_PER_JULIAN_YEAR
    estimated_hours = ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR * model_years
    padded_seconds = int(math.ceil(WALLTIME_SAFETY_FACTOR * estimated_hours * 3600.0))
    unclamped_seconds = max(MIN_WALLTIME_SECONDS, padded_seconds)
    walltime_capped = unclamped_seconds > cap_seconds
    clamped_seconds = min(cap_seconds, unclamped_seconds)

    clamped_str = validate_slurm_time(_format_walltime_hms(clamped_seconds))
    unclamped_str = _format_walltime_hms(unclamped_seconds)
    return clamped_str, unclamped_str, walltime_capped


def estimate_slurm_walltime(
    steps: int,
    timestep_hours: int = 6,
    n_trajectories: int = 1,
    max_walltime: str | None = None,
) -> str:
    """Estimate SLURM `--time` as `min(max_time, max(20min, 3 * 0.0644 GPU-h/yr * years))`."""
    clamped_str, _, _ = compute_slurm_walltime_details(
        steps=steps,
        timestep_hours=timestep_hours,
        n_trajectories=n_trajectories,
        max_walltime=max_walltime,
    )
    return clamped_str


def resolve_active_lock_sha256() -> str | None:
    """Return SHA-256 hex digest of the active `aimip.lock` file when present on disk."""
    candidates: list[str] = []
    env_lock = os.environ.get("AIMIP_LOCK", "").strip()
    if env_lock:
        candidates.append(env_lock)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_parent = os.path.abspath(os.path.join(pkg_dir, ".."))
    candidates.append(os.path.join(pkg_dir, "aimip.lock"))
    candidates.append(os.path.join(pkg_parent, "aimip.lock"))
    candidates.append("/opt/aimip/src/src/aimip/aimip.lock")
    candidates.append("/opt/aimip/src/aimip.lock")
    for cand in candidates:
        try:
            if cand and os.path.isfile(cand):
                return sha256_file(cand)
        except OSError:
            continue
    return None


def get_fme_version() -> str:
    """Query installed `fme` package version via stdlib metadata without importing `fme`."""
    try:
        return importlib.metadata.version("fme")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _sanitize_env_string(raw: str | None, max_len: int = 256) -> str:
    """Return a printable single-line string from an environment variable."""
    if not raw:
        return ""
    cleaned = "".join(ch for ch in raw if 32 <= ord(ch) < 127)
    return cleaned[:max_len]


def _resolve_default_out_root() -> str:
    """Resolve `$AIMIP_DEFAULT_OUT_ROOT` (set by `bin/aimip` from `site.json`), falling back to `/scratch/<u>/<user>/aimip-runs`."""
    env_root = os.environ.get("AIMIP_DEFAULT_OUT_ROOT")
    if env_root:
        return validate_run_path(env_root, label="AIMIP_DEFAULT_OUT_ROOT")

    raw_user = (
        os.environ.get("AIMIP_USER")
        or os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or getpass.getuser()
        or "aimip"
    )
    if not _SAFE_USER_RE.match(raw_user):
        raise ValidationError(
            f"Cannot derive safe default --out root from username {raw_user!r}; "
            "set $AIMIP_DEFAULT_OUT_ROOT or pass --out explicitly."
        )
    return validate_run_path(
        f"/scratch/{raw_user[0]}/{raw_user}/aimip-runs",
        label="default output root",
    )


def default_out_dir(
    model_name: str,
    start_iso: str,
    *,
    extra_entropy: str = "",
) -> str:
    """Construct `$AIMIP_DEFAULT_OUT_ROOT/<model>-<start>-<short_hash>`.

    Colons are stripped from `<start>` (`1979-01-01T000000`) so the generated path
    passes `safe_path()` character allowlist `[A-Za-z0-9._/+-]`.
    """
    root = _resolve_default_out_root().rstrip("/")
    norm_start = validate_iso_timestamp(start_iso, label="start timestamp")
    start_slug = norm_start.replace(":", "")
    seed_material = (
        f"{model_name}|{norm_start}|{extra_entropy}|{time.time_ns()}|{os.getpid()}|{uuid.uuid4().hex}"
    )
    short_hash = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8]
    path = f"{root}/{model_name}-{start_slug}-{short_hash}"
    return validate_run_path(path, label="default --out directory")


def _resolve_wrapper_bin(playground_dir: str | None) -> str:
    """Resolve and validate host wrapper path `$AIMIP_PLAYGROUND/bin/aimip`."""
    resolved_pg = resolve_playground_dir(playground_dir) or DEFAULT_PLAYGROUND_FALLBACK
    base = resolved_pg.rstrip("/") or "/"
    return validate_safe_path(f"{base}/bin/aimip", label="host wrapper path")


def render_sbatch_script(
    *,
    model_name: str,
    out_dir: str,
    playground_dir: str | None,
    slurm_opts: SlurmOptions,
    default_walltime: str,
    exec_dirs: Sequence[str] | None = None,
) -> str:
    """Render a strictly validated `job.sbatch` script carrying `# aimip-generated v1`.

    Matches `container/aimip-wrapper.in` and `aimip.submit.check_job_script`:
    - Line 2 is `# aimip-generated v1`.
    - When `scheduler == "slurm"`, emits `#SBATCH` lines (omitting `--account` or `--gres`
      when empty so every `#SBATCH` line has a non-empty `[A-Za-z0-9._/:%+-]+` value).
    - When `scheduler == "none"` (`local.json`), omits `#SBATCH` directives so `nohup bash`
      runs the same validated script directly.
    """
    opts = slurm_opts.validate()
    validated_out = validate_run_path(out_dir, label="output directory")
    wrapper_bin = _resolve_wrapper_bin(playground_dir)
    job_name = validate_slurm_token(
        opts.job_name or f"aimip-{model_name}", "SLURM --job-name"
    )
    walltime = validate_slurm_time(opts.time or default_walltime)
    targets = [
        validate_run_path(d, label="exec-fme target directory")
        for d in (exec_dirs if exec_dirs is not None else [validated_out])
    ]
    if not targets:
        raise ValidationError(
            "render_sbatch_script requires at least one exec-fme target directory."
        )
    exec_lines = "\n".join(f'"{wrapper_bin}" exec-fme "{target}"' for target in targets)

    lines = [
        "#!/bin/bash",
        "# aimip-generated v1",
    ]
    if opts.scheduler == "slurm":
        lines.append(f"#SBATCH --job-name={job_name}")
        if opts.partition:
            lines.append(f"#SBATCH --partition={opts.partition}")
        if opts.qos:
            # MN5: the QOS selects the partition (acc_ehpc, acc_debug, ...); there may be no partition line
            lines.append(f"#SBATCH --qos={opts.qos}")
        if opts.account:
            lines.append(f"#SBATCH --account={opts.account}")
        if opts.gres:
            lines.append(f"#SBATCH --gres={opts.gres}")
        lines.append(f"#SBATCH --cpus-per-task={opts.cpus_per_task}")
        # --cpus-per-task counts physical cores: without the hint Slurm on SMT nodes (Levante: 128 cores, 256
        # threads) hands out half as many cores with two threads each. Harmless where there is no SMT.
        lines.append("#SBATCH --hint=nomultithread")
        if opts.mem:
            # MN5 refuses any job that sets --mem (memory is given per core); its site.json has mem "" -> no line
            lines.append(f"#SBATCH --mem={opts.mem}")
        lines.append(f"#SBATCH --time={walltime}")
        lines.append(f"#SBATCH --output={validated_out}/slurm-%j.out")
        lines.append(f"#SBATCH --error={validated_out}/slurm-%j.err")

    lines.append("")
    lines.append("set -euo pipefail")
    lines.append(exec_lines)
    return "\n".join(lines) + "\n"


def _list_forcing_files(forcing_dir: str) -> list[dict[str, Any]]:
    """List regular files inside *forcing_dir* with byte sizes, sorted deterministically."""
    validated_dir = validate_safe_path(forcing_dir, label="forcing_set.dir")
    if not os.path.isdir(validated_dir):
        raise ValidationError(f"Forcing directory not found: {validated_dir}")

    entries: list[dict[str, Any]] = []
    for name in sorted(os.listdir(validated_dir)):
        full = os.path.join(validated_dir, name)
        if os.path.isfile(full) and not os.path.islink(full):
            st = os.stat(full)
            entries.append({"name": name, "size_bytes": int(st.st_size)})
    return entries


def _hash_and_verify_forcing_files(
    forcing_dir: str,
    file_pattern: str | None = None,
    *,
    model: ModelSpec | None = None,
    data_dir: str | None = None,
) -> dict[str, str]:
    """Hash matched forcing files in *forcing_dir* and verify pinned `required_files` against `aimip.lock`."""
    validated_dir = validate_safe_path(forcing_dir, label="forcing_set.dir")
    if not os.path.isdir(validated_dir) or os.path.islink(validated_dir):
        raise ValidationError(f"Forcing directory not found or symlinked: {validated_dir}")

    pat = file_pattern or "*.nc"
    required_set = set(model.required_files) if (model is not None and model.required_files) else set()
    lock_files: dict[str, Any] = {}
    derived_overrides: dict[str, str] = {}
    base_data = data_dir.rstrip("/") if data_dir else ""
    if required_set and base_data:
        from .families import _load_derived_overrides, _load_lock_entries

        lock_files = _load_lock_entries(data_dir=base_data)
        derived_overrides = _load_derived_overrides(data_dir=base_data)

    hashes: dict[str, str] = {}
    for name in sorted(os.listdir(validated_dir)):
        if not fnmatch.fnmatch(name, pat):
            continue
        full = os.path.join(validated_dir, name)
        if os.path.islink(full):
            raise ValidationError(f"Refusing symlinked forcing file: {full}")
        if not os.path.isfile(full):
            continue
        actual_sha = sha256_file(full)
        if required_set and base_data and full.startswith(base_data + "/"):
            rel = full[len(base_data) + 1 :]
            if rel in required_set:
                expected_sha = derived_overrides.get(rel) or (lock_files.get(rel) or {}).get("sha256")
                if expected_sha and actual_sha.lower() != str(expected_sha).lower():
                    raise ValidationError(
                        f"Pinned forcing SHA-256 mismatch for {rel} ({full}): "
                        f"expected {expected_sha}, got {actual_sha}. "
                        f"Re-run `aimip fetch -m {model.name} --force-rebuild`."
                    )
        hashes[full] = actual_sha
    return hashes


def _write_json_atomic(target_path: str, payload: dict[str, Any]) -> None:
    """Write *payload* to *target_path* atomically via sibling temporary file."""
    validated_target = validate_run_path(target_path, label="manifest path")
    parent = os.path.dirname(validated_target) or "/"
    if parent != "/":
        makedirs_no_group_other_write(parent)
    tmp_path = os.path.join(parent, f".aimip_tmp_{uuid.uuid4().hex}.json")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
        strip_group_other_write(tmp_path)
        os.replace(tmp_path, validated_target)
        strip_group_other_write(validated_target)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _count_shipped_ic_timestamps(model: ModelSpec) -> list[str]:
    """Return sorted distinct timestamps across all shipped IC files of *model*."""
    distinct: set[str] = set()
    for inf in list_model_ics(model):
        distinct.update(inf.timestamps)
    return sorted(distinct)


def _resolve_custom_coupled_ic_paths(
    ic_path: str, ocean_ic_path: str | None
) -> tuple[str, str]:
    """Resolve custom coupled `(atmos_ic_path, ocean_ic_path)` from `--ic` (file or run dir) and optional `--ocean-ic`."""
    validated_ic = validate_visible_input_path(ic_path, label="--ic")
    if os.path.isdir(validated_ic):
        atmos_cand = validate_visible_input_path(
            f"{validated_ic}/atmosphere/restart.nc",
            label="--ic atmosphere/restart.nc",
        )
        ocean_cand = (
            validate_visible_input_path(ocean_ic_path, label="--ocean-ic")
            if ocean_ic_path is not None
            else validate_visible_input_path(
                f"{validated_ic}/ocean/restart.nc",
                label="--ic ocean/restart.nc",
            )
        )
        return atmos_cand, ocean_cand

    if ocean_ic_path is not None:
        return validated_ic, validate_visible_input_path(
            ocean_ic_path, label="--ocean-ic"
        )

    # Auto-pair `<prev_dir>/atmosphere/restart.nc` -> `<prev_dir>/ocean/restart.nc`,
    # or `<spun>.nc` -> `<spun>_ocean.nc` from `aimip collect`.
    parent_dir = os.path.dirname(validated_ic)
    if (
        os.path.basename(validated_ic) == "restart.nc"
        and os.path.basename(parent_dir) == "atmosphere"
    ):
        run_root = os.path.dirname(parent_dir)
        sibling_ocean = os.path.join(run_root, "ocean", "restart.nc")
        if os.path.isfile(sibling_ocean):
            return validated_ic, validate_visible_input_path(
                sibling_ocean, label="paired ocean/restart.nc"
            )

    companion_ocean = _default_companion_ocean_ic_path(validated_ic)
    if os.path.isfile(companion_ocean):
        return validated_ic, validate_visible_input_path(
            companion_ocean, label="companion ocean IC"
        )

    raise ConfigError(
        f"Coupled model requires both atmosphere and ocean initial conditions: "
        f"pass `--ocean-ic <ocean.nc>` alongside `--ic {validated_ic}`, "
        "or pass `--ic <prev-run-dir>` containing `atmosphere/restart.nc` and `ocean/restart.nc`."
    )


def _resolve_coupled_ic_and_members(
    *,
    model: ModelSpec,
    norm_start: str,
    ic_path: str | None,
    ocean_ic_path: str | None,
    members: int | None,
    candidate_ic_files: Sequence[str] | None,
    candidate_ocean_ic_files: Sequence[str] | None,
) -> ICResolution:
    """Resolve paired `(atmosphere, ocean)` ICs, `start_indices`, and ensemble plan for `fme.coupled.inference`."""
    if ic_path is None:
        if ocean_ic_path is not None:
            raise ConfigError(
                "Specify both --ic and --ocean-ic (or --ic <prev-run-dir>) for custom coupled initial conditions."
            )
        atmos_info, ocean_info, first_idx = find_shipped_coupled_ic_for_time(
            model,
            norm_start,
            candidate_ic_files=candidate_ic_files,
            candidate_ocean_ic_files=candidate_ocean_ic_files,
        )
        req_members = (
            members if members is not None else model.default_n_ensemble_per_ic
        )
        avail_contig = 0
        while (
            first_idx + avail_contig < ocean_info.n_samples
            and ocean_info.timestamps[first_idx + avail_contig] == norm_start
        ):
            avail_contig += 1

        if not model.stochastic:
            if req_members > avail_contig:
                distinct_shipped = _count_shipped_ic_timestamps(model)
                if len(distinct_shipped) < req_members:
                    raise ConfigError(
                        f"Model {model.name!r} is deterministic and ships only "
                        f"{len(distinct_shipped)} initial-condition state(s) "
                        f"({', '.join(distinct_shipped) or 'none'}), so `aimip ic spin` from "
                        f"shipped ICs cannot produce {req_members} distinct members. "
                        f"Run with `--members 1` or pass custom multi-sample `--ic` + `--ocean-ic` "
                        f"containing at least {req_members} distinct initial states."
                    )
                raise ConfigError(
                    f"Model {model.name!r} is deterministic and shipped IC {atmos_info.path} "
                    f"provides only {avail_contig} initial state(s) at {norm_start} (requested --members {req_members}). "
                    f"Use `aimip ic spin -m {model.name} --from T1,T2,... --to {norm_start} --out <spin-dir>` "
                    "and `aimip collect <spin-dir> --out <ic.nc>` to generate K distinct initial states first."
                )
            if ocean_info.has_sample_coord:
                return ICResolution(
                    source_ic=atmos_info,
                    source_ocean_ic=ocean_info,
                    start_indices={
                        "first": first_idx,
                        "n_initial_conditions": req_members,
                    },
                    total_members=req_members,
                    n_ic_samples=req_members,
                    n_ensemble_per_ic=1,
                    replicate_repeats=1,
                    replicate_select_single=False,
                    ensemble_strategy=(
                        "multi_sample_ic" if req_members > 1 else "single_member"
                    ),
                )
            if first_idx == 0 and req_members == ocean_info.n_samples:
                return ICResolution(
                    source_ic=atmos_info,
                    source_ocean_ic=ocean_info,
                    start_indices=None,
                    total_members=req_members,
                    n_ic_samples=req_members,
                    n_ensemble_per_ic=1,
                    replicate_repeats=1,
                    replicate_select_single=False,
                    ensemble_strategy=(
                        "multi_sample_ic" if req_members > 1 else "single_member"
                    ),
                )
            return ICResolution(
                source_ic=atmos_info,
                source_ocean_ic=ocean_info,
                start_indices=None,
                total_members=req_members,
                n_ic_samples=req_members,
                n_ensemble_per_ic=1,
                replicate_repeats=1,
                replicate_select_single=False,
                replicate_sample_slice=(first_idx, req_members),
                ensemble_strategy=(
                    "multi_sample_ic" if req_members > 1 else "single_member"
                ),
            )

        # Stochastic coupled model (e.g. `samudrace-e3smv3` with 3-state `ICx3`).
        if req_members <= avail_contig:
            if ocean_info.has_sample_coord:
                return ICResolution(
                    source_ic=atmos_info,
                    source_ocean_ic=ocean_info,
                    start_indices={
                        "first": first_idx,
                        "n_initial_conditions": req_members,
                    },
                    total_members=req_members,
                    n_ic_samples=req_members,
                    n_ensemble_per_ic=1,
                    replicate_repeats=1,
                    replicate_select_single=False,
                    ensemble_strategy=(
                        "multi_sample_ic" if req_members > 1 else "single_member"
                    ),
                )
            if first_idx == 0 and req_members == ocean_info.n_samples:
                return ICResolution(
                    source_ic=atmos_info,
                    source_ocean_ic=ocean_info,
                    start_indices=None,
                    total_members=req_members,
                    n_ic_samples=req_members,
                    n_ensemble_per_ic=1,
                    replicate_repeats=1,
                    replicate_select_single=False,
                    ensemble_strategy=(
                        "multi_sample_ic" if req_members > 1 else "single_member"
                    ),
                )
            return ICResolution(
                source_ic=atmos_info,
                source_ocean_ic=ocean_info,
                start_indices=None,
                total_members=req_members,
                n_ic_samples=req_members,
                n_ensemble_per_ic=1,
                replicate_repeats=1,
                replicate_select_single=False,
                replicate_sample_slice=(first_idx, req_members),
                ensemble_strategy=(
                    "multi_sample_ic" if req_members > 1 else "single_member"
                ),
            )

        if avail_contig > 1 and req_members % avail_contig == 0:
            return ICResolution(
                source_ic=atmos_info,
                source_ocean_ic=ocean_info,
                start_indices=None,
                total_members=req_members,
                n_ic_samples=req_members,
                n_ensemble_per_ic=1,
                replicate_repeats=req_members // avail_contig,
                replicate_select_single=False,
                replicate_sample_slice=(first_idx, avail_contig),
                ensemble_strategy="replicated_ic_sample_dim",
            )

        return ICResolution(
            source_ic=atmos_info,
            source_ocean_ic=ocean_info,
            start_indices=None,
            total_members=req_members,
            n_ic_samples=req_members,
            n_ensemble_per_ic=1,
            replicate_repeats=req_members,
            replicate_select_single=False,
            replicate_sample_slice=(first_idx, 1),
            ensemble_strategy="replicated_ic_sample_dim",
        )

    # Custom coupled `--ic` (+ `--ocean-ic` or `<prev-run-dir>`).
    atmos_custom, ocean_custom = _resolve_custom_coupled_ic_paths(ic_path, ocean_ic_path)
    atmos_info = inspect_ic_file(atmos_custom)
    ocean_info = inspect_ic_file(ocean_custom)
    if atmos_info.n_samples != ocean_info.n_samples:
        raise ConfigError(
            f"Custom coupled IC sample-count mismatch: atmosphere {atmos_info.path} has "
            f"{atmos_info.n_samples} sample(s) vs ocean {ocean_info.path} with {ocean_info.n_samples}."
        )
    if atmos_info.calendar != ocean_info.calendar:
        raise ConfigError(
            f"Custom coupled IC calendar mismatch: atmosphere ({atmos_info.calendar}) != "
            f"ocean ({ocean_info.calendar})."
        )
    for comp_label, comp_info in (("atmosphere", atmos_info), ("ocean", ocean_info)):
        mismatched = sorted({ts for ts in comp_info.timestamps if ts != norm_start})
        if mismatched:
            raise ConfigError(
                f"All sample timestamps in custom {comp_label} IC {comp_info.path} must equal "
                f"--start {norm_start}, but found: {', '.join(mismatched[:8])}."
            )

    if not model.stochastic:
        req_members = members if members is not None else ocean_info.n_samples
        if ocean_info.n_samples < req_members:
            raise ConfigError(
                f"Deterministic coupled model {model.name!r} requires at least {req_members} samples "
                f"in --ic {atmos_info.path}, but file contains only {ocean_info.n_samples}."
            )
        if req_members == ocean_info.n_samples:
            return ICResolution(
                source_ic=atmos_info,
                source_ocean_ic=ocean_info,
                start_indices=None,
                total_members=req_members,
                n_ic_samples=req_members,
                n_ensemble_per_ic=1,
                replicate_repeats=1,
                replicate_select_single=False,
                ensemble_strategy=(
                    "multi_sample_ic" if req_members > 1 else "single_member"
                ),
            )
        if ocean_info.has_sample_coord:
            return ICResolution(
                source_ic=atmos_info,
                source_ocean_ic=ocean_info,
                start_indices={
                    "first": 0,
                    "n_initial_conditions": req_members,
                },
                total_members=req_members,
                n_ic_samples=req_members,
                n_ensemble_per_ic=1,
                replicate_repeats=1,
                replicate_select_single=False,
                ensemble_strategy=(
                    "multi_sample_ic" if req_members > 1 else "single_member"
                ),
            )
        return ICResolution(
            source_ic=atmos_info,
            source_ocean_ic=ocean_info,
            start_indices=None,
            total_members=req_members,
            n_ic_samples=req_members,
            n_ensemble_per_ic=1,
            replicate_repeats=1,
            replicate_select_single=False,
            replicate_sample_slice=(0, req_members),
            ensemble_strategy=(
                "multi_sample_ic" if req_members > 1 else "single_member"
            ),
        )

    # Stochastic coupled model with custom `--ic`.
    if members is None or members == ocean_info.n_samples:
        return ICResolution(
            source_ic=atmos_info,
            source_ocean_ic=ocean_info,
            start_indices=None,
            total_members=ocean_info.n_samples,
            n_ic_samples=ocean_info.n_samples,
            n_ensemble_per_ic=1,
            replicate_repeats=1,
            replicate_select_single=False,
            ensemble_strategy=(
                "multi_sample_ic" if ocean_info.n_samples > 1 else "single_member"
            ),
        )
    if members < ocean_info.n_samples:
        if ocean_info.has_sample_coord:
            return ICResolution(
                source_ic=atmos_info,
                source_ocean_ic=ocean_info,
                start_indices={
                    "first": 0,
                    "n_initial_conditions": members,
                },
                total_members=members,
                n_ic_samples=members,
                n_ensemble_per_ic=1,
                replicate_repeats=1,
                replicate_select_single=False,
                ensemble_strategy=(
                    "multi_sample_ic" if members > 1 else "single_member"
                ),
            )
        return ICResolution(
            source_ic=atmos_info,
            source_ocean_ic=ocean_info,
            start_indices=None,
            total_members=members,
            n_ic_samples=members,
            n_ensemble_per_ic=1,
            replicate_repeats=1,
            replicate_select_single=False,
            replicate_sample_slice=(0, members),
            ensemble_strategy=(
                "multi_sample_ic" if members > 1 else "single_member"
            ),
        )
    if members % ocean_info.n_samples != 0:
        raise ConfigError(
            f"--ic {atmos_info.path} contains {ocean_info.n_samples} sample(s), which does not "
            f"evenly divide requested --members {members}."
        )
    return ICResolution(
        source_ic=atmos_info,
        source_ocean_ic=ocean_info,
        start_indices=None,
        total_members=members,
        n_ic_samples=members,
        n_ensemble_per_ic=1,
        replicate_repeats=members // ocean_info.n_samples,
        replicate_select_single=False,
        ensemble_strategy="replicated_ic_sample_dim",
    )


def resolve_ic_and_members(
    *,
    model: ModelSpec,
    norm_start: str,
    ic_path: str | None,
    members: int | None,
    ocean_ic_path: str | None = None,
    candidate_ic_files: Sequence[str] | None = None,
    candidate_ocean_ic_files: Sequence[str] | None = None,
) -> ICResolution:
    """Resolve IC dataset(s), `start_indices`, ensemble counts, and `ic_members*.nc` replication plan.

    Always produces `n_ensemble_per_ic == 1` so `fme` 2026.5.1 never hits the double
    forcing broadcast bug in `Stepper.predict_paired` (`single_module.py:1238-1241`).
    - When `model.stochastic` and `members > 1`: flags replication along `"sample"` so
      `prepare_run` writes `<out>/ic_members.nc` (or `<out>/ic_members_{atmosphere,ocean}.nc`).
    - When `members == 1` (or omitted with default 1): uses the original IC path untouched.
    """
    if members is not None:
        validate_int_range(members, "members", min_value=1, max_value=10_000)

    if model.is_coupled:
        return _resolve_coupled_ic_and_members(
            model=model,
            norm_start=norm_start,
            ic_path=ic_path,
            ocean_ic_path=ocean_ic_path,
            members=members,
            candidate_ic_files=candidate_ic_files,
            candidate_ocean_ic_files=candidate_ocean_ic_files,
        )

    if ocean_ic_path is not None:
        raise ConfigError(
            f"--ocean-ic is only valid for coupled models (`fme.coupled.inference`), not {model.name!r}."
        )

    if ic_path is None:
        ic_info = find_shipped_ic_for_time(
            model, norm_start, candidate_ic_files=candidate_ic_files
        )
        req_members = (
            members if members is not None else model.default_n_ensemble_per_ic
        )
        if not model.stochastic:
            if req_members > 1:
                distinct_shipped = _count_shipped_ic_timestamps(model)
                if len(distinct_shipped) < req_members:
                    raise ConfigError(
                        f"Model {model.name!r} is deterministic and ships only "
                        f"{len(distinct_shipped)} initial-condition state(s) "
                        f"({', '.join(distinct_shipped) or 'none'}), so `aimip ic spin` from "
                        f"shipped ICs cannot produce {req_members} distinct members. "
                        f"Run with `--members 1` or pass a custom multi-sample `--ic <file.nc>` "
                        f"containing at least {req_members} distinct initial states."
                    )
                raise ConfigError(
                    f"Model {model.name!r} is deterministic and shipped IC {ic_info.path} "
                    f"provides only 1 initial state at {norm_start} (requested --members {req_members}). "
                    f"Use `aimip ic spin -m {model.name} --from T1,T2,... --to {norm_start} --out <spin-dir>` "
                    "and `aimip collect <spin-dir> --out <ic.nc>` to generate K distinct initial states first."
                )
            return ICResolution(
                source_ic=ic_info,
                start_indices={"times": [norm_start]},
                total_members=1,
                n_ic_samples=1,
                n_ensemble_per_ic=1,
                replicate_repeats=1,
                replicate_select_single=True,
                ensemble_strategy="single_member",
            )

        if req_members == 1:
            return ICResolution(
                source_ic=ic_info,
                start_indices={"times": [norm_start]},
                total_members=1,
                n_ic_samples=1,
                n_ensemble_per_ic=1,
                replicate_repeats=1,
                replicate_select_single=True,
                ensemble_strategy="single_member",
            )

        return ICResolution(
            source_ic=ic_info,
            start_indices=None,
            total_members=req_members,
            n_ic_samples=req_members,
            n_ensemble_per_ic=1,
            replicate_repeats=req_members,
            replicate_select_single=True,
            ensemble_strategy="replicated_ic_sample_dim",
        )

    # Custom `--ic FILE` (including `<prev>/restart.nc` or merged spin `ic.nc`).
    validated_custom_ic = validate_visible_input_path(ic_path, label="--ic")
    ic_info = inspect_ic_file(validated_custom_ic)
    mismatched = sorted({ts for ts in ic_info.timestamps if ts != norm_start})
    if mismatched:
        raise ConfigError(
            f"All sample timestamps in --ic {ic_info.path} must equal --start {norm_start}, "
            f"but found: {', '.join(mismatched[:8])}."
        )

    if not model.stochastic:
        req_members = members if members is not None else ic_info.n_samples
        if ic_info.n_samples < req_members:
            distinct_shipped = _count_shipped_ic_timestamps(model)
            if len(distinct_shipped) >= req_members:
                hint = (
                    f"Generate {req_members} distinct states with "
                    f"`aimip ic spin -m {model.name} --from T1,T2,... --to {norm_start} --out <spin-dir>`."
                )
            else:
                hint = (
                    f"Model {model.name!r} ships only {len(distinct_shipped)} IC state(s), "
                    f"so provide a custom multi-sample `--ic <file.nc>` with at least {req_members} states."
                )
            raise ConfigError(
                f"Deterministic model {model.name!r} requires at least {req_members} samples "
                f"in --ic {ic_info.path}, but file contains only {ic_info.n_samples}. {hint}"
            )
        start_indices: dict[str, Any] | None = (
            None
            if req_members == ic_info.n_samples
            else {
                "n_initial_conditions": req_members,
                "first": 0,
                "interval": 1,
            }
        )
        return ICResolution(
            source_ic=ic_info,
            start_indices=start_indices,
            total_members=req_members,
            n_ic_samples=req_members,
            n_ensemble_per_ic=1,
            replicate_repeats=1,
            replicate_select_single=False,
            ensemble_strategy="multi_sample_ic" if req_members > 1 else "single_member",
        )

    # Stochastic model with custom `--ic FILE` or `<prev>/restart.nc`.
    if members is None or members == ic_info.n_samples:
        return ICResolution(
            source_ic=ic_info,
            start_indices=None,
            total_members=ic_info.n_samples,
            n_ic_samples=ic_info.n_samples,
            n_ensemble_per_ic=1,
            replicate_repeats=1,
            replicate_select_single=False,
            ensemble_strategy=(
                "multi_sample_ic" if ic_info.n_samples > 1 else "single_member"
            ),
        )

    if members < ic_info.n_samples:
        return ICResolution(
            source_ic=ic_info,
            start_indices={
                "n_initial_conditions": members,
                "first": 0,
                "interval": 1,
            },
            total_members=members,
            n_ic_samples=members,
            n_ensemble_per_ic=1,
            replicate_repeats=1,
            replicate_select_single=False,
            ensemble_strategy="multi_sample_ic" if members > 1 else "single_member",
        )

    if members % ic_info.n_samples != 0:
        raise ConfigError(
            f"--ic {ic_info.path} contains {ic_info.n_samples} sample(s), which does not "
            f"evenly divide requested --members {members}."
        )
    repeats = members // ic_info.n_samples
    return ICResolution(
        source_ic=ic_info,
        start_indices=None,
        total_members=members,
        n_ic_samples=members,
        n_ensemble_per_ic=1,
        replicate_repeats=repeats,
        replicate_select_single=False,
        ensemble_strategy="replicated_ic_sample_dim",
    )


def compute_coupled_steps_in_memory(steps: int, *, max_in_memory: int = 6) -> int:
    """Choose the largest divisor `<= max_in_memory` of *steps* (`fme/coupled/inference/evaluator.py:160-162`)."""
    validated_steps = validate_int_range(steps, "steps", min_value=1, max_value=10_000_000)
    upper = min(max(1, int(max_in_memory)), validated_steps)
    for cand in range(upper, 0, -1):
        if validated_steps % cand == 0:
            return cand
    return 1


def compute_forcing_n_repeats(
    end_iso: str,
    *,
    base_start_iso: str,
    repeat_period_hours: int,
    calendar: str,
    inner_step_hours: int = 6,
) -> int:
    """Compute `XarrayDataConfig.n_repeats` (`fme 2026.5.1`) needed to cover up to *end_iso* inclusive."""
    end_dt = parse_cftime(end_iso, calendar=calendar)
    base_dt = parse_cftime(base_start_iso, calendar=calendar)
    delta_hours = (end_dt - base_dt).total_seconds() / 3600.0
    if delta_hours < 0:
        return 1
    return max(
        1,
        int(
            math.ceil(
                (delta_hours + float(inner_step_hours)) / float(repeat_period_hours)
            )
        ),
    )


def build_inference_config(
    *,
    experiment_dir: str,
    steps: int,
    checkpoint_path: str,
    ic_path: str,
    start_indices: dict[str, Any] | None,
    forcing_dir: str,
    file_pattern: str | None = None,
    n_ensemble_per_ic: int = 1,
    save_monthly: bool = True,
    save_prediction: bool = False,
    num_data_workers: int = 4,
    seed: int | None = None,
    fme_version: str = "2026.5.1",
) -> dict[str, Any]:
    """Construct a strict `fme.ace.InferenceConfig`-compatible dictionary (`dacite(strict=True)`)."""
    validated_exp = validate_run_path(experiment_dir, label="experiment_dir")
    validated_steps = validate_int_range(steps, "steps", min_value=1, max_value=10_000_000)
    validated_ckpt = validate_safe_path(checkpoint_path, label="checkpoint_path")
    validated_ic = validate_safe_path(ic_path, label="initial_condition.path")
    validated_forcing = validate_safe_path(
        forcing_dir, label="forcing_loader.dataset.data_path"
    )
    validated_ens = validate_int_range(
        n_ensemble_per_ic, "n_ensemble_per_ic", min_value=1, max_value=10_000
    )
    validated_workers = validate_int_range(
        num_data_workers, "num_data_workers", min_value=0, max_value=64
    )

    ic_section: dict[str, Any] = {"path": validated_ic}
    if start_indices is not None:
        ic_section["start_indices"] = start_indices

    dataset_cfg: dict[str, Any] = {"data_path": validated_forcing}
    if file_pattern is not None:
        dataset_cfg["file_pattern"] = file_pattern

    cfg: dict[str, Any] = {
        "experiment_dir": validated_exp,
        "n_forward_steps": validated_steps,
        "forward_steps_in_memory": min(40, validated_steps),
        "checkpoint_path": validated_ckpt,
        "n_ensemble_per_ic": validated_ens,
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
            "project": "ace",
        },
        "initial_condition": ic_section,
        "forcing_loader": {
            "dataset": dataset_cfg,
            "num_data_workers": validated_workers,
        },
        "data_writer": {
            "save_prediction_files": bool(save_prediction),
            "save_monthly_files": bool(save_monthly),
        },
    }
    # `fme 2026.10.0` adds explicit top-level `seed: int | None` on `InferenceConfig`,
    # whereas `fme 2026.5.1` rejects unknown keys under `dacite(strict=True)`.
    if fme_version == "2026.10.0" and seed is not None:
        cfg["seed"] = validate_seed(seed)
    return cfg


def build_coupled_inference_config(
    *,
    experiment_dir: str,
    steps: int,
    checkpoint_path: str,
    atmos_ic_path: str,
    ocean_ic_path: str,
    start_indices: dict[str, Any] | None,
    atmos_forcing_dir: str,
    atmos_file_pattern: str | None = None,
    atmos_n_repeats: int | None = None,
    ocean_forcing_dir: str | None = None,
    ocean_file_pattern: str | None = None,
    n_ensemble_per_ic: int = 1,
    save_monthly: bool = True,
    save_prediction: bool = False,
    num_data_workers: int = 1,
    seed: int | None = None,
    fme_version: str = "2026.5.1",
) -> dict[str, Any]:
    """Construct a strict `fme.coupled.inference.InferenceConfig` dictionary."""
    validated_exp = validate_run_path(experiment_dir, label="experiment_dir")
    validated_steps = validate_int_range(
        steps, "n_coupled_steps", min_value=1, max_value=10_000_000
    )
    validated_ckpt = validate_safe_path(checkpoint_path, label="checkpoint_path")
    validated_atmos_ic = validate_safe_path(
        atmos_ic_path, label="initial_condition.atmosphere.path"
    )
    validated_ocean_ic = validate_safe_path(
        ocean_ic_path, label="initial_condition.ocean.path"
    )
    validated_atmos_forcing = validate_safe_path(
        atmos_forcing_dir, label="forcing_loader.atmosphere.dataset.data_path"
    )
    validated_ens = validate_int_range(
        n_ensemble_per_ic, "n_ensemble_per_ic", min_value=1, max_value=10_000
    )
    validated_workers = validate_int_range(
        num_data_workers, "num_data_workers", min_value=0, max_value=64
    )

    in_memory = compute_coupled_steps_in_memory(validated_steps, max_in_memory=6)

    atmos_ds: dict[str, Any] = {"data_path": validated_atmos_forcing}
    if atmos_file_pattern is not None:
        atmos_ds["file_pattern"] = atmos_file_pattern
    if (
        fme_version == "2026.5.1"
        and atmos_n_repeats is not None
        and int(atmos_n_repeats) > 1
    ):
        atmos_ds["n_repeats"] = int(atmos_n_repeats)

    forcing_section: dict[str, Any] = {
        "atmosphere": {"dataset": atmos_ds},
    }
    if ocean_forcing_dir is not None:
        validated_ocean_forcing = validate_safe_path(
            ocean_forcing_dir, label="forcing_loader.ocean.dataset.data_path"
        )
        ocean_ds: dict[str, Any] = {"data_path": validated_ocean_forcing}
        if ocean_file_pattern is not None:
            ocean_ds["file_pattern"] = ocean_file_pattern
        forcing_section["ocean"] = {"dataset": ocean_ds}
    forcing_section["num_data_workers"] = validated_workers

    ic_section: dict[str, Any] = {
        "ocean": {"path": validated_ocean_ic},
        "atmosphere": {"path": validated_atmos_ic},
    }
    if start_indices is not None:
        ic_section["start_indices"] = start_indices

    cfg: dict[str, Any] = {
        "experiment_dir": validated_exp,
        "n_coupled_steps": validated_steps,
        "coupled_steps_in_memory": in_memory,
        "checkpoint_path": validated_ckpt,
        "n_ensemble_per_ic": validated_ens,
        "data_writer": {
            "ocean": {
                "save_prediction_files": bool(save_prediction),
                "save_monthly_files": bool(save_monthly),
            },
            "atmosphere": {
                "save_prediction_files": bool(save_prediction),
                "save_monthly_files": bool(save_monthly),
            },
        },
        "forcing_loader": forcing_section,
        "initial_condition": ic_section,
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
        },
    }
    if fme_version == "2026.10.0" and seed is not None:
        cfg["seed"] = validate_seed(seed)
    return cfg


def _scenario_for_ic(model: ModelSpec, ic_path: str, norm_start: str, steps: int) -> str | None:
    """Return the forcing set an --ic state belongs to when several scenario sets cover the run; None if unambiguous."""
    scenario_sets = [fs for fs in model.forcing_sets if fs.ic_files or fs.ocean_ic_files]
    covering = []
    for fs in scenario_sets:
        try:
            select_forcing_set(norm_start, steps, model.timestep_hours, [fs], calendar=model.calendar,
                               model_name=model.name)
            covering.append(fs.name)
        except ConfigError:
            pass
    if len(covering) < 2:
        return None
    real_ic = os.path.realpath(ic_path)
    base = real_ic if os.path.isdir(real_ic) else os.path.dirname(real_ic)
    for d in (base, os.path.dirname(base)):  # <run>/restart.nc or <run>/atmosphere/restart.nc
        man = os.path.join(d, "aimip-manifest.json")
        if os.path.islink(man):
            continue
        try:
            with open(man, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        manifest_model = (data.get("registry_entry") or {}).get("name") or data.get("model")
        if manifest_model is not None and manifest_model != model.name:
            continue
        name = (data.get("forcing_set") or {}).get("name")
        if name in covering:
            print(f"aimip: --ic comes from a {name} run ({man}); using forcing set {name}", file=sys.stderr)
            return name
    raise ConfigError(
        f"{model.name}: forcing sets {', '.join(covering)} all cover this window and --ic does not say which "
        f"scenario its state belongs to. Pass --forcing-set <name> (the scenario of the run the state came from)."
    )


def prepare_run(
    *,
    model: ModelSpec,
    start_iso: str,
    steps: int,
    out_dir: str | None = None,
    ic_path: str | None = None,
    ocean_ic_path: str | None = None,
    members: int | None = None,
    seed: int | None = None,
    lora_path: str | None = None,
    forcing_set_name: str | None = None,
    save_monthly: bool = True,
    save_prediction: bool = False,
    mode: str = "dry-run",
    slurm_opts: SlurmOptions | None = None,
    site_config: SiteConfig | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    auto_fetch: bool = True,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate inputs, auto-fetch missing model files on first use, and write `config.yaml`, `job.sbatch`, and `aimip-manifest.json`."""
    if mode not in ("dry-run", "submit", "local"):
        raise ValidationError(
            f"Invalid run mode {mode!r}: expected 'dry-run', 'submit', or 'local'."
        )
    if model.family != "fme":
        # DLESyM and ArchesClimate-SSP run in the emulator environment (aimip/emu.py).
        from .emu import prepare_emu_run

        return prepare_emu_run(
            model=model, start_iso=start_iso, steps=steps, out_dir=out_dir, ic_path=ic_path,
            ocean_ic_path=ocean_ic_path, members=members, seed=seed, lora_path=lora_path,
            forcing_set_name=forcing_set_name, mode=mode, slurm_opts=slurm_opts, site_config=site_config,
            playground_dir=playground_dir, data_dir=data_dir, auto_fetch=auto_fetch, argv=argv,
        )

    norm_start = validate_iso_timestamp(start_iso, label="--start")
    validated_steps = validate_int_range(steps, "--steps", min_value=1, max_value=10_000_000)
    resolved_seed = validate_seed(0 if seed is None else seed)
    resolved_playground = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_playground)
    active_site = site_config if site_config is not None else load_site_config()

    if ocean_ic_path is not None and not model.is_coupled:
        raise ConfigError(
            f"--ocean-ic is only valid for coupled models (fme.coupled.inference), not {model.name!r}."
        )

    if out_dir is not None:
        safe_out_pre = validate_safe_path(out_dir, label="--out")
        if resolved_playground:
            pg = resolved_playground.rstrip("/")
            if safe_out_pre == pg or safe_out_pre.startswith(pg + "/"):
                raise ConfigError(
                    f"Refusing to write run outputs ({safe_out_pre}) inside "
                    f"$AIMIP_PLAYGROUND ({pg}); choose an --out directory outside the playground."
                )
        validated_out = validate_run_path(out_dir, label="--out")
    else:
        validated_out = default_out_dir(
            model.name,
            norm_start,
            extra_entropy=(
                f"{validated_steps}|{ic_path}|{ocean_ic_path}|{members}|"
                f"{resolved_seed}|{lora_path}|{forcing_set_name}"
            ),
        )
        if resolved_playground:
            pg = resolved_playground.rstrip("/")
            if validated_out == pg or validated_out.startswith(pg + "/"):
                raise ConfigError(
                    f"Refusing to write run outputs ({validated_out}) inside "
                    f"$AIMIP_PLAYGROUND ({pg}); choose an --out directory outside the playground."
                )

    manifest_path = validate_run_path(
        f"{validated_out}/aimip-manifest.json", label="aimip-manifest.json"
    )
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as prev_fh:
                prev_manifest = json.load(prev_fh)
            if isinstance(prev_manifest, dict) and "finished" in prev_manifest:
                raise ConfigError(
                    f"Refusing to overwrite completed run directory {validated_out}; "
                    "choose a new --out path."
                )
        except (OSError, json.JSONDecodeError):
            pass

    # Auto-fetch on first use when canonical model files are missing on disk.
    # Pre-validate the forcing window BEFORE fetching so invalid `--start`/`--steps`/`--forcing-set`
    # fail immediately without downloading multi-GB model files.
    if model_has_missing_files(model, playground_dir=playground_dir, data_dir=data_dir):
        select_forcing_set(
            norm_start,
            validated_steps,
            model.timestep_hours,
            model.forcing_sets,
            calendar=model.calendar,
            model_name=model.name,
            preferred_set_name=forcing_set_name,
        )
        model = ensure_model_files_ready(
            model,
            playground_dir=playground_dir,
            data_dir=data_dir,
            auto_fetch=auto_fetch,
        )

    # Scenario/era forcing sets (e.g. `ace2-som` `1xCO2` vs `3xCO2`, `samudrace-cm4`
    # `picontrol-0311` vs `picontrol-0151`) carry paired `ic_files` / `ocean_ic_files`.
    cand_ic_files: Sequence[str] | None = None
    cand_ocean_ic_files: Sequence[str] | None = None
    if ic_path is None and any(fs.ic_files for fs in model.forcing_sets):
        pre_set, _ = select_forcing_set(
            norm_start,
            validated_steps,
            model.timestep_hours,
            model.forcing_sets,
            calendar=model.calendar,
            model_name=model.name,
            preferred_set_name=forcing_set_name,
        )
        cand_ic_files = pre_set.ic_files or None
        cand_ocean_ic_files = pre_set.ocean_ic_files or None

    # Resolve IC file(s) (shipped IC, spun IC, or `<prev>/restart.nc` / `<prev>/{atmosphere,ocean}/restart.nc`)
    # and ensemble counts.
    ic_res = resolve_ic_and_members(
        model=model,
        norm_start=norm_start,
        ic_path=ic_path,
        ocean_ic_path=ocean_ic_path,
        members=members,
        candidate_ic_files=cand_ic_files,
        candidate_ocean_ic_files=cand_ocean_ic_files,
    )
    source_ic_info = ic_res.source_ic

    # Scenario sets (ace2-som 1xCO2/3xCO2) share one window: a state passed with --ic must keep its scenario.
    # Inherit it from the source run's manifest, or refuse.
    if ic_path is not None and forcing_set_name is None:
        forcing_set_name = _scenario_for_ic(model, ic_path, norm_start, validated_steps)

    # Pick the single forcing set whose window covers [start, start + steps * dt].
    selected_set, end_iso = select_forcing_set(
        norm_start,
        validated_steps,
        model.timestep_hours,
        model.forcing_sets,
        calendar=source_ic_info.calendar,
        model_name=model.name,
        preferred_set_name=forcing_set_name,
    )

    # Resolve checkpoint (and validate LoRA checkpoint via `inspect_lora` when `--lora` is set).
    validated_lora: str | None = None
    lora_summary: dict[str, Any] | None = None
    if lora_path is not None:
        if model.is_coupled:
            raise ConfigError(
                f"--lora single-module checkpoint override is not supported for coupled model {model.name!r}."
            )
        validated_lora = validate_visible_input_path(
            lora_path, label="--lora checkpoint"
        )
        info = inspect_lora(validated_lora)
        # Lightweight model-compatibility check: never loads the 2-4 GB base
        # checkpoint; instead checks stepper `in_names ∩ out_names` (prognostic variables)
        # against the resolved IC dataset variables when `in_names`/`out_names` are present.
        out_name_set = set(info.out_names)
        ckpt_prognostic = [v for v in info.in_names if v in out_name_set]
        compat_checked = bool(ckpt_prognostic and source_ic_info.variable_names)
        if compat_checked:
            ic_var_set = set(source_ic_info.variable_names)
            missing_prog = sorted(v for v in ckpt_prognostic if v not in ic_var_set)
            if missing_prog:
                shown_missing = ", ".join(missing_prog[:8])
                if len(missing_prog) > 8:
                    shown_missing += f", ... ({len(missing_prog)} total)"
                raise ConfigError(
                    f"--lora checkpoint {validated_lora} appears incompatible with model "
                    f"{model.name!r}: checkpoint prognostic variable(s) [{shown_missing}] "
                    f"are missing from initial condition {source_ic_info.path}."
                )
        lora_summary = {
            "lora_rank": info.lora_rank,
            "lora_alpha": info.lora_alpha,
            "effective_lora_alpha": info.effective_lora_alpha,
            "spectral_lora_rank": info.spectral_lora_rank,
            "spectral_lora_alpha": info.spectral_lora_alpha,
            "effective_spectral_lora_alpha": info.effective_spectral_lora_alpha,
            "tensor_count": len(info.tensors),
            "spectral_tensor_count": len(info.spectral_tensors),
            "compatibility_checked": compat_checked,
        }
        active_checkpoint = validated_lora
        ckpt_sha = sha256_file(active_checkpoint)
    else:
        active_checkpoint = model.checkpoint
        if not os.path.isfile(active_checkpoint):
            raise ValidationError(
                f"Model checkpoint file not found: {active_checkpoint}"
            )
        ckpt_sha = sha256_file(active_checkpoint)
        if (
            model.checkpoint_sha256
            and ckpt_sha.lower() != model.checkpoint_sha256.lower()
        ):
            raise ValidationError(
                f"Checkpoint SHA-256 mismatch for model {model.name!r}: "
                f"registry expects {model.checkpoint_sha256}, got {ckpt_sha}."
            )

    forcing_files = _list_forcing_files(selected_set.dir)
    forcing_files_sha256 = _hash_and_verify_forcing_files(
        selected_set.dir,
        selected_set.file_pattern,
        model=model,
        data_dir=resolved_data,
    )
    ocean_forcing_files: list[dict[str, Any]] | None = None
    ocean_forcing_files_sha256: dict[str, str] | None = None
    if selected_set.ocean_dir is not None:
        ocean_forcing_files = _list_forcing_files(selected_set.ocean_dir)
        ocean_forcing_files_sha256 = _hash_and_verify_forcing_files(
            selected_set.ocean_dir,
            selected_set.ocean_file_pattern,
            model=model,
            data_dir=resolved_data,
        )

    effective_slurm = (
        slurm_opts.validate()
        if slurm_opts is not None
        else SlurmOptions.from_site(active_site)
    )

    makedirs_no_group_other_write(validated_out)

    seed_warning: str | None = None
    if model.stochastic and ic_path is not None and seed is None:
        seed_warning = (
            "Omitting --seed on a stochastic --ic continuation defaults to seed=0, "
            "which replays the same random noise sequence across segments; pass an "
            "explicit per-segment --seed <N> (e.g. --seed 1, --seed 2) for independent noise."
        )
        print(f"aimip warning: {seed_warning}", file=sys.stderr)

    source_ic_sha = sha256_file(source_ic_info.path)
    need_rep = ic_res.replicate_repeats > 1 or ic_res.replicate_sample_slice is not None
    ensemble_note: str | None = None
    effective_ocean_ic_info: ICFileInfo | None = None
    effective_ocean_ic_sha: str | None = None
    source_ocean_ic_info: ICFileInfo | None = ic_res.source_ocean_ic
    source_ocean_ic_sha: str | None = None

    if model.is_coupled:
        if source_ocean_ic_info is None:
            raise ConfigError(
                f"Internal error: coupled model {model.name!r} resolved without ocean initial condition."
            )
        source_ocean_ic_sha = sha256_file(source_ocean_ic_info.path)
        if need_rep:
            rep_atmos_target = validate_run_path(
                f"{validated_out}/ic_members_atmosphere.nc",
                label="ic_members_atmosphere.nc",
            )
            rep_ocean_target = validate_run_path(
                f"{validated_out}/ic_members_ocean.nc",
                label="ic_members_ocean.nc",
            )
            effective_ic_info = replicate_ic_for_ensemble(
                source_ic_info.path,
                rep_atmos_target,
                target_time=norm_start,
                repeats_per_sample=ic_res.replicate_repeats,
                select_single_timestamp=ic_res.replicate_select_single,
                sample_slice=ic_res.replicate_sample_slice,
                playground_dir=resolved_playground,
            )
            effective_ocean_ic_info = replicate_ic_for_ensemble(
                source_ocean_ic_info.path,
                rep_ocean_target,
                target_time=norm_start,
                repeats_per_sample=ic_res.replicate_repeats,
                select_single_timestamp=ic_res.replicate_select_single,
                sample_slice=ic_res.replicate_sample_slice,
                playground_dir=resolved_playground,
            )
            effective_ic_sha = sha256_file(effective_ic_info.path)
            effective_ocean_ic_sha = sha256_file(effective_ocean_ic_info.path)
            effective_start_indices = None
            ensemble_note = (
                f"Prepared paired coupled ICs (repeats={ic_res.replicate_repeats}, "
                f"slice={ic_res.replicate_sample_slice}) in ic_members_atmosphere.nc and "
                "ic_members_ocean.nc with n_ensemble_per_ic=1."
            )
        else:
            effective_ic_info = source_ic_info
            effective_ic_sha = source_ic_sha
            effective_ocean_ic_info = source_ocean_ic_info
            effective_ocean_ic_sha = source_ocean_ic_sha
            effective_start_indices = ic_res.start_indices

        atmos_n_repeats: int | None = None
        if (
            selected_set.repeat_period_hours is not None
            and selected_set.repeat_base_start is not None
        ):
            atmos_n_repeats = compute_forcing_n_repeats(
                end_iso=end_iso,
                base_start_iso=selected_set.repeat_base_start,
                repeat_period_hours=selected_set.repeat_period_hours,
                calendar=source_ic_info.calendar,
            )

        config_dict = build_coupled_inference_config(
            experiment_dir=validated_out,
            steps=validated_steps,
            checkpoint_path=active_checkpoint,
            atmos_ic_path=effective_ic_info.path,
            ocean_ic_path=effective_ocean_ic_info.path,
            start_indices=effective_start_indices,
            atmos_forcing_dir=selected_set.dir,
            atmos_file_pattern=selected_set.file_pattern,
            atmos_n_repeats=atmos_n_repeats,
            ocean_forcing_dir=selected_set.ocean_dir,
            ocean_file_pattern=selected_set.ocean_file_pattern,
            n_ensemble_per_ic=ic_res.n_ensemble_per_ic,
            save_monthly=save_monthly,
            save_prediction=save_prediction,
            seed=resolved_seed,
            fme_version=model.fme_version,
        )
    else:
        if need_rep:
            replicated_ic_target = validate_run_path(
                f"{validated_out}/ic_members.nc", label="ic_members.nc"
            )
            effective_ic_info = replicate_ic_for_ensemble(
                source_ic_info.path,
                replicated_ic_target,
                target_time=norm_start,
                repeats_per_sample=ic_res.replicate_repeats,
                select_single_timestamp=ic_res.replicate_select_single,
                sample_slice=ic_res.replicate_sample_slice,
                playground_dir=resolved_playground,
            )
            effective_ic_sha = sha256_file(effective_ic_info.path)
            effective_start_indices = None
            ensemble_note = (
                f"Replicated IC {ic_res.replicate_repeats}x along 'sample' into ic_members.nc "
                "with n_ensemble_per_ic=1; stochastic members diverge through per-sample model noise "
                "(avoids fme 2026.5.1 Stepper.predict_paired double forcing broadcast)."
            )
        else:
            effective_ic_info = source_ic_info
            effective_ic_sha = source_ic_sha
            effective_start_indices = ic_res.start_indices

        config_dict = build_inference_config(
            experiment_dir=validated_out,
            steps=validated_steps,
            checkpoint_path=active_checkpoint,
            ic_path=effective_ic_info.path,
            start_indices=effective_start_indices,
            forcing_dir=selected_set.dir,
            file_pattern=selected_set.file_pattern,
            n_ensemble_per_ic=ic_res.n_ensemble_per_ic,
            save_monthly=save_monthly,
            save_prediction=save_prediction,
            seed=resolved_seed,
            fme_version=model.fme_version,
        )

    n_trajectories = ic_res.n_ic_samples * ic_res.n_ensemble_per_ic
    default_walltime, unclamped_walltime, auto_walltime_capped = (
        compute_slurm_walltime_details(
            steps=validated_steps,
            timestep_hours=model.timestep_hours,
            n_trajectories=n_trajectories,
            max_walltime=active_site.max_time,
        )
    )
    walltime_capped = bool(effective_slurm.time is None and auto_walltime_capped)
    if effective_slurm.scheduler == "slurm" and walltime_capped:
        print(
            f"aimip warning: estimated SLURM walltime ({unclamped_walltime}) exceeds "
            f"site max_time ({active_site.max_time}) and was clamped to {default_walltime}; "
            "pass --time explicitly or split long runs into segments with --ic <prev>/restart.nc.",
            file=sys.stderr,
        )

    sbatch_script = render_sbatch_script(
        model_name=model.name,
        out_dir=validated_out,
        playground_dir=resolved_playground,
        slurm_opts=effective_slurm,
        default_walltime=default_walltime,
    )

    config_path = validate_run_path(f"{validated_out}/config.yaml", label="config.yaml")
    sbatch_path = validate_run_path(f"{validated_out}/job.sbatch", label="job.sbatch")

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_dict, fh, sort_keys=False, default_flow_style=False)
    strip_group_other_write(config_path)

    with open(sbatch_path, "w", encoding="utf-8") as fh:
        fh.write(sbatch_script)
    strip_group_other_write(sbatch_path)

    raw_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_job_id: str | None = None
    if raw_job_id:
        try:
            slurm_job_id = validate_slurm_job_id(raw_job_id.strip())
        except ValidationError:
            slurm_job_id = None

    ic_provenance = (
        get_model_ic_provenance(model.name, model.notes)
        if ic_path is None
        else (
            f"Custom paired coupled ICs: atmosphere={source_ic_info.path}, ocean={source_ocean_ic_info.path}"
            if source_ocean_ic_info is not None
            else f"Custom IC provided via --ic: {source_ic_info.path}"
        )
    )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "aimip_version": __version__,
        "aimip_lock_sha256": resolve_active_lock_sha256(),
        "argv": list(argv) if argv is not None else [],
        "mode": mode,
        "out_dir": validated_out,
        "config_path": config_path,
        "sbatch_path": sbatch_path,
        "registry_entry": model.to_dict(),
        "checkpoint_path": active_checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "lora_checkpoint": validated_lora,
        "lora_info": lora_summary,
        "ic_path": effective_ic_info.path,
        "ic_sha256": effective_ic_sha,
        "source_ic_path": source_ic_info.path,
        "source_ic_sha256": source_ic_sha,
        "ocean_ic_path": (
            effective_ocean_ic_info.path if effective_ocean_ic_info is not None else None
        ),
        "ocean_ic_sha256": effective_ocean_ic_sha,
        "source_ocean_ic_path": (
            source_ocean_ic_info.path if source_ocean_ic_info is not None else None
        ),
        "source_ocean_ic_sha256": source_ocean_ic_sha,
        "ic_is_custom": ic_path is not None,
        "ic_provenance": ic_provenance,
        "ic_calendar": effective_ic_info.calendar,
        "ic_spatial_dims": (
            list(effective_ic_info.spatial_dims)
            if effective_ic_info.spatial_dims is not None
            else None
        ),
        "start_time": norm_start,
        "end_time": end_iso,
        "steps": validated_steps,
        "members": ic_res.total_members,
        "n_ic_samples": ic_res.n_ic_samples,
        "n_ensemble_per_ic": ic_res.n_ensemble_per_ic,
        "ensemble_strategy": ic_res.ensemble_strategy,
        "ensemble_note": ensemble_note,
        "seed": resolved_seed,
        "seed_explicit": seed is not None,
        "seed_warning": seed_warning,
        "forcing_set": selected_set.to_dict(),
        "forcing_dir": selected_set.dir,
        "forcing_files": forcing_files,
        "forcing_files_sha256": forcing_files_sha256,
        "ocean_forcing_dir": selected_set.ocean_dir,
        "ocean_forcing_files": ocean_forcing_files,
        "ocean_forcing_files_sha256": ocean_forcing_files_sha256,
        "site": {
            "name": active_site.name,
            "scheduler": effective_slurm.scheduler,
            "runtime": active_site.runtime,
        },
        "slurm": {
            "scheduler": effective_slurm.scheduler,
            "partition": effective_slurm.partition,
            "account": effective_slurm.account,
            "gres": effective_slurm.gres,
            "cpus_per_task": effective_slurm.cpus_per_task,
            "mem": effective_slurm.mem,
            "time": effective_slurm.time or default_walltime,
            "unclamped_estimated_time": unclamped_walltime,
            "walltime_capped": walltime_capped,
            **({"qos": effective_slurm.qos} if effective_slurm.qos else {}),  # MN5; only when set
        },
        "slurm_job_id": slurm_job_id,
        "image_sha": _sanitize_env_string(os.environ.get("AIMIP_IMAGE_SHA")),
        "fme_version": get_fme_version(),
        "fme_version_expected": model.fme_version,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }

    _write_json_atomic(manifest_path, manifest)

    if mode == "local":
        return exec_fme(validated_out)

    return manifest


def pin_cudnn_benchmark_off() -> Any:
    """Keep `torch.backends.cudnn.benchmark` False while fme runs; return a restore callable.

    fme's inference sets `torch.backends.cudnn.benchmark = True` on GPUs. cuDNN then picks convolution algorithms
    by timing them; for the coupled models (`fme/coupled/inference/inference.py`) the same run then gave one of
    two bit-different results. `exec_fme` applies this pin to coupled models only. With benchmark off, cuDNN uses
    its fixed heuristics and repeated runs are bit-identical. Set `AIMIP_CUDNN_BENCHMARK=1` to keep fme's
    upstream behaviour.
    """
    if os.environ.get("AIMIP_CUDNN_BENCHMARK") == "1":
        return lambda: None
    try:
        import torch.backends.cudnn as cudnn_mod
    except ImportError:
        return lambda: None
    cls = type(cudnn_mod)
    had_own = "benchmark" in cls.__dict__
    original = cls.__dict__.get("benchmark")
    prior_value = bool(cudnn_mod.benchmark)
    cudnn_mod.benchmark = False
    cls.benchmark = property(lambda self: False, lambda self, value: None)

    def restore() -> None:
        if had_own:
            cls.benchmark = original
        else:
            del cls.benchmark
        cudnn_mod.benchmark = prior_value

    return restore


def _quiet_git_revision_short_hash() -> str | None:
    """Replacement for `fme.core.training_history.git_revision_short_hash` (`training_history.py:97-105`).

    Silences `fatal: not a git repository` on stderr inside read-only containers while
    still returning `None` outside git repositories (preserving `git_sha: "None"`
    metadata parity).
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return None


def _evict_overlay_modules(*, only_under: str | None = None, except_under: str | None = None) -> None:
    """Evict `fme` / `torch_harmonics` / `zarrs` from `sys.modules` when switching `fme` runtimes."""
    norm_only = os.path.realpath(only_under) + os.sep if only_under else None
    norm_except = os.path.realpath(except_under) + os.sep if except_under else None
    for mod_key in list(sys.modules.keys()):
        if not any(
            mod_key == pfx or mod_key.startswith(pfx + ".")
            for pfx in _OVERLAY_EVICT_PREFIXES
        ):
            continue
        mod_obj = sys.modules.get(mod_key)
        mod_file = getattr(mod_obj, "__file__", None) if mod_obj is not None else None
        real_file = os.path.realpath(mod_file) if isinstance(mod_file, str) and mod_file else ""
        if norm_only is not None and real_file and not real_file.startswith(norm_only):
            continue
        if norm_except is not None and real_file and real_file.startswith(norm_except):
            continue
        sys.modules.pop(mod_key, None)
    importlib.invalidate_caches()


def _detect_imported_fme_version(fme_pkg: Any) -> str:
    """Read `fme.__version__` from the imported `fme` package (or ordered `sys.path` distributions)."""
    dunder_ver = getattr(fme_pkg, "__version__", None)
    if isinstance(dunder_ver, str) and dunder_ver.strip():
        return dunder_ver.strip()
    for dist in importlib.metadata.distributions(path=list(sys.path)):
        try:
            if (dist.metadata.get("Name") or "").lower() == "fme" and dist.version:
                return str(dist.version).strip()
        except Exception:
            continue
    return "unknown"


def _activate_and_verify_fme_runtime(
    expected_version: str,
    *,
    model_name: str,
    fme_module: str,
) -> tuple[str, str | None, bool]:
    """Activate `/opt/aimip/overlay-fme-2026.10.0` when `expected_version == '2026.10.0'` and verify imported `fme`.

    Returns `(imported_version, active_overlay_dir, inserted_sys_path)`.
    """
    overlay_dir: str | None = None
    active_overlay_dir: str | None = None
    inserted_sys_path = False

    if expected_version == "2026.10.0":
        env_overlay = os.environ.get("AIMIP_FME_2026_10_OVERLAY", "").strip()
        default_overlay = DEFAULT_FME_2026_10_OVERLAY
        if not env_overlay:
            home = os.environ.get("AIMIP_HOME") or os.environ.get("AIMIP_PLAYGROUND")
            if home:
                cand_overlay = os.path.join(home.rstrip("/"), "overlay-fme-2026.10.0")
                if os.path.isdir(cand_overlay):
                    default_overlay = cand_overlay
        overlay_dir = validate_safe_path(
            env_overlay or default_overlay,
            label="fme 2026.10.0 overlay directory",
        )
        if env_overlay and not os.path.isdir(overlay_dir):
            raise ConfigError(
                f"Configured AIMIP_FME_2026_10_OVERLAY directory not found: {overlay_dir}"
            )
        if os.path.isdir(overlay_dir):
            active_overlay_dir = overlay_dir
            if not sys.path or sys.path[0] != overlay_dir:
                sys.path.insert(0, overlay_dir)
                inserted_sys_path = True
            _evict_overlay_modules(except_under=overlay_dir)

    try:
        fme_pkg = importlib.import_module("fme")
    except ImportError as exc:
        if inserted_sys_path and active_overlay_dir and active_overlay_dir in sys.path:
            sys.path.remove(active_overlay_dir)
        raise ConfigError(
            f"Cannot execute {fme_module} for model {model_name!r}: `fme` package is not installed ({exc})."
        ) from exc

    imported_version = _detect_imported_fme_version(fme_pkg)
    if imported_version != expected_version:
        overlay_hint = ""
        if expected_version == "2026.10.0" and active_overlay_dir is None:
            overlay_hint = f" (overlay directory {overlay_dir} not found)"
        if inserted_sys_path and active_overlay_dir:
            if active_overlay_dir in sys.path:
                sys.path.remove(active_overlay_dir)
            _evict_overlay_modules(only_under=active_overlay_dir)
        raise ConfigError(
            f"fme version mismatch for model {model_name!r}: registry requires "
            f"fme=={expected_version}, but imported fme=={imported_version!r}{overlay_hint}."
        )

    return imported_version, active_overlay_dir, inserted_sys_path


def exec_fme(out_dir: str) -> dict[str, Any]:
    """Execute `<out>/config.yaml` inside the container with `fme.core.set_seed` and `runpy`.

    Activates `/opt/aimip/overlay-fme-2026.10.0` (or `$AIMIP_FME_2026_10_OVERLAY`) when
    `registry_entry.fme_version == "2026.10.0"`, strictly verifies the imported `fme`
    version against `registry_entry.fme_version`, provides a harmless git environment
    (`cwd = validated_out`, `GIT_CEILING_DIRECTORIES`, and quiet `git_revision_short_hash`
    wrapper), and records `fme_version_imported`, `started_utc`, `finished`, `rc`, and
    `wall_seconds` in `<out>/aimip-manifest.json`.
    """
    validated_out = validate_run_path(out_dir, label="run output directory")
    manifest_path = validate_run_path(
        f"{validated_out}/aimip-manifest.json", label="aimip-manifest.json"
    )
    config_path = validate_run_path(f"{validated_out}/config.yaml", label="config.yaml")

    if not os.path.isfile(manifest_path):
        raise ConfigError(
            f"Run manifest not found: {manifest_path}. Prepare the run with `aimip run` first."
        )
    if not os.path.isfile(config_path):
        raise ConfigError(
            f"fme config not found: {config_path}. Prepare the run with `aimip run` first."
        )

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to read manifest {manifest_path}: {exc}") from exc

    reg_entry = manifest.get("registry_entry")
    if not isinstance(reg_entry, dict) or "fme_module" not in reg_entry:
        raise ConfigError(
            f"Manifest {manifest_path} is missing 'registry_entry.fme_module'."
        )
    if str(reg_entry.get("family", "fme")) != "fme":
        from .emu import EMU_FAMILIES, exec_emu

        if reg_entry.get("family") not in EMU_FAMILIES or manifest.get("family") != reg_entry.get("family"):
            raise ConfigError(f"Manifest {manifest_path}: unknown or inconsistent model family {reg_entry.get('family')!r}.")
        return exec_emu(validated_out, manifest, manifest_path)

    fme_module = validate_python_module(
        str(reg_entry["fme_module"]), label="registry_entry.fme_module"
    )
    model_name = str(reg_entry.get("name", "<unknown>"))
    expected_fme_version = str(reg_entry.get("fme_version") or "2026.5.1").strip()
    is_coupled = fme_module == "fme.coupled.inference"
    seed = validate_seed(int(manifest.get("seed", 0)))

    # Re-verify `config.yaml` paths and re-hash `checkpoint_path`, `ic_path`, and
    # (for coupled models) `ocean_ic_path` against `aimip-manifest.json` BEFORE loading `fme`.
    try:
        with open(config_path, encoding="utf-8") as cfg_fh:
            loaded_cfg = yaml.safe_load(cfg_fh)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Failed to read {config_path}: {exc}") from exc
    if not isinstance(loaded_cfg, dict):
        raise ValidationError(f"Invalid config.yaml in {validated_out}: expected mapping.")

    expected_ckpt = validate_safe_path(
        str(manifest.get("checkpoint_path", "")), label="manifest checkpoint_path"
    )
    expected_ckpt_sha = str(manifest.get("checkpoint_sha256", "")).strip().lower()
    expected_ic = validate_safe_path(
        str(manifest.get("ic_path", "")), label="manifest ic_path"
    )
    expected_ic_sha = str(manifest.get("ic_sha256", "")).strip().lower()
    if len(expected_ckpt_sha) != 64 or len(expected_ic_sha) != 64:
        raise ValidationError(
            f"Manifest {manifest_path} is missing 64-hex checkpoint_sha256 or ic_sha256."
        )

    cfg_ckpt = str(loaded_cfg.get("checkpoint_path", ""))
    ic_block = loaded_cfg.get("initial_condition")
    cfg_exp = str(loaded_cfg.get("experiment_dir", ""))
    if cfg_ckpt != expected_ckpt:
        raise ValidationError(
            f"Integrity check failed in {config_path}: checkpoint_path ({cfg_ckpt!r}) "
            f"does not match aimip-manifest.json ({expected_ckpt!r})."
        )
    if cfg_exp != validated_out:
        raise ValidationError(
            f"Integrity check failed in {config_path}: experiment_dir ({cfg_exp!r}) "
            f"does not match run directory ({validated_out!r})."
        )

    expected_ocean_ic: str | None = None
    expected_ocean_ic_sha: str | None = None
    if is_coupled:
        expected_ocean_ic = validate_safe_path(
            str(manifest.get("ocean_ic_path", "")), label="manifest ocean_ic_path"
        )
        expected_ocean_ic_sha = str(manifest.get("ocean_ic_sha256", "")).strip().lower()
        if len(expected_ocean_ic_sha) != 64:
            raise ValidationError(
                f"Manifest {manifest_path} is missing 64-hex ocean_ic_sha256."
            )
        atmos_block = ic_block.get("atmosphere") if isinstance(ic_block, dict) else None
        ocean_block = ic_block.get("ocean") if isinstance(ic_block, dict) else None
        cfg_atmos_ic = (
            str(atmos_block.get("path", "")) if isinstance(atmos_block, dict) else ""
        )
        cfg_ocean_ic = (
            str(ocean_block.get("path", "")) if isinstance(ocean_block, dict) else ""
        )
        if cfg_atmos_ic != expected_ic:
            raise ValidationError(
                f"Integrity check failed in {config_path}: initial_condition.atmosphere.path "
                f"({cfg_atmos_ic!r}) does not match aimip-manifest.json ({expected_ic!r})."
            )
        if cfg_ocean_ic != expected_ocean_ic:
            raise ValidationError(
                f"Integrity check failed in {config_path}: initial_condition.ocean.path "
                f"({cfg_ocean_ic!r}) does not match aimip-manifest.json ({expected_ocean_ic!r})."
            )
    else:
        cfg_ic = str(ic_block.get("path", "")) if isinstance(ic_block, dict) else ""
        if cfg_ic != expected_ic:
            raise ValidationError(
                f"Integrity check failed in {config_path}: initial_condition.path ({cfg_ic!r}) "
                f"does not match aimip-manifest.json ({expected_ic!r})."
            )

    if not os.path.isfile(expected_ckpt):
        raise ValidationError(
            f"Checkpoint file recorded in manifest not found at job time: {expected_ckpt}"
        )
    actual_ckpt_sha = sha256_file(expected_ckpt).lower()
    if actual_ckpt_sha != expected_ckpt_sha:
        raise ValidationError(
            f"Checkpoint SHA-256 mismatch at exec-fme time for {expected_ckpt}: "
            f"manifest expects {expected_ckpt_sha}, got {actual_ckpt_sha}."
        )

    if not os.path.isfile(expected_ic):
        raise ValidationError(
            f"Initial-condition file recorded in manifest not found at job time: {expected_ic}"
        )
    actual_ic_sha = sha256_file(expected_ic).lower()
    if actual_ic_sha != expected_ic_sha:
        raise ValidationError(
            f"Initial-condition SHA-256 mismatch at exec-fme time for {expected_ic}: "
            f"manifest expects {expected_ic_sha}, got {actual_ic_sha}."
        )

    if is_coupled and expected_ocean_ic is not None and expected_ocean_ic_sha is not None:
        if not os.path.isfile(expected_ocean_ic):
            raise ValidationError(
                f"Ocean initial-condition file recorded in manifest not found at job time: {expected_ocean_ic}"
            )
        actual_ocean_sha = sha256_file(expected_ocean_ic).lower()
        if actual_ocean_sha != expected_ocean_ic_sha:
            raise ValidationError(
                f"Ocean initial-condition SHA-256 mismatch at exec-fme time for {expected_ocean_ic}: "
                f"manifest expects {expected_ocean_ic_sha}, got {actual_ocean_sha}."
            )

    fs_manifest = manifest.get("forcing_set") if isinstance(manifest.get("forcing_set"), dict) else {}
    for dir_key, sha_key, pat_key, label in (
        ("forcing_dir", "forcing_files_sha256", "file_pattern", "forcing"),
        ("ocean_forcing_dir", "ocean_forcing_files_sha256", "ocean_file_pattern", "ocean forcing"),
    ):
        raw_fdir = manifest.get(dir_key)
        pinned_fshas = manifest.get(sha_key)
        if isinstance(pinned_fshas, dict) and pinned_fshas:
            if isinstance(raw_fdir, str) and raw_fdir:
                checked_fdir = validate_safe_path(raw_fdir, label=dir_key)
                if not os.path.isdir(checked_fdir) or os.path.islink(checked_fdir):
                    raise ValidationError(f"Manifest {label} directory missing or symlinked at job time: {checked_fdir}")
                pat = str(fs_manifest.get(pat_key) or "*.nc")
                live_matched = {
                    os.path.join(checked_fdir, fn)
                    for fn in os.listdir(checked_fdir)
                    if fnmatch.fnmatch(fn, pat) and (os.path.isfile(os.path.join(checked_fdir, fn)) or os.path.islink(os.path.join(checked_fdir, fn)))
                }
                if live_matched != set(pinned_fshas.keys()):
                    raise ValidationError(
                        f"Manifest {label} file set changed after preparation in {checked_fdir}."
                    )
            for fpath, want_fsha in pinned_fshas.items():
                checked_fp = validate_safe_path(str(fpath), label=f"{label} file")
                if not os.path.isfile(checked_fp) or os.path.islink(checked_fp):
                    raise ValidationError(f"Pinned {label} file missing or symlinked at job time: {checked_fp}")
                got_fsha = sha256_file(checked_fp).lower()
                if got_fsha != str(want_fsha).strip().lower():
                    raise ValidationError(
                        f"SHA-256 mismatch at exec-fme time for {label} file {checked_fp}: "
                        f"manifest expects {want_fsha}, got {got_fsha}."
                    )

    raw_job_id = os.environ.get("SLURM_JOB_ID")
    if raw_job_id:
        try:
            manifest["slurm_job_id"] = validate_slurm_job_id(raw_job_id.strip())
        except ValidationError:
            pass

    manifest["fme_version"] = get_fme_version()
    manifest["fme_version_expected"] = expected_fme_version
    manifest["started_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_json_atomic(manifest_path, manifest)

    try:
        imported_fme_version, active_overlay_dir, inserted_overlay = (
            _activate_and_verify_fme_runtime(
                expected_fme_version,
                model_name=model_name,
                fme_module=fme_module,
            )
        )
    except ConfigError as exc:
        manifest["finished"] = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        manifest["rc"] = 1
        manifest["wall_seconds"] = 0.0
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest["fme_version_imported"] = imported_fme_version
    # cuDNN autotuning is turned off only for the coupled models (their runs were not reproducible with it on);
    # uncoupled ACE runs keep fme's default (on), which has been bit-reproducible in all our tests.
    manifest["cudnn_benchmark"] = (not is_coupled) or os.environ.get("AIMIP_CUDNN_BENCHMARK") == "1"
    manifest["fme_overlay_dir"] = active_overlay_dir
    _write_json_atomic(manifest_path, manifest)

    try:
        try:
            from fme.core import set_seed
        except ImportError as exc:
            raise ConfigError(
                f"Cannot execute {fme_module}: `fme.core.set_seed` could not be imported ({exc})."
            ) from exc

        training_history_mod: Any = None
        orig_git_fn: Any = None
        try:
            import fme.core.training_history as _th_mod

            training_history_mod = _th_mod
            orig_git_fn = getattr(_th_mod, "git_revision_short_hash", None)
        except ImportError:
            training_history_mod = None

        set_seed(seed)

        saved_argv = list(sys.argv)
        saved_cwd: str | None = None
        try:
            saved_cwd = os.getcwd()
        except OSError:
            saved_cwd = None
        saved_ceiling = os.environ.get("GIT_CEILING_DIRECTORIES")
        ceiling_dir = os.path.dirname(validated_out) or "/"

        t0 = time.monotonic()
        rc = 0
        caught_error: BaseException | None = None

        # fme writes every netCDF file to a temporary directory first (fme/core/cloud.py:34-39). Apptainer's
        # --containall /tmp defaults to a small 64 MB tmpfs on some HPC sites, which overflows on multi-member
        # coupled initial conditions ("NetCDF: HDF error"). Use a private directory inside the run directory instead.
        tmp_dir = os.path.join(validated_out, ".aimip-tmp")
        if os.path.islink(tmp_dir):
            raise ConfigError(f"Refusing a symlinked temporary directory: {tmp_dir}")
        os.makedirs(tmp_dir, mode=0o700, exist_ok=True)
        # Initialize multiprocessing's temp dir (used by torch DataLoader workers for AF_UNIX sockets) in the short
        # system /tmp BEFORE redirecting TMPDIR, because socket paths under a long run directory can exceed the
        # 107-character AF_UNIX path limit.
        import multiprocessing.util

        multiprocessing.util.get_temp_dir()
        saved_tmpdir_env = os.environ.get("TMPDIR")
        saved_tempfile_dir = tempfile.tempdir
        os.environ["TMPDIR"] = tmp_dir
        tempfile.tempdir = tmp_dir
        try:
            os.chdir(validated_out)
            os.environ["GIT_CEILING_DIRECTORIES"] = ceiling_dir
            if training_history_mod is not None and orig_git_fn is not None:
                training_history_mod.git_revision_short_hash = _quiet_git_revision_short_hash

            # Both `fme.ace.inference.__main__` and `fme.coupled.inference.__main__` parse a
            # single positional `yaml_config` argument (`fme/{ace,coupled}/inference/__main__.py`).
            sys.argv = [fme_module, config_path]
            restore_cudnn = pin_cudnn_benchmark_off() if is_coupled else (lambda: None)
            try:
                runpy.run_module(fme_module, run_name="__main__")
            finally:
                restore_cudnn()
        except SystemExit as sys_exit:
            if sys_exit.code is None:
                rc = 0
            elif isinstance(sys_exit.code, int):
                rc = sys_exit.code
            else:
                rc = 1
                caught_error = sys_exit
        except BaseException as exc:
            rc = 1
            caught_error = exc
        finally:
            sys.argv = saved_argv
            tempfile.tempdir = saved_tempfile_dir
            if saved_tmpdir_env is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = saved_tmpdir_env
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if training_history_mod is not None and orig_git_fn is not None:
                training_history_mod.git_revision_short_hash = orig_git_fn
            if saved_ceiling is None:
                os.environ.pop("GIT_CEILING_DIRECTORIES", None)
            else:
                os.environ["GIT_CEILING_DIRECTORIES"] = saved_ceiling
            if saved_cwd is not None:
                try:
                    os.chdir(saved_cwd)
                except OSError:
                    pass

            wall_seconds = round(time.monotonic() - t0, 3)
            manifest["finished"] = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            manifest["rc"] = int(rc)
            manifest["wall_seconds"] = wall_seconds
            if caught_error is not None:
                manifest["error"] = f"{type(caught_error).__name__}: {caught_error}"
            _write_json_atomic(manifest_path, manifest)
    finally:
        if inserted_overlay and active_overlay_dir:
            if active_overlay_dir in sys.path:
                sys.path.remove(active_overlay_dir)
            _evict_overlay_modules(only_under=active_overlay_dir)

    if caught_error is not None:
        raise caught_error
    if rc != 0:
        raise SystemExit(rc)

    return manifest
