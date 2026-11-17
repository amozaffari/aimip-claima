"""Non-fme model families: DLESyM (Earth2Studio) and ArchesClimate-SSP.

Both run in a dedicated Python environment (`/opt/aimip/venv-emu` inside the container, or `<prefix>/venv-emu` in
native mode: torch 2.14 cu130, earth2studio @ 8fa7acb, physicsnemo @ 1dacf8d, earth2grid @ 11dcf1b, plus
ArchesClimate-SSP's Python dependencies; ArchesClimate-SSP @ 06ce82dc itself is downloaded into
`$AIMIP_DATA/code/ArchesClimate-SSP` on first use). This module runs in the main environment and only uses the
standard library plus the CLI helpers:

- `prepare_emu_run`: validate the request, fetch the model on first use (same lock/fetch path as fme models), stage
  the ERA5 initial state for DLESyM (ARCO-ERA5, login node), and write `config.yaml` (JSON), `job.sbatch` (the same
  `aimip exec-fme <dir>` line every run uses) and `aimip-manifest.json`.
- `exec_emu`: at job time re-hash the checkpoint, the model package files and the initial state against the manifest,
  then run `python -I -m aimip.emu_job run <dir>` in the emulator environment and record `finished`, `rc`,
  `wall_seconds`.

Model facts:
- DLESyM v1.0 / v1.1: HF nvidia/dlesym-v1-era5 @ b88155c (`checkpoints/v1.0`, `checkpoints/v1.1`); v1.1 samples
  conditional-layer-norm noise every step, so members differ by seed. DLESyMv0-ISCCP: HF nvidia/dlesym-v0-isccp-era5
  @ 924b2d6 (the climate checkpoint Earth2Studio documents for 100-1000-year rollouts; `use_ttr=True` converts the
  initial ERA5 OLR to the model's ISCCP-like OLR). The V1 checkpoints were built for S2S forecasts. Atmosphere outputs every
  6 h, ocean every 48 h, on the 0.25 degree lat-lon grid; no forcing (insolation is computed).
- ArchesClimate-SSP deterministic: HF gclyne/ArchesClimate @ 06a914d, data gclyne/ArchesClimate-data @ b9f5458.
  Monthly steps on 144x144 (1.25 x 2.5 degree), IPSL-CM6A-LR r1i1p1f1 scenario forcing. The authors' rollout starts from
  dataset item 0 (state 2015-02, previous state 2015-01), so the first output is March 2015.
"""
from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

from . import __version__
from .errors import EXIT_UNSTABLE, ConfigError, MissingAssetsError, ValidationError
from .emu_catalog import DLESYM_IC_WINDOW
from .lora import sha256_file
from .registry import ModelSpec, resolve_data_dir, resolve_playground_dir
from .site_config import SiteConfig, load_site_config
from .validators import (
    makedirs_no_group_other_write,
    strip_group_other_write,
    validate_forcing_set_name,
    validate_int_range,
    validate_iso_timestamp,
    validate_run_path,
    validate_safe_path,
    validate_seed,
    validate_slurm_job_id,
    validate_slurm_time,
)

EMU_FAMILIES: tuple[str, ...] = ("e2s-dlesym", "arches")
EMU_JOB_MODULE = "aimip.emu_job"
DEFAULT_EMU_PYTHON = "/opt/aimip/venv-emu/bin/python"

# ARCO-ERA5 (public, anonymous GCS) supplies the DLESyM initial state. Inputs reach back 48 h and the v1.1 OLR input
# (ttr03) is a trailing 3-hour sum; with a day of margin the first start is 1940-01-04T00 (emu_catalog.DLESYM_IC_WINDOW).
DLESYM_MODELS: dict[str, dict[str, Any]] = {
    "dlesym-v1.0": {"package": "models/dlesym-v1", "version": "v1.0", "flavor": "v1", "members": False},
    "dlesym-v1.1": {"package": "models/dlesym-v1", "version": "v1.1", "flavor": "v1", "members": True},
    "dlesym-v0-isccp": {"package": "models/dlesym-v0-isccp", "version": "v0", "flavor": "v0", "members": False},
}
ARCHES_START = "2015-01-01T00:00:00"
ARCHES_FIRST_OUTPUT_MONTH = (2015, 3)       # item 0 = state 2015-02 with previous state 2015-01 -> first output 2015-03
ARCHES_MAX_STEPS = 1029                     # 1032 monthly states (2015-01..2100-12) minus previous state and offsets
ARCHES_SCENARIOS: tuple[str, ...] = ("ssp434", "ssp585")
ARCHES_DOMAIN = {"ssp434": "val", "ssp585": "test"}   # the checkpoint config's val_filter / test_filter

# Wall-time estimate (measured on one A100 GPU), padded 3x with a 45-minute floor.
DLESYM_SECONDS_PER_YEAR = 40.0
ARCHES_SECONDS_PER_MONTH = 2.0
EMU_FIXED_SECONDS = 900
EMU_SAFETY = 3.0
EMU_MIN_WALLTIME = 2700


def emu_python() -> str:
    """Return the absolute executable path of the DLESyM / ArchesClimate Python environment."""
    raw = os.environ.get("AIMIP_EMU_PYTHON")
    if not raw:
        home = os.environ.get("AIMIP_HOME") or os.environ.get("AIMIP_PLAYGROUND")
        cand = os.path.join(home.rstrip("/"), "venv-emu", "bin", "python") if home else ""
        if cand and (os.path.exists(cand) or not os.path.exists(DEFAULT_EMU_PYTHON)):
            raw = cand
        else:
            raw = DEFAULT_EMU_PYTHON
    path = validate_safe_path(raw, label="emulator python")
    if not os.path.isabs(path) or not os.access(path, os.X_OK):
        raise ConfigError(
            f"The DLESyM/ArchesClimate environment is missing ({path}). "
            "For native installs re-run `install/install-native.sh --with-emu` (or set $AIMIP_EMU_PYTHON); "
            "for containers rebuild the image with `install/install-container.sh`."
        )
    return path


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(target: str, payload: dict[str, Any]) -> None:
    tmp = f"{target}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    strip_group_other_write(tmp)
    os.replace(tmp, target)


def _format_hms(total_seconds: float) -> str:
    s = max(0, int(total_seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _parse_hms(value: str) -> int:
    parts = [int(p) for p in value.split("-")[-1].split(":")]
    days = int(value.split("-")[0]) if "-" in value else 0
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _walltime(family: str, steps: int, members: int, max_time: str) -> tuple[str, str, bool]:
    if family == "e2s-dlesym":
        work = steps * 6 / (365.25 * 24) * DLESYM_SECONDS_PER_YEAR * members
    else:
        work = steps * ARCHES_SECONDS_PER_MONTH * members
    want = max(EMU_MIN_WALLTIME, int(EMU_SAFETY * (EMU_FIXED_SECONDS + work)))
    cap = _parse_hms(max_time) if max_time else 12 * 3600
    return validate_slurm_time(_format_hms(min(want, cap))), _format_hms(want), want > cap


def _gpu_hours(family: str, steps: int, members: int) -> float:
    """Expected GPU time (no padding): rollout plus about one minute of loading."""
    if family == "e2s-dlesym":
        return (60 + steps * 6 / (365.25 * 24) * DLESYM_SECONDS_PER_YEAR * members) / 3600
    return (60 + steps * ARCHES_SECONDS_PER_MONTH * members) / 3600


# ---------------------------------------------------------------- pinned files per model

def model_package_files(model: ModelSpec, data_dir: str) -> dict[str, str]:
    """Absolute paths of the files the job loads and hashes (checkpoint, package constants, statistics)."""
    out: dict[str, str] = {}
    for rel in model.required_files:
        if model.family == "arches" and rel.startswith("inputs/") and rel.endswith(".memmap"):
            continue      # scenario arrays (GBs): recorded by size, like fme forcing files; side files are hashed
        out[rel] = os.path.join(data_dir, rel)
    return out


def _scenario_files(model: ModelSpec, data_dir: str, scenario: str) -> list[dict[str, Any]]:
    tag = f"r1i1p1f1_{scenario}_interpolation.memmap/"
    rows = []
    for rel in model.required_files:
        if rel.startswith("inputs/") and tag in rel and rel.endswith(".memmap"):
            p = os.path.join(data_dir, rel)
            ok_file = os.path.isfile(p) and not os.path.islink(p)
            rows.append(
                {
                    "rel_path": rel,
                    "path": p,
                    "size": os.path.getsize(p) if ok_file else None,
                    "sha256": sha256_file(p) if ok_file else None,
                }
            )
    return rows


def _check_against_lock(name: str, pkg: dict[str, str], pkg_sha: dict[str, str], data_root: str) -> None:
    """Every model file (checkpoints, configs, constants, statistics) must match the active aimip.lock."""
    from .setup_data import load_lock_file, resolve_lock_path

    lock = load_lock_file(resolve_lock_path(None, dest_dir=data_root))["files"]
    bad = []
    for rel, path in pkg.items():
        entry = lock.get(rel)
        if not isinstance(entry, dict) or str(entry.get("sha256", "")).lower() != pkg_sha[path].lower():
            bad.append(rel)
    if bad:
        raise ValidationError(
            f"{name}: {len(bad)} model file(s) do not match aimip.lock ({', '.join(bad[:3])}). "
            f"Ask the install owner to re-fetch: delete them and run `aimip fetch -m {name}`."
        )


def _config_sha(path: str) -> str:
    return sha256_file(path)


# ---------------------------------------------------------------- prepare

def precheck_emu_request(
    model: ModelSpec,
    *,
    start_iso: str,
    steps: int,
    members: int | None,
    forcing_set_name: str | None,
    ic_path: str | None,
    ocean_ic_path: str | None = None,
    lora_path: str | None = None,
) -> dict[str, Any]:
    """Validate a run request for a DLESyM / ArchesClimate model WITHOUT touching disk (runs before any download)."""
    norm_start = validate_iso_timestamp(start_iso, label="--start")
    n_steps = validate_int_range(steps, "--steps", min_value=1, max_value=10_000_000)
    n_members = validate_int_range(1 if members is None else members, "--members", min_value=1, max_value=64)
    for flag, value in (("--ic", ic_path), ("--ocean-ic", ocean_ic_path), ("--lora", lora_path)):
        if value is not None:
            raise ConfigError(
                f"{flag} is not supported for {model.name!r}: "
                + ("DLESyM starts from ERA5 at --start (fetched from ARCO-ERA5) and cannot be continued yet; "
                   "run the whole period in one job (about 40 s of GPU time per simulated year)."
                   if model.family == "e2s-dlesym"
                   else "ArchesClimate-SSP always starts from the scenario's 2015 state.")
            )
    if model.family == "e2s-dlesym":
        info = DLESYM_MODELS.get(model.name)
        if info is None:
            raise ConfigError(f"Unknown DLESyM model {model.name!r}.")
        if norm_start[11:] not in ("00:00:00", "06:00:00", "12:00:00", "18:00:00"):
            raise ConfigError(f"--start for {model.name} must be at 00, 06, 12 or 18 UTC, got {norm_start}.")
        if not (DLESYM_IC_WINDOW[0] <= norm_start <= DLESYM_IC_WINDOW[1]):
            raise ConfigError(
                f"--start {norm_start} is outside the ERA5 initial states this model can use "
                f"({DLESYM_IC_WINDOW[0]} .. {DLESYM_IC_WINDOW[1]}, ARCO-ERA5)."
            )
        if n_members > 1 and not info["members"]:
            raise ConfigError(
                f"{model.name} is deterministic: --members must be 1 (dlesym-v1.1 samples noise and supports members)."
            )
        if forcing_set_name is not None and forcing_set_name != model.forcing_sets[0].name:
            raise ConfigError(f"{model.name} has no forcing sets to choose from (it is not forced; insolation is computed).")
        scenario = model.forcing_sets[0].name
        start_dt = datetime.datetime.fromisoformat(norm_start)
        end_iso = (start_dt + datetime.timedelta(hours=6 * n_steps)).strftime("%Y-%m-%dT%H:%M:%S")
    elif model.family == "arches":
        if norm_start != ARCHES_START:
            raise ConfigError(
                f"{model.name} starts from the scenario's 2015 state only: use --start {ARCHES_START} "
                f"(the first output month is {ARCHES_FIRST_OUTPUT_MONTH[0]}-{ARCHES_FIRST_OUTPUT_MONTH[1]:02d})."
            )
        if n_members > 1:
            raise ConfigError(f"{model.name} (deterministic checkpoint) supports one member only.")
        if n_steps > ARCHES_MAX_STEPS:
            raise ConfigError(
                f"--steps {n_steps} exceeds the scenario forcing: at most {ARCHES_MAX_STEPS} monthly steps "
                "(March 2015 .. November 2100)."
            )
        names = [fs.name for fs in model.forcing_sets]
        if forcing_set_name is None:
            scenario = names[0]
        else:
            scenario = validate_forcing_set_name(forcing_set_name)
            if scenario not in names:
                raise ConfigError(f"Unknown --forcing-set {scenario!r} for {model.name}: choose one of {', '.join(names)}.")
        y, m = ARCHES_FIRST_OUTPUT_MONTH
        k = m - 1 + n_steps - 1
        end_iso = f"{y + k // 12:04d}-{k % 12 + 1:02d}-01T00:00:00"
    else:
        raise ConfigError(f"Model {model.name!r} is not a DLESyM/ArchesClimate model.")
    return {"start": norm_start, "steps": n_steps, "members": n_members, "scenario": scenario, "end": end_iso}


def _stage_dlesym_ic(out_dir: str) -> str:
    """Fetch the ERA5 state at --start into <out>/ic_era5.pt (login node, needs internet)."""
    py = emu_python()
    print(f"aimip: fetching the ERA5 initial state from ARCO-ERA5 (public Google Cloud bucket) into {out_dir}", file=sys.stderr)
    t0 = time.monotonic()
    try:
        # Redirect child stdout to stderr so `--json` and `aimip mcp` stdio streams remain pure protocol/JSON.
        try:
            sys.stderr.fileno()
            has_stderr_fd = True
        except (AttributeError, OSError, io.UnsupportedOperation):
            has_stderr_fd = False
        if has_stderr_fd:
            res = subprocess.run(
                [py, "-I", "-m", EMU_JOB_MODULE, "stage", out_dir],
                env=_job_env(out_dir, py),
                stdout=sys.stderr,
                stderr=sys.stderr,
                check=False,
            )
        else:
            res = subprocess.run(
                [py, "-I", "-m", EMU_JOB_MODULE, "stage", out_dir],
                env=_job_env(out_dir, py),
                capture_output=True,
                text=True,
                check=False,
            )
            if res.stdout:
                sys.stderr.write(res.stdout)
            if res.stderr:
                sys.stderr.write(res.stderr)
    finally:
        shutil.rmtree(os.path.join(out_dir, ".aimip-tmp"), ignore_errors=True)
    print(f"aimip: ERA5 state staged in {time.monotonic() - t0:.0f} s", file=sys.stderr)
    ic = os.path.join(out_dir, "ic_era5.pt")
    if res.returncode != 0 or not os.path.isfile(ic):
        raise ConfigError(
            f"Could not fetch the ERA5 initial state (rc={res.returncode}); see the messages above. "
            "This step needs internet access on the node where `aimip run` is called."
        )
    strip_group_other_write(ic)
    return ic


def _job_env(out_dir: str, py_exe: str | None = None) -> dict[str, str]:
    from .submit import sanitized_subprocess_env

    clean_parent = sanitized_subprocess_env()
    env = {k: v for k, v in clean_parent.items() if k.startswith(("AIMIP_", "SLURM_", "CUDA_", "NVIDIA_"))}
    if os.environ.get("LD_LIBRARY_PATH"):
        # Apptainer --nv and HPC CUDA modules expose host driver libraries via LD_LIBRARY_PATH.
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    tmp = os.path.join(out_dir, ".aimip-tmp")
    emu_bin = os.path.dirname(py_exe) if py_exe else "/opt/aimip/venv-emu/bin"
    env.update({
        "PATH": f"{emu_bin}:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "AIMIP_EMU_TMP": tmp,
        "EARTH2STUDIO_CACHE": os.path.join(tmp, "e2s-cache"),
        "MPLCONFIGDIR": os.path.join(tmp, "mpl"),
        "HF_HUB_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "16"),
    })
    return env


def prepare_emu_run(
    *,
    model: ModelSpec,
    start_iso: str,
    steps: int,
    out_dir: str | None,
    ic_path: str | None,
    ocean_ic_path: str | None,
    members: int | None,
    seed: int | None,
    lora_path: str | None,
    forcing_set_name: str | None,
    mode: str,
    slurm_opts: Any,
    site_config: SiteConfig | None,
    playground_dir: str | None,
    data_dir: str | None,
    auto_fetch: bool,
    argv: Sequence[str] | None,
) -> dict[str, Any]:
    from .runner import (   # local import: runner imports this module
        SlurmOptions,
        default_out_dir,
        ensure_model_files_ready,
        model_has_missing_files,
        render_sbatch_script,
        resolve_active_lock_sha256,
    )

    req = precheck_emu_request(
        model, start_iso=start_iso, steps=steps, members=members, forcing_set_name=forcing_set_name,
        ic_path=ic_path, ocean_ic_path=ocean_ic_path, lora_path=lora_path,
    )
    if model.family == "e2s-dlesym" and not auto_fetch:
        # --no-fetch forbids network downloads during preparation.
        raise MissingAssetsError(
            f"{model.name} reads its ERA5 initial state from ARCO-ERA5 when the run is prepared, and --no-fetch forbids "
            f"downloads. Prepare it on the host: aimip run -m {model.name} --start {req['start']} --steps {req['steps']} "
            "--submit",
            field="model",
            next_command=f"aimip run -m {model.name} --start {req['start']} --steps {req['steps']} --submit",
        )
    resolved_seed = validate_seed(0 if seed is None else seed)
    resolved_playground = resolve_playground_dir(playground_dir)
    active_site = site_config if site_config is not None else load_site_config()

    if out_dir is not None:
        validated_out = validate_run_path(out_dir, label="--out")
    else:
        validated_out = default_out_dir(
            model.name, req["start"], extra_entropy=f"{req['steps']}|{req['members']}|{resolved_seed}|{req['scenario']}"
        )
    if resolved_playground:
        pg = resolved_playground.rstrip("/")
        if validated_out == pg or validated_out.startswith(pg + "/"):
            raise ConfigError(
                f"Refusing to write run outputs ({validated_out}) inside $AIMIP_PLAYGROUND ({pg}); choose another --out."
            )
    manifest_path = validate_run_path(f"{validated_out}/aimip-manifest.json", label="aimip-manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                prev = json.load(fh)
            if isinstance(prev, dict) and "finished" in prev:
                raise ConfigError(f"Refusing to overwrite completed run directory {validated_out}; choose a new --out path.")
        except (OSError, json.JSONDecodeError):
            pass

    if model_has_missing_files(model, playground_dir=playground_dir, data_dir=data_dir):
        model = ensure_model_files_ready(model, playground_dir=playground_dir, data_dir=data_dir, auto_fetch=auto_fetch)
    data_root = resolve_data_dir(data_dir, playground_dir=resolved_playground)

    pkg = model_package_files(model, data_root)
    missing = [p for p in pkg.values() if not os.path.isfile(p)]
    if missing:
        raise ConfigError(f"{model.name}: model files missing after fetch: {', '.join(missing[:3])}")
    print(f"aimip: verifying {len(pkg)} model file(s) of {model.name}", file=sys.stderr)
    pkg_sha = {p: sha256_file(p) for p in pkg.values()}
    _check_against_lock(model.name, pkg, pkg_sha, data_root)
    if model.checkpoint not in pkg_sha:
        raise ConfigError(f"Internal error: checkpoint {model.checkpoint} is not among the pinned files of {model.name}.")
    if model.checkpoint_sha256 and pkg_sha[model.checkpoint].lower() != model.checkpoint_sha256.lower():
        raise ValidationError(
            f"Checkpoint SHA-256 mismatch for model {model.name!r}: registry expects {model.checkpoint_sha256}, "
            f"got {pkg_sha[model.checkpoint]}."
        )

    effective_slurm = slurm_opts.validate() if slurm_opts is not None else SlurmOptions.from_site(active_site)
    makedirs_no_group_other_write(validated_out)

    scenario_rows: list[dict[str, Any]] = []
    if model.family == "e2s-dlesym":
        info = DLESYM_MODELS[model.name]
        package_dir = os.path.join(data_root, info["package"])
        job = {"family": model.family, "model": model.name, "version": info["version"], "flavor": info["flavor"],
               "package_dir": package_dir}
        ic_provenance = f"ERA5 state at {req['start']} from ARCO-ERA5 (gs://gcp-public-data-arco-era5), staged at prepare time"
    else:
        scen = req["scenario"]
        scenario_rows = _scenario_files(model, data_root, scen)
        if not scenario_rows or any(r["size"] is None or not r.get("sha256") for r in scenario_rows):
            raise ConfigError(f"{model.name}: scenario {scen} files are missing; run `aimip fetch -m {model.name}`.")
        _check_against_lock(
            model.name,
            {str(r["rel_path"]): str(r["path"]) for r in scenario_rows},
            {str(r["path"]): str(r["sha256"]) for r in scenario_rows},
            data_root,
        )
        job = {"family": model.family, "model": model.name, "scenario": scen, "domain": ARCHES_DOMAIN[scen],
               "model_dir": os.path.dirname(model.checkpoint),
               "memmap_dir": os.path.join(data_root, "inputs/archesclimate-ssp/memmap_filled_in"),
               "reference_dir": os.path.join(data_root, "models/archesclimate-ssp/reference_data"),
               "stats_file": os.path.join(data_root, "models/archesclimate-ssp/stats/cmip_stats_new_ozone.pt"),
               "first_output_month": "%04d-%02d" % ARCHES_FIRST_OUTPUT_MONTH}
        ic_provenance = (f"IPSL-CM6A-LR r1i1p1f1 {scen} state of 2015-02 (previous state 2015-01) from "
                         "gclyne/ArchesClimate-data; first output month 2015-03")
    job.update({"out_dir": validated_out, "start": req["start"], "end": req["end"], "steps": req["steps"],
                "members": req["members"], "seed": resolved_seed, "checkpoint": model.checkpoint})

    config_path = validate_run_path(f"{validated_out}/config.yaml", label="config.yaml")
    with open(config_path, "w", encoding="utf-8") as fh:    # JSON is valid YAML; the job reads it with json
        json.dump(job, fh, indent=2)
        fh.write("\n")
    strip_group_other_write(config_path)

    if model.family == "e2s-dlesym":
        ic_file = _stage_dlesym_ic(validated_out)
    else:
        ic_file = os.path.join(data_root, f"inputs/archesclimate-ssp/memmap_filled_in/r1i1p1f1_{req['scenario']}_interpolation.memmap/meta.json")
    ic_sha = sha256_file(ic_file)

    max_time = getattr(active_site, "max_time", "12:00:00") or "12:00:00"
    default_walltime, unclamped, capped = _walltime(model.family, req["steps"], req["members"], max_time)
    walltime_capped = bool(effective_slurm.time is None and capped)
    if effective_slurm.scheduler == "slurm" and walltime_capped:
        print(f"aimip warning: estimated walltime {unclamped} exceeds the site limit {max_time}; clamped. "
              "Split the run or pass --time.", file=sys.stderr)
    sbatch_path = validate_run_path(f"{validated_out}/job.sbatch", label="job.sbatch")
    with open(sbatch_path, "w", encoding="utf-8") as fh:
        fh.write(render_sbatch_script(model_name=model.name, out_dir=validated_out, playground_dir=resolved_playground,
                                      slurm_opts=effective_slurm, default_walltime=default_walltime))
    strip_group_other_write(sbatch_path)

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
        "family": model.family,
        "checkpoint_path": model.checkpoint,
        "checkpoint_sha256": pkg_sha[model.checkpoint],
        "package_files_sha256": pkg_sha,
        "ic_path": ic_file,
        "ic_sha256": ic_sha,
        "source_ic_path": ic_file,
        "source_ic_sha256": ic_sha,
        "ic_calendar": model.calendar,
        "ic_is_custom": False,
        "ensemble_strategy": "seeded_noise_members" if req["members"] > 1 else "single_member",
        "ensemble_note": ("members share the ERA5 state and differ by the conditional-layer-norm noise seed (seed + k)"
                          if req["members"] > 1 else None),
        "lora_checkpoint": None,
        "lora_info": None,
        "n_ic_samples": 1,
        "n_ensemble_per_ic": req["members"],
        "gpu_hours_estimate": round(_gpu_hours(model.family, req["steps"], req["members"]), 6),
        "ic_provenance": ic_provenance,
        "start_time": req["start"],
        "end_time": req["end"],
        "steps": req["steps"],
        "step_unit": "6h" if model.family == "e2s-dlesym" else "month",
        "members": req["members"],
        "seed": resolved_seed,
        "seed_explicit": seed is not None,
        "member_seeds": [resolved_seed + k for k in range(req["members"])],
        "forcing_set": next(fs.to_dict() for fs in model.forcing_sets if fs.name == req["scenario"]),
        "forcing_files": scenario_rows,
        "site": {"name": active_site.name, "scheduler": effective_slurm.scheduler, "runtime": active_site.runtime},
        "slurm": {
            "scheduler": effective_slurm.scheduler, "partition": effective_slurm.partition,
            "account": effective_slurm.account, "gres": effective_slurm.gres,
            "cpus_per_task": effective_slurm.cpus_per_task, "mem": effective_slurm.mem,
            "time": effective_slurm.time or default_walltime, "unclamped_estimated_time": unclamped,
            "walltime_capped": walltime_capped,
            **({"qos": effective_slurm.qos} if effective_slurm.qos else {}),  # MN5; only when set
        },
        "slurm_job_id": None,
        "image_sha": (os.environ.get("AIMIP_IMAGE_SHA") or "")[:256],
        "runtime_env": os.path.dirname(os.path.dirname(emu_python())),
        "runtime_python": emu_python(),
        "config_sha256": _config_sha(config_path),
        "created_utc": _now_utc(),
    }
    _write_json_atomic(manifest_path, manifest)
    if mode == "local":
        return exec_emu(validated_out, manifest, manifest_path)
    return manifest


# ---------------------------------------------------------------- job time

def _check_job_matches_manifest(job: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Every field the job uses must agree with the manifest and the pinned files."""
    def fail(what: str) -> None:
        raise ValidationError(f"Integrity check failed: config.yaml {what} does not match aimip-manifest.json.")

    reg = manifest.get("registry_entry") or {}
    for key, mkey in (("model", None), ("start", "start_time"), ("end", "end_time"), ("steps", "steps"),
                      ("members", "members"), ("seed", "seed"), ("family", "family"), ("checkpoint", "checkpoint_path")):
        want = reg.get("name") if mkey is None else manifest.get(mkey)
        if str(job.get(key)) != str(want):
            fail(key)
    pinned = set((manifest.get("package_files_sha256") or {}).keys())
    if job["checkpoint"] not in pinned:
        fail("checkpoint")
    if job["family"] == "e2s-dlesym":
        info = DLESYM_MODELS.get(job["model"])
        if info is None or job.get("version") != info["version"] or job.get("flavor") != info["flavor"]:
            fail("version/flavor")
        pkg = str(job.get("package_dir", "")).rstrip("/") + "/"
        if not job["checkpoint"].startswith(pkg) or not all(p.startswith(pkg) for p in pinned):
            fail("package_dir")
        if str(manifest.get("ic_path")) != os.path.join(job["out_dir"], "ic_era5.pt"):
            fail("initial state")
    elif job["family"] == "arches":
        scen = (manifest.get("forcing_set") or {}).get("name")
        if job.get("scenario") != scen or job.get("domain") != ARCHES_DOMAIN.get(str(scen)):
            fail("scenario/domain")
        if job.get("model_dir") != os.path.dirname(job["checkpoint"]) or job.get("stats_file") not in pinned:
            fail("model_dir/stats_file")
        if os.path.join(job["model_dir"], "config.yaml") not in pinned:
            fail("model config")
        if job.get("first_output_month") != "%04d-%02d" % ARCHES_FIRST_OUTPUT_MONTH:
            fail("first_output_month")
        mm = str(job.get("memmap_dir", "")).rstrip("/") + f"/r1i1p1f1_{scen}_interpolation.memmap/"
        rows = manifest.get("forcing_files") or []
        if not rows or not all(str(r.get("path", "")).startswith(mm) for r in rows):
            fail("memmap_dir")
        if os.path.dirname(job.get("reference_dir", "")) != os.path.dirname(os.path.dirname(job["checkpoint"])):
            fail("reference_dir")
    else:
        fail("family")


def exec_emu(validated_out: str, manifest: dict[str, Any], manifest_path: str) -> dict[str, Any]:
    """Re-verify the pinned files, run the job in the second environment, record the outcome."""
    config_path = validate_run_path(f"{validated_out}/config.yaml", label="config.yaml")
    try:
        with open(config_path, encoding="utf-8") as fh:
            job = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to read {config_path}: {exc}") from exc
    if not isinstance(job, dict) or job.get("out_dir") != validated_out:
        raise ValidationError(f"Integrity check failed in {config_path}: out_dir does not match {validated_out}.")
    if job.get("checkpoint") != manifest.get("checkpoint_path") or job.get("family") != manifest.get("family"):
        raise ValidationError(f"Integrity check failed in {config_path}: checkpoint/family differ from the manifest.")
    want_cfg = str(manifest.get("config_sha256") or "")
    if len(want_cfg) != 64 or _config_sha(config_path).lower() != want_cfg.lower():
        raise ValidationError(f"Integrity check failed: {config_path} changed after `aimip run` prepared it.")
    _check_job_matches_manifest(job, manifest)

    pinned = dict(manifest.get("package_files_sha256") or {})
    pinned[str(manifest.get("ic_path"))] = str(manifest.get("ic_sha256"))
    if str(manifest.get("checkpoint_path")) not in pinned:
        raise ValidationError("Manifest does not pin the checkpoint.")
    for path, want in pinned.items():
        p = validate_safe_path(path, label="pinned file")
        if not os.path.isfile(p):
            raise ValidationError(f"Pinned file not found at job time: {p}")
        got = sha256_file(p).lower()
        if got != str(want).lower():
            raise ValidationError(f"SHA-256 mismatch at job time for {p}: manifest expects {want}, got {got}.")
    for row in manifest.get("forcing_files") or []:
        p = validate_safe_path(str(row.get("path")), label="scenario file")
        if not os.path.isfile(p) or os.path.islink(p) or os.path.getsize(p) != row.get("size"):
            raise ValidationError(f"Scenario file changed or missing at job time: {p}")
        want_scen_sha = str(row.get("sha256") or "").strip().lower()
        if len(want_scen_sha) != 64:
            raise ValidationError(f"Manifest does not pin SHA-256 for scenario file: {p}")
        got_scen_sha = sha256_file(p).lower()
        if got_scen_sha != want_scen_sha:
            raise ValidationError(
                f"SHA-256 mismatch at job time for scenario file {p}: "
                f"manifest expects {want_scen_sha}, got {got_scen_sha}."
            )

    raw_job_id = os.environ.get("SLURM_JOB_ID")
    if raw_job_id:
        try:
            manifest["slurm_job_id"] = validate_slurm_job_id(raw_job_id.strip())
        except ValidationError:
            pass
    manifest["started_utc"] = _now_utc()
    _write_json_atomic(manifest_path, manifest)

    tmp_dir = os.path.join(validated_out, ".aimip-tmp")
    if os.path.islink(tmp_dir):
        raise ConfigError(f"Refusing a symlinked temporary directory: {tmp_dir}")
    os.makedirs(tmp_dir, mode=0o700, exist_ok=True)
    t0 = time.monotonic()
    rc = 1
    err: str | None = None
    try:
        py = emu_python()
        res = subprocess.run([py, "-I", "-m", EMU_JOB_MODULE, "run", validated_out],
                             env=_job_env(validated_out, py), check=False)
        rc = int(res.returncode)
        if rc == EXIT_UNSTABLE:
            err = "the rollout became unstable (non-finite or out-of-band month): see result.unstable_from"
        elif rc != 0:
            err = f"emulator job exited with rc={rc} (see slurm-*.err)"
    except BaseException as exc:   # noqa: BLE001 - recorded, then re-raised below
        err = f"{type(exc).__name__}: {exc}"
        rc = 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        manifest["finished"] = _now_utc()
        manifest["rc"] = rc
        manifest["wall_seconds"] = round(time.monotonic() - t0, 3)
        if err:
            manifest["error"] = err
        result_file = os.path.join(validated_out, "emu-result.json")
        if os.path.isfile(result_file) and not os.path.islink(result_file):
            try:
                with open(result_file, encoding="utf-8") as fh:
                    manifest["result"] = json.load(fh)
            except (OSError, json.JSONDecodeError):
                pass
        _write_json_atomic(manifest_path, manifest)
    if rc != 0:
        raise SystemExit(rc)
    return manifest
