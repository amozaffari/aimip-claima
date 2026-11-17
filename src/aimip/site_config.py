"""Portable HPC site configuration (`site.json`) presets, loader, and `aimip site init`.

Matches the installer contract (`install/install-container.sh`, `install/install-native.sh`,
`install/presets/*.json`, and `container/aimip-wrapper.in`):
- `scheduler`: `"slurm"` or `"none"`
- `runtime`: `"apptainer"`, `"singularity"`, `"docker"`, `"podman"`, or `"native"`
- `sbatch`: `{"partition", "qos", "account", "gres", "cpus_per_task", "mem", "max_time"}`; `qos` is optional, and
  with a `qos` the `partition` may be empty or `"none"` (MN5: the QOS places the job, there is no partition to name)
- `run_root`: template or absolute path (supports `${USER}`, `${U1}`, `${HOME}`)
- `extra_ro_binds`: optional list of read-only host bind paths (e.g. `["/work"]` on Levante)
- `trusted_group_parents`: optional list of root-owned, group-writable directories accepted above a run directory
  (MN5's `/gpfs/scratch/<group>` layout; see `submit.trusted_group_parents`)
"""

from __future__ import annotations

import dataclasses
import getpass
import json
import os
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .errors import ConfigError, ValidationError
from .validators import (
    validate_int_range,
    validate_safe_path,
    validate_slurm_mem,
    validate_slurm_time,
    validate_slurm_token,
)

VALID_SCHEDULERS = frozenset({"slurm", "none"})
VALID_RUNTIMES = frozenset({"apptainer", "singularity", "docker", "podman", "native"})
_SAFE_USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def parse_slurm_walltime_seconds(value: str) -> int:
    """Convert a validated SLURM `--time` string (`HH:MM:SS`, `D-HH:MM:SS`, or `MM`) into seconds."""
    spec = validate_slurm_time(value)
    if ":" not in spec:
        return int(spec) * 60
    days = 0
    time_part = spec
    if "-" in spec:
        day_str, time_part = spec.split("-", 1)
        days = int(day_str)
    hh_s, mm_s, ss_s = time_part.split(":")
    return ((days * 24 + int(hh_s)) * 3600) + (int(mm_s) * 60) + int(ss_s)


def expand_run_root_template(template: str) -> str | None:
    """Expand `${USER}`, `${U1}`, and `${HOME}` in a `site.json` `run_root` string safely."""
    if not isinstance(template, str) or not template.strip():
        return None
    raw_user = (
        os.environ.get("AIMIP_USER")
        or os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or getpass.getuser()
        or "aimip"
    )
    if not _SAFE_USER_RE.match(raw_user):
        raw_user = "aimip"
    u1 = raw_user[0]
    raw_home = os.environ.get("HOME") or "/tmp"
    expanded = (
        template.replace("${USER}", raw_user)
        .replace("${U1}", u1)
        .replace("${HOME}", raw_home.rstrip("/") or "/tmp")
    )
    if "${" in expanded or "$" in expanded:
        return None
    try:
        return validate_safe_path(expanded, label="site.json run_root")
    except ValidationError:
        return None


@dataclasses.dataclass(frozen=True)
class SiteConfig:
    """Validated site profile controlling SLURM batch defaults, runtime, and run roots."""

    name: str = "levante"
    scheduler: str = "slurm"
    runtime: str = "apptainer"
    partition: str = "gpu"
    account: str = "CHANGE_ME"
    gres: str = "gpu:a100_80:1"
    cpus_per_task: int = 32
    # Empty (or "none" in site.json/CLI): no `--mem` line. MN5 refuses jobs that set --mem; it gives memory per
    # core (8 GB per core on ACC nodes), so --cpus-per-task sets the memory there.
    mem: str = "64G"
    max_time: str = "12:00:00"
    run_root: str = "/scratch/${U1}/${USER}/aimip-runs"
    extra_ro_binds: tuple[str, ...] = ("/work",)
    allowed_run_roots: tuple[str, ...] = ()
    gpu_check: str = "nvidia-smi"
    # MN5 (BSC MareNostrum 5): jobs are placed by QOS (`#SBATCH --qos=acc_ehpc`), not by partition. Empty: no
    # `--qos` line, as before.
    qos: str = ""
    # MN5: root-owned group directories (drwxrws---) that may sit above a run directory; see
    # `submit.trusted_group_parents` for why. Empty: the strict ancestry rule applies everywhere, as before.
    trusted_group_parents: tuple[str, ...] = ()

    def validate(self, *, allow_change_me: bool = False) -> SiteConfig:
        """Validate all fields of this `SiteConfig`."""
        if self.scheduler not in VALID_SCHEDULERS:
            raise ValidationError(
                f"Invalid site scheduler {self.scheduler!r}: expected one of {sorted(VALID_SCHEDULERS)}."
            )
        if self.runtime not in VALID_RUNTIMES:
            raise ValidationError(
                f"Invalid site container runtime {self.runtime!r}: expected one of {sorted(VALID_RUNTIMES)}."
            )
        validate_slurm_token(self.name, "site name")

        if self.scheduler == "slurm":
            # A partition is required unless a QOS is given: on MN5 the QOS (acc_debug, acc_ehpc, ...) selects the
            # partition and no partition line is written.
            if not self.partition and not self.qos:
                raise ValidationError(
                    "Site config with scheduler='slurm' requires a non-empty 'partition' "
                    "(or a 'qos' that selects it, as on MN5)."
                )
            if self.partition:
                if not allow_change_me and self.partition == "CHANGE_ME":
                    raise ConfigError(
                        "Site config 'partition' is still set to 'CHANGE_ME'. "
                        "Pass --partition <name> or run `aimip site init` to configure your cluster."
                    )
                validate_slurm_token(self.partition, "site partition")
            if self.qos:
                validate_slurm_token(self.qos, "site qos")

            if self.account:
                if not allow_change_me and self.account == "CHANGE_ME":
                    raise ConfigError(
                        "Site config 'account' is still set to 'CHANGE_ME'. "
                        "Pass --account <acct> or run `aimip site init` to configure your cluster."
                    )
                validate_slurm_token(self.account, "site account")

            if self.gres:
                validate_slurm_token(self.gres, "site gres")

            validate_int_range(
                self.cpus_per_task, "site cpus_per_task", min_value=1, max_value=256
            )
            if self.mem:  # empty: no --mem line (MN5)
                validate_slurm_mem(self.mem)
            validate_slurm_time(self.max_time)
        else:
            if self.partition and self.partition != "CHANGE_ME":
                validate_slurm_token(self.partition, "site partition")
            if self.account and self.account != "CHANGE_ME":
                validate_slurm_token(self.account, "site account")
            if self.gres:
                validate_slurm_token(self.gres, "site gres")
            validate_int_range(
                self.cpus_per_task, "site cpus_per_task", min_value=1, max_value=256
            )
            if self.mem:  # empty: no --mem line (MN5)
                validate_slurm_mem(self.mem)
            validate_slurm_time(self.max_time)

        for bind_path in self.extra_ro_binds:
            validate_safe_path(bind_path, label="site extra_ro_binds entry")
        for root_path in self.allowed_run_roots:
            validate_safe_path(root_path, label="site allowed_run_roots entry")
        for group_dir in self.trusted_group_parents:
            norm = validate_safe_path(group_dir, label="site trusted_group_parents entry")
            if os.path.normpath(norm) == "/":
                raise ValidationError("site trusted_group_parents may not contain '/'.")

        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical `install/presets/*.json` JSON structure."""
        sbatch_block: dict[str, Any]
        if self.scheduler == "slurm":
            sbatch_block = {
                "partition": self.partition,
                "account": self.account,
                "gres": self.gres,
                "cpus_per_task": self.cpus_per_task,
                "mem": self.mem,
                "max_time": self.max_time,
            }
            if self.qos:  # MN5; written only when set, so other sites' site.json is unchanged
                sbatch_block["qos"] = self.qos
        else:
            sbatch_block = {}

        payload: dict[str, Any] = {
            "name": self.name,
            "scheduler": self.scheduler,
            "runtime": self.runtime,
            "sbatch": sbatch_block,
            "run_root": self.run_root,
            "extra_ro_binds": list(self.extra_ro_binds),
            "gpu_check": self.gpu_check,
        }
        if self.allowed_run_roots:
            payload["allowed_run_roots"] = list(self.allowed_run_roots)
        if self.trusted_group_parents:  # MN5
            payload["trusted_group_parents"] = list(self.trusted_group_parents)
        return payload

    @classmethod
    def preset(cls, preset_name: str) -> SiteConfig:
        """Return a built-in `SiteConfig` preset (`levante`, `generic-slurm`, `mn5`, or `local`/`none`)."""
        key = preset_name.strip().lower()
        if key == "mn5":
            # BSC MareNostrum 5, accelerated (ACC) partition: 4 x H100-64GB and 80 cores per node. No partition: the
            # QOS places the job (acc_ehpc: up to 3 days; acc_debug: up to 2 h, one job at a time). MN5 asks for 20
            # cores per GPU. Group directories on /gpfs are root-owned and group-writable, hence
            # trusted_group_parents (see submit.trusted_group_parents). The account and run root are per user.
            return cls(
                name="mn5",
                scheduler="slurm",
                runtime="native",
                partition="",
                qos="acc_ehpc",
                account="CHANGE_ME",
                gres="gpu:1",
                cpus_per_task=20,
                mem="",  # MN5 refuses --mem: memory comes with the cores (8 GB each, 160 GB for 20)
                max_time="12:00:00",
                run_root="${HOME}/aimip-runs",
                extra_ro_binds=(),
                gpu_check="nvidia-smi",
            )
        if key == "levante":
            return cls(
                name="levante",
                scheduler="slurm",
                runtime="apptainer",
                partition="gpu",
                account="CHANGE_ME",
                gres="gpu:a100_80:1",
                cpus_per_task=32,
                mem="64G",
                max_time="12:00:00",
                run_root="/scratch/${U1}/${USER}/aimip-runs",
                extra_ro_binds=("/work",),
                gpu_check="nvidia-smi",
            )
        if key == "generic-slurm":
            return cls(
                name="generic-slurm",
                scheduler="slurm",
                runtime="apptainer",
                partition="CHANGE_ME",
                account="CHANGE_ME",
                gres="gpu:1",
                cpus_per_task=16,
                mem="64G",
                max_time="12:00:00",
                run_root="${HOME}/aimip-runs",
                extra_ro_binds=(),
                gpu_check="nvidia-smi",
            )
        if key in ("local", "none"):
            return cls(
                name="local",
                scheduler="none",
                runtime="docker",
                partition="",
                account="",
                gres="",
                cpus_per_task=8,
                mem="16G",
                max_time="12:00:00",
                run_root="${HOME}/aimip-runs",
                extra_ro_binds=(),
                gpu_check="nvidia-smi",
            )
        raise ValidationError(
            f"Unknown site preset {preset_name!r}: expected 'levante', 'generic-slurm', 'mn5', or 'local'."
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SiteConfig:
        """Parse either nested `sbatch` (`install/presets/*.json`) or flat `site.json` dicts."""
        if not isinstance(raw, dict):
            raise ValidationError(
                f"site.json top-level value must be an object, got {type(raw).__name__}."
            )

        scheduler = str(raw.get("scheduler", "slurm")).strip().lower()
        runtime = str(
            raw.get("runtime") or raw.get("container_runtime") or "apptainer"
        ).strip().lower()
        name = str(raw.get("name") or scheduler or "site").strip()

        sbatch_raw = raw.get("sbatch")
        if sbatch_raw is not None and not isinstance(sbatch_raw, dict):
            raise ValidationError("site.json 'sbatch' field must be an object.")
        sb = sbatch_raw if isinstance(sbatch_raw, dict) else {}

        def _pick(sb_key: str, *top_keys: str, default: Any) -> Any:
            if sb_key in sb and sb[sb_key] is not None:
                return sb[sb_key]
            for tk in top_keys:
                if tk in raw and raw[tk] is not None:
                    return raw[tk]
            return default

        is_none = scheduler == "none"
        def_partition = "" if is_none else "gpu"
        def_account = ""
        def_gres = "" if is_none else "gpu:1"
        def_cpus = 8 if is_none else 16
        def_mem = "16G" if is_none else "64G"
        def_max_time = "12:00:00"
        def_run_root = "${HOME}/aimip-runs"

        partition_raw = str(_pick("partition", "partition", default=def_partition))
        # "none" (as for account and gres) means no partition line; with a qos that is the MN5 setup
        partition = "" if partition_raw.strip().lower() == "none" else partition_raw
        qos_raw = str(_pick("qos", "qos", default=""))
        qos = "" if qos_raw.strip().lower() == "none" else qos_raw
        account_raw = str(_pick("account", "account", default=def_account))
        account = "" if account_raw.strip().lower() == "none" else account_raw
        gres_raw = str(_pick("gres", "gres", default=def_gres))
        gres = "" if gres_raw.strip().lower() == "none" else gres_raw
        cpus_val = _pick(
            "cpus_per_task", "cpus_per_task", "cpus", default=def_cpus
        )
        mem = str(_pick("mem", "mem", default=def_mem))
        mem = "" if mem.strip().lower() == "none" else mem  # no --mem line (MN5 refuses it)
        max_time = str(
            _pick(
                "max_time",
                "max_time",
                "default_time_cap",
                "max_walltime",
                default=def_max_time,
            )
        )
        run_root = str(raw.get("run_root") or def_run_root)

        raw_ro = raw.get("extra_ro_binds", [])
        if not isinstance(raw_ro, list):
            raise ValidationError("site.json 'extra_ro_binds' must be a list of paths.")
        extra_ro_binds = tuple(str(x) for x in raw_ro if x)

        raw_allowed = raw.get("allowed_run_roots", [])
        if not isinstance(raw_allowed, list):
            raise ValidationError("site.json 'allowed_run_roots' must be a list of paths.")
        allowed_run_roots = tuple(str(x) for x in raw_allowed if x)

        raw_trusted = raw.get("trusted_group_parents", [])
        if not isinstance(raw_trusted, list):
            raise ValidationError("site.json 'trusted_group_parents' must be a list of paths.")
        trusted_group_parents = tuple(str(x) for x in raw_trusted if x)

        gpu_check = str(raw.get("gpu_check") or "nvidia-smi")

        return cls(
            name=name,
            scheduler=scheduler,
            runtime=runtime,
            partition=partition,
            account=account,
            gres=gres,
            cpus_per_task=cpus_val,
            mem=mem,
            max_time=max_time,
            run_root=run_root,
            extra_ro_binds=extra_ro_binds,
            allowed_run_roots=allowed_run_roots,
            gpu_check=gpu_check,
            qos=qos,
            trusted_group_parents=trusted_group_parents,
        ).validate(allow_change_me=True)


def resolve_site_config_path(
    explicit_path: str | None = None,
    *,
    require_target: bool = False,
) -> tuple[str | None, bool]:
    """Resolve `(site_json_path, is_explicit)` from args or `$AIMIP_SITE_CONFIG` / `$AIMIP_HOME` / `$AIMIP_PLAYGROUND`."""
    if explicit_path:
        return validate_safe_path(explicit_path, label="--out site.json"), True

    env_explicit = os.environ.get("AIMIP_SITE_CONFIG")
    if env_explicit:
        return validate_safe_path(env_explicit, label="AIMIP_SITE_CONFIG"), True

    for env_var in ("AIMIP_HOME", "AIMIP_PLAYGROUND"):
        root = os.environ.get(env_var)
        if root:
            base = validate_safe_path(root, label=env_var).rstrip("/")
            return validate_safe_path(f"{base}/site.json", label=f"{env_var}/site.json"), False

    if require_target:
        raise ConfigError(
            "Cannot determine target path for site.json: set $AIMIP_HOME or pass --out <path>."
        )
    return None, False


def load_site_config(explicit_path: str | None = None) -> SiteConfig:
    """Load `SiteConfig` from `site.json`, or fall back to built-in Levante/env defaults when absent."""
    env_sched = (os.environ.get("AIMIP_SCHEDULER") or "").strip().lower()
    path, is_explicit = resolve_site_config_path(explicit_path, require_target=False)
    if path is not None and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Failed to parse site config {path}: {exc}") from exc
        cfg = SiteConfig.from_dict(payload)
        if not is_explicit and env_sched in VALID_SCHEDULERS and env_sched != cfg.scheduler:
            cfg = dataclasses.replace(cfg, scheduler=env_sched).validate(allow_change_me=True)
        return cfg

    if is_explicit and path is not None:
        raise ConfigError(f"Site configuration file not found: {path}")

    if env_sched == "none":
        return SiteConfig.preset("local")
    return SiteConfig.preset("levante")


def init_site_config(
    *,
    preset: str = "levante",
    out_path: str | None = None,
    scheduler: str | None = None,
    runtime: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    gres: str | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    max_time: str | None = None,
    run_root: str | None = None,
    extra_ro_binds: Sequence[str] | None = None,
    allowed_run_roots: Sequence[str] | None = None,
    overwrite: bool = False,
    qos: str | None = None,
    trusted_group_parents: Sequence[str] | None = None,
) -> tuple[str, SiteConfig]:
    """Build, validate, and write `site.json` (`$AIMIP_HOME/site.json` or `out_path`)."""
    target_path, _ = resolve_site_config_path(out_path, require_target=True)
    assert target_path is not None

    if os.path.exists(target_path) and not overwrite:
        raise ConfigError(
            f"Refusing to overwrite existing {target_path} (pass --overwrite to replace it)."
        )

    base = SiteConfig.preset(preset)
    norm_account: str | None = None
    if account is not None:
        norm_account = "" if account.strip().lower() == "none" else account.strip()
    norm_gres: str | None = None
    if gres is not None:
        norm_gres = "" if gres.strip().lower() == "none" else gres.strip()
    norm_partition: str | None = None
    if partition is not None:
        norm_partition = "" if partition.strip().lower() == "none" else partition.strip()
    norm_qos: str | None = None
    if qos is not None:
        norm_qos = "" if qos.strip().lower() == "none" else qos.strip()

    cfg = dataclasses.replace(
        base,
        scheduler=scheduler.strip().lower() if scheduler is not None else base.scheduler,
        runtime=runtime.strip().lower() if runtime is not None else base.runtime,
        partition=norm_partition if norm_partition is not None else base.partition,
        qos=norm_qos if norm_qos is not None else base.qos,
        account=norm_account if norm_account is not None else base.account,
        gres=norm_gres if norm_gres is not None else base.gres,
        cpus_per_task=cpus_per_task if cpus_per_task is not None else base.cpus_per_task,
        mem=(("" if mem.strip().lower() == "none" else mem.strip()) if mem is not None else base.mem),
        max_time=max_time if max_time is not None else base.max_time,
        run_root=run_root if run_root is not None else base.run_root,
        extra_ro_binds=(
            tuple(extra_ro_binds) if extra_ro_binds is not None else base.extra_ro_binds
        ),
        allowed_run_roots=(
            tuple(allowed_run_roots)
            if allowed_run_roots is not None
            else base.allowed_run_roots
        ),
        trusted_group_parents=(
            tuple(trusted_group_parents)
            if trusted_group_parents is not None
            else base.trusted_group_parents
        ),
    ).validate(allow_change_me=False)

    parent = os.path.dirname(target_path) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".site_{uuid.uuid4().hex}.json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(cfg.to_dict(), fh, indent=2)
            fh.write("\n")
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return target_path, cfg
