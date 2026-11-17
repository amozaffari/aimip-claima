"""Host-side `aimip submit` gate and run-directory user/permission checks.

Implements the exact same path, UID, ancestry, private-copy, and `job.sbatch`
validation rules enforced by `container/aimip-wrapper.in`, so native installs and
direct `aimip submit <run-dir>` calls have identical submission guarantees.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import fcntl
import json
import os
import pwd
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
from typing import Any

from .errors import ConfigError, ExecutionError, ValidationError
from .site_config import load_site_config

_WRAPPER_SAFE_PATH_RE = re.compile(r"\A/[A-Za-z0-9._/+-]*\Z")
_SBATCH_VAL_RE = re.compile(r"\A[A-Za-z0-9._/:%+-]+\Z")

_SCRUBBED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AIMIP_GEMINI_KEY_FILE",
        "AIMIP_GEMINI_VERTEX_TOKEN_FILE",
        "AIMIP_GEMINI_VERTEX_PROJECT",
        "AIMIP_GEMINI_VERTEX_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
)
_SCRUBBED_ENV_PREFIXES: tuple[str, ...] = (
    "LANGCHAIN_",
    "LANGSMITH_",
    "APPTAINERENV_GEMINI",
    "APPTAINERENV_GOOGLE",
    "SINGULARITYENV_GEMINI",
    "SINGULARITYENV_GOOGLE",
)


def sanitized_subprocess_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of *base_env* (or `os.environ`) with chat/LLM credentials and tracing keys removed."""
    source = os.environ if base_env is None else base_env
    return {
        k: v
        for k, v in source.items()
        if k not in _SCRUBBED_ENV_KEYS and not k.startswith(_SCRUBBED_ENV_PREFIXES)
    }


ALLOWED_SBATCH_KEYS: frozenset[str] = frozenset(
    {
        "job-name",
        "partition",
        # MN5 (BSC MareNostrum 5): a job is placed by its QOS (acc_debug, acc_ehpc, ...), not by a partition,
        # so `job.sbatch` may carry `#SBATCH --qos=` (site.json `sbatch.qos`, or `--qos` on the command line).
        "qos",
        "account",
        "gres",
        "cpus-per-task",
        "mem",
        "time",
        "output",
        "error",
    }
)

TRUSTED_RUN_FILES: tuple[str, ...] = (
    "job.sbatch",
    "config.yaml",
    "emu-job.json",
    "ic_era5.pt",
    "aimip-manifest.json",
    "spin-manifest.json",
    "plan.json",
    "submit-intent.json",
    "cancel-requested.txt",
    "experiment.yaml",
    ".submit.lock",
    "slurm.jobid",
    "sacct.txt",
    "local.pid",
    "local-run.log",
    "ic_members.nc",
    "ic_members_atmosphere.nc",
    "ic_members_ocean.nc",
)


def check_safe_path(path: str) -> str:
    """Validate *path* against the wrapper's strict absolute path allowlist `[A-Za-z0-9._/+-]`."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValidationError(f"path must be absolute: {path}")
    if ".." in path or not _WRAPPER_SAFE_PATH_RE.match(path):
        raise ValidationError(f"refusing path: {path}")
    return path


def _uid_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


# MN5 (BSC MareNostrum 5) exception to the ancestry rule below.
#
# The rule: no directory above a run directory may be writable by other users unless it has the sticky bit,
# because such a user could rename the run directory away and put their own in its place between our checks
# and the job reading it.
#
# MN5's file systems are laid out differently from the clusters this rule was written for: every user directory
# sits in a root-owned directory of its Unix group that the whole group can write, without a sticky bit
# (`drwxrws--- root bsc32 /gpfs/scratch/bsc32`, the same for `/gpfs/projects/<group>` and the EuroHPC project
# groups such as `/gpfs/scratch/ehpc819`). The user's own directory below it is theirs alone
# (`drwxr-sr-x bsc032013 bsc32 /gpfs/scratch/bsc32/bsc032013`). There is no location with room for model output
# that passes the rule (the home directory does, but its quota is 80 GB), so on MN5 aimip could not run at all.
#
# The exception is opt-in and narrow: `site.json` lists the group directories to accept in
# `trusted_group_parents`. A listed directory passes only if, at the time of the check, it is owned by root and is
# writable by its group but not by everyone. Every other directory in the path is checked as before. Accepting it
# means trusting the members of that Unix group (colleagues of the same department or project) not to swap
# directories under you; that is the trust MN5 already asks of its users.
def trusted_group_parents() -> frozenset[str]:
    """Group directories `site.json` (`trusted_group_parents`) accepts above a run directory (MN5); empty if none."""
    try:
        site = load_site_config()
    except Exception:
        # No readable site configuration: keep the strict rule.
        return frozenset()
    return frozenset(os.path.normpath(p) for p in site.trusted_group_parents)


def is_trusted_group_parent(path: str, st: os.stat_result, trusted: frozenset[str]) -> bool:
    """Whether *path* (already lstat'ed as *st*) is a listed MN5-style group directory: root-owned, group- but not world-writable."""
    return (
        os.path.normpath(path) in trusted
        and stat.S_ISDIR(st.st_mode)
        and st.st_uid == 0
        and not (st.st_mode & stat.S_IWOTH)
    )


def check_ancestry(path: str, *, caller_uid: int | None = None) -> None:
    """Verify no parent directory above *path* allows unprivileged rename/replacement.

    Directories listed in `site.json` `trusted_group_parents` are exempt when they match the MN5 layout; see
    `trusted_group_parents` above.
    """
    euid = os.geteuid() if caller_uid is None else caller_uid
    trusted = trusted_group_parents()
    cur = path
    while True:
        parent = os.path.dirname(cur)
        if not parent or parent == cur or parent == "/":
            break
        cur = parent
        try:
            ast = os.lstat(cur)
        except OSError as exc:
            raise ValidationError(f"cannot inspect parent directory {cur}: {exc}") from exc

        if stat.S_ISLNK(ast.st_mode):
            # Root-owned system links (e.g. /var -> /private/var) and caller-owned links are allowed;
            # physical target ancestry is checked separately after realpath resolution.
            if ast.st_uid not in (0, euid):
                uname = _uid_name(ast.st_uid)
                raise ValidationError(
                    f"a symlink owned by {uname} is in the run directory path: {cur}"
                )
            continue

        if (ast.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) and not (
            ast.st_mode & stat.S_ISVTX
        ):
            if is_trusted_group_parent(cur, ast, trusted):
                continue  # MN5 group directory listed in site.json (see trusted_group_parents)
            mode_str = stat.filemode(ast.st_mode)
            raise ValidationError(
                f"a parent directory is writable by others without sticky bit ({mode_str}): {cur}"
            )


def check_trusted_run_dir(run_dir: str) -> str:
    """Verify *run_dir* and its control files are caller-owned, non-symlinked, and non-group/world-writable."""
    check_safe_path(run_dir)
    d = run_dir.rstrip("/") or "/"
    euid = os.geteuid()

    try:
        st = os.lstat(d)
    except FileNotFoundError as exc:
        raise ValidationError(f"not a directory: {d}") from exc
    except OSError as exc:
        raise ValidationError(f"cannot inspect {d}: {exc}") from exc

    if stat.S_ISLNK(st.st_mode):
        raise ValidationError(f"refusing a symlinked run directory: {d}")
    if not stat.S_ISDIR(st.st_mode):
        raise ValidationError(f"not a directory: {d}")
    if st.st_uid != euid:
        raise ValidationError(f"run directory is not owned by you: {d}")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        mode_str = stat.filemode(st.st_mode)
        raise ValidationError(
            f"run directory is group- or world-writable ({mode_str}): {d} — fix with: chmod go-w {d}"
        )

    for fname in TRUSTED_RUN_FILES:
        fpath = f"{d}/{fname}"
        try:
            fst = os.lstat(fpath)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValidationError(f"cannot inspect {fpath}: {exc}") from exc

        if stat.S_ISLNK(fst.st_mode):
            raise ValidationError(f"refusing symlinked {fpath}")
        if fst.st_uid != euid:
            raise ValidationError(f"{fpath} is not owned by you")
        if fst.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            mode_str = stat.filemode(fst.st_mode)
            raise ValidationError(
                f"{fpath} is group- or world-writable ({mode_str}) — fix with: chmod go-w {fpath}"
            )

    check_ancestry(d, caller_uid=euid)
    try:
        phys = os.path.realpath(d)
    except OSError as exc:
        raise ValidationError(f"cannot resolve {d}: {exc}") from exc
    if not os.path.isdir(phys):
        raise ValidationError(f"cannot resolve {d}")
    if phys != d:
        check_ancestry(phys, caller_uid=euid)

    return d


def is_trusted_run_dir(run_dir: str) -> bool:
    """Return True when *run_dir* passes `check_trusted_run_dir` without raising."""
    try:
        check_trusted_run_dir(run_dir)
        return True
    except (ValidationError, ConfigError):
        return False


def write_in_trusted_dir(trusted_dir: str, name: str, content: str) -> str:
    """Atomically write *content* (with trailing newline) to `<trusted_dir>/<name>`, never following symlinks."""
    target = os.path.join(trusted_dir, name)
    if os.path.islink(target):
        raise ValidationError(f"refusing symlinked {target}")
    fd, tmp_path = tempfile.mkstemp(prefix=f".{name}.", dir=trusted_dir)
    try:
        payload = content if content.endswith("\n") else f"{content}\n"
        os.write(fd, payload.encode("utf-8"))
        os.fchmod(fd, 0o644)
        os.close(fd)
        fd = -1
        os.replace(tmp_path, target)
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    return target


def check_job_script(script_path: str, run_dir: str, prefix: str) -> None:
    """Validate *script_path* against the exact `aimip-generated v1` job script grammar."""
    d = run_dir.rstrip("/") or "/"
    norm_prefix = check_safe_path(prefix).rstrip("/")
    want_exec = f'"{norm_prefix}/bin/aimip" exec-fme "'

    try:
        with open(script_path, "rb") as fh:
            raw_bytes = fh.read()
    except OSError as exc:
        raise ConfigError(f"cannot read {script_path}: {exc}") from exc

    if b"\x00" in raw_bytes:
        raise ValidationError(f"{script_path} contains NUL bytes")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{script_path} is not valid UTF-8") from exc

    # Match bash `while IFS= read -r line || [ -n "$line" ]` strictly on '\n' boundaries
    # so CR/FF/Unicode line breaks are never silently normalized away.
    if not text:
        lines: list[str] = []
    elif text.endswith("\n"):
        lines = text[:-1].split("\n")
    else:
        lines = text.split("\n")

    execs = 0
    for idx, line in enumerate(lines, start=1):
        if idx == 1:
            if line != "#!/bin/bash":
                raise ValidationError("line 1 must be #!/bin/bash")
            continue
        if idx == 2:
            if line != "# aimip-generated v1":
                raise ValidationError(f"{script_path} is not an aimip-generated script")
            continue

        if line in ("", "set -euo pipefail", "#SBATCH --hint=nomultithread"):
            continue

        if line.startswith("#SBATCH --"):
            rest = line[len("#SBATCH --") :]
            if "=" not in rest:
                raise ValidationError(f"unexpected #SBATCH line: {line}")
            key, val = rest.split("=", 1)
            if key not in ALLOWED_SBATCH_KEYS:
                raise ValidationError(f"unexpected #SBATCH line: {line}")
            if not val or ".." in val or not _SBATCH_VAL_RE.match(val):
                raise ValidationError(f"unexpected #SBATCH value: {line}")
            if key in ("output", "error"):
                if not val.startswith(f"{d}/"):
                    raise ValidationError(f"slurm log outside the run directory: {line}")
                log_rel = val[len(d) + 1 :]
                if not log_rel or log_rel == "." or "/" in log_rel:
                    raise ValidationError(
                        f"slurm log must be a file directly in the run directory: {line}"
                    )
                log_dst = f"{d}/{log_rel}"
                if os.path.islink(log_dst):
                    raise ValidationError(f"refusing symlinked slurm log: {log_dst}")
                if os.path.isdir(log_dst):
                    raise ValidationError(f"slurm log destination is a directory: {log_dst}")
                if "%" in log_rel:
                    parts = log_rel.split("%j")
                    if len(parts) != 2 or "%" in parts[0] or "%" in parts[1]:
                        raise ValidationError(f"unsupported slurm log substitution: {line}")
                    pfx, sfx = parts
                    try:
                        entries = os.listdir(d)
                    except OSError as exc:
                        raise ValidationError(f"cannot list run directory {d}: {exc}") from exc
                    for entry in entries:
                        if (
                            len(entry) > len(pfx) + len(sfx)
                            and entry.startswith(pfx)
                            and (not sfx or entry.endswith(sfx))
                        ):
                            mid = entry[len(pfx) : len(entry) - len(sfx) if sfx else len(entry)]
                            if mid.isascii() and mid.isdigit():
                                cand = f"{d}/{entry}"
                                if os.path.islink(cand):
                                    raise ValidationError(f"refusing symlinked slurm log: {cand}")
                                if os.path.isdir(cand):
                                    raise ValidationError(
                                        f"slurm log destination is a directory: {cand}"
                                    )
            continue

        if line.startswith(want_exec):
            rest = line[len(want_exec) :]
            if not (rest.startswith("/") and rest.endswith('"')):
                raise ValidationError(f"unexpected exec line: {line}")
            target = rest[:-1]
            check_safe_path(target)
            if target != d and not target.startswith(f"{d}/"):
                raise ValidationError(f"exec target outside the run directory: {target}")
            execs += 1
            continue

        raise ValidationError(f"{script_path} has an unexpected line: {line}")

    if execs < 1:
        raise ValidationError(f"{script_path} has no exec-fme line")


def _resolve_submit_prefix(playground_dir: str | None = None) -> str:
    raw = (
        playground_dir
        or os.environ.get("AIMIP_HOME")
        or os.environ.get("AIMIP_PLAYGROUND")
    )
    if not raw:
        raise ConfigError(
            "Cannot determine install prefix for `aimip submit`: set $AIMIP_HOME "
            "(or invoke `<prefix>/bin/aimip submit <run-dir>`)."
        )
    return check_safe_path(raw).rstrip("/")


def _extract_sbatch_job_name(script_bytes: bytes, run_dir: str) -> str:
    """Extract `#SBATCH --job-name=...` from a validated `job.sbatch` (or fall back to `os.path.basename(run_dir)`)."""
    for line in script_bytes.decode("utf-8", errors="ignore").splitlines():
        if line.startswith("#SBATCH --job-name="):
            return line.split("=", 1)[1].strip()
    return os.path.basename(run_dir.rstrip("/")) or "aimip"


def _reconcile_squeue_job(
    job_name: str,
    user: str,
    attempt_nonce: str | None,
) -> tuple[str | None, str]:
    """Query `squeue` for a live job matching `(user, job_name, comment=attempt_nonce)`.

    A same-name job without the exact per-attempt nonce is NEVER adopted.
    Returns `(job_id, status_note)` where `status_note` is `'found'`, `'absent'`, `'ambiguous'`, or `'unavailable'`.
    """
    if not attempt_nonce or not attempt_nonce.startswith("aimip-attempt-"):
        return None, "absent"
    squeue_bin = shutil.which("squeue")
    if not squeue_bin:
        return None, "unavailable"
    try:
        proc = subprocess.run(
            [squeue_bin, "--noheader", "-u", user, "-n", job_name, "-o", "%i|%j|%T|%k"],
            capture_output=True,
            text=True,
            env=sanitized_subprocess_env(),
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "unavailable"
    if proc.returncode != 0:
        return None, "unavailable"

    matches: list[str] = []
    for raw_line in proc.stdout.splitlines():
        parts = [p.strip() for p in raw_line.strip().split("|", 3)]
        if (
            len(parts) >= 4
            and parts[0].isdigit()
            and parts[1] == job_name
            and parts[3] == attempt_nonce
        ):
            matches.append(parts[0])
    if len(matches) == 1:
        return matches[0], "found"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "absent"


def _acquire_run_submit_lock(run_dir: str) -> int:
    """Acquire an exclusive `flock` on caller-owned `<run_dir>/.submit.lock` without following symlinks."""
    lock_path = f"{run_dir}/.submit.lock"
    if os.path.islink(lock_path):
        raise ValidationError(f"refusing symlinked {lock_path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ConfigError(f"cannot open submit lock {lock_path}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValidationError(f"submit lock is not a regular file: {lock_path}")
        if st.st_uid != os.geteuid():
            raise ValidationError(f"submit lock {lock_path} is not owned by the caller")
        if st.st_mode & 0o077:
            os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except BaseException:
        os.close(fd)
        raise


def is_pid_zombie(pid: int) -> bool:
    """Return True when *pid* is an exited zombie process awaiting `wait`."""
    if pid <= 1:
        return False
    stat_path = f"/proc/{pid}/stat"
    if os.path.isfile(stat_path):
        try:
            with open(stat_path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            after = raw.rsplit(")", 1)[1].split()
            return bool(after and after[0] == "Z")
        except (OSError, IndexError):
            return False
    try:
        proc = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            env=sanitized_subprocess_env(),
            timeout=2,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().startswith("Z")
    except (OSError, subprocess.SubprocessError):
        pass
    return False


def read_pid_start_id(pid: int) -> str | None:
    """Return a kernel start-time fingerprint for *pid* (`/proc/<pid>/stat` field 22, fallback `ps -o lstart=`)."""
    if pid <= 1:
        return None
    stat_path = f"/proc/{pid}/stat"
    if os.path.isfile(stat_path):
        try:
            with open(stat_path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            after = raw.rsplit(")", 1)[1].split()
            if len(after) > 19:
                return f"proc:{after[19]}"
        except (OSError, IndexError):
            pass
    try:
        proc = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            env=sanitized_subprocess_env(),
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return f"ps:{' '.join(proc.stdout.split())}"
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def inspect_local_process(run_dir: str, local_pid: int) -> tuple[bool, int | None, str | None]:
    """Verify whether *local_pid* for *run_dir* is still alive and matches its recorded start identity and session group.

    Returns `(alive, verified_pgid, reason)` where `reason` is:
    - `None`: process is alive and both `local_start_id` and `local_pgid` are present and match the live process
    - `"not_found"`: process has exited or is a zombie
    - `"reused_pid"`: recorded start identity (or UID) disagrees with the live process
    - `"unverified_identity"`: process is alive on this UID, but recorded or live start identity / session group cannot be verified
    - `"invalid_pid"`: `local_pid <= 1`
    """
    if local_pid <= 1:
        return False, None, "invalid_pid"

    recorded_pgid: int | None = None
    recorded_start_id: str | None = None
    intent_path = os.path.join(run_dir, "submit-intent.json")
    if os.path.isfile(intent_path) and not os.path.islink(intent_path):
        try:
            with open(intent_path, encoding="utf-8") as fh:
                intent = json.load(fh)
            if isinstance(intent, dict) and intent.get("local_pid") == local_pid:
                raw_pgid = intent.get("local_pgid")
                if isinstance(raw_pgid, int) and not isinstance(raw_pgid, bool) and raw_pgid > 1:
                    recorded_pgid = raw_pgid
                raw_start = intent.get("local_start_id")
                if isinstance(raw_start, str) and raw_start.strip():
                    recorded_start_id = raw_start.strip()
        except (OSError, json.JSONDecodeError):
            pass

    try:
        os.kill(local_pid, 0)
    except ProcessLookupError:
        return False, recorded_pgid, "not_found"
    except PermissionError:
        return False, None, "reused_pid"

    if is_pid_zombie(local_pid):
        return False, recorded_pgid, "not_found"

    live_start = read_pid_start_id(local_pid)
    if recorded_start_id is not None and live_start is not None and live_start != recorded_start_id:
        return False, None, "reused_pid"
    if not recorded_start_id or not live_start:
        return True, None, "unverified_identity"

    if recorded_pgid is None or recorded_pgid <= 1:
        return True, None, "unverified_identity"
    try:
        live_pgid = int(os.getpgid(local_pid))
    except ProcessLookupError:
        return False, recorded_pgid, "not_found"
    except OSError:
        return True, None, "unverified_identity"
    if live_pgid <= 1 or live_pgid != recorded_pgid:
        return True, None, "unverified_identity"

    return True, recorded_pgid, None


def submit_run(
    run_dir: str,
    *,
    playground_dir: str | None = None,
    site_config_path: str | None = None,
    scheduler_override: str | None = None,
    sbatch_timeout: float = 60.0,
    verify_private_script_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Validate `<run_dir>/job.sbatch` in a caller-private 0700 copy and submit via `sbatch` or detached `bash`.

    Serializes concurrent submissions per run directory via `flock`, records submission intent
    (including a unique `attempt_nonce`) to `<run_dir>/submit-intent.json` BEFORE invoking `sbatch`,
    and reconciles uncertain replies strictly by `attempt_nonce`.
    """
    import datetime
    import hashlib
    import json

    if not run_dir:
        raise ConfigError("usage: aimip submit <run-dir>")
    check_safe_path(run_dir)
    d = check_trusted_run_dir(run_dir)

    sbatch_src = f"{d}/job.sbatch"
    if os.path.islink(sbatch_src) or not os.path.isfile(sbatch_src):
        raise ConfigError(
            f"{d} has no job.sbatch (prepare it with: aimip run ... --out {d})"
        )

    lock_fd = _acquire_run_submit_lock(d)
    try:
        if os.path.lexists(f"{d}/slurm.jobid") or os.path.lexists(f"{d}/local.pid"):
            raise ConfigError("already submitted")

        prefix = _resolve_submit_prefix(playground_dir)
        site_cfg = load_site_config(site_config_path)
        scheduler = scheduler_override if scheduler_override in ("slurm", "none") else site_cfg.scheduler
        caller_user = (
            os.environ.get("AIMIP_USER")
            or os.environ.get("USER")
            or _uid_name(os.geteuid())
        )

        intent_file = f"{d}/submit-intent.json"
        if scheduler == "slurm" and os.path.isfile(intent_file) and not os.path.islink(intent_file):
            try:
                with open(intent_file, encoding="utf-8") as ifh:
                    prev_intent = json.load(ifh)
            except (OSError, json.JSONDecodeError):
                prev_intent = {}
            if prev_intent.get("state") in ("submitting", "uncertain"):
                jname = str(prev_intent.get("job_name") or os.path.basename(d))
                prev_nonce = (
                    str(prev_intent["attempt_nonce"])
                    if prev_intent.get("attempt_nonce")
                    else None
                )
                rec_id, rec_note = _reconcile_squeue_job(jname, caller_user, prev_nonce)
                if rec_note == "found" and rec_id:
                    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    write_in_trusted_dir(d, "slurm.jobid", rec_id)
                    write_in_trusted_dir(
                        d,
                        "submitted.txt",
                        f"job={rec_id} submitted={now_iso} user={caller_user} reconciled=true",
                    )
                    prev_intent["state"] = "reconciled"
                    prev_intent["job_id"] = rec_id
                    write_in_trusted_dir(d, "submit-intent.json", json.dumps(prev_intent, indent=2))
                    return {
                        "out_dir": d,
                        "scheduler": "slurm",
                        "slurm_job_id": rec_id,
                        "local_pid": None,
                        "log_path": None,
                        "reconciled": True,
                    }
                raise ExecutionError(
                    f"Previous submission for {d} had an uncertain scheduler reply (squeue status: {rec_note}); "
                    "refusing blind retry. Inspect `squeue` / `sacct` or prepare a fresh run directory."
                )

        try:
            priv_dir = tempfile.mkdtemp(prefix="aimip-submit.")
            os.chmod(priv_dir, 0o700)
        except OSError as exc:
            raise ConfigError(f"cannot create a private directory: {exc}") from exc

        keep_priv_dir = False
        try:
            priv_script = os.path.join(priv_dir, "job.sbatch")
            read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                src_fd = os.open(sbatch_src, read_flags)
                try:
                    with os.fdopen(src_fd, "rb") as rfh:
                        script_bytes = rfh.read()
                except BaseException:
                    try:
                        os.close(src_fd)
                    except OSError:
                        pass
                    raise
                dst_fd = os.open(
                    priv_script,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o400,
                )
                try:
                    view = memoryview(script_bytes)
                    while view:
                        n = os.write(dst_fd, view)
                        if n <= 0:
                            raise OSError("short write copying job.sbatch")
                        view = view[n:]
                finally:
                    os.close(dst_fd)
            except OSError as exc:
                raise ConfigError(f"cannot read {sbatch_src}: {exc}") from exc

            check_job_script(priv_script, d, prefix)
            if verify_private_script_fn is not None:
                verify_private_script_fn(priv_script)
            clean_env = sanitized_subprocess_env()

            if scheduler == "slurm":
                job_name = _extract_sbatch_job_name(script_bytes, d)
                attempt_nonce = f"aimip-attempt-{secrets.token_hex(12)}"
                started_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                intent_doc: dict[str, Any] = {
                    "state": "submitting",
                    "run_dir": d,
                    "job_name": job_name,
                    "user": caller_user,
                    "attempt_nonce": attempt_nonce,
                    "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
                    "started_at": started_iso,
                }
                write_in_trusted_dir(d, "submit-intent.json", json.dumps(intent_doc, indent=2))

                uncertain_reason: str | None = None
                proc: subprocess.CompletedProcess[str] | None = None
                try:
                    proc = subprocess.run(
                        ["sbatch", "--parsable", f"--comment={attempt_nonce}", priv_script],
                        capture_output=True,
                        text=True,
                        env=clean_env,
                        timeout=sbatch_timeout,
                        check=False,
                    )
                except FileNotFoundError as exc:
                    intent_doc["state"] = "failed_before_submit"
                    write_in_trusted_dir(d, "submit-intent.json", json.dumps(intent_doc, indent=2))
                    raise ConfigError("sbatch not found on PATH") from exc
                except subprocess.TimeoutExpired:
                    uncertain_reason = f"sbatch timed out after {sbatch_timeout:g}s"
                except OSError as exc:
                    intent_doc["state"] = "failed_before_submit"
                    write_in_trusted_dir(d, "submit-intent.json", json.dumps(intent_doc, indent=2))
                    raise ConfigError(f"sbatch failed: {exc}") from exc

                if proc is not None and uncertain_reason is None:
                    detail = (proc.stderr or proc.stdout or "").strip()
                    low_detail = detail.lower()
                    if proc.returncode != 0:
                        if any(
                            tok in low_detail
                            for tok in ("socket timed out", "slurm_receive_msg", "unable to contact", "transport endpoint")
                        ):
                            uncertain_reason = f"sbatch returned uncertain transport error ({detail})"
                        else:
                            intent_doc["state"] = "failed_before_submit"
                            intent_doc["error"] = detail
                            write_in_trusted_dir(d, "submit-intent.json", json.dumps(intent_doc, indent=2))
                            raise ConfigError(f"sbatch failed: {detail}" if detail else "sbatch failed")

                if proc is not None and uncertain_reason is None:
                    first_line = (
                        proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
                    )
                    job_id = first_line.split(";", 1)[0].strip()
                    if not job_id or not job_id.isdigit():
                        uncertain_reason = f"sbatch exited 0 with non-numeric/empty job ID ({first_line!r})"
                    else:
                        write_in_trusted_dir(d, "slurm.jobid", job_id)
                        write_in_trusted_dir(
                            d,
                            "submitted.txt",
                            f"job={job_id} submitted={started_iso} user={caller_user}",
                        )
                        intent_doc["state"] = "submitted"
                        intent_doc["job_id"] = job_id
                        write_in_trusted_dir(d, "submit-intent.json", json.dumps(intent_doc, indent=2))
                        return {
                            "out_dir": d,
                            "scheduler": "slurm",
                            "slurm_job_id": job_id,
                            "local_pid": None,
                            "log_path": None,
                            "reconciled": False,
                        }

                # Uncertain sbatch outcome: record `"uncertain"` and reconcile once by `attempt_nonce`.
                intent_doc["state"] = "uncertain"
                intent_doc["reason"] = uncertain_reason
                write_in_trusted_dir(d, "submit-intent.json", json.dumps(intent_doc, indent=2))
                rec_id, rec_note = _reconcile_squeue_job(job_name, caller_user, attempt_nonce)
                if rec_note == "found" and rec_id:
                    write_in_trusted_dir(d, "slurm.jobid", rec_id)
                    write_in_trusted_dir(
                        d,
                        "submitted.txt",
                        f"job={rec_id} submitted={started_iso} user={caller_user} reconciled=true",
                    )
                    intent_doc["state"] = "reconciled"
                    intent_doc["job_id"] = rec_id
                    write_in_trusted_dir(d, "submit-intent.json", json.dumps(intent_doc, indent=2))
                    return {
                        "out_dir": d,
                        "scheduler": "slurm",
                        "slurm_job_id": rec_id,
                        "local_pid": None,
                        "log_path": None,
                        "reconciled": True,
                    }
                raise ExecutionError(
                    f"Uncertain scheduler reply ({uncertain_reason}; squeue reconciliation: {rec_note}); "
                    "refusing blind retry."
                )

            # scheduler == "none": run the validated private copy in a detached background bash session.
            log_path = f"{d}/local-run.log"
            if os.path.islink(log_path):
                raise ValidationError(f"refusing symlinked {log_path}")
            log_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_TRUNC
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                log_fd = os.open(log_path, log_flags, 0o644)
            except OSError as exc:
                raise ConfigError(f"cannot open {log_path}: {exc}") from exc

            write_in_trusted_dir(
                d,
                "submit-intent.json",
                json.dumps({"state": "submitting_local", "run_dir": d, "user": caller_user}, indent=2),
            )
            with os.fdopen(log_fd, "wb") as log_fh:
                try:
                    bg_proc = subprocess.Popen(
                        ["bash", priv_script],
                        stdin=subprocess.DEVNULL,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        env=clean_env,
                        start_new_session=True,
                        close_fds=True,
                    )
                except OSError as exc:
                    raise ConfigError(f"failed to launch background bash job: {exc}") from exc

            local_pid_int = int(bg_proc.pid)
            try:
                local_pgid = int(os.getpgid(local_pid_int))
            except OSError:
                local_pgid = 0
            local_start_id = read_pid_start_id(local_pid_int)
            caller_pgid = os.getpgrp()
            if bg_proc.poll() is None and (
                not local_start_id or local_pgid <= 1 or local_pgid == caller_pgid
            ):
                if local_pgid > 1 and local_pgid != caller_pgid:
                    try:
                        os.killpg(local_pgid, signal.SIGTERM)
                    except OSError:
                        pass
                else:
                    try:
                        bg_proc.terminate()
                    except OSError:
                        pass
                raise ExecutionError(
                    f"Failed to verify dedicated session group or start identity for local PID {local_pid_int}; launch aborted."
                )
            if local_pgid <= 1:
                local_pgid = local_pid_int
            keep_priv_dir = True
            write_in_trusted_dir(d, "local.pid", str(local_pid_int))
            write_in_trusted_dir(
                d,
                "submit-intent.json",
                json.dumps(
                    {
                        "state": "submitted_local",
                        "run_dir": d,
                        "user": caller_user,
                        "local_pid": local_pid_int,
                        "local_pgid": local_pgid,
                        "local_start_id": local_start_id,
                    },
                    indent=2,
                ),
            )
            return {
                "out_dir": d,
                "scheduler": "none",
                "slurm_job_id": None,
                "local_pid": local_pid_int,
                "log_path": log_path,
                "reconciled": False,
            }
        finally:
            if not keep_priv_dir:
                shutil.rmtree(priv_dir, ignore_errors=True)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)
