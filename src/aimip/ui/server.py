"""Stdlib `ThreadingHTTPServer` and `/api/v1/` JSON router for `aimip ui`.

Security invariants:
- Binds to `127.0.0.1` by default; refuses any non-loopback address unless `--allow-remote`.
- Generates a fresh 256-bit `secrets.token_urlsafe(32)` token per server session.
- Never sets or accepts HTTP cookies (which are shared across ports on `127.0.0.1`).
  `GET /?token=<token>` validates the startup token and serves `index.html` (`Cache-Control: no-store`),
  which stores the token in origin-isolated `window.sessionStorage`, strips `?token=...` via
  `history.replaceState`, and sends `X-AIMIP-Token` on every `/api/v1/*` request.
- Enforces strict `Host`, `Origin`, and `Referer` checks on every request to defeat
  DNS-rebinding and cross-origin CSRF attacks.
- Bounds concurrent HTTP handler threads (`BoundedSemaphore`), enforces socket header/body read
  deadlines, caps JSON request bodies, and bounds concurrent NetCDF summary/plot workers.
- Serves all responses with `Content-Security-Policy: default-src 'self'`, `X-Frame-Options: DENY`,
  and `X-Content-Type-Options: nosniff`.
- Calls `aimip.services` directly in-process — never spawns an `aimip` subprocess.
- Uses the exact same single-use 10-minute `Authorization` rules (`MemoryApprovalBroker`)
  for `prepare`, `submit` (with separate download consent), and `cancel`.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import stat
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .. import __version__
from ..agent.security import trusted_lora_path
from ..errors import (
    AimipError,
    ApprovalError,
    ConfigError,
    ExecutionError,
    MissingAssetsError,
    ValidationError,
)
from ..ic import list_model_ics, list_model_ocean_ics
from ..registry import get_model, get_model_ic_provenance, resolve_data_dir
from ..services import (
    APPROVAL_TTL_SECONDS,
    TERMINAL_RUN_STATES,
    Authorization,
    MemoryApprovalBroker,
    ResolvedPlan,
    RunSpec,
    can_download_into_data_dir,
    cancel as svc_cancel,
    cancel_digest_for,
    cite as svc_cite,
    current_user,
    doctor as svc_doctor,
    format_bytes,
    logs as svc_logs,
    ls as svc_ls,
    models as svc_models,
    plan as svc_plan,
    prepare as svc_prepare,
    render_plan_card,
    resolve_configured_run_root,
    resolve_run_target,
    show as svc_show,
    status as svc_status,
    submit as svc_submit,
    summary as svc_summary,
    validate_run_name,
)
from ..services.metadata import TIER_DEFINITIONS, card_for_model
from ..site_config import load_site_config
from ..status import get_run_status
from ..submit import check_trusted_run_dir
from ..validators import (
    get_allowed_run_roots,
    get_visible_input_roots,
    validate_model_name,
    validate_visible_input_path,
)

CSP_HEADER_VALUE = "default-src 'self'"
MAX_JSON_BODY_BYTES = 256 * 1024
MAX_UI_LOG_BYTES = 65536
DEFAULT_UI_LOG_BYTES = 16384
MAX_PICKER_FILES = 200
DEFAULT_MAX_UI_CONNECTIONS = 16
DEFAULT_UI_REQUEST_TIMEOUT = 30.0
DEFAULT_MAX_DIAG_WORKERS = 2
MAX_PLOT_POINTS = 2400
MAX_PLOT_MEMBER_TRACES = 16
UI_SACCT_POLL_INTERVAL_SECONDS = 15.0

_HEX_PREFIX_RE = re.compile(r"\A[0-9a-f]{8,64}\Z")
_FULL_DIGEST_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_INT_STR_RE = re.compile(r"\A[+-]?[0-9]+\Z")

_EXCLUDED_IC_FILENAMES: frozenset[str] = frozenset(
    {
        "monthly_mean_predictions.nc",
        "autoregressive_predictions.nc",
        "mean_diagnostics.nc",
    }
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_ALLOWED_STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


def _is_loopback_host(host: str) -> bool:
    cleaned = (host or "").strip()
    if not cleaned:
        return False
    if cleaned.lower() == "localhost":
        return True
    cand = cleaned[1:-1] if (cleaned.startswith("[") and cleaned.endswith("]")) else cleaned
    try:
        return bool(ipaddress.ip_address(cand).is_loopback)
    except ValueError:
        return False


def validate_bind_host(host: str, *, allow_remote: bool = False) -> str:
    """Validate `--host`; refuse any non-loopback address unless `allow_remote=True`."""
    cleaned = (host or "").strip()
    if not cleaned:
        raise ValidationError("Host must be a non-empty address (default: 127.0.0.1).", field="host")
    if not allow_remote and not _is_loopback_host(cleaned):
        raise ConfigError(
            f"Refusing non-loopback bind host {cleaned!r}: `aimip ui` binds only to loopback "
            "(127.0.0.1 / ::1) by default. Use an SSH tunnel on clusters "
            "(`ssh -N -L <port>:127.0.0.1:<port> <login-node>`), or pass `--allow-remote` explicitly.",
            field="host",
        )
    return cleaned


def _format_host_for_url(host: str) -> str:
    cleaned = host.strip()
    if ":" in cleaned and not cleaned.startswith("["):
        return f"[{cleaned}]"
    return cleaned


def _build_allowed_netlocs(bind_host: str, bound_port: int, *, allow_remote: bool) -> set[str]:
    """Build exact `host:port` allowlist to defeat DNS-rebinding attacks."""
    allowed = {
        f"127.0.0.1:{bound_port}",
        f"localhost:{bound_port}",
        f"[::1]:{bound_port}",
    }
    formatted = _format_host_for_url(bind_host).lower()
    if _is_loopback_host(bind_host):
        allowed.add(f"{formatted}:{bound_port}")
        return allowed

    if allow_remote and bind_host not in ("0.0.0.0", "::", "[::]"):
        allowed.add(f"{formatted}:{bound_port}")
    return allowed


def _is_matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def _is_agent_available() -> bool:
    return (
        importlib.util.find_spec("langgraph") is not None
        and importlib.util.find_spec("langchain_core") is not None
    )


def _agent_credentials_configured() -> bool:
    for env_key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AIMIP_GEMINI_KEY_FILE",
        "AIMIP_GEMINI_VERTEX_PROJECT",
    ):
        if os.environ.get(env_key, "").strip():
            return True
    return False


def _ui_runtime_capabilities() -> dict[str, Any]:
    """Return scheduler and container-submission capability metadata for honest UI messaging."""
    in_container = os.environ.get("AIMIP_IMAGE_SHA", "native") not in ("", "native")
    try:
        site_cfg = load_site_config(None)
        scheduler = str(site_cfg.scheduler or "slurm")
    except AimipError:
        scheduler = os.environ.get("AIMIP_SCHEDULER") or "slurm"
    sbatch_available = shutil.which("sbatch") is not None
    can_submit_directly = bool(
        not in_container or (scheduler == "slurm" and sbatch_available)
    )
    return {
        "in_container": in_container,
        "scheduler": scheduler,
        "can_submit_directly": can_submit_directly,
    }


def _modal_positive_step_count(series: list[dict[str, Any]]) -> int | None:
    counts = [
        int(row["counts"])
        for row in series
        if row.get("counts") is not None and int(row["counts"]) > 0
    ]
    if not counts:
        return None
    freq: dict[int, int] = {}
    for c in counts:
        freq[c] = freq.get(c, 0) + 1
    return max(freq.items(), key=lambda kv: (kv[1], kv[0]))[0]


def render_summary_plot_png(summary_dict: dict[str, Any], *, var_name: str | None = None) -> bytes | None:
    """Render monthly area-weighted global-mean series to PNG bytes via headless `FigureCanvasAgg`.

    Uses object-oriented `Figure` + `FigureCanvasAgg` (thread-safe, no `pyplot` global state,
    no GUI backend). Returns `None` if `matplotlib` is not installed.
    """
    if not _is_matplotlib_available():
        return None

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    variables = summary_dict.get("variables") or {}
    if not isinstance(variables, dict) or not variables:
        raise ValidationError("Run summary contains no variables to plot.", field="var")

    if var_name:
        if var_name not in variables:
            raise ValidationError(
                f"Variable {var_name!r} not in run summary (available: {', '.join(variables.keys())}).",
                field="var",
                allowed=list(variables.keys()),
            )
        selected_keys = [var_name]
    else:
        selected_keys = list(variables.keys())[:2]

    n_panels = len(selected_keys)
    fig = Figure(figsize=(8.0, 3.5 * n_panels), dpi=120, facecolor="#ffffff")
    FigureCanvasAgg(fig)

    raw_run_name = str(summary_dict.get("run_name") or "").strip()
    raw_dir = str(summary_dict.get("run_dir") or summary_dict.get("out_dir") or "").rstrip("/")
    run_label = raw_run_name or os.path.basename(raw_dir) or "run"
    model_label = str(summary_dict.get("model") or "").strip()
    if model_label == "unknown":
        model_label = ""

    for idx, key in enumerate(selected_keys, start=1):
        meta = variables[key]
        raw_series = meta.get("series") or []
        series = list(raw_series[:MAX_PLOT_POINTS])
        ax = fig.add_subplot(n_panels, 1, idx, facecolor="#ffffff")
        months = [str(row.get("month") or row.get("month_index")) for row in series]
        xs = list(range(len(months)))
        ys = [row.get("ensemble_mean") for row in series]

        n_members = (
            len(series[0]["members"])
            if series and isinstance(series[0].get("members"), list)
            else int(summary_dict.get("n_samples") or 1)
        )
        has_legend_items = False

        if n_members > 1 and series:
            drawn_members = min(n_members, MAX_PLOT_MEMBER_TRACES)
            member_legend = (
                "Individual members"
                if drawn_members == n_members
                else f"Individual members ({drawn_members} of {n_members})"
            )
            labeled_member = False
            for m_i in range(drawn_members):
                m_vals = [
                    (row["members"][m_i] if m_i < len(row.get("members") or []) else None)
                    for row in series
                ]
                clean_pts = [(x, float(v)) for x, v in zip(xs, m_vals) if v is not None]
                if clean_pts:
                    ax.plot(
                        [p[0] for p in clean_pts],
                        [p[1] for p in clean_pts],
                        color="#93c5fd",
                        linewidth=1.0,
                        alpha=0.7,
                        label=(member_legend if not labeled_member else None),
                    )
                    labeled_member = True
                    has_legend_items = True

        clean_mean = [(x, float(v)) for x, v in zip(xs, ys) if v is not None]
        if clean_mean:
            mean_label = (
                f"Ensemble mean ({n_members} members)"
                if n_members > 1
                else "Area-weighted global mean"
            )
            ax.plot(
                [p[0] for p in clean_mean],
                [p[1] for p in clean_mean],
                color="#1d4ed8",
                linewidth=2.0,
                marker="o",
                markersize=4.0,
                label=mean_label,
            )
            has_legend_items = True

        modal_steps = _modal_positive_step_count(series)
        if modal_steps is not None and len(series) > 1:
            partial_pts = [
                (x, float(row["ensemble_mean"]))
                for x, row in zip(xs, series)
                if row.get("ensemble_mean") is not None
                and row.get("counts") is not None
                and 0 < int(row["counts"]) < modal_steps
            ]
            if partial_pts:
                ax.plot(
                    [p[0] for p in partial_pts],
                    [p[1] for p in partial_pts],
                    linestyle="None",
                    marker="o",
                    markersize=7.0,
                    markerfacecolor="#fef08a",
                    markeredgecolor="#b45309",
                    markeredgewidth=1.4,
                    label="Partial month",
                )
                has_legend_items = True

        units = meta.get("units") or ""
        long_name = meta.get("long_name") or key
        title_prefix = f"{run_label} ({model_label}) — " if model_label else f"{run_label} — "
        ax.set_title(f"{title_prefix}{key}: {long_name}", fontsize=10, color="#0f172a", pad=8)
        ax.set_ylabel(f"{key} [{units}]" if units else key, fontsize=9, color="#334155")
        if xs:
            step = max(1, len(xs) // 8)
            tick_idx = xs[::step]
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([months[i] for i in tick_idx], rotation=25, ha="right", fontsize=8, color="#334155")
        ax.tick_params(axis="y", labelsize=8, colors="#334155")
        ax.grid(True, color="#e2e8f0", linewidth=0.8)
        if has_legend_items:
            ax.legend(loc="best", fontsize=8, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#ffffff")
    return buf.getvalue()


def _parse_strict_int(raw: Any, field_name: str) -> int:
    if isinstance(raw, bool):
        raise ValidationError(f"Field {field_name!r} must be an integer.", field=field_name)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ValidationError(
                f"Field {field_name!r} must be an integer (got non-integral value {raw!r}).",
                field=field_name,
            )
        return int(raw)
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not _INT_STR_RE.match(cleaned):
            raise ValidationError(f"Field {field_name!r} must be an integer.", field=field_name)
        return int(cleaned)
    raise ValidationError(f"Field {field_name!r} must be an integer.", field=field_name)


def _spec_from_payload(body: dict[str, Any]) -> RunSpec:
    """Validate and construct a `RunSpec` from a `/api/v1/plan` JSON request body."""
    if not isinstance(body, dict):
        raise ValidationError("Request body must be a JSON object.")

    model = str(body.get("model") or "").strip()
    start = str(body.get("start") or "").strip()
    if not model:
        raise ValidationError("Field 'model' is required.", field="model")
    if not start:
        raise ValidationError("Field 'start' is required (ISO date/time, e.g. 1979-01-01).", field="start")

    def _opt_int(field_name: str) -> int | None:
        raw = body.get(field_name)
        if raw is None or raw == "":
            return None
        return _parse_strict_int(raw, field_name)

    def _opt_str(field_name: str) -> str | None:
        raw = body.get(field_name)
        if raw is None:
            return None
        s = str(raw).strip()
        return s if s else None

    steps = _opt_int("steps")
    length = _opt_str("length")
    end = _opt_str("end")
    members = _opt_int("members")
    seed = _opt_int("seed")
    ic = _opt_str("ic")
    ocean_ic = _opt_str("ocean_ic")
    lora = _opt_str("lora")
    forcing_set = _opt_str("forcing_set")
    name = _opt_str("name")
    out = _opt_str("out")
    mode = _opt_str("mode") or "slurm"
    if mode not in ("slurm", "local"):
        raise ValidationError("Field 'mode' must be 'slurm' or 'local'.", field="mode", allowed=["slurm", "local"])

    if ic is not None:
        ic = validate_visible_input_path(ic, label="ic")
    if ocean_ic is not None:
        ocean_ic = validate_visible_input_path(ocean_ic, label="ocean_ic")
    if lora is not None:
        try:
            lora = trusted_lora_path(lora)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Untrusted LoRA checkpoint: {exc}", field="lora") from exc

    monthly = bool(body.get("monthly", True))
    prediction = bool(body.get("prediction", False))

    raw_res = body.get("resources") or {}
    if not isinstance(raw_res, dict):
        raise ValidationError("Field 'resources' must be an object.", field="resources")
    allowed_res_keys = {"partition", "qos", "account", "gres", "cpus_per_task", "mem", "time", "job_name"}  # qos: MN5
    clean_res: dict[str, Any] = {}
    for rk, rv in raw_res.items():
        if rk not in allowed_res_keys:
            raise ValidationError(
                f"Unknown resource key {rk!r}.",
                field=f"resources.{rk}",
                allowed=sorted(allowed_res_keys),
            )
        if rv is None or rv == "":
            continue
        if rk == "cpus_per_task":
            clean_res[rk] = _parse_strict_int(rv, "resources.cpus_per_task")
        else:
            clean_res[rk] = str(rv).strip()

    return RunSpec(
        model=model,
        start=start,
        steps=steps,
        length=length,
        end=end,
        members=members,
        seed=seed,
        ic=ic,
        ocean_ic=ocean_ic,
        lora=lora,
        forcing_set=forcing_set,
        name=name,
        out=out,
        monthly=monthly,
        prediction=prediction,
        resources=clean_res,
        mode=mode,
    )


def _ic_candidate_priority(fname: str) -> int:
    low = fname.lower()
    if low == "restart.nc":
        return 0
    if "restart" in low or low.startswith("ic") or "_ic" in low or "-ic" in low:
        return 1
    return 2


def _list_picker_files(kind: str, directory: str | None = None) -> dict[str, Any]:
    """List candidate IC (`.nc`) or LoRA (`.tar`/`.ckpt`/`.pt`) files strictly inside allowed visible roots."""
    if kind not in ("ic", "lora"):
        raise ValidationError("Query parameter 'kind' must be 'ic' or 'lora'.", field="kind", allowed=["ic", "lora"])

    visible_roots = list(get_visible_input_roots())
    if directory is not None and directory.strip():
        validated_dir = validate_visible_input_path(directory.strip(), label="picker directory")
        search_roots = [validated_dir]
    else:
        default_candidates: list[str] = []
        try:
            default_candidates.append(resolve_configured_run_root())
        except AimipError:
            pass
        try:
            data_cand = resolve_data_dir(None)
            if data_cand:
                default_candidates.append(data_cand)
        except AimipError:
            pass
        for env_key in ("AIMIP_PLAYGROUND", "AIMIP_HOME"):
            env_val = (os.environ.get(env_key) or "").strip()
            if env_val:
                default_candidates.append(env_val)
        search_roots = []
        for cand_r in default_candidates:
            trimmed = cand_r.rstrip("/")
            if (
                trimmed
                and trimmed.count("/") >= 2
                and trimmed not in search_roots
                and os.path.isdir(trimmed)
            ):
                search_roots.append(trimmed)

    suffixes = (".nc",) if kind == "ic" else (".tar", ".ckpt", ".pt", ".pth")
    found: list[dict[str, Any]] = []
    seen_real: set[str] = set()
    visited_dirs = 0
    max_visited_dirs = 250
    truncated = False

    for root in search_roots:
        if not os.path.isdir(root) or os.path.islink(root):
            continue
        base_depth = root.rstrip("/").count("/")
        for cur_dir, dirnames, filenames in os.walk(root, followlinks=False):
            visited_dirs += 1
            if visited_dirs > max_visited_dirs:
                truncated = True
                break
            depth = cur_dir.count("/") - base_depth
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not d.startswith(".")
                and depth < 4
                and not os.path.islink(os.path.join(cur_dir, d))
            )
            ordered_names = (
                sorted(filenames, key=lambda n: (_ic_candidate_priority(n), n))
                if kind == "ic"
                else sorted(filenames)
            )
            for fname in ordered_names:
                if fname.startswith(".") or not fname.endswith(suffixes):
                    continue
                if kind == "ic":
                    low_name = fname.lower()
                    if low_name in _EXCLUDED_IC_FILENAMES or low_name.startswith("forcing_"):
                        continue
                cand = os.path.join(cur_dir, fname)
                if os.path.islink(cand):
                    continue
                try:
                    validated = validate_visible_input_path(cand, label=f"{kind} candidate")
                    if kind == "lora":
                        validated = trusted_lora_path(validated)
                    st = os.stat(validated)
                except (AimipError, OSError, ValueError):
                    continue
                if not stat.S_ISREG(st.st_mode) or validated in seen_real:
                    continue
                seen_real.add(validated)
                found.append(
                    {
                        "path": validated,
                        "name": fname,
                        "size_bytes": int(st.st_size),
                    }
                )
                if len(found) >= MAX_PICKER_FILES:
                    truncated = True
                    break
            if len(found) >= MAX_PICKER_FILES:
                break
        if len(found) >= MAX_PICKER_FILES or visited_dirs > max_visited_dirs:
            break

    if kind == "ic":
        found.sort(key=lambda item: (_ic_candidate_priority(str(item.get("name") or "")), str(item.get("path") or "")))

    return {
        "kind": kind,
        "allowed_roots": visible_roots,
        "truncated": truncated,
        "files": found,
    }


def _list_run_directory_files(run_dir: str) -> list[dict[str, Any]]:
    """List regular non-symlink files inside a run directory (max 100 entries, depth <= 2)."""
    items: list[dict[str, Any]] = []
    if not os.path.isdir(run_dir) or os.path.islink(run_dir):
        return items
    base_len = len(run_dir.rstrip("/")) + 1
    for cur_dir, dirnames, filenames in os.walk(run_dir, followlinks=False):
        rel_dir = cur_dir[base_len:] if len(cur_dir) >= base_len else ""
        depth = 0 if not rel_dir else rel_dir.count("/") + 1
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and depth < 2
            and not os.path.islink(os.path.join(cur_dir, d))
        )
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            full = os.path.join(cur_dir, fname)
            if os.path.islink(full):
                continue
            try:
                lst = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(lst.st_mode):
                continue
            rel_name = f"{rel_dir}/{fname}" if rel_dir else fname
            items.append(
                {
                    "name": rel_name,
                    "size_bytes": int(lst.st_size),
                    "size_human": format_bytes(int(lst.st_size)),
                    "mtime": float(lst.st_mtime),
                }
            )
            if len(items) >= 100:
                return items
    return items


def _plan_decision_context(resolved: ResolvedPlan) -> dict[str, Any]:
    """Attach model tier definition, licence, limitations, IC provenance, and container submission honesty to a proposal."""
    mcard = card_for_model(resolved.model)
    in_container = os.environ.get("AIMIP_IMAGE_SHA", "native") not in ("", "native")
    eff_sched = (
        "none"
        if resolved.spec.mode == "local"
        else str(resolved.runtime_identity.get("scheduler", "slurm"))
    )
    will_prepare_only = bool(
        in_container
        and (eff_sched == "none" or (eff_sched == "slurm" and shutil.which("sbatch") is None))
    )
    return {
        "model_card": {
            "status_tier": resolved.status_tier,
            "status_definition": mcard.get(
                "status_definition",
                TIER_DEFINITIONS.get(resolved.status_tier, ""),
            ),
            "licence": mcard.get("licence", ""),
            "ic_provenance": resolved.ic_provenance or get_model_ic_provenance(resolved.model),
            "limitations": mcard.get("limitations", ""),
        },
        "will_prepare_only_in_container": will_prepare_only,
    }


class UiApprovalManager:
    """Thread-safe human-approval state for `aimip ui` backed by `MemoryApprovalBroker`.

    Enforces the exact same rules as the console broker:
    - Full plan card + 8-hex `approval_prefix` is shown to the human in the UI.
    - Clicking Approve posts `digest_prefix` (`>= 8` hex chars), optional full `digest` of
      the drawn card, and explicit `allow_download` when assets are missing.
    - Mints a single-use 10-minute `Authorization` bound to `(action, digest, user)`.
    - Replay, expiry, wrong prefix, superseded card digest, missing download consent, and
      post-approval plan drift are all refused with `ApprovalError`.
    """

    def __init__(self, *, now_fn: Any = time.time) -> None:
        self._lock = threading.RLock()
        self._now_fn = now_fn
        self.broker = MemoryApprovalBroker(now_fn=now_fn)
        self._pending_by_action: dict[str, dict[str, Any]] = {}
        self._minted_meta: dict[str, dict[str, Any]] = {}

    def register_submit_proposal(
        self,
        *,
        spec: RunSpec,
        resolved: ResolvedPlan,
        card_text: str,
        prepared_run_dir: str | None = None,
    ) -> dict[str, Any]:
        ctx = _plan_decision_context(resolved)
        with self._lock:
            for nonce, meta in list(self._minted_meta.items()):
                if meta.get("action") in ("prepare", "submit"):
                    self._minted_meta.pop(nonce, None)
                    self.broker.burn_token(meta.get("auth"))
            req_action = "submit" if prepared_run_dir else resolved.required_approval_action
            entry: dict[str, Any] = {
                "action": req_action,
                "requires_staging": False if prepared_run_dir else resolved.requires_staging,
                "model": resolved.model,
                "spec": spec,
                "digest": resolved.digest,
                "short_digest": resolved.short_digest,
                "approval_prefix": resolved.approval_prefix,
                "card": card_text,
                "plan": resolved.to_dict(),
                "model_card": ctx["model_card"],
                "will_prepare_only_in_container": ctx["will_prepare_only_in_container"],
                "missing_bytes": 0 if prepared_run_dir else resolved.missing_bytes,
                "missing_count": 0 if prepared_run_dir else len(resolved.missing_assets),
                "requires_data_dir_write": False
                if prepared_run_dir
                else bool(any(not a.prospective for a in resolved.missing_assets)),
                "download_required": False
                if prepared_run_dir
                else bool(resolved.missing_bytes > 0 or len(resolved.missing_assets) > 0),
                "missing_size_human": format_bytes(0 if prepared_run_dir else resolved.missing_bytes),
            }
            if prepared_run_dir:
                entry["prepared_run_dir"] = prepared_run_dir
            self._pending_by_action["submit"] = entry
            return dict(entry)

    def register_cancel_proposal(
        self,
        *,
        run_name: str,
        run_dir: str,
        attempt_id: str,
        digest: str,
    ) -> dict[str, Any]:
        with self._lock:
            for nonce, meta in list(self._minted_meta.items()):
                if meta.get("action") == "cancel":
                    self._minted_meta.pop(nonce, None)
                    self.broker.burn_token(meta.get("auth"))
            card_text = (
                f"Cancel proposal [{digest[:12]}]\n"
                f"  Run:        {run_name}\n"
                f"  Directory:  {run_dir}\n"
                f"  Attempt ID: {attempt_id}\n"
            )
            entry = {
                "action": "cancel",
                "run_name": run_name,
                "run_dir": run_dir,
                "attempt_id": attempt_id,
                "digest": digest,
                "short_digest": digest[:12],
                "approval_prefix": digest[:8],
                "card": card_text,
            }
            self._pending_by_action["cancel"] = entry
            return dict(entry)

    def get_pending(self, action: str = "submit") -> dict[str, Any] | None:
        with self._lock:
            slot = "submit" if action in ("prepare", "submit") else action
            item = self._pending_by_action.get(slot)
            if item is None:
                return None
            out = {k: v for k, v in item.items() if k != "spec"}
            return json.loads(json.dumps(out))

    def approve(
        self,
        *,
        digest_prefix: str,
        expected_digest: str | None = None,
        action: str | None = None,
        allow_download: bool = False,
    ) -> dict[str, Any]:
        req_action = (action or "").strip() or None
        if req_action is not None and req_action not in ("prepare", "submit", "cancel"):
            raise ValidationError(
                "Field 'action' must be 'prepare', 'submit', or 'cancel'.",
                field="action",
                allowed=["prepare", "submit", "cancel"],
            )
        cleaned = (digest_prefix or "").strip().lower()
        if not _HEX_PREFIX_RE.match(cleaned):
            raise ApprovalError(
                "Approval refused: provide at least the first 8 hex characters of the plan digest.",
                field="digest_prefix",
            )
        cleaned_full: str | None = None
        if expected_digest is not None and str(expected_digest).strip():
            cleaned_full = str(expected_digest).strip().lower()
            if not _FULL_DIGEST_RE.match(cleaned_full):
                raise ApprovalError(
                    "Approval refused: 'digest' must be a 64-character lowercase hex SHA-256 digest.",
                    field="digest",
                )

        with self._lock:
            slot = "cancel" if req_action == "cancel" else "submit"
            proposal = self._pending_by_action.get(slot)
            if proposal is None:
                raise ApprovalError(
                    f"No pending {req_action or 'run'} proposal to approve (prepare a plan first or already approved).",
                    field="digest_prefix",
                )
            eff_action = str(proposal["action"])
            if req_action is not None and req_action != eff_action:
                raise ApprovalError(
                    f"Approval refused: pending proposal requires action {eff_action!r}, not {req_action!r}.",
                    field="action",
                )
            prop_digest = str(proposal["digest"])
            if cleaned_full is not None and cleaned_full != prop_digest:
                raise ApprovalError(
                    f"Approval refused: displayed plan card [{cleaned_full[:8]}] was superseded by newer proposal "
                    f"[{proposal['approval_prefix']}]; review the updated card before approving.",
                    field="digest",
                )
            if not prop_digest.startswith(cleaned):
                raise ApprovalError(
                    f"Approval refused: prefix {cleaned!r} does not match expected digest prefix "
                    f"{proposal['approval_prefix']!r} (review the current plan card before approving).",
                    field="digest_prefix",
                )
            if eff_action in ("prepare", "submit") and proposal["download_required"]:
                if not allow_download:
                    raise ApprovalError(
                        f"Download consent required: plan needs {proposal['missing_count']} missing asset(s) "
                        f"(~{proposal['missing_size_human']}). Tick the download consent box to confirm.",
                        field="allow_download",
                    )
                if proposal.get("requires_data_dir_write", True):
                    dl_ok, dl_reason = can_download_into_data_dir(
                        model_name=str(proposal.get("model") or "") or None,
                    )
                    if not dl_ok:
                        m_hint = str(proposal.get("model") or "")
                        raise ApprovalError(
                            dl_reason,
                            field="download",
                            next_command=f"aimip fetch -m {m_hint}" if m_hint else None,
                        )

            self._pending_by_action.pop(slot, None)
            now_ts = float(self._now_fn())
            auth = Authorization(
                action=eff_action,
                digest=prop_digest,
                user=current_user(),
                issued_at=now_ts,
                expires_at=now_ts + APPROVAL_TTL_SECONDS,
                nonce=secrets.token_hex(16),
                allow_download=bool(allow_download) if eff_action in ("prepare", "submit") else False,
            )
            self.broker.store_minted(auth)
            self._minted_meta[auth.nonce] = {"action": eff_action, "auth": auth, "proposal": proposal}
            return {
                "authorization_nonce": auth.nonce,
                "action": auth.action,
                "digest": auth.digest,
                "approval_prefix": auth.digest[:8],
                "allow_download": auth.allow_download,
                "issued_at": auth.issued_at,
                "expires_at": auth.expires_at,
            }

    def execute_prepare(self, *, nonce: str | None = None) -> dict[str, Any]:
        """Execute Step 1 (`action='prepare'`) staging without holding `_lock` across I/O."""
        with self._lock:
            eff_nonce = (nonce or "").strip() or None
            if eff_nonce is None:
                prep_nonces = [
                    n for n, m in self._minted_meta.items() if m.get("action") == "prepare"
                ]
                if len(prep_nonces) == 1:
                    eff_nonce = prep_nonces[0]
            if not eff_nonce or eff_nonce not in self._minted_meta:
                raise ApprovalError(
                    "Preparation refused: missing, already-used, or invalid prepare approval token.",
                    field="nonce",
                )
            meta = self._minted_meta.pop(eff_nonce)
            auth: Authorization = meta["auth"]
            proposal: dict[str, Any] = meta["proposal"]
            if meta.get("action") != "prepare":
                self.broker.burn_token(auth)
                raise ApprovalError("Authorization action mismatch: expected 'prepare'.", field="nonce")

        try:
            spec: RunSpec = proposal["spec"]
            prepared = svc_prepare(
                spec,
                authorization=auth,
                require_authorization=True,
                broker=self.broker,
            )
            post_entry = self.register_submit_proposal(
                spec=prepared.plan.spec,
                resolved=prepared.plan,
                card_text=render_plan_card(prepared.plan, is_tty=False),
                prepared_run_dir=prepared.run_dir,
            )
            post_out = {k: v for k, v in post_entry.items() if k != "spec"}
            return {
                "stage": "prepared_awaiting_submit_approval",
                "run_dir": prepared.run_dir,
                "digest": prepared.plan.digest,
                "approval_prefix": prepared.plan.approval_prefix,
                "proposal": post_out,
            }
        except BaseException:
            self.broker.burn_token(auth)
            raise

    def execute_submit(self, *, nonce: str | None = None) -> dict[str, Any]:
        """Execute `submit` without holding `_lock` across file preparation or `sbatch`."""
        with self._lock:
            eff_nonce = (nonce or "").strip() or None
            if eff_nonce is None:
                submit_nonces = [
                    n for n, m in self._minted_meta.items() if m.get("action") == "submit"
                ]
                if len(submit_nonces) == 1:
                    eff_nonce = submit_nonces[0]
            if not eff_nonce or eff_nonce not in self._minted_meta:
                raise ApprovalError(
                    "Submission refused: missing, already-used, or invalid human submit approval token.",
                    field="nonce",
                )
            meta = self._minted_meta.pop(eff_nonce)
            auth: Authorization = meta["auth"]
            proposal: dict[str, Any] = meta["proposal"]
            if meta.get("action") != "submit":
                self.broker.burn_token(auth)
                raise ApprovalError(
                    "Prepare-only authorization can never submit; run `/api/v1/prepare` first and approve the final staged digest.",
                    field="nonce",
                )
            now_ts = float(self._now_fn())

        try:
            MemoryApprovalBroker.verify_authorization_fields(
                auth,
                expected_action="submit",
                expected_digest=str(proposal["digest"]),
                now_ts=now_ts,
            )
            prepared_run_dir = proposal.get("prepared_run_dir")
            if isinstance(prepared_run_dir, str) and prepared_run_dir:
                attempt = svc_submit(
                    prepared_run_dir,
                    authorization=auth,
                    require_authorization=True,
                    broker=self.broker,
                )
                return attempt.to_dict()

            spec: RunSpec = proposal["spec"]
            prepared = svc_prepare(
                spec,
                authorization=auth,
                require_authorization=True,
                broker=self.broker,
            )
            attempt = svc_submit(
                prepared,
                authorization=auth,
                require_authorization=True,
                broker=self.broker,
            )
            return attempt.to_dict()
        except BaseException:
            self.broker.burn_token(auth)
            raise

    def execute_cancel(self, run_name: str, *, nonce: str | None = None) -> dict[str, Any]:
        validated_name = validate_run_name(run_name)
        with self._lock:
            eff_nonce = (nonce or "").strip() or None
            if eff_nonce is None:
                cancel_nonces = [
                    n
                    for n, m in self._minted_meta.items()
                    if m.get("action") == "cancel"
                    and m.get("proposal", {}).get("run_name") == validated_name
                ]
                if len(cancel_nonces) == 1:
                    eff_nonce = cancel_nonces[0]
            if not eff_nonce or eff_nonce not in self._minted_meta:
                raise ApprovalError(
                    f"Cancelling {validated_name!r} requires human approval of its cancel digest first.",
                    field="nonce",
                )
            meta = self._minted_meta.pop(eff_nonce)
            auth: Authorization = meta["auth"]
            if meta.get("action") != "cancel":
                self.broker.burn_token(auth)
                raise ApprovalError("Authorization action mismatch: expected 'cancel'.", field="nonce")

        try:
            return svc_cancel(
                validated_name,
                authorization=auth,
                require_authorization=True,
                broker=self.broker,
            )
        except BaseException:
            self.broker.burn_token(auth)
            raise


class UiApp:
    """Application controller shared across HTTP request threads."""

    def __init__(
        self,
        *,
        token: str | None = None,
        now_fn: Any = time.time,
        chat_model: Any = None,
        max_diag_workers: int = DEFAULT_MAX_DIAG_WORKERS,
    ) -> None:
        self.token = token or secrets.token_urlsafe(32)
        self.now_fn = now_fn
        self.approvals = UiApprovalManager(now_fn=now_fn)
        self._diag_sem = threading.BoundedSemaphore(max(1, int(max_diag_workers)))
        self._chat_lock = threading.Lock()
        self._injected_chat_model = chat_model
        self._chat_agent: Any = None
        self._chat_tools: Any = None

    def get_chat_log_dir(self) -> str | None:
        raw = os.environ.get("AIMIP_CHAT_LOG_DIR", "").strip()
        if raw:
            return raw
        try:
            return os.path.join(resolve_configured_run_root(), "chat-logs")
        except AimipError:
            return None

    def _get_or_create_chat_agent(self) -> Any:
        if not _is_agent_available():
            raise ConfigError(
                "Optional chat assistant requires the `agent` extra (`pip install -r container/requirements-agent.txt`)."
            )
        if self._chat_agent is not None:
            return self._chat_agent

        from ..agent.chat import ChatAgent, Transcript, read_api_key, read_secret_file
        from ..agent.tools import AgentTools, _build_run_detail, _with_fetch_hint

        ui_approvals = self.approvals

        class _UiAgentTools(AgentTools):
            """Resolves `propose_run` ONCE and registers that exact `ResolvedPlan` in `UiApprovalManager`."""

            def __init__(self, *, now_fn: Any = time.time) -> None:
                super().__init__(now_fn=now_fn)
                self.last_turn_proposed: dict[str, Any] | None = None

            def propose_run(
                self,
                model: str,
                start: str,
                steps: int | None = None,
                length: str | None = None,
                end: str | None = None,
                members: int | None = None,
                ic: str | None = None,
                ocean_ic: str | None = None,
                seed: int | None = None,
                forcing_set: str | None = None,
                lora: str | None = None,
                name: str | None = None,
                out: str | None = None,
            ) -> dict[str, Any]:
                self.revoke()
                try:
                    checked_ic = validate_visible_input_path(ic, label="ic") if ic is not None else None
                    checked_ocean_ic = (
                        validate_visible_input_path(ocean_ic, label="ocean_ic")
                        if ocean_ic is not None
                        else None
                    )
                    checked_lora = trusted_lora_path(lora) if lora is not None else None
                    self._check_run_size(model, start, members, checked_ic)
                    spec = RunSpec(
                        model=model,
                        start=start,
                        steps=steps,
                        length=length,
                        end=end,
                        members=members,
                        seed=seed,
                        ic=checked_ic,
                        ocean_ic=checked_ocean_ic,
                        lora=checked_lora,
                        forcing_set=forcing_set,
                        name=name,
                        out=out,
                    )
                    resolved = svc_plan(spec)
                except AimipError as exc:
                    self.revoke()
                    raise ValueError(
                        _with_fetch_hint(model, str(exc), custom_ic=ic is not None)
                    ) from exc

                card_text = render_plan_card(resolved, is_tty=False)
                entry = ui_approvals.register_submit_proposal(
                    spec=spec,
                    resolved=resolved,
                    card_text=card_text,
                )
                self.last_turn_proposed = json.loads(
                    json.dumps({k: v for k, v in entry.items() if k != "spec"})
                )
                run_detail = _build_run_detail(resolved)
                result = {
                    "kind": "run",
                    "action": resolved.required_approval_action,
                    "requires_staging": resolved.requires_staging,
                    "run_dir": resolved.output_dir,
                    "digest": resolved.digest,
                    "short_digest": resolved.short_digest,
                    "approval_prefix": resolved.approval_prefix,
                    "card": card_text,
                    "plan": resolved.to_dict(),
                    "runs": [run_detail],
                    "gpu_hours_estimate": resolved.resources["gpu_hours_estimate"],
                    "walltime_capped": bool(resolved.resources.get("walltime_capped", False)),
                    "missing_bytes": resolved.missing_bytes,
                    "missing_count": len(resolved.missing_assets),
                    "approval_required": (
                        "Shown in the Plan & Approve tab — only the human's click in the UI can approve this digest."
                    ),
                }
                return json.loads(json.dumps(result))

            def plan_spin(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise ValueError(
                    "Multi-member spin workflows require the interactive `aimip chat` CLI on `/dev/tty`; "
                    "use `propose_run` to propose a run card in the UI."
                )

            def plan_collect(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise ValueError(
                    "Spin-collect workflows require the interactive `aimip chat` CLI on `/dev/tty`; "
                    "use `propose_run` to propose a run card in the UI."
                )

            def approve_and_execute_on_tty(self, *, tty_path: str = "/dev/tty") -> dict[str, Any]:
                raise ApprovalError(
                    "The UI assistant can never approve or execute runs; only the human's click in the browser can approve."
                )

        log_dir_raw = self.get_chat_log_dir()
        if not log_dir_raw:
            rroot = resolve_configured_run_root()
            log_dir_raw = os.path.join(rroot, "chat-logs")

        tools_inst = _UiAgentTools(now_fn=self.now_fn)
        if self._injected_chat_model is not None:
            transcript = Transcript(Path(log_dir_raw))
            self._chat_tools = tools_inst
            self._chat_agent = ChatAgent(self._injected_chat_model, tools_inst, transcript)
            return self._chat_agent

        from langchain_google_genai import ChatGoogleGenerativeAI

        vertex_project = os.environ.get("AIMIP_GEMINI_VERTEX_PROJECT", "").strip()
        backend_kw: dict[str, Any] = {}
        secret_key = ""
        if vertex_project:
            backend_kw = {
                "vertexai": True,
                "project": vertex_project,
                "location": os.environ.get("AIMIP_GEMINI_VERTEX_LOCATION", "global").strip() or "global",
            }
            token_file = os.environ.get("AIMIP_GEMINI_VERTEX_TOKEN_FILE", "").strip()
            if token_file:
                secret_key = read_secret_file(token_file)
                from google.oauth2.credentials import Credentials

                backend_kw["credentials"] = Credentials(token=secret_key)
            elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
                raise ConfigError(
                    "Vertex AI needs AIMIP_GEMINI_VERTEX_TOKEN_FILE or GOOGLE_APPLICATION_CREDENTIALS."
                )
        else:
            try:
                secret_key = read_api_key()
            except (OSError, ValueError) as exc:
                raise ConfigError(str(exc)) from exc
            backend_kw["google_api_key"] = secret_key

        model_id = os.environ.get("AIMIP_AGENT_MODEL") or "gemini-3.8-flash"
        llm = ChatGoogleGenerativeAI(model=model_id, temperature=0, timeout=60, max_retries=1, **backend_kw)
        transcript = Transcript(
            Path(log_dir_raw),
            (secret_key, os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")),
        )
        self._chat_tools = tools_inst
        self._chat_agent = ChatAgent(llm, tools_inst, transcript)
        return self._chat_agent

    def chat_turn(self, message: str) -> dict[str, Any]:
        text = (message or "").strip()
        if not text:
            raise ValidationError("Chat message must be non-empty.", field="message")
        with self._chat_lock:
            agent = self._get_or_create_chat_agent()
            if self._chat_tools is not None:
                self._chat_tools.last_turn_proposed = None
            try:
                reply = agent.turn(text)
            except RuntimeError as exc:
                raise ExecutionError(str(exc)) from exc
            turn_plan = (
                self._chat_tools.last_turn_proposed
                if self._chat_tools is not None
                else None
            )
            return {
                "reply": reply,
                "proposed_plan": turn_plan,
            }


class UiHTTPServer(ThreadingHTTPServer):
    """Bounded-concurrency HTTP server carrying `UiApp`, socket timeouts, and DNS-rebinding allowlist."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        app: UiApp,
        bind_host: str,
        allow_remote: bool,
        max_connections: int = DEFAULT_MAX_UI_CONNECTIONS,
        request_timeout: float = DEFAULT_UI_REQUEST_TIMEOUT,
    ) -> None:
        raw_bind = (
            bind_host[1:-1]
            if (bind_host.startswith("[") and bind_host.endswith("]"))
            else bind_host
        )
        if ":" in raw_bind:
            self.address_family = socket.AF_INET6
        super().__init__((raw_bind, server_address[1]), handler_cls)
        self.app = app
        self.bind_host = raw_bind
        self.allow_remote = allow_remote
        self.max_connections = max(1, int(max_connections))
        self.request_timeout = max(0.05, float(request_timeout))
        self._conn_sem = threading.BoundedSemaphore(self.max_connections)
        self._active_lock = threading.Lock()
        self.active_connections = 0
        bound_port = int(self.server_address[1])
        self.allowed_netlocs = _build_allowed_netlocs(
            bind_host, bound_port, allow_remote=allow_remote
        )

    def get_request(self) -> tuple[socket.socket, Any]:
        conn, addr = super().get_request()
        conn.settimeout(self.request_timeout)
        return conn, addr

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        acquire_wait = min(0.2, self.request_timeout)
        if not self._conn_sem.acquire(timeout=acquire_wait):
            busy_body = (
                b'{"aimip":"1","ok":false,"error":{"code":"busy",'
                b'"message":"UI server connection pool is saturated; retry shortly."}}\n'
            )
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json; charset=utf-8\r\n"
                    b"Cache-Control: no-store\r\n"
                    b"Connection: close\r\n"
                    + f"Content-Length: {len(busy_body)}\r\n\r\n".encode("ascii")
                    + busy_body
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return

        with self._active_lock:
            self.active_connections += 1
        try:
            t = threading.Thread(
                target=self.process_request_thread,
                args=(request, client_address),
                daemon=self.daemon_threads,
            )
            t.start()
        except BaseException:
            with self._active_lock:
                self.active_connections -= 1
            self._conn_sem.release()
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._active_lock:
                self.active_connections -= 1
            self._conn_sem.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        return


def _status_for_aimip_error(exc: AimipError) -> int:
    if isinstance(exc, ApprovalError):
        return HTTPStatus.FORBIDDEN
    if isinstance(exc, MissingAssetsError):
        return HTTPStatus.CONFLICT
    if isinstance(exc, ExecutionError):
        return HTTPStatus.INTERNAL_SERVER_ERROR
    return HTTPStatus.BAD_REQUEST


class UiRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler enforcing Host/Origin, CSP, header session token, and `/api/v1/*` routing."""

    server: UiHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _write_security_headers(self, *, no_store: bool = True) -> None:
        self.send_header("Content-Security-Policy", CSP_HEADER_VALUE)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.close_connection = True

    def _send_json(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(int(status_code))
        self._write_security_headers(no_store=True)
        if extra_headers:
            for hk, hv in extra_headers.items():
                self.send_header(hk, hv)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_error_json(
        self, status_code: int, code: str, message: str, *, field: str | None = None
    ) -> None:
        self._send_json(
            status_code,
            {
                "aimip": "1",
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                    "field": field,
                    "allowed": None,
                    "next_command": None,
                },
            },
        )

    def _send_aimip_error(self, exc: AimipError) -> None:
        self._send_json(
            _status_for_aimip_error(exc),
            {
                "aimip": "1",
                "ok": False,
                "error": exc.to_dict(),
            },
        )

    def _send_bytes(
        self, status_code: int, content_type: str, data: bytes, *, no_store: bool = True
    ) -> None:
        self.send_response(int(status_code))
        self._write_security_headers(no_store=no_store)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _check_host_and_origin(self) -> bool:
        """Verify `Host`, `Origin`, and `Referer` against `self.server.allowed_netlocs`."""
        host_hdr = (self.headers.get("Host") or "").strip().lower()
        if not host_hdr or host_hdr not in self.server.allowed_netlocs:
            self._send_error_json(
                HTTPStatus.FORBIDDEN,
                "forbidden_host",
                f"Rejected Host header {host_hdr!r} (DNS rebinding protection).",
            )
            return False

        origin_hdr = self.headers.get("Origin")
        if origin_hdr is not None:
            parsed_origin = urlsplit(origin_hdr.strip())
            if (
                parsed_origin.scheme not in ("http", "https")
                or parsed_origin.netloc.lower() not in self.server.allowed_netlocs
            ):
                self._send_error_json(
                    HTTPStatus.FORBIDDEN,
                    "forbidden_origin",
                    f"Rejected Origin header {origin_hdr!r}.",
                )
                return False

        referer_hdr = self.headers.get("Referer")
        if referer_hdr is not None:
            parsed_ref = urlsplit(referer_hdr.strip())
            if (
                parsed_ref.scheme not in ("http", "https")
                or parsed_ref.netloc.lower() not in self.server.allowed_netlocs
            ):
                self._send_error_json(
                    HTTPStatus.FORBIDDEN,
                    "forbidden_referer",
                    "Rejected Referer header.",
                )
                return False

        return True

    def _extract_request_token(self) -> str | None:
        """Extract session token ONLY from explicit `X-AIMIP-Token` or `Authorization: Bearer` headers.

        Cookies are deliberately never accepted because RFC 6265 cookies are shared across all ports
        of `127.0.0.1` / `localhost`.
        """
        direct = self.headers.get("X-AIMIP-Token")
        if direct and direct.strip():
            return direct.strip()
        auth_hdr = self.headers.get("Authorization") or ""
        if auth_hdr.lower().startswith("bearer "):
            cand = auth_hdr[7:].strip()
            if cand:
                return cand
        return None

    def _require_api_token(self) -> bool:
        cand = self._extract_request_token()
        if not cand or not secrets.compare_digest(cand, self.server.app.token):
            self._send_error_json(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "Missing or invalid UI session token (open the printed startup URL or pass X-AIMIP-Token).",
            )
            return False
        return True

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return {}
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValidationError("Invalid Content-Length header.") from exc
        if length < 0:
            raise ValidationError("Negative Content-Length header.")
        if length > MAX_JSON_BODY_BYTES:
            raise ValidationError(f"Request body exceeds {MAX_JSON_BODY_BYTES} byte limit.")
        if length == 0:
            return {}

        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            raise ValidationError(
                "POST request body must use Content-Type: application/json.",
                field="content_type",
            )

        deadline = time.monotonic() + self.server.request_timeout
        chunks: list[bytes] = []
        remaining = length
        while remaining > 0:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("Request body read timed out.")
            self.connection.settimeout(max(0.01, left))
            piece = self.rfile.read(min(65536, remaining))
            if not piece:
                raise TimeoutError("Incomplete request body.")
            chunks.append(piece)
            remaining -= len(piece)

        raw_bytes = b"".join(chunks)
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"Malformed JSON body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValidationError("JSON request body must be an object.")
        return parsed

    @staticmethod
    def _reject_raw_path_traversal(raw_path: str) -> None:
        low = raw_path.lower()
        if "%2f" in low or "%5c" in low or "%00" in low:
            raise ValidationError("Encoded path separators or NUL bytes in URL path are forbidden.", field="path")
        unquoted = unquote(raw_path)
        if "\x00" in unquoted or "\\" in unquoted:
            raise ValidationError("Invalid character in URL path.", field="path")
        segments = [seg for seg in unquoted.split("/") if seg]
        if any(seg in (".", "..") or ".." in seg for seg in segments):
            raise ValidationError("Path traversal ('..') in URL path is forbidden.", field="path")

    def do_GET(self) -> None:  # noqa: N802
        if not self._check_host_and_origin():
            return

        split = urlsplit(self.path)
        raw_path = split.path or "/"
        query = parse_qs(split.query, keep_blank_values=False)

        try:
            self._reject_raw_path_traversal(raw_path)
        except AimipError as exc:
            self._send_aimip_error(exc)
            return

        # Startup URL `GET /?token=<token>` validates `<token>` and serves `index.html` directly
        # with `Cache-Control: no-store` and NO `Set-Cookie` header (preventing cross-port cookie leakage).
        if raw_path in ("/", "/index.html") and "token" in query:
            q_tok = (query["token"][0] if query["token"] else "").strip()
            if not q_tok or not secrets.compare_digest(q_tok, self.server.app.token):
                self._send_error_json(
                    HTTPStatus.UNAUTHORIZED,
                    "unauthorized",
                    "Invalid startup token in URL.",
                )
                return
            asset_path = _STATIC_DIR / "index.html"
            if not asset_path.is_file() or asset_path.is_symlink():
                self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Static asset index.html missing.")
                return
            self._send_bytes(
                HTTPStatus.OK,
                "text/html; charset=utf-8",
                asset_path.read_bytes(),
                no_store=True,
            )
            return

        if raw_path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self._write_security_headers(no_store=False)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if raw_path in _ALLOWED_STATIC_FILES:
            fname, ctype = _ALLOWED_STATIC_FILES[raw_path]
            asset_path = _STATIC_DIR / fname
            if not asset_path.is_file() or asset_path.is_symlink():
                self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", f"Static asset {fname} missing.")
                return
            self._send_bytes(
                HTTPStatus.OK,
                ctype,
                asset_path.read_bytes(),
                no_store=(raw_path in ("/", "/index.html")),
            )
            return

        if not raw_path.startswith("/api/v1/"):
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", f"Unknown route {raw_path!r}.")
            return

        if not self._require_api_token():
            return

        try:
            self._dispatch_api_get(raw_path, query)
        except TimeoutError as exc:
            self._send_error_json(HTTPStatus.REQUEST_TIMEOUT, "request_timeout", str(exc))
        except AimipError as exc:
            self._send_aimip_error(exc)
        except Exception as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                f"{type(exc).__name__}: {exc}",
            )

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._check_host_and_origin():
            return
        split = urlsplit(self.path)
        raw_path = split.path or "/"
        try:
            self._reject_raw_path_traversal(raw_path)
        except AimipError as exc:
            self._send_aimip_error(exc)
            return
        if raw_path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self._write_security_headers(no_store=False)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if raw_path in _ALLOWED_STATIC_FILES:
            fname, ctype = _ALLOWED_STATIC_FILES[raw_path]
            asset_path = _STATIC_DIR / fname
            if not asset_path.is_file() or asset_path.is_symlink():
                self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", f"Static asset {fname} missing.")
                return
            self._send_bytes(
                HTTPStatus.OK,
                ctype,
                asset_path.read_bytes(),
                no_store=(raw_path in ("/", "/index.html")),
            )
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", f"Unknown route {raw_path!r}.")

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_host_and_origin():
            return

        split = urlsplit(self.path)
        raw_path = split.path or "/"

        try:
            self._reject_raw_path_traversal(raw_path)
        except AimipError as exc:
            self._send_aimip_error(exc)
            return

        if not raw_path.startswith("/api/v1/"):
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", f"Unknown route {raw_path!r}.")
            return

        if not self._require_api_token():
            return

        try:
            body = self._read_json_body()
            self._dispatch_api_post(raw_path, body)
        except TimeoutError as exc:
            self._send_error_json(HTTPStatus.REQUEST_TIMEOUT, "request_timeout", str(exc))
        except AimipError as exc:
            self._send_aimip_error(exc)
        except Exception as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                f"{type(exc).__name__}: {exc}",
            )

    def _dispatch_api_get(self, raw_path: str, query: dict[str, list[str]]) -> None:
        parts = [unquote(seg) for seg in raw_path[len("/api/v1/") :].split("/") if seg]
        app = self.server.app

        def _q1(key: str) -> str | None:
            vals = query.get(key)
            return vals[0].strip() if vals and vals[0].strip() else None

        if parts == ["session"]:
            try:
                run_root = resolve_configured_run_root()
            except AimipError:
                run_root = None
            caps = _ui_runtime_capabilities()
            self._send_json(
                HTTPStatus.OK,
                {
                    "aimip": "1",
                    "ok": True,
                    "result": {
                        "version": __version__,
                        "run_root": run_root,
                        "allowed_run_roots": list(get_allowed_run_roots()),
                        "visible_input_roots": list(get_visible_input_roots()),
                        "matplotlib_available": _is_matplotlib_available(),
                        "agent_available": _is_agent_available(),
                        "agent_ready": bool(
                            app._injected_chat_model is not None
                            or (_is_agent_available() and _agent_credentials_configured())
                        ),
                        "chat_log_dir": app.get_chat_log_dir(),
                        "in_container": caps["in_container"],
                        "scheduler": caps["scheduler"],
                        "can_submit_directly": caps["can_submit_directly"],
                        "poll_interval_seconds": int(UI_SACCT_POLL_INTERVAL_SECONDS),
                        "pending_submit": app.approvals.get_pending("submit"),
                        "pending_cancel": app.approvals.get_pending("cancel"),
                    },
                },
            )
            return

        if parts == ["doctor"]:
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": svc_doctor()})
            return

        if parts == ["models"]:
            installed_flag = (_q1("installed") or "").lower() in ("1", "true", "yes")
            res = svc_models(
                task=_q1("task"),
                family=_q1("family"),
                status=_q1("status"),
                installed_only=installed_flag,
            )
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": res})
            return

        if len(parts) == 2 and parts[0] in ("models", "show"):
            model_name = validate_model_name(parts[1])
            inc_ex = (_q1("examples") or "1").lower() in ("1", "true", "yes")
            card = dict(svc_show(model_name, include_examples=inc_ex))
            card["citation"] = svc_cite(model_name)
            try:
                mspec = get_model(model_name)
                if mspec.family == "fme":
                    card["shipped_ics"] = [ic.to_dict() for ic in list_model_ics(mspec)]
                    if mspec.is_coupled:
                        card["shipped_ocean_ics"] = [ic.to_dict() for ic in list_model_ocean_ics(mspec)]
            except AimipError:
                card["shipped_ics"] = []
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": card})
            return

        if len(parts) == 3 and parts[0] == "models" and parts[2] == "cite":
            model_name = validate_model_name(parts[1])
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": svc_cite(model_name)})
            return

        if parts == ["pickers", "files"]:
            kind = _q1("kind") or "ic"
            self._send_json(
                HTTPStatus.OK,
                {"aimip": "1", "ok": True, "result": _list_picker_files(kind, _q1("dir"))},
            )
            return

        if parts == ["runs"]:
            runs_list = svc_ls(min_sacct_interval_seconds=UI_SACCT_POLL_INTERVAL_SECONDS)
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": runs_list})
            return

        if len(parts) >= 2 and parts[0] == "runs":
            run_name = validate_run_name(parts[1])
            sub = parts[2:]
            if sub in ([], ["status"]):
                st = dict(
                    svc_status(
                        run_name,
                        min_sacct_interval_seconds=UI_SACCT_POLL_INTERVAL_SECONDS,
                    )
                )
                rdir = st.get("out_dir") or st.get("run_dir")
                st["files"] = _list_run_directory_files(str(rdir)) if rdir else []
                self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": st})
                return
            if sub == ["logs"]:
                raw_mb = _q1("max_bytes")
                max_b = DEFAULT_UI_LOG_BYTES
                if raw_mb is not None:
                    max_b = _parse_strict_int(raw_mb, "max_bytes")
                bounded = min(max(64, max_b), MAX_UI_LOG_BYTES)
                self._send_json(
                    HTTPStatus.OK,
                    {"aimip": "1", "ok": True, "result": svc_logs(run_name, follow=False, max_bytes=bounded)},
                )
                return
            if sub == ["summary"]:
                diag_wait = max(30.0, self.server.request_timeout * 2.0)
                if not app._diag_sem.acquire(timeout=diag_wait):
                    self._send_error_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "diagnostic_busy",
                        "Summary worker pool is busy; retry in a few seconds.",
                    )
                    return
                try:
                    sum_dict = svc_summary(run_name).to_dict()
                    sum_dict["run_name"] = run_name
                    has_mpl = _is_matplotlib_available()
                    sum_dict["plot_available"] = has_mpl
                    sum_dict["plot_urls"] = (
                        {
                            vname: f"/api/v1/runs/{quote(run_name)}/plot.png?var={quote(vname)}"
                            for vname in (sum_dict.get("variables") or {})
                        }
                        if has_mpl
                        else {}
                    )
                finally:
                    app._diag_sem.release()
                self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": sum_dict})
                return
            if sub == ["plot.png"]:
                diag_wait = max(30.0, self.server.request_timeout * 2.0)
                if not app._diag_sem.acquire(timeout=diag_wait):
                    self._send_error_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "diagnostic_busy",
                        "Plot worker pool is busy; retry in a few seconds.",
                    )
                    return
                try:
                    sum_dict = svc_summary(run_name).to_dict()
                    sum_dict["run_name"] = run_name
                    png = render_summary_plot_png(sum_dict, var_name=_q1("var"))
                finally:
                    app._diag_sem.release()
                if png is None:
                    self._send_error_json(
                        HTTPStatus.NOT_FOUND,
                        "matplotlib_unavailable",
                        "matplotlib is not installed; use numerical summary JSON.",
                    )
                    return
                self._send_bytes(HTTPStatus.OK, "image/png", png, no_store=True)
                return

        if parts == ["chat", "status"]:
            self._send_json(
                HTTPStatus.OK,
                {
                    "aimip": "1",
                    "ok": True,
                    "result": {
                        "agent_available": _is_agent_available(),
                        "credentials_configured": bool(
                            app._injected_chat_model is not None or _agent_credentials_configured()
                        ),
                    },
                },
            )
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", f"Unknown API endpoint {raw_path!r}.")

    def _dispatch_api_post(self, raw_path: str, body: dict[str, Any]) -> None:
        parts = [unquote(seg) for seg in raw_path[len("/api/v1/") :].split("/") if seg]
        app = self.server.app

        if parts == ["plan"]:
            spec = _spec_from_payload(body)
            resolved = svc_plan(spec)
            card_text = render_plan_card(resolved, is_tty=False)
            proposal = app.approvals.register_submit_proposal(
                spec=spec,
                resolved=resolved,
                card_text=card_text,
            )
            proposal_out = {k: v for k, v in proposal.items() if k != "spec"}
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": proposal_out})
            return

        if parts == ["approve"]:
            prefix = str(body.get("digest_prefix") or "")
            raw_digest = body.get("digest") if body.get("digest") is not None else body.get("expected_digest")
            expected_digest = str(raw_digest).strip() if raw_digest is not None and str(raw_digest).strip() else None
            raw_act = body.get("action")
            action = str(raw_act).strip() if raw_act is not None else None
            allow_dl = bool(body.get("allow_download") is True)
            approved = app.approvals.approve(
                digest_prefix=prefix,
                expected_digest=expected_digest,
                action=action or None,
                allow_download=allow_dl,
            )
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": approved})
            return

        if parts == ["prepare"]:
            nonce = body.get("nonce") or body.get("authorization_nonce")
            prep_res = app.approvals.execute_prepare(
                nonce=str(nonce) if nonce is not None else None
            )
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": prep_res})
            return

        if parts == ["submit"]:
            nonce = body.get("nonce") or body.get("authorization_nonce")
            attempt = app.approvals.execute_submit(
                nonce=str(nonce) if nonce is not None else None
            )
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": attempt})
            return

        if len(parts) == 4 and parts[0] == "runs" and parts[2] == "cancel" and parts[3] == "propose":
            run_name = validate_run_name(parts[1])
            run_dir, _ = resolve_run_target(run_name)
            trusted = check_trusted_run_dir(run_dir)
            snap = get_run_status(trusted)
            job_id = snap.get("slurm_job_id") or snap.get("job_id")
            local_pid = snap.get("local_pid")
            if not job_id and not local_pid:
                raise ConfigError(
                    f"Run {run_name!r} has no recorded Slurm job ID or local PID to cancel.",
                    field="run",
                )
            cur_state = str(snap.get("state") or "unknown").strip()
            if (cur_state.split() or [""])[0].rstrip("+").upper() in TERMINAL_RUN_STATES or cur_state.upper() in TERMINAL_RUN_STATES:
                raise ConfigError(
                    f"Run {run_name!r} is already in terminal state {cur_state!r} and cannot be cancelled.",
                    field="run",
                )
            attempt_id = str(job_id) if job_id else str(local_pid)
            digest = cancel_digest_for(trusted, attempt_id)
            prop = app.approvals.register_cancel_proposal(
                run_name=run_name,
                run_dir=trusted,
                attempt_id=attempt_id,
                digest=digest,
            )
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": prop})
            return

        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "cancel":
            run_name = validate_run_name(parts[1])
            nonce = body.get("nonce") or body.get("authorization_nonce")
            cancelled = app.approvals.execute_cancel(
                run_name,
                nonce=str(nonce) if nonce is not None else None,
            )
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": cancelled})
            return

        if parts == ["chat"]:
            res = app.chat_turn(str(body.get("message") or ""))
            self._send_json(HTTPStatus.OK, {"aimip": "1", "ok": True, "result": res})
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", f"Unknown API endpoint {raw_path!r}.")


def _resolve_default_request_timeout() -> float:
    raw = os.environ.get("AIMIP_UI_REQUEST_TIMEOUT", "").strip()
    if raw:
        try:
            return max(0.05, float(raw))
        except ValueError:
            pass
    return DEFAULT_UI_REQUEST_TIMEOUT


def _resolve_default_max_connections() -> int:
    raw = os.environ.get("AIMIP_UI_MAX_CONNECTIONS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_UI_CONNECTIONS


def create_ui_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_remote: bool = False,
    token: str | None = None,
    now_fn: Any = time.time,
    chat_model: Any = None,
    max_connections: int | None = None,
    request_timeout: float | None = None,
    max_diag_workers: int = DEFAULT_MAX_DIAG_WORKERS,
) -> UiHTTPServer:
    """Validate bind address and construct a bound `UiHTTPServer` instance."""
    validated_host = validate_bind_host(host, allow_remote=allow_remote)
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValidationError("Port must be an integer in 0..65535.", field="port")
    eff_max_conn = _resolve_default_max_connections() if max_connections is None else int(max_connections)
    eff_req_timeout = _resolve_default_request_timeout() if request_timeout is None else float(request_timeout)
    app = UiApp(
        token=token,
        now_fn=now_fn,
        chat_model=chat_model,
        max_diag_workers=max_diag_workers,
    )
    return UiHTTPServer(
        (validated_host, port),
        UiRequestHandler,
        app=app,
        bind_host=validated_host,
        allow_remote=allow_remote,
        max_connections=eff_max_conn,
        request_timeout=eff_req_timeout,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `aimip ui [--port N] [--host 127.0.0.1] [--allow-remote]`."""
    parser = argparse.ArgumentParser(
        prog="aimip ui",
        description=(
            "Start the local AIMIP web UI on 127.0.0.1 with a random per-session token. "
            "On an HPC cluster, connect from your laptop over an SSH tunnel: "
            "`ssh -N -L <port>:127.0.0.1:<port> <login-node>`."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback address to bind (default: 127.0.0.1; non-loopback refused unless --allow-remote).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port to bind (default: 8765; pass 0 for an ephemeral port).",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding a non-loopback address (session token and Host/Origin checks remain enforced).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Socket header/body read timeout in seconds (default: 30, or AIMIP_UI_REQUEST_TIMEOUT).",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Maximum concurrent HTTP connections (default: 16, or AIMIP_UI_MAX_CONNECTIONS).",
    )
    args = parser.parse_args(argv)

    try:
        server = create_ui_server(
            host=args.host,
            port=args.port,
            allow_remote=bool(args.allow_remote),
            max_connections=args.max_connections,
            request_timeout=args.request_timeout,
        )
    except AimipError as exc:
        print(f"aimip ui error: {exc.message}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"aimip ui could not bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    bound_port = int(server.server_address[1])
    url_host = _format_host_for_url(server.bind_host)
    url = f"http://{url_host}:{bound_port}/?token={server.app.token}"
    print(f"AIMIP UI listening on {url}", flush=True)
    print(
        f"SSH tunnel from laptop: ssh -N -L {bound_port}:127.0.0.1:{bound_port} <cluster-login>",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
