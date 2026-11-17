"""MN5 (BSC MareNostrum 5) support: QOS instead of partition, and trusted group directories above run dirs.

See docs/mn5.md for why each change was needed.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
from typing import Any

import pytest

from aimip import submit
from aimip.errors import ConfigError, ValidationError
from aimip.runner import SlurmOptions, render_sbatch_script
from aimip.services import core
from aimip.site_config import SiteConfig, init_site_config, load_site_config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _write_site(path: pathlib.Path, **overrides: Any) -> pathlib.Path:
    payload: dict[str, Any] = {
        "name": "mn5",
        "scheduler": "slurm",
        "runtime": "native",
        "sbatch": {
            "partition": "",
            "qos": "acc_ehpc",
            "account": "ehpc819",
            "gres": "gpu:1",
            "cpus_per_task": 20,
            "mem": "",
            "max_time": "12:00:00",
        },
        "run_root": str(path.parent / "runs"),
        "extra_ro_binds": [],
        "gpu_check": "nvidia-smi",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---- site.json: qos, optional partition, trusted_group_parents


def test_site_qos_without_partition_round_trips(tmp_path: pathlib.Path) -> None:
    site_path = _write_site(
        tmp_path / "site.json", trusted_group_parents=["/gpfs/scratch/bsc32", "/gpfs/projects/bsc32"]
    )
    cfg = load_site_config(str(site_path))
    assert cfg.partition == ""
    assert cfg.qos == "acc_ehpc"
    assert cfg.trusted_group_parents == ("/gpfs/scratch/bsc32", "/gpfs/projects/bsc32")
    out = cfg.to_dict()
    assert out["sbatch"]["qos"] == "acc_ehpc"
    assert out["trusted_group_parents"] == ["/gpfs/scratch/bsc32", "/gpfs/projects/bsc32"]


def test_site_partition_none_means_no_partition(tmp_path: pathlib.Path) -> None:
    site_path = _write_site(tmp_path / "site.json")
    raw = json.loads(site_path.read_text())
    raw["sbatch"]["partition"] = "none"
    site_path.write_text(json.dumps(raw))
    assert load_site_config(str(site_path)).partition == ""


def test_site_without_partition_or_qos_is_refused(tmp_path: pathlib.Path) -> None:
    site_path = _write_site(tmp_path / "site.json")
    raw = json.loads(site_path.read_text())
    raw["sbatch"]["qos"] = ""
    site_path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match="requires a non-empty 'partition'"):
        load_site_config(str(site_path))


def test_other_sites_unchanged_without_qos() -> None:
    """Sites without a qos serialize exactly as before (no new keys in site.json)."""
    cfg = SiteConfig.preset("levante")
    out = cfg.to_dict()
    assert "qos" not in out["sbatch"]
    assert "trusted_group_parents" not in out


@pytest.mark.parametrize("bad", ["/", "relative/dir", "/gpfs/../etc"])
def test_trusted_group_parents_entries_are_validated(tmp_path: pathlib.Path, bad: str) -> None:
    site_path = _write_site(tmp_path / "site.json", trusted_group_parents=[bad])
    with pytest.raises(ValidationError):
        load_site_config(str(site_path))


def test_mn5_preset_and_site_init(tmp_path: pathlib.Path) -> None:
    preset = SiteConfig.preset("mn5")
    assert preset.partition == "" and preset.qos == "acc_ehpc" and preset.cpus_per_task == 20
    assert preset.mem == ""  # MN5 refuses --mem
    target, cfg = init_site_config(
        preset="mn5",
        out_path=str(tmp_path / "site.json"),
        account="ehpc819",
        run_root=str(tmp_path / "runs"),
        trusted_group_parents=["/gpfs/scratch/bsc32"],
    )
    written = json.loads(pathlib.Path(target).read_text())
    assert written["sbatch"]["qos"] == "acc_ehpc"
    assert written["sbatch"]["partition"] == ""
    assert written["trusted_group_parents"] == ["/gpfs/scratch/bsc32"]
    assert cfg.account == "ehpc819"


def test_install_preset_mn5_parses() -> None:
    raw = json.loads((REPO_ROOT / "install" / "presets" / "mn5.json").read_text())
    raw["sbatch"]["account"] = "ehpc819"  # the installer requires --account for CHANGE_ME
    cfg = SiteConfig.from_dict(raw)
    assert cfg.qos == "acc_ehpc" and cfg.partition == ""
    assert "/gpfs/scratch/bsc32" in cfg.trusted_group_parents


# ---- job.sbatch: --qos line, no --partition line


def test_mem_none_means_no_mem(tmp_path: pathlib.Path) -> None:
    site_path = _write_site(tmp_path / "site.json")
    raw = json.loads(site_path.read_text())
    raw["sbatch"]["mem"] = "none"
    site_path.write_text(json.dumps(raw))
    cfg = load_site_config(str(site_path))
    assert cfg.mem == ""
    assert SlurmOptions.from_site(cfg).mem == ""
    assert SlurmOptions.from_site(cfg, mem="32G").mem == "32G"
    assert SlurmOptions.from_site(SiteConfig.preset("levante"), account="a", mem="none").mem == ""
    with pytest.raises(ValidationError):
        SlurmOptions.from_site(cfg, mem="lots")


def test_slurm_options_need_partition_or_qos() -> None:
    with pytest.raises(ConfigError, match="partition must be non-empty"):
        SlurmOptions(partition="", qos="", account="a").validate()
    SlurmOptions(partition="", qos="acc_ehpc", account="a").validate()


def test_qos_override_and_none(tmp_path: pathlib.Path) -> None:
    cfg = load_site_config(str(_write_site(tmp_path / "site.json")))
    assert SlurmOptions.from_site(cfg).qos == "acc_ehpc"
    assert SlurmOptions.from_site(cfg, qos="acc_debug").qos == "acc_debug"
    with pytest.raises(ConfigError):  # no qos and no partition left
        SlurmOptions.from_site(cfg, qos="none")
    opts = SlurmOptions.from_site(cfg, qos="none", partition="acc")
    assert opts.qos == "" and opts.partition == "acc"
    with pytest.raises(ValidationError):
        SlurmOptions.from_site(cfg, qos="acc ehpc")


def test_rendered_script_has_qos_and_passes_both_gates(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_path = _write_site(tmp_path / "site.json")
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    run_dir = tmp_path / "runs" / "check"
    run_dir.mkdir(parents=True)
    os.chmod(run_dir, 0o755)
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    cfg = load_site_config(str(site_path))
    script = render_sbatch_script(
        model_name="ace2-era5",
        out_dir=str(run_dir),
        playground_dir=str(prefix),
        slurm_opts=SlurmOptions.from_site(cfg),
        default_walltime="00:20:00",
    )
    assert "#SBATCH --qos=acc_ehpc" in script
    assert "--partition" not in script
    assert "--mem" not in script  # MN5: "You cannot submit a job requesting memory parameters"
    assert "#SBATCH --account=ehpc819" in script
    job = run_dir / "job.sbatch"
    job.write_text(script, encoding="utf-8")
    submit.check_job_script(str(job), str(run_dir), str(prefix))


def test_rendered_script_without_qos_is_unchanged(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site_path = _write_site(tmp_path / "site.json")
    raw = json.loads(site_path.read_text())
    raw["sbatch"].update({"partition": "gpu", "qos": "", "mem": "64G"})
    site_path.write_text(json.dumps(raw))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    run_dir = tmp_path / "runs" / "plain"
    run_dir.mkdir(parents=True)
    (tmp_path / "prefix" / "bin").mkdir(parents=True)
    cfg = load_site_config(str(site_path))
    script = render_sbatch_script(
        model_name="ace2-era5",
        out_dir=str(run_dir),
        playground_dir=str(tmp_path / "prefix"),
        slurm_opts=SlurmOptions.from_site(cfg),
        default_walltime="00:20:00",
    )
    lines = script.splitlines()
    assert lines[2:4] == ["#SBATCH --job-name=aimip-ace2-era5", "#SBATCH --partition=gpu"]
    assert "#SBATCH --mem=64G" in lines
    assert "--qos" not in script


# ---- ancestry: MN5 group directories (root-owned, drwxrws---) listed in site.json


@pytest.fixture
def mn5_tree(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """`<tmp>/gpfs/grp/user/run` where `grp` looks like MN5's /gpfs/scratch/bsc32 (root-owned, drwxrws---).

    The test cannot create a root-owned directory, so `os.lstat` reports `grp` with the owner and mode chosen by
    the test; everything else is the real file system.
    """
    base = tmp_path / "gpfs"
    grp = base / "grp"
    run = grp / "user" / "run"
    run.mkdir(parents=True)
    for d in (base, grp, grp / "user", run):
        os.chmod(d, 0o755)
    state = {"uid": 0, "mode": stat.S_IFDIR | 0o2770}
    real_lstat = os.lstat

    def fake_lstat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        st = real_lstat(path, *args, **kwargs)
        if os.path.normpath(os.fspath(path)) == str(grp):
            fields = list(st)
            fields[stat.ST_MODE] = state["mode"]
            fields[stat.ST_UID] = state["uid"]
            return os.stat_result(fields)
        return st

    monkeypatch.setattr(os, "lstat", fake_lstat)

    def set_trusted(paths: list[str]) -> None:
        site_path = _write_site(tmp_path / "site.json", trusted_group_parents=paths)
        monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))

    return {"grp": str(grp), "run": str(run), "state": state, "set_trusted": set_trusted}


def test_group_writable_parent_refused_when_not_listed(mn5_tree: dict[str, Any]) -> None:
    mn5_tree["set_trusted"]([])
    with pytest.raises(ValidationError, match="writable by others without sticky bit"):
        submit.check_ancestry(mn5_tree["run"])
    with pytest.raises(ValidationError, match="writable by others without sticky bit"):
        core._check_existing_ancestry(mn5_tree["run"], label="run directory", field="out")


def test_listed_mn5_group_parent_is_accepted(mn5_tree: dict[str, Any]) -> None:
    mn5_tree["set_trusted"]([mn5_tree["grp"]])
    submit.check_ancestry(mn5_tree["run"])
    core._check_existing_ancestry(mn5_tree["run"], label="run directory", field="out")


def test_listed_parent_still_refused_if_world_writable(mn5_tree: dict[str, Any]) -> None:
    mn5_tree["set_trusted"]([mn5_tree["grp"]])
    mn5_tree["state"]["mode"] = stat.S_IFDIR | 0o2777
    with pytest.raises(ValidationError, match="writable by others without sticky bit"):
        submit.check_ancestry(mn5_tree["run"])
    with pytest.raises(ValidationError, match="writable by others without sticky bit"):
        core._check_existing_ancestry(mn5_tree["run"], label="run directory", field="out")


def test_listed_parent_still_refused_if_not_root_owned(mn5_tree: dict[str, Any]) -> None:
    mn5_tree["set_trusted"]([mn5_tree["grp"]])
    mn5_tree["state"]["uid"] = os.geteuid() + 1 if os.geteuid() != 0 else 4242
    with pytest.raises(ValidationError, match="writable by others without sticky bit"):
        submit.check_ancestry(mn5_tree["run"])


def test_trusted_group_parents_empty_without_site_config() -> None:
    assert submit.trusted_group_parents() == frozenset()
