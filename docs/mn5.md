# Running aimip on MareNostrum 5 (BSC)

This page lists what had to change for `aimip` to run on MareNostrum 5 (MN5), and how it was installed there. The
changes are opt-in through `site.json`: on every other site `aimip` behaves exactly as before.

## Why MN5 needed changes

MN5 differs from the clusters `aimip` was written for (Levante, generic Slurm) in three ways.

1. **Group directories are writable by the whole group.** Every user directory sits in a root-owned directory of
   its Unix group that all members can write, without a sticky bit:

   ```text
   drwxrws---  root       bsc32  /gpfs/scratch/bsc32
   drwxr-sr-x  bsc032013  bsc32  /gpfs/scratch/bsc32/bsc032013
   ```

   The same holds for `/gpfs/projects/<group>` and for the EuroHPC project groups (`/gpfs/scratch/ehpc819`, …).
   `aimip` refuses to prepare or submit a run when any directory above the run directory is writable by other users
   without a sticky bit (another user could rename the run directory away and put their own in its place). On MN5
   that rule refused every location with room for model output; only the home directory passed, and its quota is
   80 GB.

2. **Jobs are placed by QOS, not by partition.** An MN5 job names its account and QOS (`acc_ehpc`: up to 3 days;
   `acc_debug`: up to 2 h, one job at a time) and no partition; the QOS selects the accelerated partition. `aimip`
   required a partition in every job script and could not write a `--qos` line.

3. **Jobs may not ask for memory.** MN5 gives memory per core (8 GB per core on the accelerated nodes, so 160 GB
   for 20 cores) and refuses any job that sets `--mem`:
   `sbatch: error: You cannot submit a job requesting memory parameters, memory is automatically set for each asked cpu`.
   `aimip` wrote `#SBATCH --mem=64G` into every job script.

## The modifications

| # | change | files |
|---|---|---|
| 1 | `site.json` may list `trusted_group_parents`: root-owned group directories that are accepted above a run directory. A listed directory passes only if, when checked, it is owned by root and writable by its group but **not** by everyone; every other directory in the path is checked as before. An empty list (the default) keeps the strict rule. | `src/aimip/submit.py` (`trusted_group_parents`, `is_trusted_group_parent`, `check_ancestry`), `src/aimip/services/core.py` (`_check_existing_ancestry`, `doctor` hint), `src/aimip/site_config.py` |
| 2 | New Slurm setting `qos`: `site.json` `sbatch.qos`, `--qos` on `plan`/`run`/`ic spin` (`--qos none` drops it), `qos:` in `--config` files and in the web UI's resources. The job script gets `#SBATCH --qos=…`. | `src/aimip/site_config.py`, `src/aimip/runner.py` (`SlurmOptions`, `render_sbatch_script`, manifest), `src/aimip/emu.py` (manifest), `src/aimip/cli.py`, `src/aimip/services/cards.py`, `src/aimip/ui/server.py`, `src/aimip/completion.py` |
| 3 | The partition may be empty (or `"none"`) when a QOS is given; then no `#SBATCH --partition` line is written. Without a QOS a partition is still required. | `src/aimip/site_config.py`, `src/aimip/runner.py` |
| 4 | The submit gates accept `#SBATCH --qos=`: the Python gate used by native installs and the shell gate in the container wrapper, which must stay identical (the test suite checks this). | `src/aimip/submit.py` (`ALLOWED_SBATCH_KEYS`), `container/aimip-wrapper.in`, `tests/wrapper/gate_cases.json` |
| 5 | The QOS is part of the plan a person approves: it is in the plan's resources and digest and on the plan card. It is added only when set, so digests on sites without a QOS are unchanged. | `src/aimip/services/core.py`, `src/aimip/services/cards.py` |
| 6 | Preset `mn5`: no partition, QOS `acc_ehpc`, no `mem`, one GPU with 20 cores (MN5 asks for 20 cores per GPU), run root and trusted group directories for BSC Earth Sciences (`bsc32`). | `install/presets/mn5.json`, `SiteConfig.preset("mn5")`, `aimip site init --preset mn5` |
| 7 | `mem` may be empty (or `"none"`, also `--mem none`): then no `#SBATCH --mem` line is written. The `mn5` preset has no `mem`. | `src/aimip/site_config.py`, `src/aimip/runner.py`, `src/aimip/cli.py` |
| 8 | Tests for all of the above. | `tests/test_mn5.py` |

Not changed:

- **The container wrapper's directory check.** `container/aimip-wrapper.in` does not read `trusted_group_parents`, so
  container mode still refuses MN5's group directories. MN5 has no Apptainer and the image for this development
  version is not published, so the MN5 install is native.
- **`agent/security.py`.** It checks the files that hold the agent's credentials, which belong in the home directory.

## Install on MN5

Only the login node `alogin4` (`MN5ACC4`) reaches the internet; compute nodes and the other login nodes do not.
Login nodes stop any process after 300 s of CPU time. A native install without `--with-emu` fits within that limit:
the pip installs and the 341-test suite each took under 300 s.

```bash
# on alogin4, in the repository copy
export PIP_CACHE_DIR=/gpfs/scratch/bsc32/$USER/.cache/pip   # the home quota is 80 GB: keep caches off it
nohup setsid bash install/install-native.sh --prefix /gpfs/projects/bsc32/$USER/aimip --site mn5 \
    --account ehpc819 --run-root /gpfs/scratch/bsc32/$USER/aimip-runs \
    --python /apps/GPP/MINICONDA/24.1.2/bin/python3.12 > /gpfs/scratch/bsc32/$USER/aimip-install.log 2>&1 &
/gpfs/projects/bsc32/$USER/aimip/bin/aimip doctor
```

For another group, edit `run_root` and `trusted_group_parents` in `<prefix>/site.json` (for example
`/gpfs/scratch/ehpc819`). An install made with `--site generic-slurm` can be switched by adding to its `site.json`:
`"qos": "acc_ehpc"`, `"partition": ""` and `"mem": ""` in `sbatch`, `"cpus_per_task": 20`, and
`"trusted_group_parents": ["/gpfs/scratch/bsc32", "/gpfs/projects/bsc32"]`.

## Running

- Download models on `alogin4` before submitting (`aimip fetch -m ace2-era5`); jobs cannot download.
- `aimip run … --qos acc_debug --time 00:30:00` for short tests (one job at a time), the default `acc_ehpc` otherwise.
- The first job recorded `billing=40` (its 40 CPUs), not a whole node; check your project's accounting rules.

## First MN5 job (2026-11-17)

`aimip run ace2-era5 --start 1979-01-01T00:00:00 --steps 8 --name mn5-check --qos acc_debug --time 00:30:00 --submit`
then `aimip submit`: job 49144909 completed (exit 0) in 7 min 11 s on one H100 (`cuda:0`), with all outputs written.

- MN5 accepted one GPU with 20 cores (`--hint=nomultithread`, no `--mem`) on partition `acc`. Slurm gave it 40 CPUs
  (20 cores, 2 threads each) and 125 000 MB of memory, and recorded `billing=40`, not a whole node.
- The forward pass took 266 s for 8 steps (33 s per step), which is very slow for ACE2 on an H100. With so few steps
  this may be one-time start-up cost on the first step; a longer run (for example one month, 124 steps) will show
  the steady speed.

## Still to check

- The steady per-step speed of ACE2 on MN5 (see above).
- `--with-emu` (DLESyM, ArchesClimate-SSP) compiles a CUDA kernel, which needs a batch job; it is not installed yet.
