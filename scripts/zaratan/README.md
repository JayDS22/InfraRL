# Running InfraRL on UMD Zaratan

Small bootstrap for running the full 3-seed experiment on Zaratan instead of a laptop. Zaratan's system Python is 3.10; our pins need 3.11+, so we use a user-space conda env.

## One-time setup on a Zaratan login node

```bash
ssh <your-umd-directory-id>@login.zaratan.umd.edu
git clone https://github.com/JayDS22/InfraRL.git && cd InfraRL
bash scripts/zaratan/setup.sh
```

The script creates a `conda` env named `infrarl` at Python 3.11, installs `requirements.txt`, and smoke-runs the test suite + verification gates. Takes ~10 minutes.

## Submit a run

```bash
sbatch scripts/zaratan/run_experiment.sbatch
squeue -u $USER         # check queue state
tail -f logs/infrarl-full-<jobid>.out
```

Results land in `results/` and figures in `results/figures/`. The default `.sbatch` requests `--cpus-per-task=4, --mem=8G, --time=02:00:00` on the `standard` partition. Tune those if you need more.

## Local-dev workflow (optional)

Edit on your laptop, sync to Zaratan, run there, pull results back:

```bash
export ZARATAN_USER=<your-directory-id>
bash scripts/zaratan/sync.sh push     # local -> remote
# then ssh in and sbatch
bash scripts/zaratan/sync.sh pull     # remote results/ -> local results/
```

`sync.sh` excludes `.git`, caches, `results/`, `logs/`, and `.claude/`. The `--delete` flag on push means Zaratan mirrors local exactly, so remove orphan files there.

## Why conda, not module load

Zaratan's `module load python/3.11` may or may not exist on your allocation, and the system Python is 3.10 which breaks the pins. A user-space conda env is the one-liner that always works, with no coordination with HPC ops.
