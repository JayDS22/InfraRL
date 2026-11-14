# CPU experiments on Zaratan

Run from an existing Git checkout using its `.venv`, an activated Conda environment,
or an explicit `INFRARL_PYTHON` path. No environment installation is performed.
The human authenticates, transfers files and approves all resource requests.

Before submitting, inspect the checkout/environment and create the log directory:

```bash
git status --short
bash scripts/zaratan/setup.sh inspect
mkdir -p results/logs
```

After explicit approval of the account and CPU/memory/time settings, submit one
phase at a time with the shared batch script. Its starting request is standard,
4 CPUs, 8G RAM, two hours, no GPUs; override limits only after review.

```bash
export INFRARL_RESOURCES_APPROVED=yes
sbatch --account=YOUR_ACCOUNT scripts/zaratan/run_experiment.sbatch checks
sbatch --account=YOUR_ACCOUNT scripts/zaratan/run_experiment.sbatch calibration
sbatch --account=YOUR_ACCOUNT scripts/zaratan/run_experiment.sbatch functional-smoke --provisional
sbatch --account=YOUR_ACCOUNT scripts/zaratan/run_experiment.sbatch ppo-update-smoke
sbatch --account=YOUR_ACCOUNT scripts/zaratan/run_experiment.sbatch development --provisional
# After freezing inventory/hyperparameters and acknowledging calibration limits:
sbatch --account=YOUR_ACCOUNT scripts/zaratan/run_experiment.sbatch frozen --hyperparameters-frozen --provisional
sbatch --account=YOUR_ACCOUNT scripts/zaratan/run_experiment.sbatch ablations --hyperparameters-frozen --provisional
```

Every sbatch command submits a job. Never run tests, exhaustive verification,
calibration or policy training directly on login nodes. Review each result before
proceeding; incomplete calibration is distinct from successful program execution.
Options include `--agents`, `--seeds`, `--epochs`, and `--ablate`.

Results/configuration/source hashes go to `results/PHASE_JOBID/artifacts/`; Slurm
logs go to `results/logs/infrarl-JOBID.out/.err`. Frozen runs also archive source.
Do not edit source while jobs are queued/running. Match source and training budgets
across frozen comparisons and ablations. Inspect `squeue`/`sacct` for COMPLETED/0:0;
then use `workflow.py validate --out ARTIFACTS --sacct ACCOUNTING_FILE` after export.

`checkout.sh` provides safe clean-tree/fast-forward update checks and metadata
inspection. `sync_safe.sh checkout --dry-run` previews a reviewed transfer list;
actual overlays preserve prior files in a separately configured backup directory.
It never uses destructive synchronization. `pull` retrieves into a fresh directory.
Legacy snapshot helpers remain optional and are exercised by the tests.
See [the evaluation protocol](../../docs/EVALUATION.md) for scientific gates.
