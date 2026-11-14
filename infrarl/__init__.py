"""InfraRL helpers shared by compute entry points."""


def require_compute_allocation():
    """Allow local WSL work; reject direct compute entry on known cluster login hosts."""
    import os
    import socket
    host = socket.gethostname().lower()
    on_cluster = (host.startswith('login') or host.endswith('.zaratan.umd.edu')
                  or os.environ.get('INFRARL_ON_CLUSTER') == '1')
    if on_cluster and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute workloads require a Slurm allocation on Zaratan; do not run on login nodes')
