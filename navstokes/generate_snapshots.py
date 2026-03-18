import argparse
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pseudo_spectral_working import (
    initial_condition_real_space,
    omega_to_omega_hat,
    time_stepping_with_snapshots,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate omega snapshots with existing pseudo-spectral solver settings"
    )
    parser.add_argument("--output", default="omega_snapshots.npy")
    parser.add_argument("--T", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--snapshot-interval", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=256)
    args = parser.parse_args()

    np.random.seed(args.seed)
    omega0 = initial_condition_real_space(args.grid)
    omega_hat0 = omega_to_omega_hat(omega0)

    snapshots = time_stepping_with_snapshots(
        omega_hat=omega_hat0,
        dt=args.dt,
        T=args.T,
        snapshot_interval=args.snapshot_interval,
    )
    snapshots = np.asarray(snapshots, dtype=np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, snapshots)
    print(f"Saved snapshots: {output} shape={snapshots.shape}")


if __name__ == "__main__":
    main()
