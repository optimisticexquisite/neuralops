import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot T=0 vorticity for 50 random samples from the Navier-Stokes HDF5 file.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="neuralself/data/NavierStokes_V1e-3_N5000_T50.mat",
        help="Path to the HDF5-backed .mat file.",
    )
    parser.add_argument("--num-samples", type=int, default=50, help="How many random samples to plot.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="neuralself/random_t0_50_samples.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data_path = Path(args.data_path).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as f:
        dataset = f["u"]
        if dataset.ndim != 4:
            raise ValueError(f"Expected 4D dataset, got shape {dataset.shape}")

        nt, nx, ny, nsamples = dataset.shape
        if args.num_samples > nsamples:
            raise ValueError(f"Requested {args.num_samples} samples, but dataset only has {nsamples}")

        indices = rng.choice(nsamples, size=args.num_samples, replace=False)
        sorted_order = np.argsort(indices)
        sorted_indices = indices[sorted_order]
        t0_fields = dataset[0, :, :, sorted_indices]
        t0_fields = np.moveaxis(t0_fields, -1, 0)

        inverse_order = np.empty_like(sorted_order)
        inverse_order[sorted_order] = np.arange(len(sorted_order))
        t0_fields = t0_fields[inverse_order]

    cols = 10
    rows = int(np.ceil(args.num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    vmin = float(t0_fields.min())
    vmax = float(t0_fields.max())
    image = None

    for ax, sample_idx, field in zip(axes, indices, t0_fields):
        image = ax.imshow(field, cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"idx={sample_idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(indices) :]:
        ax.axis("off")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.92)
        cbar.set_label("Vorticity")

    fig.suptitle("Random T=0 vorticity samples", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Dataset shape: {(nt, nx, ny, nsamples)}")
    print(f"Saved figure to: {output_path}")
    print(f"Random sample indices: {indices.tolist()}")


if __name__ == "__main__":
    main()
