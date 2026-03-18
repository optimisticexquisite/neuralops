import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def load_snapshots(path: str) -> np.ndarray:
    snapshots = np.load(path)
    if snapshots.ndim == 4 and snapshots.shape[0] == 1:
        snapshots = snapshots[0]
    if snapshots.ndim != 3:
        raise ValueError(f"Expected snapshots shape (T,H,W), got {snapshots.shape}")
    return snapshots.astype(np.float32)


def build_animation(
    snapshots: np.ndarray,
    interval_ms: int,
    title_prefix: str,
):
    vmin = float(np.min(snapshots))
    vmax = float(np.max(snapshots))

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image = ax.imshow(snapshots[0], cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title_prefix} 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Vorticity")

    def update(frame_idx: int):
        image.set_data(snapshots[frame_idx])
        ax.set_title(f"{title_prefix} {frame_idx}")
        return (image,)

    animation = FuncAnimation(
        fig,
        update,
        frames=snapshots.shape[0],
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    return fig, animation


def main():
    parser = argparse.ArgumentParser(description="Animate Navier-Stokes vorticity snapshots")
    parser.add_argument("--data-path", default="navstokes/omega_snapshots.npy")
    parser.add_argument("--start", type=int, default=0, help="First frame to include")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame (exclusive)")
    parser.add_argument("--step", type=int, default=1, help="Frame subsampling step")
    parser.add_argument("--interval", type=int, default=40, help="Milliseconds between frames")
    parser.add_argument("--save-gif", default=None, help="Optional path to save a GIF")
    parser.add_argument("--title-prefix", default="Omega frame", help="Title prefix for each frame")
    args = parser.parse_args()

    snapshots = load_snapshots(args.data_path)
    frames = snapshots[args.start : args.stop : args.step]
    if frames.size == 0:
        raise ValueError("No frames selected. Check --start, --stop, and --step.")

    fig, animation = build_animation(
        snapshots=frames,
        interval_ms=args.interval,
        title_prefix=args.title_prefix,
    )

    if args.save_gif:
        output = Path(args.save_gif)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, round(1000 / args.interval))
        animation.save(output, writer="pillow", fps=fps)
        print(f"Saved GIF: {output}")

    plt.show()


if __name__ == "__main__":
    main()
