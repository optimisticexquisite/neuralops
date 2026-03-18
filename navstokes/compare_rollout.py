import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from fno2d import FNO2d


@torch.no_grad()
def rollout_neural(model, omega0, steps, stride, mean, std, device):
    model.eval()
    curr = omega0.copy()
    preds = [curr.copy()]
    for _ in range(steps):
        x = ((curr - mean) / std)[None, None]
        x = torch.from_numpy(x).float().to(device)
        y = model(x).cpu().numpy()[0, 0]
        curr = y * std + mean
        preds.append(curr.copy())
    return np.array(preds)


def build_figure(pseudo_seq, pred_seq, sample_every):
    ncols = pseudo_seq.shape[0]
    err_seq = np.abs(pred_seq - pseudo_seq)

    vmax = np.max(np.abs(pseudo_seq))
    emax = np.max(err_seq)

    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 9), constrained_layout=True)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(ncols):
        t_idx = i * sample_every

        im0 = axes[0, i].imshow(pseudo_seq[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(f"Pseudo-spectral\nt+{t_idx}")
        axes[0, i].axis("off")

        im1 = axes[1, i].imshow(pred_seq[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, i].set_title(f"FNO rollout\nt+{t_idx}")
        axes[1, i].axis("off")

        im2 = axes[2, i].imshow(err_seq[i], cmap="magma", vmin=0.0, vmax=emax)
        axes[2, i].set_title(f"|Error|\nt+{t_idx}")
        axes[2, i].axis("off")

    fig.colorbar(im0, ax=axes[0, :].ravel().tolist(), fraction=0.02, pad=0.01)
    fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), fraction=0.02, pad=0.01)
    fig.colorbar(im2, ax=axes[2, :].ravel().tolist(), fraction=0.02, pad=0.01)
    return fig


def save_animation(pseudo, pred, out_path, interval_ms=300):
    err = np.abs(pred - pseudo)
    vmax = np.max(np.abs(pseudo))
    emax = np.max(err)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    ims = [
        axes[0].imshow(pseudo[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax),
        axes[1].imshow(pred[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax),
        axes[2].imshow(err[0], cmap="magma", vmin=0.0, vmax=emax),
    ]
    titles = ["Pseudo-spectral", "FNO rollout", "Absolute error"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")

    def update(frame):
        ims[0].set_data(pseudo[frame])
        ims[1].set_data(pred[frame])
        ims[2].set_data(np.abs(pred[frame] - pseudo[frame]))
        fig.suptitle(f"Rollout frame {frame}")
        return ims

    anim = FuncAnimation(fig, update, frames=len(pseudo), interval=interval_ms, blit=False)
    anim.save(out_path, writer="pillow", fps=max(1, 1000 // interval_ms))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visual compare pseudo-spectral and FNO rollouts")
    parser.add_argument("--checkpoint", default="navstokes/fno_navstokes.pt")
    parser.add_argument("--data-path", default="omega_snapshots.npy")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--output-dir", default="navstokes/artifacts")
    parser.add_argument("--save-gif", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    snapshots = np.load(args.data_path).astype(np.float32)
    if snapshots.ndim == 4 and snapshots.shape[0] == 1:
        snapshots = snapshots[0]

    stride = int(cfg["stride"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d(modes1=cfg["modes"], modes2=cfg["modes"], width=cfg["width"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    start = args.start_index
    total_jump = args.rollout_steps * stride
    if start + total_jump >= snapshots.shape[0]:
        raise ValueError("Rollout exceeds snapshot range. Reduce --rollout-steps or start index.")

    omega0 = snapshots[start]
    pseudo = snapshots[start : start + total_jump + 1 : stride]
    pred = rollout_neural(
        model,
        omega0,
        steps=args.rollout_steps,
        stride=stride,
        mean=float(ckpt["mean"]),
        std=float(ckpt["std"]),
        device=device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure(
        pseudo_seq=pseudo[:: args.sample_every],
        pred_seq=pred[:: args.sample_every],
        sample_every=args.sample_every * stride,
    )
    comparison_png = out_dir / "rollout_comparison.png"
    fig.savefig(comparison_png, dpi=180)
    plt.close(fig)
    print(f"Saved image: {comparison_png}")

    mse_by_step = ((pred - pseudo) ** 2).mean(axis=(1, 2))
    np.save(out_dir / "rollout_mse.npy", mse_by_step)
    print(f"Saved per-step MSE curve: {out_dir / 'rollout_mse.npy'}")

    if args.save_gif:
        gif_path = out_dir / "rollout_comparison.gif"
        save_animation(pseudo, pred, gif_path)
        print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()
