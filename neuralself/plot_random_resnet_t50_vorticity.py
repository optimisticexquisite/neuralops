import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from train_resnet50_t50_vorticity import (
    ResNet50FieldDecoder,
    TrainConfig,
    build_datasets,
    set_seed,
)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = (Path.cwd() / path).resolve()
    if candidate.exists():
        return candidate
    return (base_dir / path).resolve()


def load_checkpoint_and_config(checkpoint_path: Path, cli_args: argparse.Namespace) -> tuple[dict, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    script_dir = Path(__file__).resolve().parent

    if cli_args.data_path:
        cfg.data_path = str(resolve_path(cli_args.data_path, script_dir))
    else:
        cfg.data_path = str(resolve_path(cfg.data_path, script_dir))

    if cli_args.ntrain is not None:
        cfg.ntrain = cli_args.ntrain
    if cli_args.ntest is not None:
        cfg.ntest = cli_args.ntest
    if cli_args.target_index is not None:
        cfg.target_index = cli_args.target_index

    return checkpoint, cfg


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a random T50 comparison for the ResNet vorticity model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="neuralself/checkpoints/resnet50_t50_ns64/best.pt",
        help="Path to the trained ResNet checkpoint.",
    )
    parser.add_argument("--data-path", type=str, default="", help="Optional override for the Navier-Stokes .mat file.")
    parser.add_argument("--ntrain", type=int, default=None, help="Override ntrain from checkpoint config.")
    parser.add_argument("--ntest", type=int, default=None, help="Override ntest from checkpoint config.")
    parser.add_argument("--target-index", type=int, default=None, help="Override target index from checkpoint config.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for random test-sample selection.")
    parser.add_argument("--sample-index", type=int, default=None, help="Optional explicit test index.")
    parser.add_argument(
        "--output",
        type=str,
        default="neuralself/random_resnet_t50_vorticity.png",
        help="Where to save the figure.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint, Path(__file__).resolve().parent)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint, cfg = load_checkpoint_and_config(checkpoint_path, args)
    set_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_dataset, y_normalizer = build_datasets(cfg)
    y_normalizer = y_normalizer.to(device)

    model = ResNet50FieldDecoder(
        in_channels=3 * cfg.t_in,
        out_size=test_dataset.targets.shape[-1],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if len(test_dataset) == 0:
        raise ValueError("The test dataset is empty.")

    sample_index = args.sample_index
    if sample_index is None:
        sample_index = random.randrange(len(test_dataset))
    if not (0 <= sample_index < len(test_dataset)):
        raise IndexError(f"sample-index {sample_index} out of range for test set size {len(test_dataset)}")

    inputs, target = test_dataset[sample_index]
    inputs = inputs.unsqueeze(0).to(device)
    prediction = model(inputs)
    prediction = y_normalizer.decode(prediction).squeeze(0).cpu()
    target = target.cpu()
    abs_err = (prediction - target).abs()

    vmin = min(prediction.min().item(), target.min().item())
    vmax = max(prediction.max().item(), target.max().item())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(target.numpy(), cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth T50")
    axes[1].imshow(prediction.numpy(), cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted T50")
    im2 = axes[2].imshow(abs_err.numpy(), cmap="jet", origin="lower")
    axes[2].set_title("Absolute Error")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar0 = fig.colorbar(im0, ax=axes[:2], shrink=0.9)
    cbar0.set_label("Vorticity")
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.9)
    cbar2.set_label("|Error|")
    fig.suptitle(f"ResNet random test sample {sample_index} at T=50")

    output_path = resolve_path(args.output, Path(__file__).resolve().parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    rel_l2 = torch.linalg.vector_norm((prediction - target).reshape(-1), ord=2) / (
        torch.linalg.vector_norm(target.reshape(-1), ord=2) + 1e-12
    )
    print(f"Saved figure to: {output_path}")
    print(f"Test sample index: {sample_index}")
    print(f"T50 frame shape: {tuple(prediction.shape)}")
    print(f"T50 relative L2: {rel_l2.item():.6f}")


if __name__ == "__main__":
    main()
