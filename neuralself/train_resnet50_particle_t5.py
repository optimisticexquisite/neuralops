import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from matplotlib.colors import Normalize
from torch.utils.data import Dataset
from tqdm import tqdm

from train_fno3d_particle_t5 import resolve_path
from train_fno3d_navier_stokes import (
    UnitGaussianNormalizer,
    build_loader,
    count_params,
    set_seed,
)


class JetRGBConverter:
    def __init__(self):
        self.cmap = matplotlib.colormaps.get_cmap("jet")

    def __call__(self, field: np.ndarray) -> np.ndarray:
        norm = Normalize(vmin=float(field.min()), vmax=float(field.max()), clip=True)
        rgba = self.cmap(norm(field))
        rgb = np.transpose(rgba[..., :3], (2, 0, 1)).astype(np.float32)
        return rgb


def rotate_displacement(target: torch.Tensor, k: int) -> torch.Tensor:
    dx = float(target[0])
    dy = float(target[1])
    if k == 0:
        rotated = [dx, dy]
    elif k == 1:
        rotated = [-dy, dx]
    elif k == 2:
        rotated = [-dx, -dy]
    elif k == 3:
        rotated = [dy, -dx]
    else:
        raise ValueError("k must be in {0,1,2,3}")
    return torch.tensor(rotated, dtype=torch.float32)


class ParticleRGBDataset(Dataset):
    def __init__(
        self,
        segment_fields: np.ndarray,
        initial_positions: np.ndarray,
        targets: torch.Tensor,
        domain_length: float,
        transform: T.Compose | None,
        samples: np.ndarray | None = None,
        augment: bool = False,
        center_particle: bool = True,
    ):
        self.segment_fields = segment_fields
        self.initial_positions = initial_positions
        self.targets = targets.contiguous()
        self.domain_length = float(domain_length)
        self.transform = transform
        self.augment = augment
        self.center_particle = center_particle
        self.converter = JetRGBConverter()

        self.num_segments = self.segment_fields.shape[0]
        self.num_particles = self.initial_positions.shape[1]
        self.grid_size = self.segment_fields.shape[1]
        self.dx = self.domain_length / self.grid_size

        if samples is None:
            self.samples = np.asarray(
                [
                    (segment_index, particle_index)
                    for segment_index in range(self.num_segments)
                    for particle_index in range(self.num_particles)
                ],
                dtype=np.int64,
            )
        else:
            self.samples = np.asarray(samples, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def _center_field(self, field: np.ndarray, position: np.ndarray) -> np.ndarray:
        center = self.domain_length / 2.0
        shift_x = int(round((center - float(position[0])) / self.dx))
        shift_y = int(round((center - float(position[1])) / self.dx))
        return np.roll(field, shift=(shift_y, shift_x), axis=(0, 1))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment_index = int(self.samples[index, 0])
        particle_index = int(self.samples[index, 1])
        field = self.segment_fields[segment_index]
        position = self.initial_positions[segment_index, particle_index]
        target = self.targets[segment_index, particle_index]

        if self.center_particle:
            field = self._center_field(field, position)

        rgb = self.converter(field)
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb_uint8 = (255.0 * rgb).clip(0, 255).astype(np.uint8)
        image = T.ToPILImage()(rgb_uint8)
        tensor = self.transform(image) if self.transform is not None else T.ToTensor()(image)

        if self.augment:
            k = np.random.randint(0, 4)
            tensor = torch.rot90(tensor, k, dims=[1, 2])
            target = rotate_displacement(target, k)

        return tensor, target


class ResNet50ParticleRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            backbone = torchvision.models.resnet50(weights=weights)
        except Exception:
            backbone = torchvision.models.resnet50(weights=None)

        self.backbone = backbone
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@dataclass
class TrainConfig:
    data_path: str = "particle_dataset_prev.h5"
    output_dir: str = "neuralself/checkpoints/resnet50_particle_prev_t5"
    horizon_time: float = 5.0
    spatial_sub: int = 1
    train_fraction: float = 0.8
    split_mode: str = "sample"
    train_limit: int = 8000
    val_limit: int = 200
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    compile: bool = False
    pretrained: bool = True
    augment: bool = True
    center_particle: bool = True
    seed: int = 0
    print_every: int = 1


def build_datasets(
    cfg: TrainConfig,
) -> Tuple[ParticleRGBDataset, ParticleRGBDataset, UnitGaussianNormalizer]:
    data_path = resolve_path(cfg.data_path)
    with h5py.File(data_path, "r") as f:
        domain_length = float(f.attrs["domain_length"])
        sample_interval = float(f.attrs["sample_interval"])
        segment_duration = float(f.attrs["segment_duration"])
        num_segments = int(f["segments/start_time"].shape[0])

        frames_per_segment = int(round(segment_duration / sample_interval))
        horizon_steps = int(round(cfg.horizon_time / sample_interval))
        if horizon_steps <= 0 or horizon_steps > frames_per_segment:
            raise ValueError("horizon_time must be within one segment")

        segment_fields = np.asarray(
            f["segments/start_vorticity"][:, :: cfg.spatial_sub, :: cfg.spatial_sub],
            dtype=np.float32,
        )
        initial_positions = np.asarray(f["segments/initial_wrapped_positions"][:], dtype=np.float32)
        initial_unwrapped = np.asarray(f["segments/initial_unwrapped_positions"][:], dtype=np.float32)

        per_sample_segment_ids = np.asarray(f["samples/segment_id"][:], dtype=np.int64)
        _, counts = np.unique(per_sample_segment_ids, return_counts=True)
        if not np.all(counts == counts[0]):
            raise ValueError("Expected the same number of samples per segment")
        segment_offsets = np.concatenate(([0], np.cumsum(counts[:-1])))

        target_indices = segment_offsets + (horizon_steps - 1)
        target_unwrapped = np.asarray(f["samples/unwrapped_positions"][target_indices], dtype=np.float32)
        target_displacement = target_unwrapped - initial_unwrapped

    target_displacement_t = torch.from_numpy(target_displacement)
    all_samples = np.asarray(
        [
            (segment_index, particle_index)
            for segment_index in range(num_segments)
            for particle_index in range(initial_positions.shape[1])
        ],
        dtype=np.int64,
    )

    rng = np.random.default_rng(cfg.seed)
    if cfg.split_mode == "segment":
        shuffled_segments = np.arange(num_segments, dtype=np.int64)
        rng.shuffle(shuffled_segments)
        ntrain_segments = max(1, min(num_segments - 1, int(round(cfg.train_fraction * num_segments))))
        train_segment_ids = np.sort(shuffled_segments[:ntrain_segments])
        val_segment_ids = np.sort(shuffled_segments[ntrain_segments:])
        train_samples = all_samples[np.isin(all_samples[:, 0], train_segment_ids)]
        val_samples = all_samples[np.isin(all_samples[:, 0], val_segment_ids)]
    elif cfg.split_mode == "sample":
        num_total = all_samples.shape[0]
        num_val = max(1, int(round((1.0 - cfg.train_fraction) * num_total)))
        num_train = max(1, num_total - num_val)
        permutation = rng.permutation(num_total)
        train_cutoff = min(num_train, cfg.train_limit if cfg.train_limit > 0 else num_train)
        val_count = min(num_val, cfg.val_limit if cfg.val_limit > 0 else num_val)
        train_samples = all_samples[permutation[:train_cutoff]]
        val_start = num_train
        val_samples = all_samples[permutation[val_start : val_start + val_count]]
        if val_samples.shape[0] == 0:
            fallback_start = train_cutoff
            fallback_stop = min(num_total, fallback_start + max(1, val_count))
            val_samples = all_samples[permutation[fallback_start:fallback_stop]]
    else:
        raise ValueError("split_mode must be 'sample' or 'segment'")

    if cfg.split_mode == "segment":
        if cfg.train_limit > 0 and train_samples.shape[0] > cfg.train_limit:
            train_samples = train_samples[rng.permutation(train_samples.shape[0])[: cfg.train_limit]]
        if cfg.val_limit > 0 and val_samples.shape[0] > cfg.val_limit:
            val_samples = val_samples[rng.permutation(val_samples.shape[0])[: cfg.val_limit]]

    y_normalizer = UnitGaussianNormalizer(
        target_displacement_t[train_samples[:, 0], train_samples[:, 1]]
    )

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = ParticleRGBDataset(
        segment_fields=segment_fields,
        initial_positions=initial_positions,
        targets=target_displacement_t,
        domain_length=domain_length,
        transform=transform,
        samples=train_samples,
        augment=cfg.augment,
        center_particle=cfg.center_particle,
    )
    test_dataset = ParticleRGBDataset(
        segment_fields=segment_fields,
        initial_positions=initial_positions,
        targets=target_displacement_t,
        domain_length=domain_length,
        transform=transform,
        samples=val_samples,
        augment=False,
        center_particle=cfg.center_particle,
    )
    return train_dataset, test_dataset, y_normalizer


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    diff = preds - targets
    mse = F.mse_loss(preds, targets)
    mae = diff.abs().mean()
    endpoint_l2 = torch.linalg.vector_norm(diff, dim=1).mean()
    return {
        "mse": mse,
        "mae": mae,
        "endpoint_l2": endpoint_l2,
    }


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    normalizer: UnitGaussianNormalizer,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "endpoint_l2": 0.0}
    num_samples = 0

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        normalized_targets = normalizer.encode(targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(inputs)
            loss = F.mse_loss(preds, normalized_targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            decoded_preds = normalizer.decode(preds.float())
            decoded_targets = targets.float()
            metrics = compute_regression_metrics(decoded_preds, decoded_targets)

        batch = inputs.shape[0]
        totals["loss"] += loss.item() * batch
        totals["mse"] += metrics["mse"].item() * batch
        totals["mae"] += metrics["mae"].item() * batch
        totals["endpoint_l2"] += metrics["endpoint_l2"].item() * batch
        num_samples += batch

    return {key: value / max(num_samples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    normalizer: UnitGaussianNormalizer,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    totals = {"mse": 0.0, "mae": 0.0, "endpoint_l2": 0.0}
    num_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(inputs)

        decoded_preds = normalizer.decode(preds.float())
        decoded_targets = targets.float()
        metrics = compute_regression_metrics(decoded_preds, decoded_targets)

        batch = inputs.shape[0]
        totals["mse"] += metrics["mse"].item() * batch
        totals["mae"] += metrics["mae"].item() * batch
        totals["endpoint_l2"] += metrics["endpoint_l2"].item() * batch
        num_samples += batch

    return {key: value / max(num_samples, 1) for key, value in totals.items()}


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler | None,
    cfg: TrainConfig,
    epoch: int,
    metrics: Dict[str, float],
    target_normalizer: UnitGaussianNormalizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "metrics": metrics,
        "target_normalizer_mean": target_normalizer.mean.detach().cpu(),
        "target_normalizer_std": target_normalizer.std.detach().cpu(),
        "target_normalizer_eps": target_normalizer.eps,
    }
    torch.save(state, output_dir / "best.pt")


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg.amp and device.type == "cuda"

    t0 = time.perf_counter()
    train_dataset, test_dataset, y_normalizer = build_datasets(cfg)
    train_loader = build_loader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = build_loader(test_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    prep_time = time.perf_counter() - t0

    model = ResNet50ParticleRegressor(pretrained=cfg.pretrained).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled) if device.type == "cuda" else None
    y_normalizer = y_normalizer.to(device)

    print(f"Using device: {device}")
    print(f"Preprocessing finished in {prep_time:.2f}s")
    print(f"Model params: {count_params(model):,}")
    print(f"Split mode: {cfg.split_mode}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(test_dataset)}")

    best_test = math.inf
    output_dir = resolve_path(cfg.output_dir)

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Epochs"):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            normalizer=y_normalizer,
            device=device,
            amp_enabled=amp_enabled,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            normalizer=y_normalizer,
            device=device,
            amp_enabled=amp_enabled,
        )
        scheduler.step(test_metrics["endpoint_l2"])
        elapsed = time.perf_counter() - epoch_start

        metrics = {
            "epoch_time_s": elapsed,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_endpoint_l2": train_metrics["endpoint_l2"],
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_endpoint_l2": test_metrics["endpoint_l2"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        if test_metrics["endpoint_l2"] < best_test:
            best_test = test_metrics["endpoint_l2"]
            save_checkpoint(output_dir, model, optimizer, scheduler, scaler, cfg, epoch, metrics, y_normalizer)

        if epoch % cfg.print_every == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"time {elapsed:6.2f}s | "
                f"train_loss {metrics['train_loss']:.6e} | "
                f"train_endpoint_l2 {metrics['train_endpoint_l2']:.6f} | "
                f"test_endpoint_l2 {metrics['test_endpoint_l2']:.6f}"
            )

    print(f"Best test endpoint L2: {best_test:.6f}")
    print(f"Checkpoint saved to: {output_dir / 'best.pt'}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ResNet50 on centered RGB vorticity for particle displacement at T=5.")
    parser.add_argument("--data-path", type=str, default=TrainConfig.data_path)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--horizon-time", type=float, default=TrainConfig.horizon_time)
    parser.add_argument("--spatial-sub", type=int, default=TrainConfig.spatial_sub)
    parser.add_argument("--train-fraction", type=float, default=TrainConfig.train_fraction)
    parser.add_argument("--split-mode", type=str, default=TrainConfig.split_mode)
    parser.add_argument("--train-limit", type=int, default=TrainConfig.train_limit)
    parser.add_argument("--val-limit", type=int, default=TrainConfig.val_limit)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--print-every", type=int, default=TrainConfig.print_every)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=TrainConfig.amp)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=TrainConfig.compile)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=TrainConfig.pretrained)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=TrainConfig.augment)
    parser.add_argument("--center-particle", action=argparse.BooleanOptionalAction, default=TrainConfig.center_particle)
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        horizon_time=args.horizon_time,
        spatial_sub=args.spatial_sub,
        train_fraction=args.train_fraction,
        split_mode=args.split_mode,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=args.amp,
        compile=args.compile,
        pretrained=args.pretrained,
        augment=args.augment,
        center_particle=args.center_particle,
        seed=args.seed,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    train(parse_args())
