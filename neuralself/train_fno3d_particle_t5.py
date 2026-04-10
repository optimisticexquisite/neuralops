import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import tqdm

from train_fno3d_navier_stokes import (
    UnitGaussianNormalizer,
    build_loader,
    count_params,
    has_complex_parameters,
    set_seed,
)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    repo_relative = Path(__file__).resolve().parent.parent / path_str
    if repo_relative.exists():
        return repo_relative
    return path


def build_time_indices(
    horizon_steps: int,
    time_stride: int,
    include_target_frame: bool,
) -> np.ndarray:
    if time_stride <= 0:
        raise ValueError("time_stride must be positive")

    last_step = horizon_steps + 1 if include_target_frame else horizon_steps
    steps = np.arange(time_stride, last_step, time_stride, dtype=np.int64)
    if include_target_frame and (steps.size == 0 or steps[-1] != horizon_steps):
        steps = np.concatenate((steps, np.array([horizon_steps], dtype=np.int64)))
    return steps


class GlobalGaussianNormalizer:
    def __init__(self, tensor: torch.Tensor, eps: float = 1e-5):
        self.mean = tensor.mean()
        self.std = tensor.std(unbiased=False)
        self.eps = eps

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + self.eps)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std + self.eps) + self.mean

    def to(self, device: torch.device) -> "GlobalGaussianNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def rotate_displacement(displacement: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return displacement
    x, y = displacement.unbind(-1)
    if k == 1:
        return torch.stack((-y, x), dim=-1)
    if k == 2:
        return torch.stack((-x, -y), dim=-1)
    if k == 3:
        return torch.stack((y, -x), dim=-1)
    raise ValueError("k must be in {0, 1, 2, 3}")


class ParticleTrajectoryFNO3DDataset(Dataset):
    def __init__(
        self,
        segment_inputs: torch.Tensor,
        initial_positions: torch.Tensor,
        targets: torch.Tensor,
        domain_length: float,
        samples: np.ndarray | None = None,
        center_particle: bool = True,
        augment_rotate: bool = False,
    ):
        self.segment_inputs = segment_inputs.contiguous()
        self.initial_positions = initial_positions.contiguous()
        self.targets = targets.contiguous()
        self.domain_length = float(domain_length)
        self.center_particle = center_particle
        self.augment_rotate = augment_rotate

        self.num_segments = self.segment_inputs.shape[0]
        self.num_particles = self.initial_positions.shape[1]
        self.size_x = self.segment_inputs.shape[1]
        self.size_y = self.segment_inputs.shape[2]
        self.dx = self.domain_length / self.size_x
        self.center = self.domain_length / 2.0

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

    def _center_volume(self, volume: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        shift_x = int(round((self.center - float(position[0])) / self.dx))
        shift_y = int(round((self.center - float(position[1])) / self.dx))
        return torch.roll(volume, shifts=(shift_y, shift_x), dims=(0, 1))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment_index = int(self.samples[index, 0])
        particle_index = int(self.samples[index, 1])
        volume = self.segment_inputs[segment_index]
        position = self.initial_positions[segment_index, particle_index]
        target = self.targets[segment_index, particle_index]

        if self.center_particle:
            volume = self._center_volume(volume, position)
        if self.augment_rotate:
            k = int(torch.randint(0, 4, (1,)).item())
            volume = torch.rot90(volume, k, dims=(0, 1))
            target = rotate_displacement(target, k)

        return volume, target


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        scale = 1.0 / (in_channels * out_channels)
        weight_shape = (in_channels, out_channels, modes1, modes2, modes3)
        self.weights1 = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.cfloat))

    @staticmethod
    def compl_mul3d(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        batch, _, size_x, size_y, size_t = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            size_x,
            size_y,
            size_t // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
            self.weights2,
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3],
            self.weights3,
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3],
            self.weights4,
        )
        x = torch.fft.irfftn(out_ft, s=(size_x, size_y, size_t), dim=(-3, -2, -1))
        return x.to(orig_dtype)


class FNO3dParticleRegressor(nn.Module):
    def __init__(
        self,
        modes1: int,
        modes2: int,
        modes3: int,
        width: int,
        padding: int = 6,
        field_channels: int = 1,
        out_dim: int = 2,
        center_patch_size: int = 9,
    ):
        super().__init__()
        self.width = width
        self.padding = padding
        self.center_patch_size = max(1, int(center_patch_size))

        self.fc0 = nn.Linear(field_channels + 3, width)
        self.conv0 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w0 = nn.Conv3d(width, width, 1)
        self.w1 = nn.Conv3d(width, width, 1)
        self.w2 = nn.Conv3d(width, width, 1)
        self.w3 = nn.Conv3d(width, width, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * width, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device, x.dtype)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        if self.padding > 0:
            x = x[..., :-self.padding]

        global_features = x.mean(dim=(2, 3, 4))
        local_patch = self.extract_center_patch(x)
        local_features = local_patch.mean(dim=(2, 3, 4))
        features = torch.cat((global_features, local_features), dim=1)
        return self.head(features)

    def extract_center_patch(self, x: torch.Tensor) -> torch.Tensor:
        size_x = x.shape[2]
        size_y = x.shape[3]
        patch_x = min(self.center_patch_size, size_x)
        patch_y = min(self.center_patch_size, size_y)
        center_x = size_x // 2
        center_y = size_y // 2
        start_x = max(0, min(center_x - patch_x // 2, size_x - patch_x))
        start_y = max(0, min(center_y - patch_y // 2, size_y - patch_y))
        return x[:, :, start_x : start_x + patch_x, start_y : start_y + patch_y, :]

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        batch, size_x, size_y, size_t = shape[:4]
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=dtype).view(1, size_x, 1, 1, 1)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=dtype).view(1, 1, size_y, 1, 1)
        gridt = torch.linspace(0, 1, size_t, device=device, dtype=dtype).view(1, 1, 1, size_t, 1)
        gridx = gridx.expand(batch, size_x, size_y, size_t, 1)
        gridy = gridy.expand(batch, size_x, size_y, size_t, 1)
        gridt = gridt.expand(batch, size_x, size_y, size_t, 1)
        return torch.cat((gridx, gridy, gridt), dim=-1)


@dataclass
class TrainConfig:
    data_path: str = "particle_dataset_prev.h5"
    output_dir: str = "neuralself/checkpoints/fno3d_particle_prev_t5"
    horizon_time: float = 5.0
    spatial_sub: int = 4
    time_stride: int = 10
    center_particle: bool = True
    include_target_frame: bool = False
    augment_rotate: bool = True
    train_fraction: float = 0.8
    split_mode: str = "sample"
    train_limit: int = 8000
    val_limit: int = 200
    modes: int = 8
    width: int = 20
    padding: int = 6
    center_patch_size: int = 9
    batch_size: int = 8
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_step: int = 3
    scheduler_gamma: float = 0.5
    num_workers: int = 4
    amp: bool = True
    compile: bool = False
    seed: int = 0
    print_every: int = 1


def build_datasets(
    cfg: TrainConfig,
) -> Tuple[ParticleTrajectoryFNO3DDataset, ParticleTrajectoryFNO3DDataset, UnitGaussianNormalizer]:
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

        start_frames = np.asarray(f["segments/start_vorticity"][:, :: cfg.spatial_sub, :: cfg.spatial_sub], dtype=np.float32)
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

        time_indices = build_time_indices(horizon_steps, cfg.time_stride, cfg.include_target_frame)
        segment_inputs = np.empty(
            (
                num_segments,
                start_frames.shape[1],
                start_frames.shape[2],
                1 + len(time_indices),
                1,
            ),
            dtype=np.float32,
        )
        segment_inputs[:, :, :, 0, 0] = start_frames
        for time_axis, step in enumerate(time_indices, start=1):
            flat_indices = segment_offsets + (step - 1)
            sampled = np.asarray(
                f["samples/vorticity_fields"][flat_indices, :: cfg.spatial_sub, :: cfg.spatial_sub],
                dtype=np.float32,
            )
            segment_inputs[:, :, :, time_axis, 0] = sampled

    segment_inputs = torch.from_numpy(segment_inputs)
    initial_positions = torch.from_numpy(initial_positions)
    target_displacement = torch.from_numpy(target_displacement)

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

    train_input_segment_ids = np.unique(train_samples[:, 0])
    a_normalizer = GlobalGaussianNormalizer(segment_inputs[train_input_segment_ids])
    y_normalizer = UnitGaussianNormalizer(target_displacement[train_samples[:, 0], train_samples[:, 1]])

    segment_inputs = a_normalizer.encode(segment_inputs)

    train_dataset = ParticleTrajectoryFNO3DDataset(
        segment_inputs=segment_inputs,
        initial_positions=initial_positions,
        targets=target_displacement,
        domain_length=domain_length,
        samples=train_samples,
        center_particle=cfg.center_particle,
        augment_rotate=cfg.augment_rotate,
    )
    test_dataset = ParticleTrajectoryFNO3DDataset(
        segment_inputs=segment_inputs,
        initial_positions=initial_positions,
        targets=target_displacement,
        domain_length=domain_length,
        samples=val_samples,
        center_particle=cfg.center_particle,
        augment_rotate=False,
    )
    return train_dataset, test_dataset, y_normalizer


def compute_zero_baseline_endpoint_l2(dataset: ParticleTrajectoryFNO3DDataset) -> float:
    sample_indices = torch.from_numpy(dataset.samples).long()
    sample_targets = dataset.targets[sample_indices[:, 0], sample_indices[:, 1]]
    return torch.linalg.vector_norm(sample_targets, dim=1).mean().item()


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
    scaler: torch.cuda.amp.GradScaler | None,
    normalizer: UnitGaussianNormalizer,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "endpoint_l2": 0.0}
    num_samples = 0

    for inputs, targets in tqdm.tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        normalized_targets = normalizer.encode(targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(inputs)
            loss = F.mse_loss(preds, normalized_targets)
            decoded_preds = normalizer.decode(preds)
            decoded_targets = targets
            metrics = compute_regression_metrics(decoded_preds, decoded_targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

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
            decoded_preds = normalizer.decode(preds)
            decoded_targets = targets
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
    scaler: torch.cuda.amp.GradScaler | None,
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
    train_dataset, val_dataset, y_normalizer = build_datasets(cfg)
    train_loader = build_loader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = build_loader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    prep_time = time.perf_counter() - t0

    sample_input = train_dataset[0][0]
    modes3 = min(cfg.modes, sample_input.shape[2] // 2 + 1)
    model = FNO3dParticleRegressor(
        modes1=cfg.modes,
        modes2=cfg.modes,
        modes3=modes3,
        width=cfg.width,
        padding=cfg.padding,
        field_channels=sample_input.shape[-1],
        out_dim=2,
        center_patch_size=cfg.center_patch_size,
    ).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_gamma,
        patience=cfg.scheduler_step,
    )
    use_grad_scaler = amp_enabled and not has_complex_parameters(model)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler) if device.type == "cuda" else None
    y_normalizer = y_normalizer.to(device)

    print(f"Using device: {device}")
    print(f"Preprocessing finished in {prep_time:.2f}s")
    print(f"Model params: {count_params(model):,}")
    if amp_enabled and not use_grad_scaler:
        print("AMP is enabled without GradScaler because the model has complex-valued spectral weights.")
    print(f"Split mode: {cfg.split_mode}")
    print(
        "Volume shape:",
        tuple(train_dataset.segment_inputs.shape),
        "Target tensor shape:",
        tuple(train_dataset.targets.shape),
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Zero-baseline val endpoint L2: {compute_zero_baseline_endpoint_l2(val_dataset):.6f}")

    best_val = math.inf
    output_dir = resolve_path(cfg.output_dir)

    for epoch in tqdm.tqdm(range(1, cfg.epochs + 1), desc="Epochs"):
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
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            normalizer=y_normalizer,
            device=device,
            amp_enabled=amp_enabled,
        )
        scheduler.step(val_metrics["endpoint_l2"])
        elapsed = time.perf_counter() - epoch_start

        metrics = {
            "epoch_time_s": elapsed,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_endpoint_l2": train_metrics["endpoint_l2"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_endpoint_l2": val_metrics["endpoint_l2"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        if val_metrics["endpoint_l2"] < best_val:
            best_val = val_metrics["endpoint_l2"]
            save_checkpoint(output_dir, model, optimizer, scheduler, scaler, cfg, epoch, metrics, y_normalizer)

        if epoch % cfg.print_every == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"time {elapsed:6.2f}s | "
                f"train_loss {metrics['train_loss']:.6e} | "
                f"train_endpoint_l2 {metrics['train_endpoint_l2']:.6f} | "
                f"val_endpoint_l2 {metrics['val_endpoint_l2']:.6f}"
            )

    print(f"Best val endpoint L2: {best_val:.6f}")
    print(f"Checkpoint saved to: {output_dir / 'best.pt'}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train an FNO3D regressor for particle displacement at T=5.")
    parser.add_argument("--data-path", type=str, default=TrainConfig.data_path)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--horizon-time", type=float, default=TrainConfig.horizon_time)
    parser.add_argument("--spatial-sub", type=int, default=TrainConfig.spatial_sub)
    parser.add_argument("--time-stride", type=int, default=TrainConfig.time_stride)
    parser.add_argument("--train-fraction", type=float, default=TrainConfig.train_fraction)
    parser.add_argument("--split-mode", type=str, default=TrainConfig.split_mode)
    parser.add_argument("--train-limit", type=int, default=TrainConfig.train_limit)
    parser.add_argument("--val-limit", type=int, default=TrainConfig.val_limit)
    parser.add_argument("--modes", type=int, default=TrainConfig.modes)
    parser.add_argument("--width", type=int, default=TrainConfig.width)
    parser.add_argument("--padding", type=int, default=TrainConfig.padding)
    parser.add_argument("--center-patch-size", type=int, default=TrainConfig.center_patch_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--scheduler-step", type=int, default=TrainConfig.scheduler_step)
    parser.add_argument("--scheduler-gamma", type=float, default=TrainConfig.scheduler_gamma)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--print-every", type=int, default=TrainConfig.print_every)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=TrainConfig.amp)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=TrainConfig.compile)
    parser.add_argument(
        "--center-particle",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.center_particle,
    )
    parser.add_argument(
        "--include-target-frame",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.include_target_frame,
    )
    parser.add_argument(
        "--augment-rotate",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.augment_rotate,
    )
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        horizon_time=args.horizon_time,
        spatial_sub=args.spatial_sub,
        time_stride=args.time_stride,
        center_particle=args.center_particle,
        include_target_frame=args.include_target_frame,
        augment_rotate=args.augment_rotate,
        train_fraction=args.train_fraction,
        split_mode=args.split_mode,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        modes=args.modes,
        width=args.width,
        padding=args.padding,
        center_patch_size=args.center_patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        num_workers=args.num_workers,
        amp=args.amp,
        compile=args.compile,
        seed=args.seed,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    train(parse_args())
