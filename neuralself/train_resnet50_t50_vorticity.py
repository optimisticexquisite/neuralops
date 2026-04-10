import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MatReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = None
        self.old_mat = True
        self._load_file()

    def _load_file(self) -> None:
        try:
            self.data = scipy.io.loadmat(self.path)
            self.old_mat = True
        except NotImplementedError:
            if h5py is None:
                raise ImportError(
                    "This .mat file requires h5py (MATLAB v7.3/HDF5 format). "
                    "Install h5py or use a non-v7.3 file."
                )
            self.data = h5py.File(self.path, "r")
            self.old_mat = False

    def read_field(self, field: str) -> np.ndarray:
        array = self.data[field]
        if self.old_mat:
            return np.asarray(array)
        array = np.asarray(array)
        axes = tuple(range(array.ndim - 1, -1, -1))
        return np.transpose(array, axes=axes)


class UnitGaussianNormalizer:
    def __init__(self, tensor: torch.Tensor, eps: float = 1e-5):
        self.mean = tensor.mean(dim=0, keepdim=True)
        self.std = tensor.std(dim=0, keepdim=True, unbiased=False)
        self.eps = eps

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + self.eps)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std + self.eps) + self.mean

    def to(self, device: torch.device) -> "UnitGaussianNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class LpLoss:
    def __init__(self, p: int = 2, reduction: str = "mean", eps: float = 1e-12):
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = pred.shape[0]
        pred = pred.reshape(batch, -1)
        target = target.reshape(batch, -1)
        diff_norm = torch.linalg.vector_norm(pred - target, ord=self.p, dim=1)
        target_norm = torch.linalg.vector_norm(target, ord=self.p, dim=1)
        rel = diff_norm / (target_norm + self.eps)
        if self.reduction == "sum":
            return rel.sum()
        if self.reduction == "none":
            return rel
        return rel.mean()


class JetRGBConverter:
    def __init__(self):
        self.cmap = matplotlib.colormaps.get_cmap("jet")

    def __call__(self, field: np.ndarray) -> np.ndarray:
        norm = Normalize(vmin=float(field.min()), vmax=float(field.max()), clip=True)
        rgba = self.cmap(norm(field))
        rgb = np.transpose(rgba[..., :3], (2, 0, 1)).astype(np.float32)
        return rgb


class NavierStokesResNetDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs.contiguous()
        self.targets = targets.contiguous()

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class ResNet50FieldDecoder(nn.Module):
    def __init__(self, in_channels: int, out_size: int = 64, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2

        backbone = torchvision.models.resnet50(weights=weights)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if weights is None:
                nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
            else:
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                backbone.conv1.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.decode4 = self._decode_block(2048, 1024)
        self.decode3 = self._decode_block(1024 + 1024, 512)
        self.decode2 = self._decode_block(512 + 512, 256)
        self.decode1 = self._decode_block(256 + 256, 128)
        self.decode0 = self._decode_block(128 + 64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.out_size = out_size

    @staticmethod
    def _decode_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decode4(y)

        y = torch.cat([y, x3], dim=1)
        y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decode3(y)

        y = torch.cat([y, x2], dim=1)
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decode2(y)

        y = torch.cat([y, x1], dim=1)
        y = F.interpolate(y, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decode1(y)

        y = torch.cat([y, x0], dim=1)
        y = F.interpolate(y, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        y = self.decode0(y)
        y = self.head(y)
        return y.squeeze(1)


@dataclass
class TrainConfig:
    data_path: str = "data/NavierStokes_V1e-3_N5000_T50.mat"
    output_dir: str = "checkpoints/resnet50_t50_ns64"
    ntrain: int = 1000
    ntest: int = 200
    sub: int = 1
    t_in: int = 10
    target_index: int = 49
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5
    num_workers: int = 4
    amp: bool = True
    compile: bool = False
    pretrained: bool = True
    seed: int = 0
    print_every: int = 1


def frames_to_rgb_stack(frames: torch.Tensor) -> torch.Tensor:
    converter = JetRGBConverter()
    stacked = []
    for sample in frames.numpy():
        channels = [converter(sample[:, :, t]) for t in range(sample.shape[-1])]
        stacked.append(np.concatenate(channels, axis=0))
    return torch.from_numpy(np.stack(stacked, axis=0)).float()


def build_datasets(cfg: TrainConfig) -> Tuple[NavierStokesResNetDataset, NavierStokesResNetDataset, UnitGaussianNormalizer]:
    reader = MatReader(Path(cfg.data_path))
    full = torch.from_numpy(reader.read_field("u")).float()

    inputs = full[:, :: cfg.sub, :: cfg.sub, : cfg.t_in]
    targets = full[:, :: cfg.sub, :: cfg.sub, cfg.target_index]

    train_inputs = inputs[: cfg.ntrain]
    train_targets = targets[: cfg.ntrain]
    test_inputs = inputs[-cfg.ntest :]
    test_targets = targets[-cfg.ntest :]

    y_normalizer = UnitGaussianNormalizer(train_targets)
    train_targets = y_normalizer.encode(train_targets)

    train_rgb = frames_to_rgb_stack(train_inputs)
    test_rgb = frames_to_rgb_stack(test_inputs)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=train_rgb.dtype).repeat(cfg.t_in)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=train_rgb.dtype).repeat(cfg.t_in)
    train_rgb = (train_rgb - imagenet_mean.view(1, -1, 1, 1)) / imagenet_std.view(1, -1, 1, 1)
    test_rgb = (test_rgb - imagenet_mean.view(1, -1, 1, 1)) / imagenet_std.view(1, -1, 1, 1)

    train_dataset = NavierStokesResNetDataset(train_rgb, train_targets)
    test_dataset = NavierStokesResNetDataset(test_rgb, test_targets)
    return train_dataset, test_dataset, y_normalizer


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    normalizer: UnitGaussianNormalizer,
    rel_loss_fn: LpLoss,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    mse_total = 0.0
    rel_total = 0.0
    num_samples = 0

    for inputs, targets in tqdm(loader, desc="Training"):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(inputs)
            mse = F.mse_loss(preds, targets)
        if scaler is not None:
            scaler.scale(mse).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            mse.backward()
            optimizer.step()

        with torch.no_grad():
            decoded_preds = normalizer.decode(preds.float())
            decoded_targets = normalizer.decode(targets.float())
            rel = rel_loss_fn(decoded_preds, decoded_targets)

        batch = inputs.shape[0]
        mse_total += mse.item() * batch
        rel_total += rel.item() * batch
        num_samples += batch

    return {
        "mse": mse_total / max(num_samples, 1),
        "rel_l2": rel_total / max(num_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    normalizer: UnitGaussianNormalizer,
    rel_loss_fn: LpLoss,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    rel_total = 0.0
    mse_total = 0.0
    num_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(inputs)
            mse = F.mse_loss(preds, targets)

        decoded_preds = normalizer.decode(preds.float())
        rel = rel_loss_fn(decoded_preds, targets.float())

        batch = inputs.shape[0]
        mse_total += mse.item() * batch
        rel_total += rel.item() * batch
        num_samples += batch

    return {
        "mse": mse_total / max(num_samples, 1),
        "rel_l2": rel_total / max(num_samples, 1),
    }


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler | None,
    cfg: TrainConfig,
    epoch: int,
    metrics: Dict[str, float],
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

    in_channels = 3 * cfg.t_in
    model = ResNet50FieldDecoder(in_channels=in_channels, out_size=train_dataset.targets.shape[-1], pretrained=cfg.pretrained).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step,
        gamma=cfg.scheduler_gamma,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled) if device.type == "cuda" else None
    rel_loss_fn = LpLoss(reduction="mean")
    y_normalizer = y_normalizer.to(device)

    print(f"Using device: {device}")
    print(f"Preprocessing finished in {prep_time:.2f}s")
    print(f"Model params: {count_params(model):,}")
    print("Train RGB shape:", tuple(train_dataset.inputs.shape), "Target shape:", tuple(train_dataset.targets.shape))

    best_test = math.inf
    output_dir = Path(cfg.output_dir)

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training epochs"):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            normalizer=y_normalizer,
            rel_loss_fn=rel_loss_fn,
            device=device,
            amp_enabled=amp_enabled,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            normalizer=y_normalizer,
            rel_loss_fn=rel_loss_fn,
            device=device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        elapsed = time.perf_counter() - epoch_start

        metrics = {
            "epoch_time_s": elapsed,
            "train_mse": train_metrics["mse"],
            "train_rel_l2": train_metrics["rel_l2"],
            "test_mse": test_metrics["mse"],
            "test_rel_l2": test_metrics["rel_l2"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        if test_metrics["rel_l2"] < best_test:
            best_test = test_metrics["rel_l2"]
            save_checkpoint(output_dir, model, optimizer, scheduler, scaler, cfg, epoch, metrics)

        if epoch % cfg.print_every == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"time {elapsed:6.2f}s | "
                f"train_mse {metrics['train_mse']:.6e} | "
                f"train_rel_l2 {metrics['train_rel_l2']:.6f} | "
                f"test_mse {metrics['test_mse']:.6e} | "
                f"test_rel_l2 {metrics['test_rel_l2']:.6f}"
            )

    print(f"Best test relative L2: {best_test:.6f}")
    print(f"Checkpoint saved to: {output_dir / 'best.pt'}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a ResNet50-style model to predict T50 vorticity from the first 10 frames.")
    parser.add_argument("--data-path", type=str, default=TrainConfig.data_path)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--ntrain", type=int, default=TrainConfig.ntrain)
    parser.add_argument("--ntest", type=int, default=TrainConfig.ntest)
    parser.add_argument("--sub", type=int, default=TrainConfig.sub)
    parser.add_argument("--t-in", type=int, default=TrainConfig.t_in)
    parser.add_argument("--target-index", type=int, default=TrainConfig.target_index, help="Frame index to predict. Use 49 for T50.")
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
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=TrainConfig.pretrained)
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        ntrain=args.ntrain,
        ntest=args.ntest,
        sub=args.sub,
        t_in=args.t_in,
        target_index=args.target_index,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        num_workers=args.num_workers,
        amp=args.amp,
        compile=args.compile,
        pretrained=args.pretrained,
        seed=args.seed,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    train(parse_args())
