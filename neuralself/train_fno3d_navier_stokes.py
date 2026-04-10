import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import tqdm

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
                    "Install h5py or use the original benchmark .mat files."
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


class NavierStokesFNO3DDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs.contiguous()
        self.targets = targets.contiguous()

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


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


class FNO3d(nn.Module):
    def __init__(self, modes1: int, modes2: int, modes3: int, width: int, padding: int = 6, in_channels: int = 13):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = padding

        self.fc0 = nn.Linear(in_channels, width)
        self.conv0 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w0 = nn.Conv3d(width, width, 1)
        self.w1 = nn.Conv3d(width, width, 1)
        self.w2 = nn.Conv3d(width, width, 1)
        self.w3 = nn.Conv3d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device, x.dtype)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = F.pad(x, (0, self.padding))

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

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
    data_path: str = ""
    train_path: str = "data/ns_data_V100_N1000_T50_1.mat"
    test_path: str = "data/ns_data_V100_N1000_T50_2.mat"
    output_dir: str = "checkpoints/fno3d_ns64"
    ntrain: int = 1000
    ntest: int = 200
    sub: int = 1
    t_in: int = 10
    t_out: int = 40
    modes: int = 8
    width: int = 20
    padding: int = 6
    batch_size: int = 10
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    num_workers: int = 4
    amp: bool = True
    compile: bool = False
    seed: int = 0
    print_every: int = 1


def build_datasets(cfg: TrainConfig) -> Tuple[NavierStokesFNO3DDataset, NavierStokesFNO3DDataset, UnitGaussianNormalizer]:
    if cfg.data_path:
        reader = MatReader(Path(cfg.data_path))
        full = torch.from_numpy(reader.read_field("u")).float()
        train_u_full = full
        test_u_full = full
    else:
        train_reader = MatReader(Path(cfg.train_path))
        test_reader = MatReader(Path(cfg.test_path))
        train_u_full = torch.from_numpy(train_reader.read_field("u")).float()
        test_u_full = torch.from_numpy(test_reader.read_field("u")).float()

    train_a = train_u_full[: cfg.ntrain, :: cfg.sub, :: cfg.sub, : cfg.t_in]
    train_u = train_u_full[: cfg.ntrain, :: cfg.sub, :: cfg.sub, cfg.t_in : cfg.t_in + cfg.t_out]
    test_a = test_u_full[-cfg.ntest :, :: cfg.sub, :: cfg.sub, : cfg.t_in]
    test_u = test_u_full[-cfg.ntest :, :: cfg.sub, :: cfg.sub, cfg.t_in : cfg.t_in + cfg.t_out]

    a_normalizer = UnitGaussianNormalizer(train_a)
    y_normalizer = UnitGaussianNormalizer(train_u)

    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)
    train_u = y_normalizer.encode(train_u)

    s1, s2 = train_a.shape[1], train_a.shape[2]
    train_inputs = train_a.unsqueeze(3).repeat(1, 1, 1, cfg.t_out, 1)
    test_inputs = test_a.unsqueeze(3).repeat(1, 1, 1, cfg.t_out, 1)

    if train_u.shape[1] != s1 or train_u.shape[2] != s2 or train_u.shape[3] != cfg.t_out:
        raise ValueError(f"Unexpected train target shape {tuple(train_u.shape)}")

    train_dataset = NavierStokesFNO3DDataset(train_inputs, train_u)
    test_dataset = NavierStokesFNO3DDataset(test_inputs, test_u)
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
    return sum(p.numel() for p in model.parameters())


def has_complex_parameters(model: nn.Module) -> bool:
    return any(torch.is_complex(param) for param in model.parameters())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    normalizer: UnitGaussianNormalizer,
    loss_fn: LpLoss,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    mse_total = 0.0
    rel_total = 0.0
    num_samples = 0

    for inputs, targets in tqdm.tqdm(loader, desc="Training"):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(inputs)
            mse = F.mse_loss(preds, targets)
            decoded_preds = normalizer.decode(preds)
            decoded_targets = normalizer.decode(targets)
            rel = loss_fn(decoded_preds, decoded_targets)

        if scaler is not None:
            scaler.scale(rel).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            rel.backward()
            optimizer.step()

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
    loss_fn: LpLoss,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    rel_total = 0.0
    num_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(inputs)
            decoded_preds = normalizer.decode(preds)
            rel = loss_fn(decoded_preds, targets)
        batch = inputs.shape[0]
        rel_total += rel.item() * batch
        num_samples += batch

    return {"rel_l2": rel_total / max(num_samples, 1)}


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
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

    model = FNO3d(cfg.modes, cfg.modes, cfg.modes, cfg.width, padding=cfg.padding).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step,
        gamma=cfg.scheduler_gamma,
    )
    use_grad_scaler = amp_enabled and not has_complex_parameters(model)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler) if device.type == "cuda" else None
    loss_fn = LpLoss(reduction="mean")
    y_normalizer = y_normalizer.to(device)

    print(f"Using device: {device}")
    print(f"Preprocessing finished in {prep_time:.2f}s")
    print(f"Model params: {count_params(model):,}")
    if amp_enabled and not use_grad_scaler:
        print("AMP is enabled without GradScaler because the model has complex-valued spectral weights.")
    print(
        "Train shape:",
        tuple(train_dataset.inputs.shape),
        "Target shape:",
        tuple(train_dataset.targets.shape),
    )

    best_test = math.inf
    output_dir = Path(cfg.output_dir)

    for epoch in tqdm.tqdm(range(1, cfg.epochs + 1), desc="Epochs"):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            normalizer=y_normalizer,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            normalizer=y_normalizer,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        elapsed = time.perf_counter() - epoch_start

        metrics = {
            "epoch_time_s": elapsed,
            "train_mse": train_metrics["mse"],
            "train_rel_l2": train_metrics["rel_l2"],
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
                f"test_rel_l2 {metrics['test_rel_l2']:.6f}"
            )

    print(f"Best test relative L2: {best_test:.6f}")
    print(f"Checkpoint saved to: {output_dir / 'best.pt'}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the paper-style FNO3D on 64x64 Navier-Stokes.")
    parser.add_argument("--data-path", type=str, default=TrainConfig.data_path)
    parser.add_argument("--train-path", type=str, default=TrainConfig.train_path)
    parser.add_argument("--test-path", type=str, default=TrainConfig.test_path)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--ntrain", type=int, default=TrainConfig.ntrain)
    parser.add_argument("--ntest", type=int, default=TrainConfig.ntest)
    parser.add_argument("--sub", type=int, default=TrainConfig.sub)
    parser.add_argument("--t-in", type=int, default=TrainConfig.t_in)
    parser.add_argument("--t-out", type=int, default=TrainConfig.t_out)
    parser.add_argument("--modes", type=int, default=TrainConfig.modes)
    parser.add_argument("--width", type=int, default=TrainConfig.width)
    parser.add_argument("--padding", type=int, default=TrainConfig.padding)
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
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        ntrain=args.ntrain,
        ntest=args.ntest,
        sub=args.sub,
        t_in=args.t_in,
        t_out=args.t_out,
        modes=args.modes,
        width=args.width,
        padding=args.padding,
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
