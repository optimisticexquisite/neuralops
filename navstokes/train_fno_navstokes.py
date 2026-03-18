import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fno2d import FNO2d


@dataclass
class TrainConfig:
    data_path: str = "omega_snapshots.npy"
    output_path: str = "navstokes/fno_navstokes.pt"
    stride: int = 4
    train_ratio: float = 0.85
    batch_size: int = 4
    epochs: int = 25
    lr: float = 2e-3
    modes: int = 16
    width: int = 48
    seed: int = 42


class SnapshotPairDataset(Dataset):
    def __init__(self, snapshots: np.ndarray, stride: int, mean: float, std: float):
        self.x = snapshots[:-stride]
        self.y = snapshots[stride:]
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = (self.x[idx] - self.mean) / self.std
        y = (self.y[idx] - self.mean) / self.std
        return (
            torch.from_numpy(x).float().unsqueeze(0),
            torch.from_numpy(y).float().unsqueeze(0),
        )


def load_snapshots(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected snapshots shape (T,H,W), got {arr.shape}")
    return arr.astype(np.float32)


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    snapshots = load_snapshots(config.data_path)
    print(f"Loaded snapshots: {snapshots.shape} from {config.data_path}")
    num_frames = snapshots.shape[0]
    split = int(num_frames * config.train_ratio)

    train_frames = snapshots[:split]
    val_frames = snapshots[split - config.stride :]

    mean = float(train_frames.mean())
    std = float(train_frames.std() + 1e-6)

    train_ds = SnapshotPairDataset(train_frames, config.stride, mean, std)
    val_ds = SnapshotPairDataset(val_frames, config.stride, mean, std)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d(modes1=config.modes, modes2=config.modes, width=config.width).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in tqdm(range(config.epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc="Training batches", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

        print(
            f"Epoch {epoch + 1:03d}/{config.epochs} | "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "mean": mean,
            "std": std,
            "config": asdict(config),
            "shape": snapshots.shape,
            "best_val": best_val,
        },
        output,
    )
    print(f"Saved checkpoint: {output}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train FNO for Navier-Stokes vorticity rollouts")
    parser.add_argument("--data-path", default="omega_snapshots.npy")
    parser.add_argument("--output-path", default="navstokes/fno_navstokes.pt")
    parser.add_argument("--stride", type=int, default=4, help="Predict t+stride from t")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
