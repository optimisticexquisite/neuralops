# ============================================================
# FNO-based Turbulence → Particle Position Regression
# Single-file, end-to-end training script
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------------------------------------------
# Requires: pip install neuraloperator
# ------------------------------------------------------------
from neuralop.models import FNO



# ============================================================
# Rotation helpers (field + target)
# ============================================================

def rotate_field(field, k):
    """
    field: Tensor of shape (1, H, W)
    k: number of 90-degree rotations
    """
    return torch.rot90(field, k, dims=[1, 2])


def rotate_target(target, k):
    """
    Rotate (x,y) around center (pi, pi) by k * 90 degrees
    """
    center = np.pi
    x, y = target[0].item(), target[1].item()

    if k == 0:
        nx, ny = x, y
    elif k == 1:
        nx, ny = 2 * center - y, x
    elif k == 2:
        nx, ny = 2 * center - x, 2 * center - y
    elif k == 3:
        nx, ny = y, 2 * center - x
    else:
        raise ValueError("k must be in {0,1,2,3}")

    return torch.tensor([nx, ny], dtype=torch.float32)


# ============================================================
# Dataset (raw vorticity, operator-ready)
# ============================================================

class TurbulenceFNODataset(Dataset):
    def __init__(self, npz_file, augment=True, normalize=True):
        data = np.load(npz_file, allow_pickle=True)

        self.omegas = data["initial_omegas"]       # (n_sims, H, W)
        self.checkpoints = data["checkpoints"]     # list of dicts
        self.n_sims = self.omegas.shape[0]
        self.n_particles = self.checkpoints[0]["500"].shape[0]

        self.augment = augment
        self.normalize = normalize

        self.samples = [
            (s, p)
            for s in range(self.n_sims)
            for p in range(self.n_particles)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sim_idx, part_idx = self.samples[idx]

        omega = torch.tensor(
            self.omegas[sim_idx],
            dtype=torch.float32
        ).unsqueeze(0)  # (1, H, W)

        if self.normalize:
            omega = (omega - omega.mean()) / (omega.std() + 1e-6)

        target = self.checkpoints[sim_idx]["500"][part_idx]
        target = torch.tensor(target, dtype=torch.float32)

        if self.augment:
            k = np.random.randint(0, 4)
            omega = rotate_field(omega, k)
            target = rotate_target(target, k)

        return omega, target


# ============================================================
# FNO Regression Model
# ============================================================

class FNORegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fno = FNO(
            n_modes=(24, 24),        # Fourier modes in (x,y)
            hidden_channels=64,
            in_channels=1,
            out_channels=32,
            n_layers=4
        )

        self.mlp = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: (B,1,H,W)
        f = self.fno(x)          # (B,32,H,W)
        f = f.mean(dim=[2, 3])   # global pooling
        return self.mlp(f)


# ============================================================
# Training Loop
# ============================================================

def train(
    model,
    dataloader,
    device,
    epochs=50,
    lr=5e-4
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for omega, target in tqdm(dataloader, leave=False):
            omega = omega.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(omega)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f}")

    return model


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TurbulenceFNODataset(
        npz_file="centered_turbulent_swimmers_1.npz",
        augment=True,
        normalize=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,        # adjust based on GPU memory
        shuffle=True,
        num_workers=32,
        pin_memory=True
    )

    model = FNORegressor()

    model = train(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=200,
        lr=5e-4
    )

    torch.save(model.state_dict(), "fno_particle_regressor.pth")
    print("Saved model to fno_particle_regressor.pth")


if __name__ == "__main__":
    main()
