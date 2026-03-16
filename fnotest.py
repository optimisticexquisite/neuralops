import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# -------------------------------
# 1. Data generation
# -------------------------------

def target_function(x):
    return (
        torch.sin(2 * np.pi * x)
        + 0.5 * torch.sin(4 * np.pi * x)
        + 0.3 * torch.cos(6 * np.pi * x)
        + 0.2 * torch.sin(8 * np.pi * x)
    )

# Training only on [0,1]
n_train = 128
x_train = torch.linspace(0, 1, n_train)
y_train = target_function(x_train)

# Validation on extended domain [-1, 2]
n_val = 512
x_val = torch.linspace(-1, 2, n_val)
y_val = target_function(x_val)

# -------------------------------
# 2. Standard Neural Network
# -------------------------------
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
class PointwiseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            Sine(),
            nn.Linear(128, 128),
            Sine(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)

nn_model = PointwiseNN()

# -------------------------------
# 3. Simple FNO (identity operator learning)
# -------------------------------

class SimpleFNO1D(nn.Module):
    def __init__(self, n_modes):
        super().__init__()
        self.n_modes = n_modes
        self.weight = nn.Parameter(torch.randn(n_modes, dtype=torch.cfloat))

    def forward(self, x):
        x_hat = fft.rfft(x)
        out_hat = torch.zeros_like(x_hat)
        out_hat[: self.n_modes] = self.weight * x_hat[: self.n_modes]
        return fft.irfft(out_hat, n=x.shape[0])

fno_model = SimpleFNO1D(n_modes=60)

# -------------------------------
# 4. Training
# -------------------------------

opt_nn = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
opt_fno = torch.optim.Adam(fno_model.parameters(), lr=1e-2)

loss_fn = nn.MSELoss()
epochs = 1000

history_nn = []
history_fno = []

for epoch in range(epochs):

    # ---- NN ----
    opt_nn.zero_grad()
    nn_pred = nn_model(x_train.unsqueeze(-1)).squeeze()
    loss_nn = loss_fn(nn_pred, y_train)
    loss_nn.backward()
    opt_nn.step()

    # ---- FNO ----
    opt_fno.zero_grad()
    fno_pred = fno_model(y_train)
    loss_fno = loss_fn(fno_pred, y_train)
    loss_fno.backward()
    opt_fno.step()

    # Validation
    with torch.no_grad():
        nn_val_pred = nn_model(x_val.unsqueeze(-1)).squeeze()
        fno_val_pred = fno_model(y_val)

    history_nn.append(nn_val_pred.numpy())
    history_fno.append(fno_val_pred.numpy())

    print(f"Epoch {epoch:03d} | NN {loss_nn:.3e} | FNO {loss_fno:.3e}")

# -------------------------------
# 5. Animation
# -------------------------------

fig, ax = plt.subplots(figsize=(9, 4))

line_gt, = ax.plot(x_val, y_val, 'k', lw=2, label="Ground Truth")
line_nn, = ax.plot([], [], 'r--', label="Neural Network")
line_fno, = ax.plot([], [], 'b-', label="FNO")

ax.set_ylim(-2, 2)
ax.set_xlim(-1, 2)
ax.legend()
ax.set_title("Periodic Generalization Test")

def update(frame):
    line_nn.set_data(x_val, history_nn[frame])
    line_fno.set_data(x_val, history_fno[frame])
    ax.set_xlabel(f"Epoch {frame}")
    return line_nn, line_fno

ani = FuncAnimation(fig, update, frames=epochs, interval=50)
ani.save("periodicity_test.gif", writer="ffmpeg", fps=20)
