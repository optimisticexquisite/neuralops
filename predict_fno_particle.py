import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# Neural Operator import (API-safe)
# ------------------------------------------------------------
from neuralop.models import FNO


# ============================================================
# FNO Regressor (must match training definition)
# ============================================================

class FNORegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fno = FNO(
            n_modes=(24, 24),
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
        f = self.fno(x)          # (B,32,H,W)
        f = f.mean(dim=[2, 3])   # global pooling
        return self.mlp(f)


# ============================================================
# Load trained FNO model
# ============================================================

def load_fno_model(weights_path="fno_particle_regressor.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FNORegressor()
    state = torch.load(
        weights_path,
        map_location=device,
        weights_only=False
    )
    if "_metadata" in state:
        del state["_metadata"]
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    print(f"Loaded FNO model on {device}")
    return model, device


# ============================================================
# Prediction + Plotting
# ============================================================

def predict_and_plot(
    npz_path="centered_turbulent_swimmers_1.npz",
    model_path="fno_particle_regressor.pth"
):
    # ----------------------------
    # Load data
    # ----------------------------
    data = np.load(npz_path, allow_pickle=True)
    omegas = data["initial_omegas"]      # (n_sim, 256, 256)
    checkpoints = data["checkpoints"]

    n_sim = omegas.shape[0]
    n_particles = checkpoints[0]["500"].shape[0]

    # Random sample
    sim_idx = random.randint(0, n_sim - 1)
    part_idx = random.randint(0, n_particles - 1)

    omega = omegas[sim_idx]                              # (256,256)
    true_pos = checkpoints[sim_idx]["500"][part_idx]    # (2,)

    # ----------------------------
    # Prepare input for FNO
    # ----------------------------
    omega_tensor = torch.tensor(
        omega, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Same normalization as training
    omega_tensor = (omega_tensor - omega_tensor.mean()) / (
        omega_tensor.std() + 1e-6
    )

    # ----------------------------
    # Load model & predict
    # ----------------------------
    model, device = load_fno_model(model_path)
    omega_tensor = omega_tensor.to(device)

    with torch.no_grad():
        pred = model(omega_tensor).squeeze().cpu().numpy()

    # ----------------------------
    # Print results
    # ----------------------------
    print(f"True Position (continuous): {true_pos}")
    print(f"Predicted Position (continuous): {pred}")

    # ----------------------------
    # Plot in physical domain
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(omega, cmap="jet")

    # Convert from [0, 2π] → grid coordinates
    true_pix = true_pos * 256 / (2 * np.pi)
    pred_pix = pred * 256 / (2 * np.pi)

    ax.plot(true_pix[0], true_pix[1], "go", label="Ground Truth")
    ax.plot(pred_pix[0], pred_pix[1], "rx", label="Prediction")
    ax.plot(128, 128, "bo", label="Center (128,128)")

    ax.set_title("FNO Prediction: Particle Position @ t=500")
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])
    ax.invert_yaxis()
    ax.legend()

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("fno_prediction_plot.png", dpi=200)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    predict_and_plot()
