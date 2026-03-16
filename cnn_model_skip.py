import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===========================
# Helper: Rotation of Target Coordinates
# ===========================
def rotate_target(target, k):
    """
    Rotate a target coordinate (x,y) about center (pi, pi) by k*90 degrees.
    target: a tensor or numpy array of shape (2,)
    k: integer in {0,1,2,3} representing multiples of 90° (counterclockwise)
    Returns: rotated target coordinate as a tensor (dtype=torch.float32)
    """
    center = np.pi
    x, y = target[0].item(), target[1].item()
    if k == 0:
        new_x, new_y = x, y
    elif k == 1:
        # 90° rotation: (x,y) -> (2pi - y, x)
        new_x = 2 * center - y
        new_y = x
    elif k == 2:
        # 180° rotation: (x,y) -> (2pi - x, 2pi - y)
        new_x = 2 * center - x
        new_y = 2 * center - y
    elif k == 3:
        # 270° rotation: (x,y) -> (y, 2pi - x)
        new_x = y
        new_y = 2 * center - x
    else:
        raise ValueError("k must be in {0,1,2,3}")
    return torch.tensor([new_x, new_y], dtype=torch.float32)

# ===========================
# 1. Dataset Definition with Rotation Augmentation
# ===========================
class TurbulenceDataset(Dataset):
    def __init__(self, npz_file, transform=None, augment=False):
        """
        npz_file: path to a npz file containing:
          - 'initial_omegas': shape (n_simulations, 256,256)
          - 'initial_positions': shape (n_simulations, n_particles, 2)
              (After transformation, these are all (pi,pi))
          - 'checkpoints': a 1D object array of dictionaries.
              For each simulation, the dictionary has a key '500' (or 500) mapping to an array of shape (n_particles, 2)
        transform: function to preprocess the omega field (e.g., normalization)
        augment: if True, apply random rotations (multiples of 90°) to both input and target
        """
        data = np.load(npz_file, allow_pickle=True)
        self.omegas = data['initial_omegas']   # (n_sim, 256,256)
        self.initial_positions = data['initial_positions']  # (n_sim, n_particles, 2)
        self.checkpoints = data['checkpoints']  # length = n_sim, each element is a dict with key '500'
        
        self.n_sim = self.omegas.shape[0]
        self.n_particles = self.initial_positions.shape[1]
        self.transform = transform
        self.augment = augment

        # Build sample list: one sample per particle per simulation
        self.samples = []
        for sim_idx in range(self.n_sim):
            for part_idx in range(self.n_particles):
                self.samples.append((sim_idx, part_idx))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sim_idx, part_idx = self.samples[idx]
        # Get the vorticity field for the simulation: shape (256,256)
        omega = self.omegas[sim_idx]
        # Apply transform if provided
        if self.transform:
            omega = self.transform(omega)
        else:
            omega = torch.tensor(omega, dtype=torch.float32).unsqueeze(0)
            
        # Get target from the checkpoint at 500 timesteps.
        cp_dict = self.checkpoints[sim_idx]
        try:
            target = cp_dict['500']
        except KeyError:
            target = cp_dict[500]
        target = target[part_idx]  # shape (2,)
        target = torch.tensor(target, dtype=torch.float32)
        
        # If augmentation is enabled, randomly rotate the input and adjust the target accordingly.
        if self.augment:
            k = np.random.randint(0, 4)  # Randomly choose 0,1,2,3 (multiples of 90°)
            # Rotate omega tensor using torch.rot90: dims 1 and 2 are spatial.
            omega = torch.rot90(omega, k, dims=[1, 2])
            # Rotate the target coordinate accordingly.
            target = rotate_target(target, k)
            
        return omega, target

# ===========================
# 2. Normalization Transform (same as before)
# ===========================
def normalize_omega(omega):
    """
    Normalize a 256x256 vorticity field per sample using its mean and standard deviation.
    Returns a tensor of shape (1,256,256).
    """
    omega = omega.astype(np.float32)
    omega = omega + 30
    omega_norm = omega / 60
    return torch.tensor(omega_norm, dtype=torch.float32).unsqueeze(0)

# ===========================
# 3. Improved CNN Regression Model with Skip Connections and Output Scaling
# ===========================
class VortexRegressorResNetPeriodic(nn.Module):
    def __init__(self):
        super(VortexRegressorResNetPeriodic, self).__init__()
        # Contraction Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 256 -> 128
        
        # Contraction Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 128 -> 64
        
        # Contraction Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)  # 64 -> 32
        
        # Skip connection from Block 2: adjust channels using 1x1 conv
        self.skip_conv = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        
        # Global Average Pooling after combining features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers (we concatenate pooled features from the main branch and skip branch)
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 2)
        # Final activation: tanh to constrain output to [-1,1]
        # self.tanh = nn.Tanh()
        # Optional: dropout can be added if needed.
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # x: (batch, 1, 256, 256)
        x1 = torch.relu(self.bn1(self.conv1(x)))   # (B,32,256,256)
        x1p = self.pool1(x1)                         # (B,32,128,128)
        
        x2 = torch.relu(self.bn2(self.conv2(x1p)))    # (B,64,128,128)
        x2p = self.pool2(x2)                          # (B,64,64,64)
        
        x3 = torch.relu(self.bn3(self.conv3(x2p)))    # (B,128,64,64)
        x3p = self.pool3(x3)                          # (B,128,32,32)
        
        # Upsample x2p to match x3p spatial dimensions and transform with 1x1 conv
        x2p_up = nn.functional.interpolate(x2p, size=x3p.shape[2:], mode='bilinear', align_corners=False)  # (B,64,32,32)
        skip = self.skip_conv(x2p_up)  # (B,128,32,32)
        
        # Combine by concatenation along channel dimension
        combined = torch.cat([x3p, skip], dim=1)  # (B,256,32,32)
        pooled = self.global_pool(combined)  # (B,256,1,1)
        pooled = pooled.view(pooled.size(0), -1)  # (B,256)
        
        x = torch.relu(self.fc1(self.dropout(pooled)))  # (B,64)
        out = self.fc2(x)  # (B,2)
        # Constrain output via tanh, then scale to [0, 2pi]:
        # out = (self.tanh(out) + 1) * np.pi  # Maps [-1,1] to [0, 2pi]
        return out

# ===========================
# 4. Periodic MSE Loss Function
# ===========================
def periodic_mse_loss(pred, target):
    """
    Compute MSE loss
    """
    diff = pred - target
    loss = torch.mean(diff ** 2)
    return loss

# ===========================
# 5. Training Loop with LR Scheduler
# ===========================
def train_model(model, dataloader, optimizer, scheduler, device, num_epochs=20):
    model.to(device)
    model.train()

    # Check the shape of data

    for omega, target in dataloader:
        print(f"Omega shape: {omega.shape}, Target shape: {target.shape}")
        break

    for inputs, targets in dataloader:
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        break
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            loss = periodic_mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ===========================
# 6. Main Function
# ===========================
def main():
    npz_file = 'centered_turbulent_swimmers.npz'
    # Set augment=True to perform random rotations during training.
    dataset = TurbulenceDataset(npz_file, transform=normalize_omega, augment=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    model = VortexRegressorResNetPeriodic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Use ReduceLROnPlateau scheduler to reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    num_epochs = 50
    train_model(model, dataloader, optimizer, scheduler, device, num_epochs=num_epochs)
    
    torch.save(model.state_dict(), 'vortex_regressor_resnet_periodic_augmented.pth')
    print("Model saved as vortex_regressor_resnet_periodic_augmented.pth")

if __name__ == "__main__":
    main()
