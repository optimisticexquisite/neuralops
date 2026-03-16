import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===========================
# 1. Dataset Definition
# ===========================
class TurbulenceDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        """
        npz_file: path to a npz file containing:
          - 'initial_omegas': shape (n_simulations, 256,256)
          - 'initial_positions': shape (n_simulations, n_particles, 2) 
              (After transformation, these are all (pi,pi))
          - 'checkpoints': a 1D object array of dictionaries.
              For each simulation, the dictionary has a key '500' mapping to an array of shape (n_particles, 2)
        transform: function to preprocess the omega field (e.g. normalization)
        """
        data = np.load(npz_file, allow_pickle=True)
        self.omegas = data['initial_omegas']   # (n_sim, 256, 256)
        self.initial_positions = data['initial_positions']  # (n_sim, n_particles, 2)
        self.checkpoints = data['checkpoints']  # length = n_sim, each element is a dict with key '500'
        
        self.n_sim = self.omegas.shape[0]
        self.n_particles = self.initial_positions.shape[1]
        self.transform = transform

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
        # Apply transform if provided (e.g., normalization)
        if self.transform:
            omega = self.transform(omega)
        else:
            # Convert to tensor and add a channel dimension to get shape (1,256,256)
            omega = torch.tensor(omega, dtype=torch.float32).unsqueeze(0)
            
        # Get the target: use the checkpoint at 500 steps.
        # The checkpoint dictionary key can be int or string, so try both.
        cp_dict = self.checkpoints[sim_idx]
        try:
            target = cp_dict['500']
        except KeyError:
            target = cp_dict[500]
        # target for the given particle: shape (2,)
        target = target[part_idx]
        target = torch.tensor(target, dtype=torch.float32)
        return omega, target

# ===========================
# 2. Normalization Transform
# ===========================
def normalize_omega(omega):
    """
    Normalize a 256x256 vorticity field.
    Instead of using fixed -20 to 20 bounds, we normalize per sample using mean and std.
    """
    omega = omega.astype(np.float32)
    mean = np.mean(omega)
    std = np.std(omega)
    if std < 1e-6:
        std = 1e-6
    omega_norm = (omega - mean) / std
    # Return as tensor with shape (1,256,256)
    return torch.tensor(omega_norm, dtype=torch.float32).unsqueeze(0)

# ===========================
# 3. CNN Regression Model
# ===========================
class VortexRegressor(nn.Module):
    def __init__(self):
        super(VortexRegressor, self).__init__()
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
        
        # Global Average Pooling: convert (B, 128, 32,32) -> (B, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # Output: predicted (x,y) position
        
    def forward(self, x):
        # x: (batch_size, 1, 256, 256)
        x = torch.relu(self.bn1(self.conv1(x)))  # -> (B,32,256,256)
        x = self.pool1(x)                        # -> (B,32,128,128)
        
        x = torch.relu(self.bn2(self.conv2(x)))  # -> (B,64,128,128)
        x = self.pool2(x)                        # -> (B,64,64,64)
        
        x = torch.relu(self.bn3(self.conv3(x)))  # -> (B,128,64,64)
        x = self.pool3(x)                        # -> (B,128,32,32)
        
        # Global average pooling
        x = self.global_pool(x)                  # -> (B,128,1,1)
        x = x.view(x.size(0), -1)                # -> (B,128)
        
        x = torch.relu(self.fc1(x))              # -> (B,64)
        out = self.fc2(x)                        # -> (B,2)
        # Optionally, you can add a tanh and then scale to [0,2pi] if desired.
        return out

# ===========================
# 4. Training Loop
# ===========================
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=20):
    model.to(device)
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ===========================
# 5. Main Function
# ===========================
def main():
    npz_file = 'centered_turbulent_swimmers.npz'  # Your transformed data file
    dataset = TurbulenceDataset(npz_file, transform=normalize_omega)
    # Adjust batch_size and num_workers as appropriate
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    model = VortexRegressor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # For regression, you can later consider periodic-aware loss
    
    num_epochs = 20
    train_model(model, dataloader, optimizer, criterion, device, num_epochs=num_epochs)
    
    # After training, save the model
    torch.save(model.state_dict(), 'vortex_regressor.pth')
    print("Model saved as vortex_regressor.pth")

if __name__ == "__main__":
    main()
