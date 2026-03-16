import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm import tqdm

# ===========================
# Helper: Convert 2D vorticity to RGB using jet colormap (with shape check)
# ===========================
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Helper: Convert 2D vorticity to RGB using the jet colormap
def to_jet_rgb(omega):
    """
    Convert a 2D numpy array (omega) into an RGB image using the jet colormap.
    If omega is 2D, returns an array of shape (3, H, W) with float32 values in [0,1].
    """
    if omega.ndim == 2:
        # Normalize data to [0, 1]
        vmin = omega.min()
        vmax = omega.max()
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = matplotlib.colormaps.get_cmap('jet')
        rgba_img = cmap(norm(omega))  # shape (H, W, 4)
        rgb_img = rgba_img[..., :3]   # shape (H, W, 3)
        rgb_img = np.transpose(rgb_img, (2, 0, 1)).astype(np.float32)  # shape (3, H, W)
        return rgb_img
    elif omega.ndim == 3:
        if omega.shape[-1] in [3, 4]:
            return np.transpose(omega, (2, 0, 1)).astype(np.float32)
        else:
            return omega.astype(np.float32)
    else:
        raise ValueError("Input omega has unsupported number of dimensions.")


# Helper: Rotation of Target Coordinates (same as before)
def rotate_target(target, k):
    """
    Rotate a target coordinate (x,y) about center (pi, pi) by k*90 degrees.
    """
    center = np.pi
    x, y = target[0].item(), target[1].item()
    if k == 0:
        new_x, new_y = x, y
    elif k == 1:
        new_x = 2 * center - y
        new_y = x
    elif k == 2:
        new_x = 2 * center - x
        new_y = 2 * center - y
    elif k == 3:
        new_x = y
        new_y = 2 * center - x
    else:
        raise ValueError("k must be in {0,1,2,3}")
    return torch.tensor([new_x, new_y], dtype=torch.float32)


# Dataset Definition with Rotation Augmentation (RGB version)
class TurbulenceRGBDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file, transform=None, augment=False):
        """
        npz_file: path to .npz with:
            - 'initial_omegas': shape (n_sims, 256,256) (expected to be 2D)
            - 'checkpoints': array of dictionaries, each dict has a '500' key -> (n_particles, 2)
        transform: a torchvision transform pipeline (e.g., Resize, Normalize)
        augment: if True, apply random rotations (multiples of 90°) to both input and target
        """
        data = np.load(npz_file, allow_pickle=True)
        self.omegas = data['initial_omegas']       # shape: (n_sims, 256,256)
        self.checkpoints = data['checkpoints']     # shape: (n_sims,) each is dict with key '500'
        self.n_sims = self.omegas.shape[0]
        self.n_particles = self.checkpoints[0]['500'].shape[0]
        self.transform = transform
        self.augment = augment
        
        # Build a list of (sim_idx, part_idx)
        self.samples = []
        for sim_idx in range(self.n_sims):
            for part_idx in range(self.n_particles):
                self.samples.append((sim_idx, part_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sim_idx, part_idx = self.samples[idx]
        # Get the 2D vorticity field
        omega_2d = self.omegas[sim_idx]  # shape (256,256)
        
        # Convert to RGB using jet colormap
        rgb_img = to_jet_rgb(omega_2d)   # shape (3,256,256), float32 in [0,1]
        
        # For PIL conversion, convert to (H,W,3)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        # Convert to uint8 by scaling
        rgb_img_uint8 = (rgb_img * 255).astype(np.uint8)
        rgb_pil = T.ToPILImage()(rgb_img_uint8)
        
        if self.transform:
            rgb_tensor = self.transform(rgb_pil)  # e.g., (3,224,224)
        else:
            rgb_tensor = T.ToTensor()(rgb_pil)
            
        # Get target from checkpoint at '500' timesteps.
        cp_dict = self.checkpoints[sim_idx]
        try:
            target = cp_dict['500']
        except KeyError:
            target = cp_dict[500]
        target = target[part_idx]  # shape (2,)
        target = torch.tensor(target, dtype=torch.float32)
        
        # Augmentation: random rotation by multiples of 90°
        if self.augment:
            k = np.random.randint(0, 4)
            rgb_tensor = torch.rot90(rgb_tensor, k, dims=[1, 2])
            target = rotate_target(target, k)
            
        return rgb_tensor, target




# ===========================
# 2. Build Pretrained ResNet50 Model for Regression
# ===========================
def build_resnet50_finetune():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    # Replace final fully connected layer: original has out_features=1000
    # Build 2 linear layers with ReLU in between
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2)
    )
    return model

# ===========================
# 3. Training Setup
# ===========================
def regression_mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)

def train_model(model, dataloader, optimizer, device, num_epochs=10, scheduler=None):
    model.to(device)
    model.train()
    criterion = regression_mse_loss
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for batch_idx, (images, targets) in tqdm(enumerate(dataloader)):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # shape (B,2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if scheduler:
            scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ===========================
# 4. Main Function
# ===========================
def main():
    # Define transform pipeline:
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    dataset = TurbulenceRGBDataset(npz_file='centered_turbulent_swimmers_1.npz',
                                   transform=transform,
                                   augment=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_resnet50_finetune()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_model(model, dataloader, optimizer, device, num_epochs=200, scheduler=scheduler)
    
    torch.save(model.state_dict(), 'resnet50_2d_regression.pth')
    print("Saved model as resnet50_2d_regression.pth")

if __name__ == "__main__":
    main()
