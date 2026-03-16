import numpy as np

data = np.load("turbulent_swimmers.npz", allow_pickle=True)

# Check the shape of the loaded data
print("Shape of initial_omegas:", data['initial_omegas'].shape)
print("Shape of initial_positions:", data['initial_positions'].shape)
print("Shape of checkpoints:", data['checkpoints'].shape)

#Check inside the checkpoints
for i in range(len(data['checkpoints'])):
    print(f"Checkpoints for simulation {i}:")
    for key, value in data['checkpoints'][i].items():
        print(f"  {key}: {value.shape}")

    
