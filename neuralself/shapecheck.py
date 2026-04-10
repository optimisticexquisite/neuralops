import h5py

with h5py.File('data/NavierStokes_V1e-3_N5000_T50.mat', 'r') as f:
    # Check shape of a specific dataset
    dataset = f['u']
    print(dataset.shape)
