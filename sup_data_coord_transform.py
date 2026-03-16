import numpy as np

def transform_and_save_centered_coordinates(input_path, output_path):
    """
    Transform all particles to appear starting at (π, π) with periodic BCs and save results
    """
    # Load original data
    with np.load(input_path, allow_pickle=True) as data:
        original_data = {
            'initial_omegas': data['initial_omegas'],
            'initial_positions': data['initial_positions'],
            'checkpoints': data['checkpoints']  # This is a 1D array of dictionaries
        }
    
    L = 2 * np.pi
    N = 256
    dx = L / N
    n_simulations = len(original_data['initial_omegas'])
    n_particles = original_data['initial_positions'].shape[1]

    # Initialize transformed data structures
    transformed_data = {
        'initial_omegas': np.empty_like(original_data['initial_omegas']),
        'initial_positions': np.empty_like(original_data['initial_positions']),
        'checkpoints': np.empty(n_simulations, dtype=object)  # Array to store dictionaries
    }

    for sim_idx in range(n_simulations):
        original_positions = original_data['initial_positions'][sim_idx]
        checkpoint_dict = original_data['checkpoints'][sim_idx]
        
        # Calculate offsets
        offsets = original_positions - np.array([np.pi, np.pi])
        
        # Transform omega field using first particle's offset
        grid_shifts = np.round(offsets[0] / dx).astype(int)
        transformed_data['initial_omegas'][sim_idx] = np.roll(
            original_data['initial_omegas'][sim_idx],
            shift=(-grid_shifts[0], -grid_shifts[1]),
            axis=(0, 1)
        )

        # Set initial positions to π
        transformed_data['initial_positions'][sim_idx] = np.full((n_particles, 2), np.pi)

        # Create new dictionary for transformed checkpoints
        transformed_checkpoints = {}
        
        # Transform checkpoints - handle both string and integer keys
        for step in [500, 1000, 1500, 2000]:
            # Try both string and integer key formats
            try:
                original = checkpoint_dict[str(step)]
            except KeyError:
                original = checkpoint_dict[step]
                
            transformed = (original - offsets) % L
            transformed[np.isnan(original)] = np.nan
            transformed_checkpoints[str(step)] = transformed  # Standardize to string keys
        
        transformed_data['checkpoints'][sim_idx] = transformed_checkpoints

    # Save transformed data
    np.savez(
        output_path,
        initial_omegas=transformed_data['initial_omegas'],
        initial_positions=transformed_data['initial_positions'],
        checkpoints=transformed_data['checkpoints']
    )
    print(f"Data saved to {output_path}")
    return transformed_data

# Usage
if __name__ == "__main__":
    transform_and_save_centered_coordinates(
        input_path='turbulent_swimmers.npz',
        output_path='centered_turbulent_swimmers_1.npz'
    )