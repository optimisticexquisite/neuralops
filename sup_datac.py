import numpy as np
from swimmers import rk2_update_turbulent
from pseudo_spectral_working import omega_to_omega_hat, omega_hat_to_omega, time_step_single
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

def run_simulation(params):
    """Modified simulation function to work with multiprocessing"""
    # print(f"Running simulation")
    omega_hat_initial, sim_id = params
    # Vectorized initialization
    num_particles = 10
    L = 2 * np.pi
    dt = 5e-4
    N = 256
    
    # Initialize particles with simulation-specific seed
    np.random.seed(sim_id)
    positions = np.random.uniform(np.pi-0.5, np.pi+0.5, (num_particles, 2))
    directions = np.random.rand(num_particles, 2) - 0.5
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.where(norms < 1e-8, np.array([1., 0.]), directions/norms)
    active = np.ones(num_particles, dtype=bool)
    
    # Store references
    omega_hat = omega_hat_initial.copy()
    checkpoint_steps = {500: None, 1000: None, 1500: None, 2000: None, 2500: None, 3000: None, 3500: None, 4000: None, 4500: None, 5000: None, 5500: None, 6000: None, 6500: None, 7000: None, 7500: None, 8000: None, 8500: None, 9000: None, 9500: None, 10000: None}
    
    # Pre-allocate arrays
    all_positions = np.empty((10000, num_particles, 2))
    all_positions[0] = positions.copy()

    for step in tqdm(range(10000), desc=f"Sim {sim_id}", leave=False):
        omega_hat = time_step_single(omega_hat, dt)
        omega = omega_hat_to_omega(omega_hat)
        
        new_pos = np.empty_like(positions)
        new_dir = np.empty_like(directions)
        
        for i in range(num_particles):
            if active[i]:
                new_pos[i], new_dir[i] = rk2_update_turbulent(
                    positions[i], directions[i], 0.0, omega,
                    [0.0, 0.0], 1e6, L, N, dt
                )
        
        active &= ~np.any(np.abs(new_pos) > L, axis=1)
        positions = new_pos
        directions = new_dir
        all_positions[step] = positions
        
        if step+1 in checkpoint_steps:
            checkpoint_steps[step+1] = positions.copy()

    return {
        'initial_omega': omega_hat_to_omega(omega_hat_initial),
        'initial_positions': all_positions[0],
        'checkpoints': checkpoint_steps
    }

def main():
    # Load data and prepare parameters
    omega_snapshots = np.load('omega_snapshots.npy')
    num_simulations = 1000
    
    # Create parameter tuples (omega_hat, unique_sim_id)
    params_list = [
        (omega_to_omega_hat(omega_snapshots[np.random.randint(len(omega_snapshots))]), 
        i
    ) for i in range(num_simulations)]

    # Parallel execution
    with Pool(5) as pool:
        results = list(tqdm(
            pool.imap(run_simulation, params_list, chunksize=10),
            total=num_simulations,
            desc="Running simulations"
        ))

    # Save results
    np.savez('turbulent_swimmers.npz',
        initial_omegas=np.array([r['initial_omega'] for r in results]),
        initial_positions=np.array([r['initial_positions'] for r in results]),
        checkpoints=np.array([r['checkpoints'] for r in results])
    )

if __name__ == '__main__':
    main()