import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.fft import fft2, ifft2, fftshift, fftfreq
from fluidfft import import_fft_class
import fluidfft
from tqdm import tqdm

# -------------------------------
# Grid and Simulation Parameters
# -------------------------------
L = 2 * np.pi
N = 256
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
nu = 0.002
X, Y = np.meshgrid(x, y)

# Choose the FFT method
method = 'fft2d.with_pyfftw'
FFT2D = import_fft_class(method)

# Create an instance of the FFT class
fft2d = FFT2D(int(N), int(N))

# Get kx and ky from fluidfft (for computing the wave vectors)
kx = fft2d.get_k_adim_loc()[1] * 2 * np.pi / L
ky = fft2d.get_k_adim_loc()[0] * 2 * np.pi / L
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
# print(np.where(K2 == 0))

# Dealiasing and forcing parameters
N_cutoff = int(2/3 * N)
dealias_mask = (K2 < (N_cutoff**2))

kinf = 1    # Lower bound for forcing wavenumbers
ksup = 4    # Upper bound for forcing wavenumbers
alpha = 0.05  # Damping coefficient

# ---------------------------------------
# Helper functions for forward/backward
# ---------------------------------------
def fluidfft_fft2d(variable):
    return fft2d.fft2d(variable)

def fluidfft_ifft2d(variable):
    return fft2d.ifft2d(variable)

# -------------------------------------
# Random Fourier Initial Condition
# -------------------------------------
def initial_condition_real_space(N):
    omega = np.random.randn(N, N)
    omega /= np.max(np.abs(omega))  # Normalize
    return omega

def initial_condition_restart(N, filename):
    omega = np.load(filename)
    return omega

# -------------------------------------
# Velocity, forcing, time stepping
# -------------------------------------
def psi_hat_fn(omega_hat):
    K2_temp = K2.copy()
    K2_temp[0, 0] = 1  # Avoid divide by zero
    psi_hat = omega_hat / K2_temp
    psi_hat[0, 0] = 0
    return psi_hat

def omega_hat_fn(psi_hat):
    K2_temp = K2.copy()
    K2_temp[0, 0] = 1  # Avoid divide by zero
    omega_hat = K2_temp * psi_hat
    omega_hat[0, 0] = 0
    return omega_hat

def velocity(omega_hat):
    psi_hat = psi_hat_fn(omega_hat)
    u = fluidfft_ifft2d(1j * KY * psi_hat).real
    v = fluidfft_ifft2d(-1j * KX * psi_hat).real
    return u, v

def calculate_nonlinear_term_hat(omega_hat, u, v):
    domega_dx_hat = 1j * KX * omega_hat
    domega_dy_hat = 1j * KY * omega_hat
    domega_dx = fluidfft_ifft2d(domega_dx_hat).real
    domega_dy = fluidfft_ifft2d(domega_dy_hat).real
    nonlinear_term = u * domega_dx + v * domega_dy
    nonlinear_term_hat = fluidfft_fft2d(nonlinear_term)
    return nonlinear_term_hat

def add_forcing(omega_hat, kinf, ksup, dt):
    psi_hat = psi_hat_fn(omega_hat)
    forcing_range = (np.sqrt(K2) >= (kinf)) & (np.sqrt(K2) <= (ksup))
    IIf = np.where(forcing_range)
    K2_valid = K2[IIf]
    K2_valid[K2_valid == 0] = 1e-30
    # print(K2_valid)
    factor = dt / (K2_valid * np.pi * (ksup - kinf) * (ksup + kinf))
    std_force = np.sqrt(factor)
    forcing = np.zeros_like(psi_hat, dtype=np.complex128)
    rand_complex = (np.random.randn(len(IIf[0])) + 1j*np.random.randn(len(IIf[0])))
    forcing[IIf] = std_force * rand_complex
    psi_hat += forcing
    omega_hat = omega_hat_fn(psi_hat)
    return omega_hat

def apply_damping(omega_hat, alpha, dt):
    # return omega_hat
    return omega_hat * np.exp(-alpha * dt * K2)

def omega_hat_to_omega(omega_hat):
    """
    Convert the spectral vorticity field to real space.
    """
    omega = fluidfft_ifft2d(omega_hat).real
    return omega

def omega_to_omega_hat(omega):
    """
    Convert the real space vorticity field to spectral space.
    """
    omega_hat = fluidfft_fft2d(omega)
    return omega_hat

def time_stepping_with_snapshots(omega_hat, dt, T, snapshot_interval):
    """
    Time stepping that stores snapshots of the vorticity field
    for animation purposes.
    """
    snapshots = []
    steps = int(T / dt)
    snapshot_steps = int(snapshot_interval / dt)
    
    for step in tqdm(range(steps)):
        # Dealias
        omega_hat *= dealias_mask
        
        # Compute velocity
        u, v = velocity(omega_hat)
        
        # Compute nonlinear + advective terms
        nonlinear_term_hat = calculate_nonlinear_term_hat(omega_hat, u, v)
        
        # Add forcing and damping
        omega_hat = add_forcing(omega_hat, kinf, ksup, dt)
        omega_hat = apply_damping(omega_hat, alpha, dt)
        
        # Apply viscous term and finalize step
        omega_hat = omega_hat - dt * nonlinear_term_hat - dt * (nu * K2) * omega_hat
        
        # Save snapshot every snapshot_interval
        if step % snapshot_steps == 0:
            snapshots.append(fluidfft_ifft2d(omega_hat).real)
    
    return snapshots

def time_step_single(omega_hat, dt):
    """
    Perform a single time step of the simulation.
    """
    # Dealias
    omega_hat *= dealias_mask
    
    # Compute velocity
    u, v = velocity(omega_hat)
    
    # Compute nonlinear + advective terms
    nonlinear_term_hat = calculate_nonlinear_term_hat(omega_hat, u, v)
    
    # Add forcing and damping
    omega_hat = add_forcing(omega_hat, kinf, ksup, dt)
    omega_hat = apply_damping(omega_hat, alpha, dt)
    
    # Apply viscous term and finalize step
    omega_hat = omega_hat - dt * nonlinear_term_hat - dt * (nu * K2) * omega_hat
    
    return omega_hat

# ----------------
# Main Script
# ----------------
if __name__ == '__main__':
    # 1) Create a random initial condition in real space
    # omega = initial_condition_real_space(N)
    import scipy.io
    mat_file = scipy.io.loadmat('vorticity_data.mat')
    keys = [key for key in mat_file.keys() if not key.startswith('__')]
    print("Variables in MAT file:", keys)
    mat_variable = mat_file[keys[0]]
    omega = np.array(mat_variable, dtype=np.complex128)
    
    
    # 2) Transform to spectral space
    omega_hat = fluidfft_fft2d(omega)
    
    # 3) Time-stepping with snapshots
    T = 500         # Total simulation time
    dt = 5e-4      # Time step size
    snapshot_interval = 0.1  # Save frame every 0.1 units of time
    snapshots = time_stepping_with_snapshots(omega_hat, dt, T, snapshot_interval)
    
    # ---------------------
    # Animation of Results
    # ---------------------
    fig, ax = plt.subplots()
    img = ax.imshow(snapshots[0], extent=(0, L, 0, L), origin='lower', cmap='jet')
    ax.set_title('Time Evolution of Vorticity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = fig.colorbar(img, ax=ax)

    def update(frame):
        img.set_array(snapshots[frame])
        ax.set_title(f'Time Evolution of Vorticity (Frame {frame})')

        # Update colorbar limits dynamically
        img.set_clim(np.min(snapshots[frame]), np.max(snapshots[frame]))
        cbar.draw_all()  # Redraw colorbar
        
        return [img]

    # Use the 'ffmpeg' writer to save as an MP4 file
    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=True)
    anim.save('fluid_vorticity_evolution.mp4', writer='ffmpeg', fps=20)
    plt.show()
