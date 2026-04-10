import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from fluidfft import import_fft_class
from matplotlib.animation import FuncAnimation
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

# Dealiasing and forcing parameters
N_cutoff = int(2 / 3 * N)
kinf = 1
ksup = 4
alpha = 0.05


class FluidFFTSolver:
    def __init__(self, n, domain_length):
        method = "fft2d.with_pyfftw"
        fft2d_cls = import_fft_class(method)
        self.fft_engine = fft2d_cls(int(n), int(n))

        kx = self.fft_engine.get_k_adim_loc()[1] * 2 * np.pi / domain_length
        ky = self.fft_engine.get_k_adim_loc()[0] * 2 * np.pi / domain_length
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.dealias_mask = self.K2 < (N_cutoff**2)
        self.viscous_scale = nu * self.K2

    def fft2d(self, variable):
        return self.fft_engine.fft2d(variable)

    def ifft2d(self, variable):
        return self.fft_engine.ifft2d(variable)

    def psi_hat_from_omega_hat(self, omega_hat):
        k2_temp = self.K2.copy()
        k2_temp[0, 0] = 1
        psi_hat = omega_hat / k2_temp
        psi_hat[0, 0] = 0
        return psi_hat

    def omega_hat_from_psi_hat(self, psi_hat):
        k2_temp = self.K2.copy()
        k2_temp[0, 0] = 1
        omega_hat = k2_temp * psi_hat
        omega_hat[0, 0] = 0
        return omega_hat

    def add_forcing(self, omega_hat, dt):
        psi_hat = self.psi_hat_from_omega_hat(omega_hat)
        forcing_range = (np.sqrt(self.K2) >= kinf) & (np.sqrt(self.K2) <= ksup)
        forcing_indices = np.where(forcing_range)
        k2_valid = self.K2[forcing_indices]
        k2_valid[k2_valid == 0] = 1e-30
        factor = dt / (k2_valid * np.pi * (ksup - kinf) * (ksup + kinf))
        std_force = np.sqrt(factor)
        forcing = np.zeros_like(psi_hat, dtype=np.complex128)
        rand_complex = (
            np.random.randn(len(forcing_indices[0])) + 1j * np.random.randn(len(forcing_indices[0]))
        )
        forcing[forcing_indices] = std_force * rand_complex
        psi_hat += forcing
        omega_hat = self.omega_hat_from_psi_hat(psi_hat)
        return omega_hat

    def step(self, omega_hat, dt):
        omega_hat *= self.dealias_mask

        psi_hat = self.psi_hat_from_omega_hat(omega_hat)
        u = self.ifft2d(1j * self.KY * psi_hat).real
        v = self.ifft2d(-1j * self.KX * psi_hat).real

        domega_dx = self.ifft2d(1j * self.KX * omega_hat).real
        domega_dy = self.ifft2d(1j * self.KY * omega_hat).real
        nonlinear_term_hat = self.fft2d(u * domega_dx + v * domega_dy)

        omega_hat = self.add_forcing(omega_hat, dt)
        omega_hat *= np.exp(-alpha * dt * self.K2)
        omega_hat_before_update = omega_hat.copy()
        omega_hat = (
            omega_hat_before_update
            - dt * nonlinear_term_hat
            - dt * self.viscous_scale * omega_hat_before_update
        )
        return omega_hat

    def to_spectral(self, omega):
        return self.fft2d(omega)

    def to_real(self, omega_hat):
        return self.ifft2d(omega_hat).real


class TorchSolver:
    def __init__(self, n, domain_length, device, dtype):
        self.n = n
        self.device = torch.device(device)
        self.real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.complex_dtype = dtype
        self.rng = None

        ky_positive = np.arange(0, n // 2 + 1, dtype=np.float64)
        ky_negative = np.arange(-n // 2 + 1, 0, dtype=np.float64)
        ky = np.concatenate((ky_positive, ky_negative)) * (2 * np.pi / domain_length)
        kx = 2 * np.pi * np.fft.rfftfreq(n, d=domain_length / n)
        kx_t = torch.as_tensor(np.tile(kx, (n, 1)), device=self.device, dtype=self.real_dtype)
        ky_t = torch.as_tensor(np.tile(ky[:, None], (1, kx.shape[0])), device=self.device, dtype=self.real_dtype)
        k2 = kx_t.square() + ky_t.square()
        k2_safe = k2.clone()
        k2_safe[0, 0] = 1.0

        self.KX = kx_t
        self.KY = ky_t
        self.K2 = k2
        self.K2_safe = k2_safe
        self.dealias_mask = (k2 < (N_cutoff**2)).to(dtype)
        self.viscous_scale = nu * k2

        forcing_mask = (torch.sqrt(k2) >= kinf) & (torch.sqrt(k2) <= ksup)
        self.forcing_i, self.forcing_j = torch.where(forcing_mask)
        forcing_k2 = k2[self.forcing_i, self.forcing_j].clone()
        forcing_k2[forcing_k2 == 0] = 1e-30
        self.forcing_factor = 1.0 / (
            forcing_k2 * np.pi * (ksup - kinf) * (ksup + kinf)
        )
        self._cached_dt = None
        self._damping_factor = None

        if self.device.type == "cuda":
            self.rng = torch.Generator(device=self.device)

    def psi_hat_from_omega_hat(self, omega_hat):
        psi_hat = omega_hat / self.K2_safe
        psi_hat[0, 0] = 0
        return psi_hat

    def omega_hat_from_psi_hat(self, psi_hat):
        omega_hat = psi_hat * self.K2_safe
        omega_hat[0, 0] = 0
        return omega_hat

    def add_forcing(self, psi_hat, dt):
        std_force = torch.sqrt(dt * self.forcing_factor)
        real_noise = torch.randn(
            std_force.shape,
            device=self.device,
            dtype=self.real_dtype,
            generator=self.rng,
        )
        imag_noise = torch.randn(
            std_force.shape,
            device=self.device,
            dtype=self.real_dtype,
            generator=self.rng,
        )
        rand_complex = torch.complex(real_noise, imag_noise)
        psi_hat[self.forcing_i, self.forcing_j] += std_force * rand_complex
        return psi_hat

    def damping_factor(self, dt):
        if self._cached_dt != dt:
            self._cached_dt = dt
            self._damping_factor = torch.exp(-alpha * dt * self.K2)
        return self._damping_factor

    def _raw_step(self, omega_hat, dt):
        omega_hat = omega_hat * self.dealias_mask

        psi_hat = self.psi_hat_from_omega_hat(omega_hat)
        u = torch.fft.irfft2(1j * self.KY * psi_hat, s=(self.n, self.n), norm="forward")
        v = torch.fft.irfft2(-1j * self.KX * psi_hat, s=(self.n, self.n), norm="forward")

        domega_dx = torch.fft.irfft2(1j * self.KX * omega_hat, s=(self.n, self.n), norm="forward")
        domega_dy = torch.fft.irfft2(1j * self.KY * omega_hat, s=(self.n, self.n), norm="forward")
        nonlinear_term_hat = torch.fft.rfft2(u * domega_dx + v * domega_dy, norm="forward")

        psi_hat = self.add_forcing(psi_hat, dt)
        omega_hat = self.omega_hat_from_psi_hat(psi_hat)
        omega_hat = omega_hat * self.damping_factor(dt)
        omega_hat_before_update = omega_hat.clone()
        omega_hat = (
            omega_hat_before_update
            - dt * nonlinear_term_hat
            - dt * self.viscous_scale * omega_hat_before_update
        )
        return omega_hat

    def step(self, omega_hat, dt):
        return self._raw_step(omega_hat, dt)

    def to_spectral(self, omega):
        if not isinstance(omega, torch.Tensor):
            omega = torch.as_tensor(np.asarray(omega).real, dtype=self.real_dtype, device=self.device)
        else:
            omega = omega.to(device=self.device, dtype=self.real_dtype).real
        return torch.fft.rfft2(omega, norm="forward")

    def to_real(self, omega_hat):
        return torch.fft.irfft2(omega_hat, s=(self.n, self.n), norm="forward").detach().cpu().numpy()


def choose_solver():
    requested_backend = os.environ.get("PSEUDO_SPECTRAL_BACKEND", "torch").lower()
    requested_device = os.environ.get("PSEUDO_SPECTRAL_DEVICE", "cuda").lower()
    requested_dtype = os.environ.get("PSEUDO_SPECTRAL_DTYPE", "complex128").lower()
    torch_dtype = torch.complex64 if requested_dtype == "complex64" else torch.complex128

    if requested_backend == "auto":
        requested_backend = "torch" if torch.cuda.is_available() else "fluidfft"

    if requested_backend == "torch":
        if requested_device == "cuda" and torch.cuda.is_available():
            return TorchSolver(N, L, "cuda", torch_dtype), "torch-cuda"
        device = "cpu"
        if requested_device != "cpu" and not torch.cuda.is_available():
            print("CUDA is not available in this Python environment. Falling back to torch CPU.")
        return TorchSolver(N, L, device, torch_dtype), f"torch-{device}"

    return FluidFFTSolver(N, L), "fluidfft-cpu"


def initial_condition_real_space(n):
    omega = np.random.randn(n, n)
    omega /= np.max(np.abs(omega))
    return omega


def load_initial_condition(filename):
    mat_file = scipy.io.loadmat(filename)
    keys = [key for key in mat_file.keys() if not key.startswith("__")]
    print("Variables in MAT file:", keys)
    omega = np.asarray(mat_file[keys[0]])
    return np.asarray(np.real(omega), dtype=np.float64)


def time_stepping_with_snapshots(solver, omega_hat, dt, total_time, snapshot_interval):
    snapshots = []
    steps = int(total_time / dt)
    snapshot_steps = int(snapshot_interval / dt)

    for step in tqdm(range(steps)):
        omega_hat = solver.step(omega_hat, dt)
        if step % snapshot_steps == 0:
            snapshots.append(solver.to_real(omega_hat))

    return snapshots


if __name__ == "__main__":
    solver, backend_name = choose_solver()
    print("Selected backend:", backend_name)

    omega = load_initial_condition("vorticity_data.mat")
    # omega = initial_condition_real_space(N)
    if isinstance(solver, TorchSolver):
        omega = torch.as_tensor(omega, dtype=solver.real_dtype, device=solver.device)

    omega_hat = solver.to_spectral(omega)

    T = 5000
    dt = 5e-4
    snapshot_interval = 0.1
    snapshots = time_stepping_with_snapshots(solver, omega_hat, dt, T, snapshot_interval)

    fig, ax = plt.subplots()
    img = ax.imshow(snapshots[0], extent=(0, L, 0, L), origin="lower", cmap="jet")
    ax.set_title("Time Evolution of Vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(img, ax=ax)

    def update(frame):
        img.set_array(snapshots[frame])
        ax.set_title(f"Time Evolution of Vorticity (Frame {frame})")
        img.set_clim(np.min(snapshots[frame]), np.max(snapshots[frame]))
        # cbar.draw_all()
        return [img]

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=True)
    anim.save("fluid_vorticity_evolution.gif", writer="ffmpeg", fps=20)
    plt.show()
