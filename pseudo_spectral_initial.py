
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.fft import fft2, ifft2, fftshift, fftfreq
# from fluidfft.fft2d import FFT2D
from fluidfft import import_fft_class
import fluidfft
from tqdm import tqdm



L = 2 * np.pi
N = 256
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
nu = 0.002
X, Y = np.meshgrid(x, y)
kx = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
ky = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

# Choose the FFT method
method = 'fft2d.with_pyfftw'
plugins = fluidfft.get_plugins()
# print(plugins)
# Import the FFT class using the chosen method
FFT2D = import_fft_class(method)

# Create an instance of the FFT class
fft2d = FFT2D(int(N), int(N))
# Get kx and ky from fluidfft
kx = fft2d.get_k_adim_loc()[1] * 2 * np.pi / L
ky = fft2d.get_k_adim_loc()[0] * 2 * np.pi / L
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
# Assuming K2 is the squared magnitude of the wavenumber vector
N_cutoff = int(2/3 * N)  # Applying the 2/3 rule
dealias_mask = (KX**2 + KY**2) < (N_cutoff**2)
kinf = 1  # Lower bound for forcing wavenumbers
ksup = 4  # Upper bound for forcing wavenumbers
alpha = 0.05  # Damping coefficient

def fluidfft_fft2d(variable):
    return fft2d.fft2d(variable)

def fluidfft_ifft2d(variable):
    return fft2d.ifft2d(variable)

def initial_condition(X, Y):
    omega = 2 * np.sin(X) * np.sin(Y)
    # print(omega.shape)
    return omega

def psi_hat_fn(omega_hat):
    K2_temp = K2.copy()
    K2_temp[0, 0] = 1
    psi_hat = omega_hat / K2_temp
    psi_hat[0, 0] = 0
    return psi_hat

def velocity(omega_hat):
    psi_hat = psi_hat_fn(omega_hat)
    u = fft2d.ifft2d(1j * KY * psi_hat).real
    v = fft2d.ifft2d(-1j * KX * psi_hat).real
    return u, v

def velocity_from_omega(omega):
    omega_hat = fft2d.fft2d(omega)
    psi_hat = psi_hat_fn(omega_hat)
    u = fft2d.ifft2d(1j * KY * psi_hat).real
    v = fft2d.ifft2d(-1j * KX * psi_hat).real
    return u, v

def calculate_nonlinear_term_hat(omega_hat, u, v):
    domega_dx_hat = 1j * KX * omega_hat
    domega_dy_hat = 1j * KY * omega_hat
    domega_dx = fft2d.ifft2d(domega_dx_hat).real
    domega_dy = fft2d.ifft2d(domega_dy_hat).real

    nonlinear_term = u * domega_dx + v * domega_dy

    nonlinear_term_hat = fft2d.fft2d(nonlinear_term)
    return nonlinear_term_hat

def add_forcing(omega_hat, kinf, ksup, dt):
    forcing_range = (K2 >= (kinf**2)) & (K2 <= (ksup**2))
    # Simple forcing: add a small amount of energy at specified wavenumbers
    forcing_amplitude = 1e-3  # Adjust based on your simulation needs
    omega_hat[forcing_range] += forcing_amplitude * dt
    return omega_hat

def apply_damping(omega_hat, alpha, dt):
    # Apply damping in the spectral domain
    return omega_hat * np.exp(-alpha * dt * K2)

def time_stepping(omega_hat, dt):
    omega_hat = omega_hat * dealias_mask
    u, v = velocity(omega_hat)
    # K2_temp = K2.copy()
    # K2_temp[0, 0] = 1
    nonlinear_term_hat = calculate_nonlinear_term_hat(omega_hat, u, v) * dealias_mask
    omega_hat = add_forcing(omega_hat, kinf, ksup, dt)
    omega_hat = apply_damping(omega_hat, alpha, dt) - dt * nonlinear_term_hat - dt * K2 * omega_hat * nu
    #omega = fft2d.ifft2d(omega_hat).real
    # print(energy(omega))
    return omega_hat

def time_stepping_analytical(omega, dt):
    omega = omega * np.exp(-2 * nu * dt)
    return omega




def time_stepping_complete(omega, omega_hat, dt, T):
    for _ in tqdm(range(int(T/dt))):
        u, v = velocity(omega_hat)
        K2_temp = K2.copy()
        # K2_temp[0, 0] = 1
        omega_hat = omega_hat - dt * calculate_nonlinear_term_hat(omega_hat, u, v) - dt * K2_temp * omega_hat * nu
        # omega = fft2d.ifft2d(omega_hat).real
        # print(energy(omega))
    return omega_hat

def energy(omega):
    return np.sum(omega**2) 

def error(omega, omega_exact):
    return omega - omega_exact




if __name__ == '__main__':

    omega = initial_condition(X, Y)
    omega_hat = fft2d.fft2d(omega)
    u, v = velocity(omega_hat)
    T = 5
    dt = 5e-4
    omega_hat = time_stepping_complete(omega, omega_hat, dt, T)
    omega = fft2d.ifft2d(omega_hat).real

    def analytical_omega(X, Y, t):
        omega = 2 * np.sin(X) * np.sin(Y) * np.exp(-2 * nu * t)
        return omega

    plt.figure()
    plt.imshow(omega, cmap='jet')
    plt.colorbar()
    plt.savefig('omega.png')


    plt.figure()
    plt.imshow(analytical_omega(X, Y, T), cmap='jet')
    plt.colorbar()
    plt.savefig('omega_exact.png')


    plt.figure()
    plt.imshow(error(omega, analytical_omega(X, Y, T)), cmap='jet')
    plt.colorbar()
    plt.savefig('error.png')
    # print(error(omega, analytical_omega(X, Y, T)))

    print(u, v)

    def analytical_velocity(X, Y, t):
        u = np.sin(X) * np.cos(Y) * np.exp(-2 * t)
        v = -np.cos(X) * np.sin(Y) * np.exp(-2 * t)
        return u, v
    print("------")
    u_exact, v_exact = analytical_velocity(X, Y, 0)
    # print(u_exact, v_exact)
    def error(u, v, u_exact, v_exact):
        return np.sqrt(np.sum((u - u_exact)**2 + (v - v_exact)**2))

    def scaling(u, v, u_exact, v_exact):
        return np.average(u_exact / u), np.average(v_exact / v)


    print(error(u, v, u_exact, v_exact))
    print(scaling(u, v, u_exact, v_exact))
    plt.figure()
    plt.imshow(u, cmap='jet')
    plt.colorbar()
    plt.title('u')
    plt.savefig('u.png')

    plt.figure()
    plt.imshow(u_exact, cmap='jet')
    plt.colorbar()
    plt.title('u_exact')
    plt.savefig('u_exact.png')

    plt.figure()
    plt.imshow(v, cmap='jet')
    plt.colorbar()
    plt.title('v')
    plt.show()

    plt.figure()
    plt.imshow(v_exact, cmap='jet')
    plt.colorbar()
    plt.title('v_exact')
    plt.show()
