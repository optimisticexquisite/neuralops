import numpy as np
import torch

import pseudo_spectral_working as ps


def velocity_from_omega_hat(solver, omega_hat):
    psi_hat = solver.psi_hat_from_omega_hat(omega_hat)

    if isinstance(solver, ps.TorchSolver):
        u = torch.fft.irfft2(1j * solver.KY * psi_hat, s=(solver.n, solver.n), norm="forward")
        v = torch.fft.irfft2(-1j * solver.KX * psi_hat, s=(solver.n, solver.n), norm="forward")
        return u, v

    u = solver.ifft2d(1j * solver.KY * psi_hat).real
    v = solver.ifft2d(-1j * solver.KX * psi_hat).real
    return u, v


def bilinear_interpolate_numpy_batch(field, positions, domain_length, n):
    dx = domain_length / n
    x_pos = np.remainder(positions[:, 0], domain_length) / dx
    y_pos = np.remainder(positions[:, 1], domain_length) / dx

    x0_float = np.floor(x_pos)
    y0_float = np.floor(y_pos)
    x0 = x0_float.astype(np.int64) % n
    y0 = y0_float.astype(np.int64) % n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n

    tx = x_pos - x0_float
    ty = y_pos - y0_float

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def bilinear_interpolate_torch_batch(field, positions, domain_length, n):
    dx = domain_length / n
    x_pos = torch.remainder(positions[:, 0], domain_length) / dx
    y_pos = torch.remainder(positions[:, 1], domain_length) / dx

    x0_float = torch.floor(x_pos)
    y0_float = torch.floor(y_pos)
    x0 = x0_float.to(torch.long) % n
    y0 = y0_float.to(torch.long) % n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n

    tx = x_pos - x0_float
    ty = y_pos - y0_float

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def bilinear_interpolate_numpy(field, position, domain_length, n):
    values = bilinear_interpolate_numpy_batch(field, np.asarray(position, dtype=np.float64)[None, :], domain_length, n)
    return float(values[0])


def bilinear_interpolate_torch(field, position, domain_length, n):
    if position.ndim == 1:
        position_batch = position[None, :]
    else:
        position_batch = position
    values = bilinear_interpolate_torch_batch(field, position_batch, domain_length, n)
    return values[0]


def advect_particles_numpy(wrapped_positions, unwrapped_positions, u_field, v_field, dt, domain_length, n):
    velocity_now = np.stack(
        [
            bilinear_interpolate_numpy_batch(u_field, wrapped_positions, domain_length, n),
            bilinear_interpolate_numpy_batch(v_field, wrapped_positions, domain_length, n),
        ],
        axis=1,
    )
    wrapped_midpoint = np.mod(wrapped_positions + 0.5 * dt * velocity_now, domain_length)
    velocity_mid = np.stack(
        [
            bilinear_interpolate_numpy_batch(u_field, wrapped_midpoint, domain_length, n),
            bilinear_interpolate_numpy_batch(v_field, wrapped_midpoint, domain_length, n),
        ],
        axis=1,
    )
    new_wrapped = np.mod(wrapped_positions + dt * velocity_mid, domain_length)
    new_unwrapped = unwrapped_positions + dt * velocity_mid
    return new_wrapped, new_unwrapped


def advect_particles_torch(wrapped_positions, unwrapped_positions, u_field, v_field, dt, domain_length, n):
    velocity_now = torch.stack(
        [
            bilinear_interpolate_torch_batch(u_field, wrapped_positions, domain_length, n),
            bilinear_interpolate_torch_batch(v_field, wrapped_positions, domain_length, n),
        ],
        dim=1,
    )
    wrapped_midpoint = torch.remainder(wrapped_positions + 0.5 * dt * velocity_now, domain_length)
    velocity_mid = torch.stack(
        [
            bilinear_interpolate_torch_batch(u_field, wrapped_midpoint, domain_length, n),
            bilinear_interpolate_torch_batch(v_field, wrapped_midpoint, domain_length, n),
        ],
        dim=1,
    )
    new_wrapped = torch.remainder(wrapped_positions + dt * velocity_mid, domain_length)
    new_unwrapped = unwrapped_positions + dt * velocity_mid
    return new_wrapped, new_unwrapped


def advect_particle_numpy(position, u_field, v_field, dt, domain_length, n):
    wrapped, _ = advect_particles_numpy(
        np.asarray(position, dtype=np.float64)[None, :],
        np.asarray(position, dtype=np.float64)[None, :],
        u_field,
        v_field,
        dt,
        domain_length,
        n,
    )
    return wrapped[0]


def advect_particle_torch(position, u_field, v_field, dt, domain_length, n):
    if position.ndim == 1:
        wrapped_positions = position[None, :]
    else:
        wrapped_positions = position
    wrapped, _ = advect_particles_torch(
        wrapped_positions,
        wrapped_positions.clone(),
        u_field,
        v_field,
        dt,
        domain_length,
        n,
    )
    return wrapped[0]
