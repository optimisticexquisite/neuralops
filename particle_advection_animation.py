import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from tqdm import tqdm

import pseudo_spectral_working as ps
from particle_tracking import advect_particle_numpy, advect_particle_torch, velocity_from_omega_hat


def simulate_particle(solver, omega_hat, dt, total_time, snapshot_interval):
    steps = int(total_time / dt)
    snapshot_steps = int(snapshot_interval / dt)

    snapshots = []
    particle_positions = []
    frame_times = []

    if isinstance(solver, ps.TorchSolver):
        position = torch.tensor(
            [ps.L / 2.0, ps.L / 2.0], device=solver.device, dtype=solver.real_dtype
        )
    else:
        position = np.array([ps.L / 2.0, ps.L / 2.0], dtype=np.float64)

    for step in tqdm(range(steps)):
        u_field, v_field = velocity_from_omega_hat(solver, omega_hat)

        if isinstance(solver, ps.TorchSolver):
            position = advect_particle_torch(position, u_field, v_field, dt, ps.L, ps.N)
        else:
            position = advect_particle_numpy(position, u_field, v_field, dt, ps.L, ps.N)

        omega_hat = solver.step(omega_hat, dt)

        if step % snapshot_steps == 0:
            snapshots.append(solver.to_real(omega_hat))
            if isinstance(solver, ps.TorchSolver):
                particle_positions.append(position.detach().cpu().numpy().copy())
            else:
                particle_positions.append(position.copy())
            frame_times.append((step + 1) * dt)

    return snapshots, particle_positions, frame_times


def build_animation(snapshots, particle_positions, frame_times, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(
        snapshots[0],
        extent=(0, ps.L, 0, ps.L),
        origin="lower",
        cmap="jet",
        animated=True,
    )
    particle = ax.scatter(
        [particle_positions[0][0]],
        [particle_positions[0][1]],
        s=45,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_xlim(0, ps.L)
    ax.set_ylim(0, ps.L)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Passive Particle in Evolving Turbulent Flow, t = {frame_times[0]:.2f}")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Vorticity")

    def update(frame_index):
        frame = snapshots[frame_index]
        img.set_array(frame)
        img.set_clim(np.min(frame), np.max(frame))
        particle.set_offsets(particle_positions[frame_index][None, :])
        ax.set_title(
            f"Passive Particle in Evolving Turbulent Flow, t = {frame_times[frame_index]:.2f}"
        )
        return [img, particle]

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=True)

    if output_path.lower().endswith(".gif"):
        anim.save(output_path, writer=PillowWriter(fps=20))
        plt.close(fig)
        return output_path

    try:
        anim.save(output_path, writer=FFMpegWriter(fps=20, codec="mpeg4"))
        plt.close(fig)
        return output_path
    except Exception:
        fallback_path = os.path.splitext(output_path)[0] + ".gif"
        anim.save(fallback_path, writer=PillowWriter(fps=20))
        plt.close(fig)
        return fallback_path


def main():
    solver, backend_name = ps.choose_solver()
    print("Selected backend:", backend_name)

    omega = ps.load_initial_condition("vorticity_data.mat")
    if isinstance(solver, ps.TorchSolver):
        omega = torch.as_tensor(omega, dtype=solver.real_dtype, device=solver.device)

    omega_hat = solver.to_spectral(omega)

    total_time = float(os.environ.get("PARTICLE_TOTAL_TIME", "5"))
    dt = float(os.environ.get("PARTICLE_DT", "5e-4"))
    snapshot_interval = float(os.environ.get("PARTICLE_SNAPSHOT_INTERVAL", "5e-4"))
    output_path = os.environ.get("PARTICLE_OUTPUT", "particle_advection_evolution.gif")

    snapshots, particle_positions, frame_times = simulate_particle(
        solver, omega_hat, dt, total_time, snapshot_interval
    )
    saved_path = build_animation(snapshots, particle_positions, frame_times, output_path)
    print(f"Saved animation to {saved_path}")


if __name__ == "__main__":
    main()
