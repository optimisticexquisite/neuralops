import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

import pseudo_spectral_working as ps
from particle_tracking import (
    advect_particles_numpy,
    advect_particles_torch,
    bilinear_interpolate_numpy_batch,
    velocity_from_omega_hat,
)


def int_steps(value, dt, name):
    steps = int(round(value / dt))
    if not np.isclose(steps * dt, value, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name}={value} is not an integer multiple of dt={dt}")
    return steps


def create_storage(output_path, total_samples, num_segments, num_particles, compression):
    compression_value = None if compression.lower() == "none" else compression

    h5_file = h5py.File(output_path, "w")
    samples_group = h5_file.create_group("samples")
    segments_group = h5_file.create_group("segments")

    grid_chunks = (1, ps.N, ps.N)
    particle_chunks = (min(256, total_samples), num_particles, 2)
    scalar_chunks = (min(1024, total_samples),)
    local_chunks = (min(256, total_samples), num_particles)

    samples_group.create_dataset(
        "vorticity_fields",
        shape=(total_samples, ps.N, ps.N),
        dtype=np.float32,
        compression=compression_value,
        chunks=grid_chunks,
    )
    samples_group.create_dataset(
        "wrapped_positions",
        shape=(total_samples, num_particles, 2),
        dtype=np.float32,
        compression=compression_value,
        chunks=particle_chunks,
    )
    samples_group.create_dataset(
        "unwrapped_positions",
        shape=(total_samples, num_particles, 2),
        dtype=np.float64,
        compression=compression_value,
        chunks=particle_chunks,
    )
    samples_group.create_dataset(
        "local_vorticity",
        shape=(total_samples, num_particles),
        dtype=np.float32,
        compression=compression_value,
        chunks=local_chunks,
    )
    samples_group.create_dataset(
        "absolute_time",
        shape=(total_samples,),
        dtype=np.float64,
        compression=compression_value,
        chunks=scalar_chunks,
    )
    samples_group.create_dataset(
        "segment_time",
        shape=(total_samples,),
        dtype=np.float64,
        compression=compression_value,
        chunks=scalar_chunks,
    )
    samples_group.create_dataset(
        "segment_id",
        shape=(total_samples,),
        dtype=np.int32,
        compression=compression_value,
        chunks=scalar_chunks,
    )

    segments_group.create_dataset(
        "start_vorticity",
        shape=(num_segments, ps.N, ps.N),
        dtype=np.float32,
        compression=compression_value,
        chunks=grid_chunks,
    )
    segments_group.create_dataset(
        "initial_wrapped_positions",
        shape=(num_segments, num_particles, 2),
        dtype=np.float32,
        compression=compression_value,
        chunks=(1, num_particles, 2),
    )
    segments_group.create_dataset(
        "initial_unwrapped_positions",
        shape=(num_segments, num_particles, 2),
        dtype=np.float64,
        compression=compression_value,
        chunks=(1, num_particles, 2),
    )
    segments_group.create_dataset(
        "start_time",
        shape=(num_segments,),
        dtype=np.float64,
        compression=compression_value,
        chunks=(min(256, num_segments),),
    )

    return h5_file


def initialize_particles(num_particles, rng, solver):
    initial_positions = rng.uniform(0.0, ps.L, size=(num_particles, 2)).astype(np.float64)

    if isinstance(solver, ps.TorchSolver):
        wrapped = torch.as_tensor(initial_positions, dtype=solver.real_dtype, device=solver.device)
        unwrapped = wrapped.clone()
    else:
        wrapped = initial_positions.copy()
        unwrapped = initial_positions.copy()

    return initial_positions, wrapped, unwrapped


def positions_to_numpy(solver, wrapped_positions, unwrapped_positions):
    if isinstance(solver, ps.TorchSolver):
        wrapped_np = wrapped_positions.detach().cpu().numpy()
        unwrapped_np = unwrapped_positions.detach().cpu().numpy()
    else:
        wrapped_np = wrapped_positions.copy()
        unwrapped_np = unwrapped_positions.copy()
    return wrapped_np, unwrapped_np


def collect_segment(solver, omega_hat, wrapped_positions, unwrapped_positions, dt, steps_per_segment, sample_every):
    samples_per_segment = steps_per_segment // sample_every
    vorticity_buffer = np.empty((samples_per_segment, ps.N, ps.N), dtype=np.float32)
    wrapped_buffer = np.empty((samples_per_segment, wrapped_positions.shape[0], 2), dtype=np.float32)
    unwrapped_buffer = np.empty((samples_per_segment, wrapped_positions.shape[0], 2), dtype=np.float64)
    local_vorticity_buffer = np.empty((samples_per_segment, wrapped_positions.shape[0]), dtype=np.float32)

    buffer_index = 0
    for step_in_segment in range(1, steps_per_segment + 1):
        u_field, v_field = velocity_from_omega_hat(solver, omega_hat)

        if isinstance(solver, ps.TorchSolver):
            wrapped_positions, unwrapped_positions = advect_particles_torch(
                wrapped_positions, unwrapped_positions, u_field, v_field, dt, ps.L, ps.N
            )
        else:
            wrapped_positions, unwrapped_positions = advect_particles_numpy(
                wrapped_positions, unwrapped_positions, u_field, v_field, dt, ps.L, ps.N
            )

        omega_hat = solver.step(omega_hat, dt)

        if step_in_segment % sample_every == 0:
            omega = solver.to_real(omega_hat)
            wrapped_np, unwrapped_np = positions_to_numpy(solver, wrapped_positions, unwrapped_positions)

            vorticity_buffer[buffer_index] = omega.astype(np.float32, copy=False)
            wrapped_buffer[buffer_index] = wrapped_np.astype(np.float32, copy=False)
            unwrapped_buffer[buffer_index] = unwrapped_np
            local_vorticity_buffer[buffer_index] = bilinear_interpolate_numpy_batch(
                omega, wrapped_np, ps.L, ps.N
            ).astype(np.float32, copy=False)
            buffer_index += 1

    return omega_hat, wrapped_positions, unwrapped_positions, vorticity_buffer, wrapped_buffer, unwrapped_buffer, local_vorticity_buffer


def write_segment(
    storage,
    segment_id,
    sample_offset,
    absolute_times,
    segment_times,
    vorticity_buffer,
    wrapped_buffer,
    unwrapped_buffer,
    local_vorticity_buffer,
    initial_positions,
    start_vorticity,
    segment_start_time,
):
    sample_slice = slice(sample_offset, sample_offset + vorticity_buffer.shape[0])

    storage["samples/vorticity_fields"][sample_slice] = vorticity_buffer
    storage["samples/wrapped_positions"][sample_slice] = wrapped_buffer
    storage["samples/unwrapped_positions"][sample_slice] = unwrapped_buffer
    storage["samples/local_vorticity"][sample_slice] = local_vorticity_buffer
    storage["samples/absolute_time"][sample_slice] = absolute_times
    storage["samples/segment_time"][sample_slice] = segment_times
    storage["samples/segment_id"][sample_slice] = segment_id

    storage["segments/start_vorticity"][segment_id] = start_vorticity
    storage["segments/initial_wrapped_positions"][segment_id] = initial_positions.astype(np.float32, copy=False)
    storage["segments/initial_unwrapped_positions"][segment_id] = initial_positions
    storage["segments/start_time"][segment_id] = segment_start_time


def main():
    solver, backend_name = ps.choose_solver()
    print("Selected backend:", backend_name)

    omega = ps.load_initial_condition("vorticity_data.mat")
    if isinstance(solver, ps.TorchSolver):
        omega = torch.as_tensor(omega, dtype=solver.real_dtype, device=solver.device)
    omega_hat = solver.to_spectral(omega)

    total_time = float(os.environ.get("PARTICLE_DATA_TOTAL_TIME", "600"))
    dt = float(os.environ.get("PARTICLE_DATA_DT", "5e-4"))
    sample_interval = float(os.environ.get("PARTICLE_DATA_SAMPLE_INTERVAL", "1e-2"))
    segment_duration = float(os.environ.get("PARTICLE_DATA_SEGMENT_DURATION", "12"))
    num_particles = int(os.environ.get("PARTICLE_DATA_NUM_PARTICLES", "750"))
    output_path = os.environ.get("PARTICLE_DATA_OUTPUT", "particle_dataset.h5")
    compression = os.environ.get("PARTICLE_DATA_COMPRESSION", "lzf")
    seed = int(os.environ.get("PARTICLE_DATA_SEED", "0"))

    steps_per_segment = int_steps(segment_duration, dt, "PARTICLE_DATA_SEGMENT_DURATION")
    sample_every = int_steps(sample_interval, dt, "PARTICLE_DATA_SAMPLE_INTERVAL")
    num_segments = int_steps(total_time, segment_duration, "PARTICLE_DATA_TOTAL_TIME / PARTICLE_DATA_SEGMENT_DURATION")

    if steps_per_segment % sample_every != 0:
        raise ValueError("sample interval must divide the segment duration exactly")

    samples_per_segment = steps_per_segment // sample_every
    total_samples = num_segments * samples_per_segment

    rng = np.random.default_rng(seed)
    storage = create_storage(output_path, total_samples, num_segments, num_particles, compression)
    storage.attrs["backend"] = backend_name
    storage.attrs["dt"] = dt
    storage.attrs["sample_interval"] = sample_interval
    storage.attrs["segment_duration"] = segment_duration
    storage.attrs["total_time"] = total_time
    storage.attrs["num_particles"] = num_particles
    storage.attrs["grid_size"] = ps.N
    storage.attrs["domain_length"] = ps.L
    storage.attrs["seed"] = seed

    sample_offset = 0
    progress = tqdm(total=num_segments * steps_per_segment, desc="Collecting particle dataset")
    try:
        for segment_id in range(num_segments):
            segment_start_time = segment_id * segment_duration
            start_vorticity = solver.to_real(omega_hat).astype(np.float32, copy=False)
            initial_positions, wrapped_positions, unwrapped_positions = initialize_particles(
                num_particles, rng, solver
            )

            (
                omega_hat,
                wrapped_positions,
                unwrapped_positions,
                vorticity_buffer,
                wrapped_buffer,
                unwrapped_buffer,
                local_vorticity_buffer,
            ) = collect_segment(
                solver,
                omega_hat,
                wrapped_positions,
                unwrapped_positions,
                dt,
                steps_per_segment,
                sample_every,
            )

            segment_times = (np.arange(samples_per_segment, dtype=np.float64) + 1.0) * sample_interval
            absolute_times = segment_start_time + segment_times
            write_segment(
                storage,
                segment_id,
                sample_offset,
                absolute_times,
                segment_times,
                vorticity_buffer,
                wrapped_buffer,
                unwrapped_buffer,
                local_vorticity_buffer,
                initial_positions,
                start_vorticity,
                segment_start_time,
            )
            sample_offset += samples_per_segment
            storage.flush()
            progress.update(steps_per_segment)
    finally:
        progress.close()
        storage.close()

    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
