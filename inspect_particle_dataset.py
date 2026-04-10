import argparse
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    repo_relative = Path(__file__).resolve().parent / path_str
    if repo_relative.exists():
        return repo_relative
    return path


def print_attrs(h5_file: h5py.File) -> None:
    print("Attributes:")
    for key in sorted(h5_file.attrs.keys()):
        print(f"  {key}: {h5_file.attrs[key]}")


def print_shapes(name: str, obj) -> None:
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")


def summarize_file(data_path: Path) -> None:
    with h5py.File(data_path, "r") as h5_file:
        print(f"File: {data_path}")
        print_attrs(h5_file)
        print("Datasets:")
        h5_file.visititems(print_shapes)


def build_segment_indices(segment_ids: np.ndarray) -> dict[int, np.ndarray]:
    unique_segments = np.unique(segment_ids)
    return {
        int(segment_id): np.flatnonzero(segment_ids == segment_id)
        for segment_id in unique_segments
    }


def format_vector(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec, dtype=np.float64), precision=6, separator=", ")


def print_trajectory_table(
    segment_times: np.ndarray,
    absolute_times: np.ndarray,
    wrapped_positions: np.ndarray,
    unwrapped_positions: np.ndarray,
    local_vorticity: np.ndarray,
    limit: int | None,
    start_index: int,
) -> None:
    num_rows = segment_times.shape[0]
    if limit is None or limit >= (num_rows - start_index):
        row_indices = np.arange(start_index, num_rows)
    else:
        head = min(limit, num_rows - start_index)
        row_indices = np.arange(start_index, start_index + head)

    print("Trajectory:")
    print("  idx | segment_t | absolute_t | wrapped_x | wrapped_y | unwrapped_x | unwrapped_y | local_omega")
    for idx in row_indices:
        print(
            "  "
            f"{idx:4d} | "
            f"{segment_times[idx]:9.4f} | "
            f"{absolute_times[idx]:10.4f} | "
            f"{wrapped_positions[idx, 0]:9.5f} | "
            f"{wrapped_positions[idx, 1]:9.5f} | "
            f"{unwrapped_positions[idx, 0]:11.5f} | "
            f"{unwrapped_positions[idx, 1]:11.5f} | "
            f"{local_vorticity[idx]:11.5f}"
        )
    if start_index > 0:
        print(f"  ... skipped rows [0, {start_index - 1}] because local_omega is unavailable there")
    if limit is not None and limit < (num_rows - start_index):
        print(f"  ... showing first {limit} of {num_rows - start_index} rows starting at idx={start_index}")


def inspect_particle(
    data_path: Path,
    segment_id: int,
    particle_id: int,
    full_trajectory: bool,
    limit: int | None,
) -> None:
    with h5py.File(data_path, "r") as h5_file:
        num_segments = int(h5_file["segments/start_time"].shape[0])
        num_particles = int(h5_file["segments/initial_wrapped_positions"].shape[1])
        if not 0 <= segment_id < num_segments:
            raise ValueError(f"segment_id must be in [0, {num_segments - 1}]")
        if not 0 <= particle_id < num_particles:
            raise ValueError(f"particle_id must be in [0, {num_particles - 1}]")

        sample_segment_ids = np.asarray(h5_file["samples/segment_id"][:], dtype=np.int64)
        segment_indices_map = build_segment_indices(sample_segment_ids)
        segment_indices = segment_indices_map[segment_id]

        domain_length = float(h5_file.attrs["domain_length"])
        sample_interval = float(h5_file.attrs["sample_interval"])
        segment_duration = float(h5_file.attrs["segment_duration"])

        start_time = float(h5_file["segments/start_time"][segment_id])
        start_vorticity = np.asarray(h5_file["segments/start_vorticity"][segment_id], dtype=np.float32)
        initial_wrapped = np.asarray(
            h5_file["segments/initial_wrapped_positions"][segment_id, particle_id],
            dtype=np.float64,
        )
        initial_unwrapped = np.asarray(
            h5_file["segments/initial_unwrapped_positions"][segment_id, particle_id],
            dtype=np.float64,
        )

        sampled_segment_times = np.asarray(
            h5_file["samples/segment_time"][segment_indices],
            dtype=np.float64,
        )
        sampled_absolute_times = np.asarray(
            h5_file["samples/absolute_time"][segment_indices],
            dtype=np.float64,
        )
        sampled_wrapped = np.asarray(
            h5_file["samples/wrapped_positions"][segment_indices, particle_id],
            dtype=np.float64,
        )
        sampled_unwrapped = np.asarray(
            h5_file["samples/unwrapped_positions"][segment_indices, particle_id],
            dtype=np.float64,
        )
        sampled_local_vorticity = np.asarray(
            h5_file["samples/local_vorticity"][segment_indices, particle_id],
            dtype=np.float64,
        )

    segment_times = np.concatenate(([0.0], sampled_segment_times))
    absolute_times = np.concatenate(([start_time], sampled_absolute_times))
    wrapped_positions = np.vstack((initial_wrapped[None, :], sampled_wrapped))
    unwrapped_positions = np.vstack((initial_unwrapped[None, :], sampled_unwrapped))
    local_vorticity = np.concatenate(([np.nan], sampled_local_vorticity))
    display_start_index = 1 if np.isnan(local_vorticity[0]) else 0

    displacement = unwrapped_positions[-1] - unwrapped_positions[0]
    wrapped_jumps = np.linalg.norm(np.diff(wrapped_positions, axis=0), axis=1)
    boundary_crossings = int(np.sum(np.any(np.abs(np.diff(wrapped_positions, axis=0)) > 0.5 * domain_length, axis=1)))

    print(f"File: {data_path}")
    print(f"Segment: {segment_id} / {num_segments - 1}")
    print(f"Particle: {particle_id} / {num_particles - 1}")
    print("Segment metadata:")
    print(f"  start_time: {start_time}")
    print(f"  sample_interval: {sample_interval}")
    print(f"  segment_duration: {segment_duration}")
    print(f"  num_saved_frames_including_start: {segment_times.shape[0]}")
    print(f"  start_vorticity_shape: {start_vorticity.shape}")
    print("Particle state:")
    print(f"  initial_wrapped: {format_vector(wrapped_positions[0])}")
    print(f"  final_wrapped:   {format_vector(wrapped_positions[-1])}")
    print(f"  initial_unwrapped: {format_vector(unwrapped_positions[0])}")
    print(f"  final_unwrapped:   {format_vector(unwrapped_positions[-1])}")
    print(f"  total_unwrapped_displacement: {format_vector(displacement)}")
    print(f"  wrapped_min_xy: {format_vector(np.min(wrapped_positions, axis=0))}")
    print(f"  wrapped_max_xy: {format_vector(np.max(wrapped_positions, axis=0))}")
    print(f"  unwrapped_min_xy: {format_vector(np.min(unwrapped_positions, axis=0))}")
    print(f"  unwrapped_max_xy: {format_vector(np.max(unwrapped_positions, axis=0))}")
    print(f"  estimated_boundary_crossings: {boundary_crossings}")
    print(f"  max_wrapped_step_norm: {wrapped_jumps.max() if wrapped_jumps.size else 0.0:.6f}")

    valid_local_vorticity = sampled_local_vorticity
    print("Local vorticity along trajectory:")
    print(f"  min:  {np.min(valid_local_vorticity):.6f}")
    print(f"  max:  {np.max(valid_local_vorticity):.6f}")
    print(f"  mean: {np.mean(valid_local_vorticity):.6f}")
    print(f"  std:  {np.std(valid_local_vorticity):.6f}")

    print_trajectory_table(
        segment_times=segment_times,
        absolute_times=absolute_times,
        wrapped_positions=wrapped_positions,
        unwrapped_positions=unwrapped_positions,
        local_vorticity=local_vorticity,
        limit=None if full_trajectory else limit,
        start_index=display_start_index,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect particle_dataset_prev.h5 or any compatible particle HDF5 dataset.",
    )
    parser.add_argument("--data-path", type=str, default="particle_dataset_prev.h5")
    parser.add_argument("--segment-id", type=int, default=None)
    parser.add_argument("--particle-id", type=int, default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print file attributes and dataset shapes.",
    )
    parser.add_argument(
        "--full-trajectory",
        action="store_true",
        help="Print every saved timestep for the chosen particle.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of trajectory rows to print when --full-trajectory is not used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_path(args.data_path)
    summarize_file(data_path)

    if args.summary_only:
        return

    if args.segment_id is None or args.particle_id is None:
        print(
            "\nSet both --segment-id and --particle-id to inspect one particle trajectory in detail."
        )
        return

    print()
    inspect_particle(
        data_path=data_path,
        segment_id=args.segment_id,
        particle_id=args.particle_id,
        full_trajectory=args.full_trajectory,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
