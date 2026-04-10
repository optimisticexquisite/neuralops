import argparse
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    repo_relative = Path(__file__).resolve().parent / path_str
    if repo_relative.exists():
        return repo_relative
    return path


def build_wrapped_polyline(wrapped_positions: np.ndarray, domain_length: float):
    xs = [float(wrapped_positions[0, 0])]
    ys = [float(wrapped_positions[0, 1])]
    lengths = [1]

    for frame_index in range(1, wrapped_positions.shape[0]):
        previous = wrapped_positions[frame_index - 1]
        current = wrapped_positions[frame_index]
        if np.any(np.abs(current - previous) > 0.5 * domain_length):
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(float(current[0]))
        ys.append(float(current[1]))
        lengths.append(len(xs))

    return np.asarray(xs), np.asarray(ys), np.asarray(lengths, dtype=np.int64)


def compute_centered_limits(unwrapped_positions: np.ndarray, domain_length: float):
    center = 0.5 * domain_length
    min_x = min(0.0, float(unwrapped_positions[:, 0].min()))
    max_x = max(domain_length, float(unwrapped_positions[:, 0].max()))
    min_y = min(0.0, float(unwrapped_positions[:, 1].min()))
    max_y = max(domain_length, float(unwrapped_positions[:, 1].max()))

    radius_x = max(abs(min_x - center), abs(max_x - center))
    radius_y = max(abs(min_y - center), abs(max_y - center))
    margin = 0.08 * max(2.0 * radius_x, 2.0 * radius_y, domain_length)

    return (
        center - radius_x - margin,
        center + radius_x + margin,
        center - radius_y - margin,
        center + radius_y + margin,
    )


def load_segment_particle_data(
    data_path: Path,
    segment_id: int,
    particle_id: int,
    frame_step: int,
    max_frames: int | None,
):
    with h5py.File(data_path, "r") as h5_file:
        domain_length = float(h5_file.attrs["domain_length"])
        num_segments = int(h5_file["segments/start_time"].shape[0])
        num_particles = int(h5_file["segments/initial_wrapped_positions"].shape[1])

        if not 0 <= segment_id < num_segments:
            raise ValueError(f"segment_id must be in [0, {num_segments - 1}]")
        if not 0 <= particle_id < num_particles:
            raise ValueError(f"particle_id must be in [0, {num_particles - 1}]")

        segment_mask = np.asarray(h5_file["samples/segment_id"][:], dtype=np.int64) == segment_id
        segment_indices = np.flatnonzero(segment_mask)
        if segment_indices.size == 0:
            raise ValueError(f"No samples found for segment_id={segment_id}")

        start_vorticity = np.asarray(h5_file["segments/start_vorticity"][segment_id], dtype=np.float32)
        start_wrapped = np.asarray(
            h5_file["segments/initial_wrapped_positions"][segment_id, particle_id],
            dtype=np.float64,
        )
        start_unwrapped = np.asarray(
            h5_file["segments/initial_unwrapped_positions"][segment_id, particle_id],
            dtype=np.float64,
        )
        start_time = float(h5_file["segments/start_time"][segment_id])

        sampled_vorticity = np.asarray(
            h5_file["samples/vorticity_fields"][segment_indices],
            dtype=np.float32,
        )
        sampled_wrapped = np.asarray(
            h5_file["samples/wrapped_positions"][segment_indices, particle_id],
            dtype=np.float64,
        )
        sampled_unwrapped = np.asarray(
            h5_file["samples/unwrapped_positions"][segment_indices, particle_id],
            dtype=np.float64,
        )
        sampled_segment_time = np.asarray(
            h5_file["samples/segment_time"][segment_indices],
            dtype=np.float64,
        )
        sampled_absolute_time = np.asarray(
            h5_file["samples/absolute_time"][segment_indices],
            dtype=np.float64,
        )

    vorticity_frames = np.concatenate((start_vorticity[None, ...], sampled_vorticity), axis=0)
    wrapped_positions = np.concatenate((start_wrapped[None, ...], sampled_wrapped), axis=0)
    unwrapped_positions = np.concatenate((start_unwrapped[None, ...], sampled_unwrapped), axis=0)
    segment_times = np.concatenate((np.array([0.0]), sampled_segment_time), axis=0)
    absolute_times = np.concatenate((np.array([start_time]), sampled_absolute_time), axis=0)

    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    frame_indices = np.arange(0, vorticity_frames.shape[0], frame_step, dtype=np.int64)
    if frame_indices[-1] != vorticity_frames.shape[0] - 1:
        frame_indices = np.concatenate((frame_indices, np.array([vorticity_frames.shape[0] - 1], dtype=np.int64)))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    vorticity_frames = vorticity_frames[frame_indices]
    wrapped_positions = wrapped_positions[frame_indices]
    unwrapped_positions = unwrapped_positions[frame_indices]
    segment_times = segment_times[frame_indices]
    absolute_times = absolute_times[frame_indices]

    return {
        "domain_length": domain_length,
        "segment_id": segment_id,
        "particle_id": particle_id,
        "vorticity_frames": vorticity_frames,
        "wrapped_positions": wrapped_positions,
        "unwrapped_positions": unwrapped_positions,
        "segment_times": segment_times,
        "absolute_times": absolute_times,
    }


def style_axes(ax):
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def save_animation(animation, output_path: Path, fps: int):
    output_str = str(output_path)
    if output_str.lower().endswith(".gif"):
        animation.save(output_str, writer=PillowWriter(fps=fps))
        return output_str

    try:
        animation.save(output_str, writer=FFMpegWriter(fps=fps, codec="mpeg4"))
        return output_str
    except Exception:
        fallback_path = output_path.with_suffix(".gif")
        animation.save(str(fallback_path), writer=PillowWriter(fps=fps))
        return str(fallback_path)


def build_animation(data, output_path: Path, fps: int):
    domain_length = data["domain_length"]
    vorticity_frames = data["vorticity_frames"]
    wrapped_positions = data["wrapped_positions"]
    unwrapped_positions = data["unwrapped_positions"]
    segment_times = data["segment_times"]
    absolute_times = data["absolute_times"]

    x_min, x_max, y_min, y_max = compute_centered_limits(unwrapped_positions, domain_length)
    wrapped_line_x, wrapped_line_y, wrapped_line_lengths = build_wrapped_polyline(
        wrapped_positions,
        domain_length,
    )

    vmin = float(vorticity_frames.min())
    vmax = float(vorticity_frames.max())
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    style_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    image = ax.imshow(
        vorticity_frames[0],
        extent=(0.0, domain_length, 0.0, domain_length),
        origin="lower",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        zorder=1,
    )
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            domain_length,
            domain_length,
            fill=False,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )
    )

    wrapped_line, = ax.plot(
        wrapped_line_x[: wrapped_line_lengths[0]],
        wrapped_line_y[: wrapped_line_lengths[0]],
        color="cyan",
        linewidth=1.2,
        alpha=0.9,
        zorder=4,
        label="Wrapped path",
    )
    unwrapped_line, = ax.plot(
        unwrapped_positions[:1, 0],
        unwrapped_positions[:1, 1],
        color="orange",
        linewidth=1.5,
        alpha=0.95,
        zorder=2,
        label="Unwrapped path",
    )
    wrapped_marker = ax.scatter(
        [wrapped_positions[0, 0]],
        [wrapped_positions[0, 1]],
        s=48,
        c="cyan",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="Wrapped position",
    )
    unwrapped_marker = ax.scatter(
        [unwrapped_positions[0, 0]],
        [unwrapped_positions[0, 1]],
        s=58,
        c="orange",
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
        label="Unwrapped position",
    )

    title = ax.set_title(
        (
            f"Segment {data['segment_id']} | Particle {data['particle_id']} | "
            f"segment t = {segment_times[0]:.2f} | abs t = {absolute_times[0]:.2f}"
        ),
        color="white",
    )

    legend = ax.legend(loc="upper right", facecolor="black", edgecolor="white", framealpha=0.85)
    for text in legend.get_texts():
        text.set_color("white")

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    colorbar.set_label("Vorticity", color="white")
    colorbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(colorbar.ax.get_yticklabels(), color="white")

    def update(frame_index):
        image.set_array(vorticity_frames[frame_index])
        wrapped_line.set_data(
            wrapped_line_x[: wrapped_line_lengths[frame_index]],
            wrapped_line_y[: wrapped_line_lengths[frame_index]],
        )
        unwrapped_line.set_data(
            unwrapped_positions[: frame_index + 1, 0],
            unwrapped_positions[: frame_index + 1, 1],
        )
        wrapped_marker.set_offsets(wrapped_positions[frame_index][None, :])
        unwrapped_marker.set_offsets(unwrapped_positions[frame_index][None, :])
        title.set_text(
            (
                f"Segment {data['segment_id']} | Particle {data['particle_id']} | "
                f"segment t = {segment_times[frame_index]:.2f} | "
                f"abs t = {absolute_times[frame_index]:.2f}"
            )
        )
        return [image, wrapped_line, unwrapped_line, wrapped_marker, unwrapped_marker, title]

    animation = FuncAnimation(
        fig,
        update,
        frames=vorticity_frames.shape[0],
        interval=max(1, int(round(1000 / max(fps, 1)))),
        blit=True,
    )

    saved_path = save_animation(animation, output_path, fps=fps)
    plt.close(fig)
    return saved_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate one particle from particle_dataset.h5 with both wrapped and unwrapped positions.",
    )
    parser.add_argument("--data-path", type=str, default="particle_dataset.h5")
    parser.add_argument("--segment-id", type=int, default=0)
    parser.add_argument("--particle-id", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default="dataset_particle_segment0_particle0.gif")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_path(args.data_path)
    output_path = resolve_path(args.output)

    data = load_segment_particle_data(
        data_path=data_path,
        segment_id=args.segment_id,
        particle_id=args.particle_id,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    saved_path = build_animation(data, output_path=output_path, fps=args.fps)
    print(f"Saved animation to {saved_path}")


if __name__ == "__main__":
    main()
