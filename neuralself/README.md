# FNO3D Navier-Stokes Benchmark

This folder contains a CUDA-oriented training script for the paper-style `FNO-3D` Navier-Stokes benchmark from `2010.08895`.

## Expected data

The script expects the original 64x64 MATLAB files from the FNO benchmark:

- `ns_data_V100_N1000_T50_1.mat`
- `ns_data_V100_N1000_T50_2.mat`

It can also work with a single larger Navier-Stokes MATLAB file such as:

- `NavierStokes_V1e-3_N5000_T50.mat`

Place them under:

```text
neuralself/data/
```

Each file should contain a field named `u` with shape compatible with:

```text
[samples, 64, 64, 50]
```

The benchmark split used by the paper-style script is:

- first `10` frames as input
- next `40` frames as target rollout
- `1000` training samples
- `200` test samples

## Run

From the repo root:

```bash
python neuralself/train_fno3d_navier_stokes.py \
  --train-path neuralself/data/ns_data_V100_N1000_T50_1.mat \
  --test-path neuralself/data/ns_data_V100_N1000_T50_2.mat \
  --output-dir neuralself/checkpoints/fno3d_ns64 \
  --amp \
  --num-workers 4
```

If you only have the monolithic file:

```bash
python neuralself/train_fno3d_navier_stokes.py \
  --data-path neuralself/data/NavierStokes_V1e-3_N5000_T50.mat \
  --ntrain 1000 \
  --ntest 200 \
  --output-dir neuralself/checkpoints/fno3d_ns64 \
  --amp \
  --num-workers 4
```

On PyTorch 2.x with a recent NVIDIA GPU, you can also try:

```bash
python neuralself/train_fno3d_navier_stokes.py \
  --train-path neuralself/data/ns_data_V100_N1000_T50_1.mat \
  --test-path neuralself/data/ns_data_V100_N1000_T50_2.mat \
  --output-dir neuralself/checkpoints/fno3d_ns64 \
  --amp \
  --compile
```

## Notes

- Defaults follow the original paper script closely: `modes=8`, `width=20`, `batch_size=10`, `epochs=500`, `lr=1e-3`, `T_in=10`, `T_out=40`.
- The implementation is modernized for speed with pinned-memory loaders, non-blocking CUDA transfers, TF32, and AMP.
- The relative `L2` metric is computed in the decoded physical space, matching the benchmark intent more closely than plain normalized MSE.
