# Navier-Stokes Fourier Neural Operator (FNO) workflow

This folder implements a compact **Fourier Neural Operator** pipeline for vorticity prediction on 2D Navier-Stokes snapshots.

## Files

- `fno2d.py`: minimal FNO-2D implementation (spectral convolution + pointwise residual layers).
- `train_fno_navstokes.py`: trains one-step operator `omega_t -> omega_(t+stride)` from snapshot data.
- `compare_rollout.py`: autoregressive rollout and direct visual comparison against pseudo-spectral snapshots.
- `generate_snapshots.py`: optional data generation with `time_stepping_with_snapshots` from `pseudo_spectral_working.py`.

## 1) Train from existing snapshots

If you already have `omega_snapshots.npy` as described (shape `(T, 256, 256)`):

```bash
python navstokes/train_fno_navstokes.py \
  --data-path omega_snapshots.npy \
  --output-path navstokes/fno_navstokes.pt \
  --stride 4 \
  --epochs 25
```

## 2) Visual rollout comparison (pseudo-spectral vs FNO)

```bash
python navstokes/compare_rollout.py \
  --checkpoint navstokes/fno_navstokes.pt \
  --data-path omega_snapshots.npy \
  --start-index 200 \
  --rollout-steps 20 \
  --sample-every 4 \
  --save-gif
```

Outputs in `navstokes/artifacts/`:

- `rollout_comparison.png` (multi-step side-by-side panel)
- `rollout_comparison.gif` (optional animation)
- `rollout_mse.npy` (per-step MSE curve)

## 3) Optional: generate snapshots with tuned pseudo-spectral settings

This uses the existing solver exactly as-is (no changes to damping/forcing/time-stepping equations):

```bash
python navstokes/generate_snapshots.py \
  --output omega_snapshots.npy \
  --T 5.0 \
  --dt 5e-4 \
  --snapshot-interval 5e-4
```

## Notes

- The FNO model uses coordinate channels and truncated Fourier modes, following the paper's architecture style.
- Rollout is autoregressive: the model repeatedly predicts the next state from its previous prediction.
- For strict apples-to-apples rollout comparisons, evaluate on pseudo-spectral snapshots generated with the same solver process.
