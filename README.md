# Neural Operators for Lagrangian Tracer Prediction in 2D Turbulence

## 1. Project Motivation

This repository is built around a very specific and interesting research question:

Can a neural operator, which is designed to learn mappings between functions, outperform a more local architecture such as a CNN when predicting the future position of a Lagrangian tracer in a turbulent flow?

The intuition behind this question is strong:

- A CNN is good at extracting local spatial patterns and composing them hierarchically.
- A Fourier Neural Operator (FNO) has direct access to global spectral structure and long-range interactions.
- Turbulent advection is not just a local image-processing problem. Even when a particle’s instantaneous motion is local, its long-time displacement can depend on coherent structures, multiscale interactions, and large-scale organization of the flow.

So the core scientific hypothesis appears to be:

For short or moderate horizons, a CNN may perform competitively because the particle displacement is dominated by local flow structure near the initial position. For longer horizons, global flow organization may matter more, and an FNO may begin to show an advantage.

That is a good hypothesis. It is not guaranteed to be true, but it is absolutely worth testing carefully.

## 2. The Central Research Problem

At a high level, the repository currently frames the learning problem as:

- Input: a 2D initial vorticity field `omega(x, y)` on a periodic `256 x 256` domain.
- Dynamics: evolve the flow using a pseudo-spectral Navier-Stokes-style solver and advect particles through the resulting velocity field.
- Output: predict the particle position after some number of timesteps, currently emphasizing step `500`.

This is a subtle operator-learning problem because the target is not another field. The mapping is:

`initial field -> particle endpoint`

That matters a lot.

Classical FNO success stories usually look like:

- `initial field -> future field`
- `forcing / coefficients -> solution field`
- `PDE parameters + initial condition -> space-time response`

Here, the final target is only a 2D coordinate. That means the network has to:

1. understand the flow field,
2. infer what parts of the field matter for the particle trajectory,
3. compress that into a latent representation,
4. regress a final point on a periodic domain.

So this repository is not merely reproducing an FNO benchmark. It is asking whether an operator model helps when the downstream observable is Lagrangian rather than Eulerian.

That is a genuinely interesting angle.

## 3. Physics and Numerical Setup

### 3.1 Domain and discretization

Across the main turbulence code, the setup is:

- periodic square domain of size `L = 2*pi`
- grid size `N = 256`
- viscosity `nu = 0.002`
- pseudo-spectral evolution in Fourier space
- FFT backend via `fluidfft`, using `fft2d.with_pyfftw`

This is important because the periodic structure is not just a convenience. It is mathematically aligned with the spectral representation and also aligned with why FNOs are attractive here.

### 3.2 Solver structure

The tuned turbulence evolution lives in [pseudo_spectral_working.py](/home/rchethan1/snehasish/neuralops/pseudo_spectral_working.py).

This file contains the main components:

- conversion between real-space vorticity and spectral coefficients
- streamfunction inversion through `psi_hat = omega_hat / K^2`
- velocity recovery from the streamfunction
- nonlinear advective term construction
- dealiasing
- stochastic forcing in a band of wavenumbers
- damping
- explicit timestep update

The most important practical note for this repository is:

The pseudo-spectral solver is the fragile, high-value scientific core and should be treated as fixed unless there is a very deliberate numerical study around it.

That matches the project constraint exactly: do not casually modify pressure/dealiasing/forcing/timestep details that were carefully tuned to prevent blow-up.

### 3.3 What is being simulated physically

The generated dataset appears to represent passive or near-passive tracers carried by the turbulent flow.

In [sup_datac.py](/home/rchethan1/snehasish/neuralops/sup_datac.py), the particle update calls:

- `rk2_update_turbulent(...)`
- with swimming speed `velocity = 0.0`
- with control direction `[0.0, 0.0]`
- and very large `B = 1e6`

This effectively means the particles are behaving close to passive tracers advected by the flow, with the local vorticity also entering the orientation dynamics inside [swimmers.py](/home/rchethan1/snehasish/neuralops/swimmers.py).

So even though some files are written in a broader “swimmer / predator-prey” language, the current machine learning task is much closer to:

predict the transport of Lagrangian tracers in a turbulent velocity field generated from an initial vorticity condition.

That clarification is useful and should shape the research framing going forward.

## 4. Current Data Pipeline

## 4.1 Raw turbulence data

The repository includes:

- `omega_snapshots.npy`
- `turbulent_swimmers.npz`

On disk, `omega_snapshots.npy` is very large, roughly 52 GB. That suggests it is being used as a bank of precomputed turbulent vorticity states from which training simulations can be sampled.

### 4.2 Supervised dataset generation

[sup_datac.py](/home/rchethan1/snehasish/neuralops/sup_datac.py) creates the supervised dataset by:

1. loading `omega_snapshots.npy`,
2. sampling an initial vorticity field for each simulation,
3. seeding `10` particles per simulation,
4. evolving the flow for `10000` steps,
5. evolving particles through the flow using RK2,
6. storing positions at checkpoints every `500` steps up to `10000`.

The current saved dataset `turbulent_swimmers.npz` contains:

- `1000` simulations
- `10` particles per simulation
- vorticity fields of shape `(256, 256)`
- checkpoint dictionaries with keys such as `500, 1000, ..., 10000`

So the raw sample count for a single horizon is currently:

`1000 simulations x 10 particles = 10000 supervised samples`

This is useful, but it is not especially large for a high-capacity model unless the train/validation setup is very careful.

### 4.3 Coordinate-centering transform

[sup_data_coord_transform.py](/home/rchethan1/snehasish/neuralops/sup_data_coord_transform.py) applies an important transformation:

- it shifts each vorticity field so that the first particle’s initial position is moved to `(pi, pi)`,
- it sets all initial positions to `(pi, pi)`,
- it transforms future checkpoints correspondingly under periodic boundary conditions.

This is a smart idea because it removes trivial translation variability and turns the task into:

given the flow field relative to a centered initial particle location, predict the future displacement outcome.

However, there is also a subtle issue:

- the flow is shifted using the offset of the first particle,
- but each simulation contains `10` particles,
- so the centering is exact for particle `0`, but the description and downstream usage should be checked carefully for the other particles.

If the centered dataset is intended to make every particle start at the exact same canonical location, then the preprocessing logic deserves explicit verification for particles `1` through `9`.

This is not a criticism of the idea. It is a research correctness point worth checking before drawing conclusions from model comparisons.

## 5. Current Learning Formulation

The main current learning task is:

- input: centered initial vorticity field
- target: particle position at `t = 500 steps`

This formulation is implemented in the baseline scripts and in [train_fno_particle_regressor.py](/home/rchethan1/snehasish/neuralops/train_fno_particle_regressor.py).

This is likely the main reason the FNO has not yet separated strongly from the CNN.

Why?

- At `500` steps, the particle may still remain in a region whose future transport is strongly determined by local or semi-local structures.
- A CNN with circular padding and some hierarchy may already capture enough local geometry to do well.
- The target is only a 2D coordinate, so the FNO is not being used in the regime where its operator-learning advantage is most obvious.

This does not mean the FNO idea is wrong. It means the current experiment may not yet be probing the regime where the difference becomes visible.

## 6. Codebase Walkthrough

## 6.1 Turbulence solver and physics helpers

[pseudo_spectral_working.py](/home/rchethan1/snehasish/neuralops/pseudo_spectral_working.py)

- Main tuned pseudo-spectral turbulence integrator.
- Provides FFT wrappers, spectral inversion, velocity recovery, forcing, damping, and single-step update.
- This should be considered the protected numerical backbone of the project.

[pseudo_spectral_initial.py](/home/rchethan1/snehasish/neuralops/pseudo_spectral_initial.py)

- Earlier or auxiliary spectral code.
- Includes `velocity_from_omega`, which is used by the particle update routines.
- Also contains analytical checks for a simpler test problem.

[swimmers.py](/home/rchethan1/snehasish/neuralops/swimmers.py)

- Contains RK2 particle/swimmer update logic.
- Includes `rk2_update_turbulent`, which interpolates fluid velocity and vorticity to particle positions and advances particle states.
- The current supervised dataset effectively uses this in a near-passive-tracer regime.

## 6.2 Data generation and analysis

[sup_datac.py](/home/rchethan1/snehasish/neuralops/sup_datac.py)

- Main dataset generation script.
- Samples initial flow states, seeds particles, runs the coupled flow-particle evolution, and saves supervised training data.

[sup_data_coord_transform.py](/home/rchethan1/snehasish/neuralops/sup_data_coord_transform.py)

- Re-centers coordinates and flow fields to make the learning problem translation-normalized.

[sup_data_analysis.py](/home/rchethan1/snehasish/neuralops/sup_data_analysis.py)

- Small inspection utility for dataset shapes and checkpoint contents.

## 6.3 CNN baselines

[cnn_model.py](/home/rchethan1/snehasish/neuralops/cnn_model.py)

- Baseline CNN regressor.
- Uses circular padding, three convolution blocks, global average pooling, and a small MLP head.
- Predicts the position directly from the vorticity field.

[cnn_model_skip.py](/home/rchethan1/snehasish/neuralops/cnn_model_skip.py)

- Slightly stronger CNN variant with:
- rotation augmentation,
- skip-style feature fusion,
- scheduler,
- a different normalization choice.

[cnn_model_resnet50.py](/home/rchethan1/snehasish/neuralops/cnn_model_resnet50.py)

- Uses ImageNet-pretrained ResNet-50 on RGB pseudocolor renderings of vorticity.
- This is more of a computer-vision transfer baseline than a physics-aligned model.
- It may perform well empirically, but it is less scientifically interpretable for a periodic turbulence problem.

## 6.4 FNO experiments

[train_fno_particle_regressor.py](/home/rchethan1/snehasish/neuralops/train_fno_particle_regressor.py)

- Main current FNO experiment.
- Loads centered vorticity fields.
- Uses per-sample normalization.
- Uses 90-degree rotation augmentation with consistent target rotation.
- Applies an FNO backbone to produce feature maps.
- Global-average-pools those features.
- Uses an MLP to regress final `(x, y)` position.

This is a reasonable first experiment, but it is worth saying very clearly:

The model is using an FNO backbone, but the final task is still “global pooled image-to-point regression.”

That means the FNO is not fully leveraged as an operator learner over space-time fields. It is instead functioning as a global spectral feature extractor for a point prediction problem.

[predict_fno_particle.py](/home/rchethan1/snehasish/neuralops/predict_fno_particle.py)

- Loads a trained FNO regressor.
- Picks a random sample.
- Predicts a single endpoint.
- Plots the result over the vorticity field.

[fnotest.py](/home/rchethan1/snehasish/neuralops/fnotest.py)

- A toy script testing periodic generalization ideas in 1D.
- It is conceptually useful for building intuition but not directly part of the main turbulence benchmark.

## 7. What the Current Results Probably Mean

Your interpretation is sensible:

- if the prediction horizon is only `500` steps,
- and the particle does not travel very far relative to the meaningful flow scales,
- then a CNN may already have enough local and mid-range context to predict well,
- so the FNO does not yet get a clean opportunity to exploit global structure.

That is a plausible explanation.

But there are several other possibilities too, and the README should be honest about them.

### 7.1 Possible reason A: the horizon is too short

This is your current main hypothesis.

If true, then increasing the horizon should increase:

- trajectory complexity,
- sensitivity to coherent structures,
- dependence on large-scale transport barriers or pathways,
- accumulated effect of global flow organization.

This is exactly where an FNO might start to help.

### 7.2 Possible reason B: the task is not aligned with FNO strengths

FNOs shine when the target is itself a field or a function on a grid. Here the target is just a 2D point.

That means:

- most of the operator structure is compressed away,
- the loss supervises only the endpoint,
- the model may not be forced to learn the correct transport mechanism.

So even if global structure matters physically, the current loss may not make the FNO advantage visible.

### 7.3 Possible reason C: the baseline CNN is already strong enough

The CNN uses:

- periodic padding,
- hierarchical receptive fields,
- global pooling.

That already bakes in several of the right inductive biases for this problem.

If the target is endpoint-only and the horizon is modest, the CNN may simply be a very good fit.

### 7.4 Possible reason D: the dataset size is still limited

For one horizon, the effective sample count is on the order of `10000`.

That may be enough for a baseline comparison, but it is not a huge amount for confidently separating:

- architecture effects,
- randomness in particle initial conditions,
- randomness in forced turbulence,
- training noise,
- horizon difficulty.

### 7.5 Possible reason E: evaluation may not yet isolate the scientific question

If train/validation splitting is done per particle rather than per simulation, leakage can happen:

- particles from the same flow realization are strongly correlated,
- multiple targets share the same initial field,
- model performance can look better than true generalization.

This is a critical methodological point. Splits should be by simulation, not by particle sample.

## 8. The Most Important Methodological Questions to Answer Next

Before scaling experiments, these are the questions that matter most.

### 8.1 What exactly is the generalization unit?

The right answer is likely:

one initial turbulent flow realization

not:

one particle

If you split randomly across particle samples, the same vorticity field can appear in both train and test through different particles.

### 8.2 What metric should matter?

Raw Euclidean MSE on `(x, y)` is not enough because the domain is periodic.

You likely want at least:

- periodic displacement error in `x`
- periodic displacement error in `y`
- periodic Euclidean endpoint error
- normalized error relative to domain size
- error as a function of prediction horizon

Also, if the scientific question is about transport, endpoint error alone may be incomplete. Useful extras:

- radial displacement error from the centered initial point
- angular error
- distributional comparison over many particles
- uncertainty or multimodality analysis for long horizons

### 8.3 Is predicting absolute endpoint the right target?

Maybe, but maybe not.

Alternative targets that may better expose the physics:

- displacement vector
- multi-horizon trajectory checkpoints
- coarse trajectory sequence
- probability density over endpoint location
- occupancy heatmap over the periodic domain

These may be more informative and may align better with operator learning.

## 9. Research Directions From Here

This section is the most important one for deciding what to do next.

## Branch A: Keep the same endpoint task, but make the test scientifically sharper

This is the lowest-risk next step and probably the first thing to do.

### A1. Evaluate across multiple horizons

Instead of only step `500`, train and compare at:

- `500`
- `1000`
- `2000`
- `4000`
- `8000`
- `10000`

Expected value of this experiment:

- It directly tests your central hypothesis.
- It gives a difficulty curve instead of a single number.
- It may reveal a crossover horizon where FNO starts helping.

This is probably the single most important immediate experiment.

### A2. Use simulation-level train/val/test splits

Split the `1000` simulations into disjoint groups, for example:

- train: 700
- validation: 150
- test: 150

Then include all particles from a simulation in the same split.

Without this, architecture comparisons are hard to trust.

### A3. Add periodic-aware error metrics

Do not report only naive Cartesian MSE.

Report:

- wrapped `dx`
- wrapped `dy`
- wrapped endpoint distance
- mean and median error
- error percentiles

This is especially important once trajectories cross periodic boundaries.

### A4. Benchmark against trivial and strong non-neural baselines

Before concluding anything architecture-specific, compare against:

- persistence or zero-displacement baseline
- local linear regression from nearby vorticity statistics
- MLP on handcrafted global spectral summaries
- CNN
- FNO

If a simple spectral-feature regressor is already competitive, that teaches something important about the task.

## Branch B: Reformulate the target so FNO has a fairer chance

This branch may be scientifically stronger than simply “push the time horizon further.”

### B1. Predict the trajectory at multiple checkpoints

Instead of one endpoint, predict:

- positions at `500, 1000, 1500, ...`

Why this matters:

- it gives richer supervision,
- it forces the model to capture temporal structure,
- it helps diagnose whether one architecture degrades earlier than another.

This can still be done without touching the turbulence solver.

### B2. Predict displacement fields or transport maps

A particularly promising direction is:

for a fixed initial vorticity field, predict the final displacement for a whole grid of initial particle locations.

That would convert the task into:

`field -> field`

which is much more naturally an operator-learning problem and much more aligned with FNO strengths.

This might actually be the best “better problem” hidden inside the current project.

Why it is exciting:

- the output becomes a transport map or displacement field,
- the target is defined on a grid,
- FNOs are designed exactly for this type of mapping,
- CNN vs FNO becomes a cleaner operator-learning comparison.

This could become a much stronger research story than endpoint-only regression.

### B3. Predict endpoint density instead of one deterministic point

At longer horizons, tracer transport may become effectively more chaotic or multimodal from the model’s perspective.

Instead of predicting one point, predict:

- a probability density map over final position,
- or a low-resolution occupancy distribution.

This handles uncertainty better and again turns the task into field prediction.

## Branch C: Study what information is actually needed

This branch asks a more scientific question:

What aspects of the initial flow determine tracer transport over different timescales?

### C1. Local patch vs full-domain ablation

Train one model on:

- only a local patch around the initial tracer position

and another on:

- the full domain

Then compare performance as horizon increases.

This directly tests your hypothesis about locality vs globality.

This experiment is extremely valuable because it is interpretable.

If local patches perform nearly as well at `500` but collapse at large horizons, that is strong evidence for your intuition.

### C2. Spectral ablations

Filter the initial vorticity field into:

- low-frequency only
- high-frequency only
- full spectrum

Then compare model performance.

This would tell you whether long-time transport is driven more by:

- coherent large scales,
- fine local eddies,
- or their interaction.

This is a very nice physics-oriented diagnostic.

### C3. Receptive field study for CNNs

If a CNN keeps up with the FNO, ask why.

Possible reason:

its effective receptive field is already global enough.

You can test:

- small CNN
- deeper CNN
- dilated CNN
- CNN with and without circular padding

This helps separate “locality” from “architecture label.”

## Branch D: Move closer to the original operator-learning literature

If the inspiration came from papers predicting the vorticity field itself, then another path is:

### D1. Reproduce field prediction first

Train an FNO to predict:

- `omega_t+Delta` from `omega_t`
- or `omega_0 -> omega_T`

If that works well on your solver outputs, then you can build from there toward tracer prediction.

Why this is useful:

- it establishes that the dataset and training setup are compatible with FNO-style learning,
- it gives a sanity check before judging the harder endpoint-regression task,
- it can later be composed with particle advection.

### D2. Two-stage pipeline

Stage 1:

- predict future vorticity or velocity field with FNO

Stage 2:

- advect particles through the predicted field

This may be more physically faithful than directly regressing endpoint coordinates.

It also creates an interpretable decomposition:

- field forecasting quality
- transport forecasting quality

## 10. The Strongest Immediate Experiments

If the goal is to make progress quickly without overcomplicating the project, this is the best order.

### Priority 1: Clean evaluation protocol

Do this before making claims.

- split by simulation, not by particle
- define periodic endpoint metrics
- run several seeds if possible

### Priority 2: Horizon sweep

This directly tests your current hypothesis.

Train/evaluate CNN and FNO for multiple checkpoints:

- `500`
- `1000`
- `2000`
- `4000`
- maybe `8000`

Plot error vs horizon.

If there is an architecture crossover, that figure will likely reveal it.

### Priority 3: Local-patch vs full-field comparison

This is one of the highest-value ablations in the whole project.

It answers:

Does the problem genuinely become more global at larger horizons?

If yes, that immediately strengthens the scientific argument for trying FNOs.

### Priority 4: Multi-horizon trajectory prediction

If endpoint-only remains inconclusive, move to predicting multiple checkpoints.

This gives richer supervision without changing the solver.

## 11. What Could Become the Best Research Story

There are at least three strong stories hidden in the current setup.

### Story 1: Crossover from local to global transport prediction

Thesis:

short-horizon tracer prediction is effectively local, but long-horizon tracer prediction depends on global flow organization.

How to prove it:

- compare local-patch and full-field models across horizons,
- compare CNN and FNO across horizons,
- show where performance diverges.

This is likely the cleanest story closest to your current intuition.

### Story 2: Endpoint regression is the wrong target for operator learning

Thesis:

FNOs do not show a strong advantage on endpoint-only regression because the operator structure is collapsed into a 2D coordinate, but they become advantageous when the output is a transport field or endpoint density.

How to prove it:

- compare endpoint regression vs transport-map prediction,
- keep the same underlying turbulence data,
- show that FNO advantage appears in the field-output setting.

Scientifically, this could be a very compelling paper or thesis chapter because it says something more nuanced than “FNO beats CNN.”

### Story 3: What spectral scales control Lagrangian transport?

Thesis:

different prediction horizons depend on different spectral bands of the initial vorticity field.

How to prove it:

- low-pass / high-pass ablations,
- performance vs horizon,
- architecture comparison.

This is a more physics-centered story and could be very interesting.

## 12. Risks and Caveats

### 12.1 Chaotic sensitivity

Longer horizons are scientifically attractive, but they also make the target harder.

If two nearly identical initial conditions produce divergent particle endpoints, deterministic point prediction may become noisy or unstable.

That does not invalidate the problem. It may simply mean the task should evolve toward:

- probabilistic prediction,
- multi-step prediction,
- or field-valued outputs.

### 12.2 Endpoint-only supervision may hide physically meaningful errors

Two predicted trajectories can end at similar points while following very different paths.

If path matters scientifically, endpoint-only evaluation is incomplete.

### 12.3 Data preprocessing can shape conclusions

Centering and periodic coordinate transforms are powerful, but they must be verified carefully to ensure they do not unintentionally introduce:

- particle-specific bias,
- leakage,
- or target simplification artifacts.

### 12.4 Architecture comparison must be capacity-aware

If one model has far more parameters or stronger pretraining, results can be hard to interpret scientifically.

The goal is not just to maximize accuracy. The goal is to learn what inductive bias is appropriate for turbulent transport prediction.

## 13. Recommended Exact Next Steps

If the question is “what should I do right now?”, this is the recommendation.

### Step 1

Keep the pseudo-spectral turbulence solver unchanged.

This is non-negotiable unless you are running a separate numerical validation study.

### Step 2

Turn the current project into a clean benchmark:

- define train/val/test splits by simulation
- document dataset sizes for each split
- use periodic endpoint metrics
- keep CNN and FNO training protocols comparable

### Step 3

Run the horizon sweep:

- train at `500`
- train at `1000`
- train at `2000`
- train at `4000`
- optionally `8000` and `10000`

This is the fastest way to test your current hypothesis.

### Step 4

Run a locality ablation:

- one model sees only a patch around the centered particle,
- one model sees the full vorticity field.

Do this for at least two horizons, one short and one long.

If the full-domain model only matters at long times, you have a strong result.

### Step 5

If endpoint regression still does not clearly separate CNN and FNO, pivot to a stronger operator-learning problem:

- predict a full displacement field / transport map for many initial particle locations.

This is the most promising “better problem” already latent in the repository.

## 14. My Overall Assessment

This is already a solid and research-worthy project.

The main issue is not that the idea is weak. The issue is that the current task formulation may be too easy, too local, or too compressed to reveal the real difference between CNNs and neural operators.

So the immediate goal should not be “make the FNO win somehow.”

The immediate goal should be:

design the experiment so that it honestly tests when global operator structure matters for Lagrangian prediction.

If the answer turns out to be:

- “only at longer horizons,” that is a result.
- “only when output is field-valued,” that is also a result.
- “CNNs remain competitive even for this task,” that is still a meaningful result if shown carefully.

The best next move is therefore not to touch the turbulence solver, but to sharpen the learning problem and evaluation protocol.

## 15. Practical Repository Notes

Current notable files:

- [pseudo_spectral_working.py](/home/rchethan1/snehasish/neuralops/pseudo_spectral_working.py): tuned turbulence solver
- [swimmers.py](/home/rchethan1/snehasish/neuralops/swimmers.py): particle update routines
- [sup_datac.py](/home/rchethan1/snehasish/neuralops/sup_datac.py): supervised data generation
- [sup_data_coord_transform.py](/home/rchethan1/snehasish/neuralops/sup_data_coord_transform.py): centering transform
- [cnn_model.py](/home/rchethan1/snehasish/neuralops/cnn_model.py): baseline CNN
- [cnn_model_skip.py](/home/rchethan1/snehasish/neuralops/cnn_model_skip.py): stronger CNN variant
- [cnn_model_resnet50.py](/home/rchethan1/snehasish/neuralops/cnn_model_resnet50.py): ResNet baseline on pseudocolor images
- [train_fno_particle_regressor.py](/home/rchethan1/snehasish/neuralops/train_fno_particle_regressor.py): current FNO endpoint regressor
- [predict_fno_particle.py](/home/rchethan1/snehasish/neuralops/predict_fno_particle.py): inference/visualization
- [fnotest.py](/home/rchethan1/snehasish/neuralops/fnotest.py): toy FNO periodicity intuition script

Dataset notes:

- `turbulent_swimmers.npz` currently contains `1000` simulations and `10` particles per simulation.
- The scripts reference centered files such as `centered_turbulent_swimmers_1.npz`; those should be treated as derived datasets generated from the raw data pipeline.

## 16. Suggested Thesis-Style Summary

If you want the project in one sentence:

This repository studies whether global spectral inductive biases from neural operators provide an advantage over convolutional locality when predicting Lagrangian transport from turbulent initial conditions, and it likely needs longer horizons or field-valued transport targets to expose that difference cleanly.
