# CUDA Field Module

This directory contains the CUDA-extension work for the 2D Gaussian field path.

Scope:

- Contract freeze and interface enforcement
- C++/CUDA extension implementation

Current status:

- `contracts.py`
  Python-side contract, tensor layout rules, thresholds, and autograd boundary
- `saved_state_protocol.md`
  Frozen saved-state and rebuild policy
- `loader.py`
  Extension source discovery and build scaffold
- `setup.py`
  Non-JIT build entry for environments without `ninja`
- `csrc/`
  C++/CUDA implementation in progress

Implementation status:

- `forward` is now a real CUDA path with four explicit stages:
  - `preprocess`
  - `prefix-sum + duplicateWithKeys`
  - `identifyTileRanges`
  - `tile-local accumulate`
- `forward.cu` now contains the active kernels for:
  - `preprocess_geometry_kernel`
  - pair emission
  - per-tile count accumulation
  - tile range identification
  - `tile_accumulate_kernel`
- binning uses CUB-based exclusive scan and key/value sort for
  point-offset construction and `(tile_id, gaussian_id)` sorting
- tile accumulation uses block-local shared-memory staging for Gaussian
  parameter chunks
- tile-accumulate launch sizing uses a small heuristic based on tile area
- a few orchestration details are still correctness-oriented, including the
  remaining host-visible `total_pairs` materialization
- `backward` now has two paths:
  - `gaussian_field_reference(...)` for validation-only reference replay
  - `gaussian_field_train(...)` for the active custom CUDA training backend
- the training backend uses explicit CUDA gradient accumulation plus the
  `d_inv_cov/d_det_cov -> d_cov -> d_chol_raw` chain
- the training backend has moved beyond pure global atomics into pair-buffered
  partial gradients plus grouped reduce-by-key
- remaining work is now concentrated in performance tuning, workload coverage,
  and broader regression testing rather than correctness fallback

Important boundary:

- this module currently provides an `extension-packaged reference path`
- it is the only tiled backend line that should continue evolving
- other tiled paths under `gaussian_peft.kernels` are reference or experimental
  baselines only
- callers should treat `gaussian_field_reference(...)` as validation-only
- reference backward is opt-in only:
  pass `allow_reference_backward=True` exclusively for explicit gradient
  alignment checks
- `gaussian_field_train(...)` is the active training backend
- a tiny local optimizer-step smoke run lives in:
  `scripts/smoke_train_cuda_field_train.py`
- platform training uses:
  `scripts/train_diffusion.py`
- platform build/verification uses:
  `scripts/build_cuda_field_extension.sh`
  and `scripts/setup_offline_env.sh`
- build artifacts must stay out of the source tree; use the configured
  artifact directory under `hpc_sd_deployment/.artifacts/cuda_field_reference`

This module now owns the live tiled CUDA path under `gaussian_peft`.
