# Saved State Protocol

This file freezes the saved-state contract for the current `cuda_field`
extension path.

It complements:

- [`contracts.py`](/home/pluto/projects/GAUSS/hpc_sd_deployment/gaussian_peft/cuda_field/contracts.py)
- [`csrc/field.h`](/home/pluto/projects/GAUSS/hpc_sd_deployment/gaussian_peft/cuda_field/csrc/field.h)

The purpose is to keep the Python `ctx.save_for_backward(...)`, the C++
`SavedTensorBundle`, and the forward/backward orchestration state fully aligned
without API drift.


## Decision Rules

The code path now uses three buckets:

1. `Persistent`
Saved through `ctx.save_for_backward(...)` and available directly in backward.

2. `Rebuild`
Not persisted across the Python boundary. Reconstructed in backward from the
persistent tensors.

3. `Forward Workspace Only`
Temporary forward orchestration buffers that must never become Python-visible
saved state.


## Decision Table

| State | Producer | Typical Shape | Bucket | Reason |
|---|---|---:|---|---|
| `row_coords` | forward input / `SavedTensorBundle.row_coords` | `[out_features]` | Persistent | backward needs tile-local coordinate replay |
| `col_coords` | forward input / `SavedTensorBundle.col_coords` | `[in_features]` | Persistent | backward needs tile-local coordinate replay |
| `mu` | forward input / `SavedTensorBundle.mu` | `[K, 2]` | Persistent | differentiable input |
| `chol_raw` | forward input / `SavedTensorBundle.chol_raw` | `[K, 3]` | Persistent | differentiable input and backward rebuild root |
| `amp` | forward input / `SavedTensorBundle.amp` | `[K, 1]` | Persistent | differentiable input |
| `tile_ptr` | forward output / `SavedTensorBundle.tile_ptr` | `[num_tiles + 1]` | Persistent | defines active tile ranges for backward |
| `gaussian_ids_sorted` | forward output / `SavedTensorBundle.gaussian_ids_sorted` | `[num_pairs]` | Persistent | defines active pair set for backward |
| `tile_keys_unsorted` | forward preprocess/binning | `[num_pairs]` | Forward Workspace Only | needed only for sort construction |
| `tile_keys_sorted` | forward sort | `[num_pairs]` | Forward Workspace Only | `tile_ptr + gaussian_ids_sorted` is sufficient after range identification |
| `point_offsets` | `BinningState.point_offsets` | `[K]` | Forward Workspace Only | used only during pair emission |
| `scan_workspace` | `BinningState.scan_workspace` | `[bytes]` | Forward Workspace Only | CUB exclusive-scan workspace owned by host orchestration |
| `sort_workspace` | `BinningState.sort_workspace` | `[bytes]` | Forward Workspace Only | CUB radix-sort workspace owned by host orchestration |
| `inv_cov` | preprocess | `[K, 2, 2]` | Rebuild | deterministic from `chol_raw`, not persisted in Stage 0 |
| `det_cov` | preprocess | `[K]` | Rebuild | deterministic from `chol_raw`, not persisted in Stage 0 |
| `tile_r0/r1/c0/c1` | preprocess | `[K]` each | Rebuild by default | can be recomputed from `mu/chol_raw/config` |
| `tiles_touched` | preprocess | `[K]` | Rebuild by default | useful for diagnostics, not required as persistent state |
| `num_pairs` | derived | scalar | Rebuild | can be read from `gaussian_ids_sorted.numel()` |
| `num_tiles` | derived | scalar | Rebuild | can be read from `tile_ptr.numel() - 1` |


## Upgrade Conditions

Some states default to `Rebuild`, but may be promoted to `Persistent` if their
reconstruction cost proves material.

### `tile_r0/r1/c0/c1`

Promote to `Persistent` only if:

- backward profiling shows preprocess-style recomputation is a non-trivial share
  of backward time, and
- this promotion reduces backward time without causing unacceptable persistent
  memory growth.

### `inv_cov` and `det_cov`

Promote to `Persistent` only if:

- repeated reconstruction from `chol_raw` becomes a measured bottleneck, and
- the added persistent memory is cheaper than recomputation.

Current default remains:

- rebuild `chol -> cov -> inv_cov/det_cov` in backward


## Non-Negotiable Constraints

1. `tile_ptr` and `gaussian_ids_sorted` are mandatory persistent tensors.
2. Forward workspaces must not leak across the Python boundary.
3. If a state is not in the persistent set, backward must be able to reconstruct
   it exactly enough to preserve the active-set semantics for the chosen
   `sigma_multiplier`.
4. Any promotion from `Rebuild` to `Persistent` must update:
   - [`contracts.py`](/home/pluto/projects/GAUSS/hpc_sd_deployment/gaussian_peft/cuda_field/contracts.py)
   - [`csrc/field.h`](/home/pluto/projects/GAUSS/hpc_sd_deployment/gaussian_peft/cuda_field/csrc/field.h)
   - this file


## Current Minimal Persistent Set

Current persistent set used by both Python and CUDA:

```text
row_coords
col_coords
mu
chol_raw
amp
tile_ptr
gaussian_ids_sorted
```

Nothing else should be persisted across the Python boundary in the first
training-facing implementation unless this protocol is explicitly revised.


## Interface Lock

The following must stay one-to-one aligned:

1. `ctx.save_for_backward(...)` in [`runtime.py`](/home/pluto/projects/GAUSS/hpc_sd_deployment/gaussian_peft/cuda_field/runtime.py)
2. `SavedTensorBundle` in [`field.h`](/home/pluto/projects/GAUSS/hpc_sd_deployment/gaussian_peft/cuda_field/csrc/field.h)
3. the `saved` field inside `ForwardOrchestrationState`

The current order is:

```text
row_coords
col_coords
mu
chol_raw
amp
tile_ptr
gaussian_ids_sorted
```

The main backward definition is now the explicit CUDA path bound through
`gaussian_field_backward(...)`. Reference replay must remain a separate
validation API only.
