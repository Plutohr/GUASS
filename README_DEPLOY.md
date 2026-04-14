# HPC Deployment Bundle

This directory is the cleaned deployment bundle for running Gaussian-PEFT
DreamBooth training on the compute platform.

Included:

- `gaussian_peft/`
- `configs/`
- `slurm/`
- `offline_packages/`
- `scripts/train_diffusion.py`
- `scripts/build_cuda_field_extension.sh`
- `scripts/setup_offline_env.sh`
- `scripts/test_cuda_env.py`
- `scripts/smoke_train_cuda_field_train.py`
- `scripts/prepare_dreambooth_data.py`
- `requirements*.txt`

Intentionally excluded:

- local build artifacts
- benchmark/report utilities that are not required for platform training
- monitor logs
- temporary loss artifacts

Recommended platform workflow:

1. Upload this directory as `hpc_sd_deployment`.
2. Prepare a Python environment.
3. If using offline wheels, run:

```bash
bash scripts/setup_offline_env.sh
```

4. Build and verify the CUDA extension:

```bash
bash scripts/build_cuda_field_extension.sh
```

5. Run a tiny CUDA smoke check before DreamBooth:

```bash
python scripts/smoke_train_cuda_field_train.py --steps 2
```

6. Run a short DreamBooth smoke:

```bash
python scripts/train_diffusion.py --config configs/dreambooth_sd_smoke.yaml
```

7. Submit the real training job:

```bash
sbatch slurm/train_diffusion.slurm
```

Notes:

- Do not upload local `.artifacts/` prebuilt `.so` files from a different Python
  version.
- If this bundle still contains any local `__pycache__` files, they are not
  required on the platform and can be ignored or removed safely.
- Keep the pretrained diffusers model and DreamBooth dataset on the platform
  separately; they are not bundled here.
- The main training entrypoint is `scripts/train_diffusion.py`.
