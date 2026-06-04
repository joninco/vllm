# 10 — Build, Image Recipes, Scripts

## Purpose

Restore the fork's build/image/script tweaks on top of upstream:
ROCm-flash-attn skip for CUDA-only builds, editable image rebuild
driver passthrough, canonical B12X commit pin, GLM/Kimi image
recipes, and FlashInfer-related serve helpers.

## Acceptance Criteria

- `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
  succeeds end-to-end on CUDA 13 from a clean `.venv`.
- The B12X submodule commit pinned by `0217ee622` (or its successor)
  resolves cleanly.
- `build_unholy_fusion.sh` / `build_nameless_ascent.sh` still build
  the image with the rebased dockerfile.
- The GLM and Kimi image recipes from `3ff722e55` are present and
  match upstream's current image structure.
- `scripts/` serve helpers (DSV4-Flash, GLM, Kimi) parse without
  error and pass `--help`.

## Constraints

- Do not regress upstream's CUDA 13 default switch (#39878 family on
  upstream). Layer the B12X pin on top.
- Keep `requirements/` deletions consistent — do not re-introduce
  rocm-flash-attn into the CUDA path.
- Do not commit secrets or local paths into the image recipes.

## Dependencies

None upstream of this. This cluster lands first because failures
here block every later cluster's `pip install -e .`.

## Commit Map (in order)

- `42caa6d38` build: skip ROCm flash-attn submodules for CUDA
- `bc0bb4ef0` build: pass GPU driver into editable image rebuild
- `0217ee622` build: pin canonical B12X commit
- `3ff722e55` build: add canonical GLM and Kimi image recipes
- `47b25dbb7` scripts: make GLM local argmax reduction opt-in
- `e95dbe741` scripts: enable FlashInfer autotune for Kimi

## Conflict Hotspots

- `requirements/build/cuda.txt`, `requirements/build/rocm.txt`,
  `requirements/cuda.txt`, `requirements/test/cuda.in` — upstream
  added cutlass-dsl[cu13]; layer B12X commit pin on the new pin set.
- `pyproject.toml` — keep both upstream's build-system entries and
  any B12X extras.
- `cmake/external_projects/vllm_flash_attn.cmake`,
  `cmake/utils.cmake` — upstream may have refactored arch detection;
  re-anchor the ROCm-skip toggle to whatever guard exists now.

## Validation

```bash
.venv/bin/python -c "import vllm; print(vllm.__version__)"
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```
