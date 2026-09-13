#!/usr/bin/env bash
# Install application wheels into an existing isolated foundation environment.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:-.venv-jovian}
uv_binary=${UV_BIN:-uv}
if ! uv_path=$(command -v "${uv_binary}"); then
  printf 'uv is required; set UV_BIN to its absolute path.\n' >&2
  exit 1
fi

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)

test -x "${venv_path}/bin/python" || {
  printf 'Foundation environment is absent: %s\n' "${venv_path}" >&2
  exit 1
}
"${venv_path}/bin/python" -I - <<'PY'
import torch

assert torch.__version__ == "2.13.0"
assert torch.version.cuda == "13.3"
assert torch._C._GLIBCXX_USE_CXX11_ABI
PY

"${uv_path}" pip install \
  --python "${venv_path}/bin/python" \
  --require-hashes \
  --no-index \
  --find-links "${bundle_dir}/wheels" \
  --no-deps \
  -r "${bundle_dir}/requirements-wheelhouse.txt"

env -u PYTHONPATH "${venv_path}/bin/python" "${bundle_dir}/verify_install.py"
