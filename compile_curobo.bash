#!/bin/bash

# Exit on any error
set -e

echo "--------------------------------------------------------"
echo "Starting cuRobo Compilation (Container Runtime)"
echo "--------------------------------------------------------"

cd /workspace/libs/curobo

echo "[Step 1/3] Upgrading setuptools for modern build hooks..."
python3.11 -m pip install --upgrade setuptools

echo "[Step 3/3] Compiling CUDA kernels. This may take 10-20 minutes..."
python3.11 -m pip install -e . --no-build-isolation

echo "--------------------------------------------------------"
echo "cuRobo successfully installed!"
echo "You can now run: python3.11 -c 'import curobo; print(\"cuRobo Version:\", curobo.__version__)'"
echo "--------------------------------------------------------"