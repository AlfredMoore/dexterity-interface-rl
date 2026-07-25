# Azure Kinect SDK and PyK4A Installation

## Environment

```text
Ubuntu 24.04.1
ROS 2 Jazzy
Conda environment: policy
Python 3.12.13
```

Installed versions:

```text
libk4a: 1.4.2
libk4a-dev: 1.4.2
pyk4a: 1.5.0
```

The version numbers of `libk4a` and `pyk4a` are independent:

```text
pyk4a 1.5.0
    ↓
libk4a 1.4.2
```

## Step 1: Check Kinect USB Access

Confirm that the Azure Kinect USB devices are visible inside the container:

```bash
lsusb | grep -Ei 'Microsoft|Kinect|045e'
```

Expected devices include:

```text
045e:097a  Azure Kinect USB 3.0 Hub
045e:097c  Azure Kinect Depth Camera
045e:097d  Azure Kinect 4K Camera
045e:097b  Azure Kinect USB 2.0 Hub
045e:097e  Azure Kinect Microphone Array
```

## Step 2: Download the Azure Kinect SDK Packages

Create an installation directory:

```bash
mkdir -p <path>/k4a-install
cd <path>/k4a-install
```

Download the runtime package:

```bash
wget \
  https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.2_amd64.deb
```

Download the development package:

```bash
wget \
  https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.2_amd64.deb
```

Verify the downloaded files:

```bash
ls -lh *.deb
```

## Step 3: Install System Dependencies

```bash
apt-get update
```

```bash
apt-get install -y \
  debconf-utils \
  libusb-1.0-0 \
  libudev1 \
  libgl1 \
  libx11-6
```

## Step 4: Accept the Azure Kinect EULA

The runtime package requires acceptance of the Microsoft EULA. In a noninteractive container, configure it through `debconf` before installation:

```bash
echo \
'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' \
| debconf-set-selections
```

```bash
echo \
'libk4a1.4 libk4a1.4/accept-eula boolean true' \
| debconf-set-selections
```

Verify the stored EULA setting:

```bash
debconf-show libk4a1.4
```

Expected output should include:

```text
libk4a1.4/accept-eula: true
```

## Step 5: Install `libk4a`

Make sure the current directory contains the downloaded packages:

```bash
cd /workspace/dep/k4a-install
```

Install the runtime and development packages:

```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ./libk4a1.4_1.4.2_amd64.deb \
  ./libk4a1.4-dev_1.4.2_amd64.deb
```

If a previous failed installation left packages in a partially configured state, run:

```bash
DEBIAN_FRONTEND=noninteractive dpkg --configure -a
apt-get -f install
```

Verify the installed packages:

```bash
dpkg -l | grep k4a
```

Expected output:

```text
ii  libk4a1.4      1.4.2
ii  libk4a1.4-dev  1.4.2
```

## Step 6: Register the Depth Engine Library Path

The main SDK library is installed in a standard path:

```text
/usr/lib/x86_64-linux-gnu/libk4a.so.1.4
```

The closed-source depth engine is installed in a subdirectory:

```text
/usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0
```

Register this directory with the Linux dynamic linker:

```bash
echo '/usr/lib/x86_64-linux-gnu/libk4a1.4' \
  > /etc/ld.so.conf.d/azure-kinect.conf
```

Refresh the linker cache:

```bash
ldconfig
```

Verify that the depth engine can be found:

```bash
ldconfig -p | grep depthengine
```

Expected output:

```text
libdepthengine.so.2.0 =>
/usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0
```

## Step 7: Verify the SDK Libraries

Find the installed SDK files:

```bash
find /usr/lib /usr/include \
  \( -name 'libk4a.so*' \
     -o -name 'libdepthengine.so*' \
     -o -name 'k4a.h' \) \
  2>/dev/null
```

Expected files include:

```text
/usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0
/usr/lib/x86_64-linux-gnu/libk4a.so.1.4
/usr/lib/x86_64-linux-gnu/libk4a.so.1.4.2
/usr/lib/x86_64-linux-gnu/libk4a.so
/usr/include/k4a/k4a.h
```

Verify that Python can load the libraries:

```bash
python - <<'PY'
import ctypes

ctypes.CDLL("libk4a.so.1.4")
ctypes.CDLL("libdepthengine.so.2.0")

print("libk4a and depth engine loaded successfully")
PY
```

Check the depth engine dependencies:

```bash
ldd \
  /usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0
```

There should be no dependency marked as:

```text
not found
```

## Step 8: Install `pyk4a`

Activate the Conda environment:

```bash
conda activate policy
```

Confirm the Python interpreter:

```bash
which python
python --version
```

Install the Python build dependencies:

```bash
python -m pip install \
  pip \
  wheel \
  cython
```

Install `pyk4a`:

```bash
python -m pip install "pyk4a==1.5.0"
```

Verify the installation:

```bash
python - <<'PY'
import sys
import pyk4a
import k4a_module

print("Python:", sys.executable)
print("pyk4a:", pyk4a.__file__)
print("k4a_module:", k4a_module.__file__)
PY
```

## Step 9: Check the PyK4A Native Extension

Find the native extension:

```bash
K4A_MODULE=$(
  python -c 'import k4a_module; print(k4a_module.__file__)'
)
```

Print its location:

```bash
echo "$K4A_MODULE"
```

Inspect its linked libraries:

```bash
ldd "$K4A_MODULE"
```

The extension should link successfully to libraries such as:

```text
libk4a.so.1.4
libk4arecord.so.1.4
libudev.so.1
libstdc++.so.6
```

There should be no dependency marked as:

```text
not found
```

## Step 10: Minimal Azure Kinect Capture Test

Create a test program:

```bash
cat >/tmp/test_kinect.py <<'PY'
from pyk4a import (
    PyK4A,
    Config,
    FPS,
    ColorResolution,
    DepthMode,
    ImageFormat,
)

camera = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        color_format=ImageFormat.COLOR_BGRA32,
        depth_mode=DepthMode.NFOV_2X2BINNED,
        camera_fps=FPS.FPS_30,
        synchronized_images_only=True,
    )
)

print("Starting Azure Kinect...")
camera.start()

try:
    capture = None

    for index in range(10):
        capture = camera.get_capture()
        print(f"Captured {index + 1}/10")

    if capture is None:
        raise RuntimeError("No capture returned")

    color = capture.color
    depth = capture.depth

    print(
        "Color:",
        None if color is None else (
            color.shape,
            color.dtype,
        ),
    )

    print(
        "Depth:",
        None if depth is None else (
            depth.shape,
            depth.dtype,
            int(depth.min()),
            int(depth.max()),
        ),
    )

finally:
    camera.stop()
    print("Camera stopped")
PY
```

Run the test:

```bash
python ./test_kinect.py
```

Expected image sizes:

```text
Color: 720 × 1280 × 4
Depth: 288 × 320
```

This corresponds to:

```text
Color: 1280 × 720 at 30 Hz
Depth: 320 × 288 at 30 Hz
```

## Installation Status

The following components have been successfully verified:

```text
Kinect USB devices visible in container       OK
libk4a 1.4.2 installed                       OK
libk4a development files installed            OK
libdepthengine.so.2.0 installed               OK
Dynamic linker can load libk4a                OK
Dynamic linker can load the depth engine      OK
Depth engine dependencies resolved            OK
pyk4a 1.5.0 installed                         OK
k4a_module native extension imported           OK
k4a_module linked to libk4a successfully       OK
```

If `camera.start()` fails with:

```text
Depth engine create and initialize failed with error code: 204
```

the installation itself is already complete. The remaining issue is related to runtime initialization of the depth engine and should be diagnosed separately.
