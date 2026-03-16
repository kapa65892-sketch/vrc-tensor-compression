[README.md](https://github.com/user-attachments/files/26038740/README.md)
# VRC Core — Lossless Tensor Compression

**Version:** 1.1.0
**Status:** Production Ready (Hardened)
**License:** MIT
**Author:** Igor Kapustin

## Overview

A C++ core library for lossless compression of float64 arrays using Byte-Shuffle + LZ4.
Designed to accelerate I/O in ML pipelines and scientific computing
(Materials Science, HPC, Finite Element Analysis).

## Specifications

| Metric        | Value                                         |
|:--------------|:----------------------------------------------|
| Integrity     | 100% Lossless (bit-for-bit exact)             |
| Compression   | 1.0x–10x+ (data-dependent, see table below)   |
| Speed         | < 50 ms per 1M float64 elements               |
| Languages     | C++17, Python 3.8+                            |
| Dependencies  | pybind11, numpy, LZ4                          |

### Real-World Compression Ratios

| Data Type                        | Expected Ratio |
|:---------------------------------|:---------------|
| Random noise (np.random.randn)   | ~1.0x          |
| Scientific tensors with patterns | 1.5x–3x        |
| Monotonic sequences              | 2x–5x          |
| Zeros / constant arrays          | 10x+           |

> **Note:** Random data is incompressible by nature — this is mathematics, not a bug.
> LZ4 cannot compress true entropy. Real scientific data compresses well.

---

## Installation

### Option 1: CMake (Recommended) — Linux / macOS / Google Cloud Shell

```bash
# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y liblz4-dev python3-pip cmake
pip3 install pybind11 numpy pytest --user

# 2. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cp vrc_core*.so ..        # Linux (.so) or Windows (.pyd)

# 3. Quick sanity check
cd ..
python3 -c "
import vrc_core, numpy as np
d = np.ones(1000)
assert (d == vrc_core.decompress(vrc_core.compress(d))).all()
print('OK — lossless verified')
"
```

### Option 2: Single Command (no CMake)

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    vrc_core_lossless.cpp \
    -o vrc_core$(python3-config --extension-suffix) \
    -llz4
```

### Windows (Visual Studio + vcpkg)

```powershell
vcpkg install lz4 pybind11
mkdir build; cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

---

## Usage

```python
import vrc_core
import numpy as np

# Your float64 data (scientific tensor, ML weights, etc.)
data = np.random.randn(1_000_000).astype(np.float64)

# Compress
compressed = vrc_core.compress(data)

# Decompress
decompressed = vrc_core.decompress(compressed)

# Verify integrity
assert np.array_equal(data, decompressed)   # always True
print(f"Compression ratio: {data.nbytes / len(compressed):.2f}x")
```

---

## Running Tests

```bash
# Run all tests
pytest test_lossless.py -v -s

# Fast tests only (skip 100M benchmark)
pytest test_lossless.py -v -s -m "not slow"

# Full suite including 100M elements
pytest test_lossless.py -v -s -m slow
```

### Test Coverage

- **Integrity:** roundtrip on random, scientific, and special (nan/inf/±0) data
- **Security:** corrupted packets, oversized headers, misaligned sizes
- **Performance:** 1M / 10M / 100M elements with MB/s throughput measurement
- **Compression ratio:** across different data types

---

## Architecture

```
Compress:
  float64[] → Byte-Shuffle → LZ4 → [8 bytes: original_size | compressed_data]

Decompress:
  bytes → read header → validate → LZ4_decompress_safe → Byte-Unshuffle → float64[]
```

**Byte-Shuffle** reorders float64 bytes so that the same byte positions across
all elements are grouped together. This dramatically improves LZ4's ability to
find repeating patterns in structured scientific data.

**LZ4** is a high-speed lossless compressor optimized for throughput over ratio.

---

## Changelog

### v1.1.0 — Security Hardening
- Fixed: `original_bytes` now validated against 2GB limit before memory allocation
- Fixed: `payload_size` computed safely without unsigned integer underflow (UB)
- Added: alignment check — size must be a multiple of `sizeof(double)`
- Added: null pointer check on input data
- Added: correct handling of empty arrays
- Clarified: algorithm name is Byte-Shuffle (not Bitshuffle/bit-level)
- Added: professional pytest suite with 4 test groups

### v1.0.0 — Initial Release

---

## License

MIT License — free to use in commercial and open-source projects.
