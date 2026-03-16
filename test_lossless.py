"""
VRC Core — Professional Test Suite
====================================
pytest script covering:
  - Data integrity (lossless roundtrip)
  - Edge cases and security (corrupted packets)
  - Performance benchmarks: 1M / 10M / 100M elements
  - Compression ratios on realistic data types
"""

import pytest
import numpy as np
import time
import struct

# Attempt to import the compiled module
try:
    import vrc_core
    VRC_AVAILABLE = True
except ImportError:
    VRC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not VRC_AVAILABLE,
    reason="vrc_core not compiled. Follow build instructions in README."
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def random_1m():
    """1M random float64 elements — incompressible by nature"""
    rng = np.random.default_rng(42)
    return rng.standard_normal(1_000_000).astype(np.float64)

@pytest.fixture
def scientific_elastic():
    """Simulated elasticity tensor — real-world patterns, compresses well"""
    rng = np.random.default_rng(123)
    size = 500_000
    x      = np.linspace(0, 100, size)
    signal = np.sin(x * 0.1) * 1e9 + np.cos(x * 0.05) * 5e8
    noise  = rng.standard_normal(size) * 1e4
    return (signal + noise).astype(np.float64)

@pytest.fixture
def zeros_array():
    return np.zeros(100_000, dtype=np.float64)

@pytest.fixture
def single_element():
    return np.array([3.14159265358979], dtype=np.float64)


# =============================================================================
# GROUP 1: DATA INTEGRITY (LOSSLESS)
# =============================================================================

class TestLosslessIntegrity:

    def test_random_data_roundtrip(self, random_1m):
        """Random data: compress → decompress must be bit-for-bit identical"""
        compressed   = vrc_core.compress(random_1m)
        decompressed = vrc_core.decompress(compressed)
        assert np.array_equal(random_1m, decompressed), \
            "LOSSLESS VIOLATION: data changed after roundtrip!"

    def test_scientific_data_roundtrip(self, scientific_elastic):
        """Scientific tensor data: full integrity required"""
        compressed   = vrc_core.compress(scientific_elastic)
        decompressed = vrc_core.decompress(compressed)
        assert np.array_equal(scientific_elastic, decompressed), \
            "LOSSLESS VIOLATION on scientific data!"

    def test_zeros_roundtrip(self, zeros_array):
        """Zero array: compresses extremely well, must restore exactly"""
        compressed   = vrc_core.compress(zeros_array)
        decompressed = vrc_core.decompress(compressed)
        assert np.array_equal(zeros_array, decompressed)

    def test_single_element_roundtrip(self, single_element):
        """Single element: edge case"""
        compressed   = vrc_core.compress(single_element)
        decompressed = vrc_core.decompress(compressed)
        assert np.array_equal(single_element, decompressed)

    def test_special_float_values(self):
        """Special values: inf, -inf, nan, min, max, -0.0"""
        data = np.array([
            np.inf, -np.inf, np.nan,
            np.finfo(np.float64).max,
            np.finfo(np.float64).min,
            0.0, -0.0, 1.0
        ], dtype=np.float64)
        compressed   = vrc_core.compress(data)
        decompressed = vrc_core.decompress(compressed)
        # Bit-level comparison (handles nan equality correctly)
        assert data.tobytes() == decompressed.tobytes(), \
            "Special float values not preserved bit-for-bit!"

    def test_constant_array(self):
        """Constant array: maximum compression case"""
        data = np.full(100_000, 42.0, dtype=np.float64)
        compressed   = vrc_core.compress(data)
        decompressed = vrc_core.decompress(compressed)
        assert np.array_equal(data, decompressed)

    def test_integer_values_as_float64(self):
        """Integer values stored as float64 — common ML pattern"""
        data = np.arange(100_000, dtype=np.float64)
        compressed   = vrc_core.compress(data)
        decompressed = vrc_core.decompress(compressed)
        assert np.array_equal(data, decompressed)


# =============================================================================
# GROUP 2: SECURITY — EDGE CASES AND ADVERSARIAL INPUTS
# =============================================================================

class TestSecurity:

    def test_empty_array(self):
        """Empty array must not crash"""
        data = np.array([], dtype=np.float64)
        compressed   = vrc_core.compress(data)
        decompressed = vrc_core.decompress(compressed)
        assert len(decompressed) == 0

    def test_corrupted_packet_too_small(self):
        """Packet smaller than header → RuntimeError, not crash"""
        with pytest.raises(RuntimeError, match="too small"):
            vrc_core.decompress(b"\x01\x02\x03")

    def test_corrupted_header_giant_size(self):
        """Header claims 999 GB allocation → must be rejected"""
        fake_size   = struct.pack('<Q', 999 * 1024**3)
        fake_packet = fake_size + b"\x00" * 100
        with pytest.raises(RuntimeError):
            vrc_core.decompress(bytes(fake_packet))

    def test_corrupted_payload(self, random_1m):
        """Flipped payload bytes → decompression error"""
        compressed = bytearray(vrc_core.compress(random_1m))
        mid = len(compressed) // 2
        for i in range(mid, min(mid + 50, len(compressed))):
            compressed[i] ^= 0xFF
        with pytest.raises(RuntimeError):
            vrc_core.decompress(bytes(compressed))

    def test_misaligned_header(self):
        """Header size not a multiple of 8 (sizeof double) → error"""
        fake_size   = struct.pack('<Q', 7)   # 7 bytes — not aligned
        fake_packet = fake_size + b"\x00" * 50
        with pytest.raises(RuntimeError):
            vrc_core.decompress(bytes(fake_packet))


# =============================================================================
# GROUP 3: PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformance:

    def _benchmark(self, size: int, label: str):
        """Helper: run compress/decompress benchmark for given array size"""
        rng  = np.random.default_rng(0)
        data = rng.standard_normal(size).astype(np.float64)
        mb   = data.nbytes / 1024 / 1024

        # Warm-up
        _ = vrc_core.decompress(vrc_core.compress(data[:1000]))

        # Compress (best of 3 runs)
        compress_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            compressed = vrc_core.compress(data)
            compress_times.append((time.perf_counter() - t0) * 1000)
        compress_ms = min(compress_times)

        # Decompress (best of 3 runs)
        decompress_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            decompressed = vrc_core.decompress(compressed)
            decompress_times.append((time.perf_counter() - t0) * 1000)
        decompress_ms = min(decompress_times)

        ratio = data.nbytes / len(compressed)

        print(f"\n  [{label}]")
        print(f"    Size:              {mb:.1f} MB ({size:,} elements)")
        print(f"    Compress:          {compress_ms:.1f} ms  ({mb/compress_ms*1000:.0f} MB/s)")
        print(f"    Decompress:        {decompress_ms:.1f} ms  ({mb/decompress_ms*1000:.0f} MB/s)")
        print(f"    Compression ratio: {ratio:.2f}x  (random data — ~1.0x expected)")

        assert np.array_equal(data, decompressed), \
            "LOSSLESS VIOLATION in performance test!"
        return compress_ms, decompress_ms, ratio

    def test_perf_1m_elements(self):
        """1M elements: compress must complete within 100 ms"""
        compress_ms, _, _ = self._benchmark(1_000_000, "1M elements")
        assert compress_ms < 100, \
            f"Compression of 1M elements too slow: {compress_ms:.1f} ms"

    def test_perf_10m_elements(self):
        """10M elements: compress must complete within 1000 ms"""
        compress_ms, _, _ = self._benchmark(10_000_000, "10M elements")
        assert compress_ms < 1000, \
            f"Compression of 10M elements too slow: {compress_ms:.1f} ms"

    @pytest.mark.slow
    def test_perf_100m_elements(self):
        """100M elements (~763 MB): long test, run with -m slow"""
        compress_ms, _, _ = self._benchmark(100_000_000, "100M elements")
        assert compress_ms < 10_000, \
            f"Compression of 100M elements too slow: {compress_ms:.1f} ms"


# =============================================================================
# GROUP 4: COMPRESSION RATIOS ON REAL DATA
# =============================================================================

class TestCompressionRatio:

    def test_ratio_random_data(self, random_1m):
        """Random data: expect ~1.0x (incompressible by nature)"""
        compressed = vrc_core.compress(random_1m)
        ratio = random_1m.nbytes / len(compressed)
        print(f"\n  Random data:            {ratio:.2f}x")
        assert ratio >= 0.9, \
            f"Output larger than input — unexpected: {ratio:.2f}x"

    def test_ratio_scientific_data(self, scientific_elastic):
        """Scientific tensor with patterns: expect 1.3x–3x"""
        compressed = vrc_core.compress(scientific_elastic)
        ratio = scientific_elastic.nbytes / len(compressed)
        print(f"\n  Scientific tensor:      {ratio:.2f}x")
        assert ratio >= 1.2, \
            f"Weak compression on structured data: {ratio:.2f}x"

    def test_ratio_zeros(self, zeros_array):
        """Zero array: expect > 10x compression"""
        compressed = vrc_core.compress(zeros_array)
        ratio = zeros_array.nbytes / len(compressed)
        print(f"\n  Zero array:             {ratio:.2f}x")
        assert ratio >= 10.0, \
            f"Zero array should compress heavily: {ratio:.2f}x"

    def test_ratio_arange(self):
        """Monotonic sequence: expect > 2x"""
        data = np.arange(1_000_000, dtype=np.float64)
        compressed = vrc_core.compress(data)
        ratio = data.nbytes / len(compressed)
        print(f"\n  Monotonic sequence:     {ratio:.2f}x")
        assert ratio >= 2.0, \
            f"Weak compression on monotonic sequence: {ratio:.2f}x"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
