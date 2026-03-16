#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <lz4.h>
#include <zlib.h>  // Для CRC32
#include <vector>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace py = pybind11;

// =============================================================================
// VRC Core — Lossless Tensor Compression
// Algorithm: Byte-Shuffle + LZ4 + CRC32
// Version: 1.2.0 (Security Hardened with CRC)
// Author: Igor Kapustin
// =============================================================================

// Safety constants
static constexpr size_t MAX_ALLOWED_BYTES = 2ULL * 1024 * 1024 * 1024; // 2 GB hard limit
static constexpr size_t HEADER_SIZE       = sizeof(size_t);
static constexpr size_t CRC_SIZE          = sizeof(uint32_t);
static constexpr size_t METADATA_SIZE     = HEADER_SIZE + CRC_SIZE; // 8 + 4 = 12 bytes

// -----------------------------------------------------------------------------
// 1. Byte-Shuffle: reorders identical bytes together for better LZ4 compression
// -----------------------------------------------------------------------------
void byte_shuffle(const uint8_t* src, uint8_t* dest,
                  size_t num_elements, size_t element_size)
{
    for (size_t b = 0; b < element_size; ++b) {
        for (size_t i = 0; i < num_elements; ++i) {
            dest[b * num_elements + i] = src[i * element_size + b];
        }
    }
}

// -----------------------------------------------------------------------------
// 2. Byte-Unshuffle: reverse operation
// -----------------------------------------------------------------------------
void byte_unshuffle(const uint8_t* src, uint8_t* dest,
                    size_t num_elements, size_t element_size)
{
    for (size_t b = 0; b < element_size; ++b) {
        for (size_t i = 0; i < num_elements; ++i) {
            dest[i * element_size + b] = src[b * num_elements + i];
        }
    }
}

// -----------------------------------------------------------------------------
// 3. COMPRESS — Lossless float64 array compression with CRC32
// -----------------------------------------------------------------------------
py::bytes compress_strict_lossless(py::array_t<double, py::array::c_style | py::array::forcecast> input_array)
{
    py::buffer_info buf = input_array.request();
    size_t num_elements = static_cast<size_t>(buf.size);
    constexpr size_t element_size = sizeof(double);
    size_t total_bytes = num_elements * element_size;

    // --- Edge case: empty array ---
    if (total_bytes == 0) {
        size_t zero = 0;
        uint32_t crc = 0;
        std::string empty_packet;
        empty_packet.append(reinterpret_cast<const char*>(&zero), HEADER_SIZE);
        empty_packet.append(reinterpret_cast<const char*>(&crc), CRC_SIZE);
        return py::bytes(empty_packet);
    }

    // --- Sanity check: reject absurdly large inputs ---
    if (total_bytes > MAX_ALLOWED_BYTES) {
        throw std::runtime_error(
            "VRC Core Error: Input too large (> 2GB). Split into chunks.");
    }

    const uint8_t* src_ptr = static_cast<const uint8_t*>(buf.ptr);
    if (!src_ptr) {
        throw std::runtime_error("VRC Core Error: Null input pointer.");
    }

    // --- Step A: Byte-Shuffle ---
    std::vector<uint8_t> shuffled(total_bytes);
    byte_shuffle(src_ptr, shuffled.data(), num_elements, element_size);

    // --- Step B: Check LZ4 integer size limit ---
    if (total_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "VRC Core Error: Chunk too large for LZ4 (> 2GB). Use chunked compress.");
    }

    int input_size_int  = static_cast<int>(total_bytes);
    int max_lz4_size    = LZ4_compressBound(input_size_int);
    if (max_lz4_size <= 0) {
        throw std::runtime_error("VRC Core Error: LZ4_compressBound failed.");
    }

    // --- Step C: LZ4 compression ---
    std::vector<char> compressed(static_cast<size_t>(max_lz4_size));
    int compressed_size = LZ4_compress_default(
        reinterpret_cast<const char*>(shuffled.data()),
        compressed.data(),
        input_size_int,
        max_lz4_size
    );

    if (compressed_size <= 0) {
        throw std::runtime_error("VRC Core Error: LZ4 compression failed.");
    }

    // --- Step D: Calculate CRC32 of compressed data ---
    uint32_t crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(compressed.data()), 
                static_cast<uInt>(compressed_size));

    // --- Step E: Build packet [header: 8 bytes] [crc: 4 bytes] [compressed_data] ---
    std::string result_packet;
    result_packet.append(reinterpret_cast<const char*>(&total_bytes), HEADER_SIZE);
    result_packet.append(reinterpret_cast<const char*>(&crc), CRC_SIZE);
    result_packet.append(compressed.data(), static_cast<size_t>(compressed_size));

    return py::bytes(result_packet);
}

// -----------------------------------------------------------------------------
// 4. DECOMPRESS — Decompress bytes to float64 array with CRC32 verification
// -----------------------------------------------------------------------------
py::array_t<double> decompress_strict_lossless(py::bytes compressed_data)
{
    std::string cpp_str = static_cast<std::string>(compressed_data);

    // --- Validate minimum packet size ---
    if (cpp_str.size() < METADATA_SIZE) {
        throw std::runtime_error(
            "VRC Core Error: Invalid packet — too small to contain header + CRC.");
    }

    // --- Read header (original size) ---
    size_t original_bytes = 0;
    std::memcpy(&original_bytes, cpp_str.data(), HEADER_SIZE);

    // --- SECURITY PATCH: Validate original_bytes before allocation ---
    if (original_bytes > MAX_ALLOWED_BYTES) {
        throw std::runtime_error(
            "VRC Core Error: Corrupted packet — claimed size > 2GB.");
    }

    // --- Edge case: empty array ---
    if (original_bytes == 0) {
        return py::array_t<double>(0);
    }

    // --- Validate alignment to sizeof(double) ---
    if (original_bytes % sizeof(double) != 0) {
        throw std::runtime_error(
            "VRC Core Error: Corrupted packet — size not aligned to float64.");
    }

    // --- Read stored CRC32 ---
    uint32_t stored_crc = 0;
    std::memcpy(&stored_crc, cpp_str.data() + HEADER_SIZE, CRC_SIZE);

    // --- Extract payload (compressed data) ---
    const char* payload = cpp_str.data() + METADATA_SIZE;
    size_t payload_size_sz = cpp_str.size() - METADATA_SIZE;

    // --- SECURITY PATCH: Safe signed cast after bounds check ---
    if (payload_size_sz > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "VRC Core Error: Payload too large for LZ4 decompressor.");
    }
    size_t num_elements = original_bytes / sizeof(double);

    if (original_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "VRC Core Error: Original size too large for LZ4 decompressor.");
    }

    int payload_size  = static_cast<int>(payload_size_sz);
    int orig_size_int = static_cast<int>(original_bytes);

    // --- Step A: Verify CRC32 BEFORE decompression ---
    uint32_t calculated_crc = crc32(0L, Z_NULL, 0);
    calculated_crc = crc32(calculated_crc, reinterpret_cast<const Bytef*>(payload), 
                           static_cast<uInt>(payload_size_sz));

    if (calculated_crc != stored_crc) {
        throw std::runtime_error(
            "VRC Core Error: CRC32 mismatch — data corrupted or tampered.");
    }

    // --- Step B: Allocate decompression buffer ---
    std::vector<uint8_t> shuffled(original_bytes);

    // --- Step C: LZ4 decompression ---
    int decompressed_size = LZ4_decompress_safe(
        payload,
        reinterpret_cast<char*>(shuffled.data()),
        payload_size,
        orig_size_int
    );

    if (decompressed_size < 0) {
        throw std::runtime_error(
            "VRC Core Error: LZ4 decompression failed — data corrupted.");
    }

    if (static_cast<size_t>(decompressed_size) != original_bytes) {
        throw std::runtime_error(
            "VRC Core Error: Decompression size mismatch — integrity check failed.");
    }

    // --- Step D: Byte-Unshuffle → final output array ---
    auto result_array = py::array_t<double>(static_cast<py::ssize_t>(num_elements));
    py::buffer_info res_buf = result_array.request();
    uint8_t* dest_ptr = static_cast<uint8_t*>(res_buf.ptr);

    byte_unshuffle(shuffled.data(), dest_ptr, num_elements, sizeof(double));

    return result_array;
}

// -----------------------------------------------------------------------------
// 5. Python module export
// -----------------------------------------------------------------------------
PYBIND11_MODULE(vrc_core, m) {
    m.doc() = "VRC Core v1.2.0 — Production-Ready Lossless Compression (Byte-Shuffle + LZ4 + CRC32)";

    m.def("compress",
          &compress_strict_lossless,
          py::arg("input_array"),
          "Compress a float64 numpy array to bytes (lossless). "
          "Returns bytes object: [8-byte header][4-byte CRC32][compressed data].");

    m.def("decompress",
          &decompress_strict_lossless,
          py::arg("compressed_data"),
          "Decompress bytes back to a float64 numpy array (lossless). "
          "Raises RuntimeError on CRC32 mismatch or any integrity violation.");
}