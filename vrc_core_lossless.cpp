#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <lz4.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace py = pybind11;

// 1. Механика Byte-Shuffle (Перемешивание байтов)
void byte_shuffle(const uint8_t* src, uint8_t* dest, size_t num_elements, size_t element_size) {
    for (size_t b = 0; b < element_size; ++b) {
        for (size_t i = 0; i < num_elements; ++i) {
            dest[b * num_elements + i] = src[i * element_size + b];
        }
    }
}

// 2. Обратная механика (Unshuffle)
void byte_unshuffle(const uint8_t* src, uint8_t* dest, size_t num_elements, size_t element_size) {
    for (size_t b = 0; b < element_size; ++b) {
        for (size_t i = 0; i < num_elements; ++i) {
            dest[i * element_size + b] = src[b * num_elements + i];
        }
    }
}

// 3. Функция СЖАТИЯ (Lossless)
py::bytes compress_strict_lossless(py::array_t<double> input_array) {
    py::buffer_info buf = input_array.request();
    size_t num_elements = buf.size;
    size_t element_size = sizeof(double);
    size_t total_bytes = num_elements * element_size;

    if (total_bytes == 0) return py::bytes("");

    const uint8_t* src_ptr = static_cast<const uint8_t*>(buf.ptr);

    // Шаг А: Перемешивание байтов
    std::vector<uint8_t> shuffled(total_bytes);
    byte_shuffle(src_ptr, shuffled.data(), num_elements, element_size);

    // Шаг Б: Выделение буфера под LZ4
    int max_lz4_size = LZ4_compressBound(total_bytes);
    std::vector<char> compressed(max_lz4_size);

    // Шаг В: Сжатие LZ4
    int compressed_size = LZ4_compress_default(
        reinterpret_cast<const char*>(shuffled.data()),
        compressed.data(),
        total_bytes,
        max_lz4_size
    );

    if (compressed_size <= 0) {
        throw std::runtime_error("VRC Core Error: LZ4 compression failed.");
    }

    // Шаг Г: Упаковка [Размер исходных данных] + [Сжатые данные]
    std::string result_packet;
    result_packet.append(reinterpret_cast<const char*>(&total_bytes), sizeof(size_t));
    result_packet.append(compressed.data(), compressed_size);

    return py::bytes(result_packet);
}

// 4. Функция РАСПАКОВКИ (Lossless)
py::array_t<double> decompress_strict_lossless(py::bytes compressed_data) {
    std::string cpp_str = static_cast<std::string>(compressed_data);

    if (cpp_str.size() <= sizeof(size_t)) {
        throw std::runtime_error("VRC Core Error: Invalid packet (too small).");
    }

    // Шаг А: Чтение заголовка
    size_t original_bytes;
    std::memcpy(&original_bytes, cpp_str.data(), sizeof(size_t));

    size_t num_elements = original_bytes / sizeof(double);
    const char* payload_ptr = cpp_str.data() + sizeof(size_t);
    int payload_size = cpp_str.size() - sizeof(size_t);

    // Шаг Б: Буфер для распаковки
    std::vector<uint8_t> shuffled(original_bytes);

    // Шаг В: Декомпрессия LZ4
    int decompressed_size = LZ4_decompress_safe(
        payload_ptr,
        reinterpret_cast<char*>(shuffled.data()),
        payload_size,
        original_bytes
    );

    if (decompressed_size < 0 || static_cast<size_t>(decompressed_size) != original_bytes) {
        throw std::runtime_error("VRC Core Error: Decompression integrity check failed.");
    }

    // Шаг Г: Обратное перемешивание
    auto result_array = py::array_t<double>(num_elements);
    py::buffer_info res_buf = result_array.request();
    uint8_t* dest_ptr = static_cast<uint8_t*>(res_buf.ptr);

    byte_unshuffle(shuffled.data(), dest_ptr, num_elements, sizeof(double));

    return result_array;
}

// 5. Экспорт модуля
PYBIND11_MODULE(vrc_core, m) {
    m.doc() = "VRC Core: Production Ready Lossless Compression (Shuffle+LZ4)";
    m.def("compress", &compress_strict_lossless, "Compress float64 array to bytes (Lossless)");
    m.def("decompress", &decompress_strict_lossless, "Decompress bytes to float64 array (Lossless)");
}
