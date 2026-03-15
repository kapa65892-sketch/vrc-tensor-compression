import vrc_core
import numpy as np
import time

print("=== VRC CORE LOSSLESS VALIDATION ===")

size = 1_000_000
original_data = np.random.randn(size).astype(np.float64) * 100.0

print(f"1. Исходный объем: {original_data.nbytes / 1024 / 1024:.2f} MB")

start = time.perf_counter()
compressed_bytes = vrc_core.compress(original_data)
compress_time = (time.perf_counter() - start) * 1000

print(f"2. Сжатый объем: {len(compressed_bytes) / 1024 / 1024:.2f} MB")
ratio = original_data.nbytes / len(compressed_bytes)
print(f"   Коэффициент сжатия: {ratio:.2f}x")
print(f"   Время сжатия: {compress_time:.2f} ms")

start = time.perf_counter()
decompressed_data = vrc_core.decompress(compressed_bytes)
decompress_time = (time.perf_counter() - start) * 1000

print(f"3. Время распаковки: {decompress_time:.2f} ms")

if np.array_equal(original_data, decompressed_data):
    print("\n✅ СТАТУС: ПОЛНАЯ ЦЕЛОСТНОСТЬ (LOSSLESS). ДАННЫЕ СОХРАНЕНЫ БИТ-В-БИТ.")
else:
    print("\n❌ СТАТУС: ОШИБКА! ДАННЫЕ ПОВРЕЖДЕНЫ.")

print("=== TEST COMPLETE ===")
