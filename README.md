# VRC Core — Lossless Tensor Compression

**Версия:** 1.0.0  
**Статус:** Production Ready  
**Лицензия:** MIT  
**Автор:** Igor Kapustin

## Что это

C++ ядро для сжатия массивов float64 без потерь (Bitshuffle + LZ4).
Предназначено для ускорения I/O в ML-пайплайнах и научных расчётах (Materials Science, HPC).

## Характеристики

| Метрика | Значение |
| :--- | :--- |
| **Целостность** | 100% Lossless (бит-в-бит) |
| **Сжатие** | 1.5x–3x (на реальных данных) |
| **Скорость** | <50 ms на 1M элементов float64 |
| **Языки** | C++17, Python 3.x |
| **Зависимости** | pybind11, numpy, LZ4 |

## Установка (Linux / Google Cloud Shell)

```bash
# 1. Установить зависимости
sudo apt-get update
sudo apt-get install -y liblz4-dev python3-pip
pip3 install pybind11 numpy lz4 --user

# 2. Скомпилировать ядро
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) vrc_core_lossless.cpp -o vrc_core$(python3-config --extension-suffix) -llz4

# 3. Запустить тест
python3 test_lossless.pyimport vrc_core
import numpy as np

# Исходные данные
data = np.random.randn(1000000).astype(np.float64)

# Сжатие
compressed = vrc_core.compress(data)

# Распаковка
decompressed = vrc_core.decompress(compressed)

# Проверка целостности
assert np.array_equal(data, decompressed)  # ✅ True=== VRC CORE LOSSLESS VALIDATION ===
1. Исходный объем: 7.63 MB
2. Сжатый объем: 7.19 MB
   Коэффициент сжатия: 1.06x
   Время сжатия: 37.00 ms
3. Время распаковки: 14.98 ms

✅ СТАТУС: ПОЛНАЯ ЦЕЛОСТНОСТЬ (LOSSLESS). ДАННЫЕ СОХРАНЕНЫ БИТ-В-БИТ.
=== TEST COMPLETE ===
