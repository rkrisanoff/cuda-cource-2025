# Lab 2: Matrix Multiplication

## Задача
Реализовать и оптимизировать перемножение матриц на CUDA. Сравнить производительность с CPU.

## Реализация
1.  **CPU**: Классический алгоритм `O(N^3)`.
2.  **GPU Basic**: Наивная реализация (глобальная память).
3.  **GPU Tiled**: Оптимизация с Shared Memory (Tile 32x32) + **Thread Coarsening**.
    *   Каждый поток вычисляет 8 элементов результата для увеличения арифметической интенсивности.
    *   Использование `#pragma unroll` для разворачивания циклов.

## Результаты (Turing GPU)

| Matrix Size | CPU (us) | CUDA Basic (us) | CUDA Tiled (us) | Speedup (Basic vs CPU) | Speedup (Tiled vs Basic) |
|---|---|---|---|---|---|
| 64x64 | 333 | 166 | **101** | 2.00x | **1.64x** |
| 128x128 | 3,405 | 32 | **20** | 106x | **1.60x** |
| 512x512 | 209,305 | 384 | **358** | 545x | **1.07x** |
| 1024x1024 | 3,389,726 | 2,466 | **2,307** | 1374x | **1.07x** |
| 2048x2048 | 38,546,840 | 17,936 | **16,820** | 2149x | **1.06x** |

**Анализ:**
*   **Tiled версия быстрее**: Благодаря использованию Shared Memory и Thread Coarsening удалось добиться ускорения относительно наивной версии на всех размерах матриц.
*   **Thread Coarsening**: Внедрение техники "1 поток считает 8 элементов" позволило существенно сократить время выполнения Tiled версии на больших матрицах, снизив количество обращений к Shared Memory.

## Запуск
```bash
make
./matrix_multiply
```

```bash
(base) jovyan@a52120adfa7c:~/repos/cuda-cource-2025/works/rvkrisanov/lab2$ make
nvcc -O2 -arch=sm_70 -std=c++14 -Iinclude -o matrix_multiply src/main.cu src/matrix_multiply_cpu.cu src/matrix_multiply_cuda.cu src/matrix_multiply_cuda_tiled.cu
./matrix_multiply
Benchmarking Matrix 64x64...
  CPU:             333 us
  CUDA Basic:      166 us (2.00x speedup) [OK]
  CUDA Tiled:      101 us (3.29x speedup) [OK]

Benchmarking Matrix 128x128...
  CPU:            3405 us
  CUDA Basic:       32 us (106.40x speedup) [OK]
  CUDA Tiled:       20 us (170.25x speedup) [OK]

Benchmarking Matrix 256x256...
  CPU:           29691 us
  CUDA Basic:       83 us (357.72x speedup) [OK]
  CUDA Tiled:       79 us (375.83x speedup) [OK]

Benchmarking Matrix 512x512...
  CPU:          209305 us
  CUDA Basic:      384 us (545.06x speedup) [OK]
  CUDA Tiled:      358 us (584.65x speedup) [OK]

Benchmarking Matrix 1024x1024...
  CPU:         3389726 us
  CUDA Basic:     2466 us (1374.58x speedup) [OK]
  CUDA Tiled:     2307 us (1469.32x speedup) [OK]

Benchmarking Matrix 2048x2048...
  CPU:        38546840 us
  CUDA Basic:    17936 us (2149.13x speedup) [OK]
  CUDA Tiled:    16820 us (2291.72x speedup) [OK]
```
