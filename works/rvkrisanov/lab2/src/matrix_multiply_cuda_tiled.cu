#include "../include/matrix_multiply.h"
#include "../include/error_check.cuh"

#define TILE_SIZE 32
#define THREAD_WORK_SIZE 8

__global__ void matrix_multiply_kernel_tiled(
    const float * __restrict__ matrix_a,
    const float * __restrict__ matrix_b,
    float * __restrict__ matrix_result,
    const int M,
    const int N,
    const int K)
{
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    float thread_results[THREAD_WORK_SIZE] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row_start = blockIdx.y * TILE_SIZE;
    int block_col_start = blockIdx.x * TILE_SIZE;

    const int stride = 4;

    int num_tiles = N / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int tiled_col = tile_idx * TILE_SIZE + tx;
        int tiled_row = tile_idx * TILE_SIZE + ty;

        #pragma unroll 8
        for (int i = 0; i < THREAD_WORK_SIZE; ++i) {
            int local_row = ty + i * stride;
            
            int global_row_a = block_row_start + local_row;
            int global_col_a = tiled_col;
            tile_a[local_row][tx] = matrix_a[global_row_a * N + global_col_a];

            int global_row_b = tiled_row + i * stride; 
            int global_col_b = block_col_start + tx;
            tile_b[local_row][tx] = matrix_b[global_row_b * K + global_col_b];
        }

        __syncthreads();

        #pragma unroll 32
        for (int k = 0; k < TILE_SIZE; ++k) {
            float b_val = tile_b[k][tx];
            
            #pragma unroll 8
            for (int i = 0; i < THREAD_WORK_SIZE; ++i) {
                int local_row = ty + i * stride;
                thread_results[i] += tile_a[local_row][k] * b_val;
            }
        }

        __syncthreads();
    }

    #pragma unroll 8
    for (int i = 0; i < THREAD_WORK_SIZE; ++i) {
        int local_row = ty + i * stride;
        int global_row = block_row_start + local_row;
        int global_col = block_col_start + tx;
        
        matrix_result[global_row * K + global_col] = thread_results[i];
    }
}

void launch_matrix_multiply_tiled(
    const float *d_matrix_a,
    const float *d_matrix_b,
    float *d_matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns,
    cudaStream_t stream)
{
    dim3 block_size(TILE_SIZE, 4);
    
    dim3 grid_size(
        matrix_b_columns / TILE_SIZE,
        matrix_a_rows / TILE_SIZE);

    matrix_multiply_kernel_tiled<<<grid_size, block_size, 0, stream>>>(
        d_matrix_a, d_matrix_b, d_matrix_result,
        matrix_a_rows, matrix_a_columns, matrix_b_columns);

    CUDA_CHECK(cudaGetLastError());
}
