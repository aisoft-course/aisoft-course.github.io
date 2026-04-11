#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N_SIZE 1024

// ============================
// CPU Matrix Multiplication
// ============================
void matmul_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================
// Naive GPU
// ============================
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================
// Tiled Kernel (模板支持不同 TILE)
// ============================
template <int TILE_SIZE>
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {

        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// ============================
// CPU 计时
// ============================
double cpu_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================
// Main
// ============================
int main() {
    int N = N_SIZE;
    size_t size = N * N * sizeof(float);

    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *hC = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    // ============================
    // CPU 测试
    // ============================
    double t1 = cpu_time();
    matmul_cpu(hA, hB, hC, N);
    double t2 = cpu_time();
    printf("CPU time: %f ms\n", t2 - t1);

    // ============================
    // GPU 初始化
    // ============================
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ============================
    // Naive GPU
    // ============================
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    cudaEventRecord(start);
    matmul_naive<<<blocks, threads>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    printf("GPU Naive: %f ms\n", time_naive);

    // ============================
    // 不同 TILE_SIZE 测试
    // ============================
#define RUN_TILE(T) \
    { \
        dim3 tpb(T, T); \
        dim3 nb((N + T - 1) / T, (N + T - 1) / T); \
        cudaEventRecord(start); \
        matmul_tiled<T><<<nb, tpb>>>(dA, dB, dC, N); \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        float t_ms; \
        cudaEventElapsedTime(&t_ms, start, stop); \
        printf("GPU Tiled (%d): %f ms\n", T, t_ms); \
    }

    RUN_TILE(8);
    RUN_TILE(16);
    RUN_TILE(32);

#undef RUN_TILE

    // ============================
    // 清理
    // ============================
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
