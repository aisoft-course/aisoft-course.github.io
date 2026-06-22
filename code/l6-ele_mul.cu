#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

// ------------------------------------------------------------
// CUDA Kernel：向量逐元素相乘
// c[i] = a[i] * b[i]
// ------------------------------------------------------------
__global__ void vec_mul(float* a, float* b, float* c) {
    // 计算全局线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (i < N) {
        c[i] = a[i] * b[i];
    }
}

int main() {
    printf("=== CUDA Vector Multiply Demo ===\n");

    // ----------------------------
    // 检查 CUDA 设备
    // ----------------------------
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("❌ CUDA 初始化失败: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("❌ 没有检测到支持 CUDA 的 GPU！\n");
        return -1;
    }

    printf("✅ 检测到 %d 个 CUDA 设备\n", deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("使用 GPU: %s\n", prop.name);

    size_t size = N * sizeof(float);

    // ----------------------------
    // 主机内存分配与初始化
    // ----------------------------
    float *ha, *hb, *hc;
    ha = (float*)malloc(size);
    hb = (float*)malloc(size);
    hc = (float*)malloc(size);

    printf("[1] 初始化主机数据...\n");
    for (int i = 0; i < N; i++) {
        ha[i] = 2.0f;
        hb[i] = 3.0f;
    }

    // ----------------------------
    // 设备内存分配
    // ----------------------------
    float *da, *db, *dc;
    printf("[2] 分配 GPU 内存...\n");
    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    printf("[3] 拷贝数据到 GPU...\n");
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    // ----------------------------
    // Kernel 启动配置
    // ----------------------------
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("[4] 启动 Kernel (blocks=%d, threads=%d)...\n", numBlocks, threadsPerBlock);
    vec_mul<<<numBlocks, threadsPerBlock>>>(da, db, dc);

    cudaDeviceSynchronize(); // 等待 GPU 完成

    // ----------------------------
    // 结果拷贝回主机
    // ----------------------------
    printf("[5] 拷贝结果回主机...\n");
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    // 输出部分结果
    printf("[6] 输出前10个结果:\n");
    for (int i = 0; i < 10; i++) {
        printf("hc[%d] = %f\n", i, hc[i]);
    }

    // 简单正确性检查
    printf("[7] 验证结果...\n");
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (hc[i] != 6.0f) {
            correct = 0;
            printf("错误: hc[%d] = %f (期望 6.0)\n", i, hc[i]);
            break;
        }
    }

    if (correct) {
        printf("✅ 结果正确！所有值都是 6.0\n");
    } else {
        printf("❌ 结果错误！\n");
    }

    // ----------------------------
    // 资源释放
    // ----------------------------
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(ha);
    free(hb);
    free(hc);

    printf("=== 程序结束 ===\n");

    return 0;
}
