#include <stddef.h>
#include <omp.h>

void matmul_blocked(const double* A, const double* B, double* C,
                    int M, int K, int N, int BS) {

    // omp_set_num_threads(4);

    #pragma omp parallel for collapse(2) schedule(static)
    // 遍历 C 的块（输出块）
    for (int ii = 0; ii < M; ii += BS) {      // 行块
        for (int jj = 0; jj < N; jj += BS) {  // 列块
            for (int kk = 0; kk < K; kk += BS) {  // 累加维度块

                // 处理边界（防止越界）
                int i_max = (ii + BS < M) ? ii + BS : M;
                int j_max = (jj + BS < N) ? jj + BS : N;
                int k_max = (kk + BS < K) ? kk + BS : K;

                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {

                        double sum = C[i*N + j];

                        for (int k = kk; k < k_max; k++) {
                            sum += A[i*K + k] * B[k*N + j];
                        }

                        C[i*N + j] = sum;
                    }
                }

            }
        }
    }
}
