#include <contract_driver_arrays.h>
#include <cblas.h>
#include <algorithm> // for std::min

// Perform C = A * B using tiling (GETT-style)
void contract_with_tiles(const double *A,
                          const double *B,
                          double *C,
                          int M, int N, int K,
                          int m_tile, int n_tile, int k_tile) {
  for (int mm = 0; mm < M; mm += m_tile) {
    int m_cur = std::min(m_tile, M - mm);
    for (int nn = 0; nn < N; nn += n_tile) {
      int n_cur = std::min(n_tile, N - nn);
      for (int kk = 0; kk < K; kk += k_tile) {
        int k_cur = std::min(k_tile, K - kk);

        // Call BLAS GEMM on tile
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m_cur, n_cur, k_cur,
            1.0,
            &A[mm*K + kk], K, // Tile from A
            &B[kk*N + nn], N, // Tile from B
            1.0,
            &C[mm*N + nn], N); // Accumulate into C
      }
    }
  }
}
