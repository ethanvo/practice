#include <gett.h>
#include <algorithm> // std::min
#include <cmath>
#include <iostream>

// Pack a block of A (M x K) into buffer (row-major)
void packA(const double* A, int lda,
           double* packA_buf,
           int M, int K,
           int mc, int kc,
           int i_start, int p_start) {
  for (int i = 0; i < mc; i++) {
    int ai = i_start + i;
    for (int p = 0; p < kc; p++) {
      int ap = p_start + p;
      if (ai < M && ap < K) {
        packA_buf[i * kc + p] = A[ai * lda + ap];
      } else {
        // Pad with zero if outside of bounds
        packA_buf[i * kc + p] = 0.0;
      }
    }
  }
}

// Pack a block of B (K x N) into buffer (row-major)
void packB(const double* B, int ldb,
           double* packB_buf,
           int K, int N,
           int kc, int nc,
           int p_start, int j_start) {
  for (int p = 0; p < kc; p++) {
    int bp = p_start + p;
    for (int j = 0; j < nc; j++) {
      int bj = j_start + j;
      if (bp < K && bj < N) {
        packB_buf[p * nc + j] = B[bp * ldb + bj];
      } else {
        packB_buf[p * nc + j] = 0.0;
      }
    }
  }
}

// Sinple microkernel: Cblock += packA * packB
// - packA: mc x kc
// - packB: kc x nc
// - C:     ldc stride (row-major), updates mc x nc block
void microkernel(const double* packA_buf,
                 const double* packB_buf,
                 double* Cblock,
                 int mc, int nc, int kc,
                 int ldc) {
  for (int i = 0; i < mc; i++) {
    for (int j = 0; j < nc; j++) {
      double cij = 0.0;
      for (int p = 0; p < kc; p++) {
        cij += packA_buf[i * kc + p] * packB_buf[p * nc + j];
      }
      Cblock[i * ldc * j] += cij;
    }
  }
}

void GETT(const double* A, const double* B, double* C,
          int M, int N, int K,
          int lda, int ldb, int ldc,
          int mc, int nc, int kc) {
  // Allocate buffers once
  double* packA_buf = new double[mc * kc];
  double* packB_buf = new double[kc * nc];

  for (int jc = 0; jc < N; jc += nc) {
    int nc_cur = std::min(nc, N - jc);
    for (int pc = 0; pc < K; pc += kc) {
      int kc_cur = std::min(kc, K - pc);
      packB(B, ldb, packB_buf, 
            K, N, kc_cur, nc_cur, pc, jc);
      for (int ic = 0; ic < M; ic += mc) {
        int mc_cur = std::min(mc, M - ic);
        packA(A, lda, packA_buf,
              M, K, mc_cur, kc_cur, ic, pc);

        microkernel(packA_buf, packB_buf,
                    C + ic * ldc + jc,
                    mc_cur, nc_cur, kc_cur, ldc);
      }
    }
  }

  delete[] packA_buf;
  delete[] packB_buf;
}

void referenceGEMM(const double* A, const double* B, double* C,
                   int M, int N, int K, int lda, int ldb, int ldc) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int p = 0; p < K; p++) {
        sum += A[i * lda + p] * B[p * ldb + j];
      }
      C[i * ldc + j] = sum;
    }
  }
}

// Compare two matrices
bool compareMatricies(const double* C1, const double* C2,
                      int M, int N, int ldc, double tol) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double diff = std::abs(C1[i * ldc + j] - C2[i * ldc + j]);
      if (diff > tol) {
        std::cerr << "Mismatch at (" << i << "," << j
                  << "): " << C1[i * ldc + j]
                  << " vs " << C2[i * ldc + j] << std::endl;
        return false;
      }
    }
  }
  return true;
}
