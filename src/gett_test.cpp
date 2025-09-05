#include <iostream>
#include <random>
#include <gett.h>

int main() {
  // Problem size
  int M = 64, N = 64, K = 64;
  int lda = K, ldb = N, ldc = N;

  // Block sizes (tune later)
  int mc = 32, nc = 32, kc = 32;

  // Allocate matrices
  double* A = new double[M*K];
  double* B = new double[K*N];
  double* C_gett = new double[M*N];
  double* C_ref = new double[M*N];

  // RNG for test matrices
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int i = 0; i < M*K; i++) A[i] = dist(gen);
  for (int i = 0; i < K*N; i++) B[i] = dist(gen);
  for (int i = 0; i < M*N; i++) C_gett[i] = 0.0;
  for (int i = 0; i < M*N; i++) C_ref[i] = 0.0;

  // Run reference GEMM
  referenceGEMM(A, B, C_ref, M, N, K, lda, ldb, ldc);

  // Run GETT
  GETT(A, B, C_gett, M, N, K, lda, ldb, ldc, mc, nc, kc);

  // Compare
  if (compareMatricies(C_gett, C_ref, M, N, ldc)) {
    std::cout << "GETT matches reference GEMM!" << std::endl;
  } else {
    std::cerr << "GETT mismatch detected." << std::endl;
  }

  return 0;
}

