
void packA(const double* A, int lda,
           double* packA_buf,
           int M, int K,
           int mc, int kc,
           int i_start, int p_start);

void packB(const double* B, int ldb,
           double* packB_buf,
           int K, int N,
           int kc, int nc,
           int p_start, int j_start);

void microkernel(const double* packA_buf,
                 const double* packB_buf,
                 double* Cblock,
                 int mc, int nc, int kc,
                 int ldc);

void GETT(const double* A, const double* B, double* C,
          int M, int N, int K,
          int lda, int ldb, int ldc,
          int mc, int nc, int kc);

void referenceGEMM(const double* A, const double* B, double* C,
                   int M, int N, int K, int lda, int ldb, int ldc);

bool compareMatricies(const double* C1, const double* C2,
                      int M, int N, int ldc, double tol = 1e-10);
