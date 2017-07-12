extern "C" __global__ __launch_bounds__(256) void sgemm_nt_128x128_vec(
  const float *param_A,
  const float *param_B,
  float *param_C,
  float param_alpha,
  float param_beta,
  int param_lda8,
  int param_ldb8,
  int param_ldc,
  int param_m,
  int param_n,
  int param_k) {
  __shared__ float share[128 * 8 * 4 + 32];
  int tid = threadIdx.x;
  share[tid] = 1;
}
