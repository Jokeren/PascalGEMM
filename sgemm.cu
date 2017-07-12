#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

std::map<std::string, CUfunction> functions;
std::vector<CUmodule> modules;

using namespace std;

void load_kernels() {
  const size_t kernel_size = 1;
  const string kernel_name[kernel_size] = {
    "sgemm_tn_128x128_vec",
  };

  for (size_t i = 0; i < kernel_size; ++i) {
    const string& name = kernel_name[i];
    const string path = name + ".cubin";

    CUmodule module;
    CUfunction function;
    CUresult res;

    // load module
    res = cuModuleLoad(&module, path.c_str());
    if (res != CUDA_SUCCESS) {
      std::cerr << "Failed to load module: " << name << std::endl;
      exit(1);
    }

    // load function
    res = cuModuleGetFunction(&function, module, name.c_str());
    if (res != CUDA_SUCCESS) {
      std::cerr << "Failed to load function: " << name << std::endl;
      exit(1);
    }

    functions[name] = function;
    modules.push_back(module);
  }
}

void sgemm_tn(float *A, float *B, float *C, int size) {
  float alpha = 1.0;
  float beta = 0.0;
  int gridA = size / 128 + (size % 128 != 0);
  int gridB = size / 128 + (size % 128 != 0);
  int lda = size * 32;
  int ldb = size * 32;
  int ldc = size;
  void *args[11] = {&A, &B, &C, &alpha, &beta, &lda, &ldb, &ldc, &size, &size, &size};
  CUresult res = cuLaunchKernel(functions["sgemm_tn_128x128_vec"], 1, gridA, gridB, 256, 1, 1, 0, 0, args, NULL); 
  if (res != CUDA_SUCCESS) {
    std::cerr << "Error launching kernel " << res << std::endl;
    exit(1);
  }
}

void sgemm_nn(float *A, float *B, float *C, int size) {
  float alpha = 1.0;
  float beta = 0.0;
  int gridA = size / 128 + (size % 128 != 0);
  int gridB = size / 128 + (size % 128 != 0);
  int lda = size;
  int ldb = size * 32;
  int ldc = size;
  void *args[11] = {&A, &B, &C, &alpha, &beta, &lda, &ldb, &ldc, &size, &size, &size};
  CUresult res = cuLaunchKernel(functions["sgemm_nn_128x128_vec"], 1, gridA, gridB, 256, 1, 1, 0, 0, args, NULL); 
  if (res != CUDA_SUCCESS) {
    std::cerr << "Error launching kernel " << res << std::endl;
    exit(1);
  }
}

void sgemm_nt(float *A, float *B, float *C, int size) {
  float alpha = 1.0;
  float beta = 0.0;
  int gridA = size / 128 + (size % 128 != 0);
  int gridB = size / 128 + (size % 128 != 0);
  int lda = size;
  int ldb = size;
  int ldc = size;
  void *args[11] = {&A, &B, &C, &alpha, &beta, &lda, &ldb, &ldc, &size, &size, &size};
  CUresult res = cuLaunchKernel(functions["sgemm_nt_128x128_vec"], 1, gridA, gridB, 256, 1, 1, 0, 0, args, NULL); 
  if (res != CUDA_SUCCESS) {
    std::cerr << "Error launching kernel " << res << std::endl;
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  cudaFree(0);
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  int size = 4096;
  h_A = (float *)malloc(size * size * sizeof(float));
  h_B = (float *)malloc(size * size * sizeof(float));
  h_C = (float *)malloc(size * size * sizeof(float));
  for (size_t i = 0; i < size * size; ++i) {
    h_A[i] = 1;
    h_B[i] = 1;
  }
  cudaMalloc((void **)&d_A, sizeof(float) * size * size);
  cudaMalloc((void **)&d_B, sizeof(float) * size * size);
  cudaMalloc((void **)&d_C, sizeof(float) * size * size);
  cudaMemcpy(d_A, h_A, sizeof(float) * size * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(float) * size * size, cudaMemcpyHostToDevice);
  load_kernels();
  std::string kernel = std::string(argv[0]);
  if (kernel == "tn") {
    sgemm_tn(d_A, d_B, d_C, size);
  } else if (kernel == "nn") {
    sgemm_nn(d_A, d_B, d_C, size);
  } else if (kernel == "nt") {
    sgemm_nt(d_A, d_B, d_C, size);
  } else {
    std::cerr << "tt kernel not supported: " << std::endl;
    exit(1);
  }
  cudaMemcpy(h_C, d_C, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < size * size; ++i) {
    if (h_C[i] != size) {
      std::cerr << "Error: " << i << ":" << h_C[i] << std::endl;
      exit(1);
    }
  }
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  // run successfully
  std::cout << "finish" << std::endl;
  return 0;
}
