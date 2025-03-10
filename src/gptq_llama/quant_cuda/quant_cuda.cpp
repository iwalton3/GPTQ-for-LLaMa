#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant2matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
);

void vecquant2matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_cuda(vec, mat, mul, scales, zeros,groupsize);
}

void vecquant3matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
);

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant4matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant8matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
);

void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_cuda(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant2matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
);

void vecquant2matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_faster_cuda(vec, mat, mul, scales, zeros, groupsize, vec_height);
}

void vecquant3matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
);

void vecquant3matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_faster_cuda(vec, mat, mul, scales, zeros, groupsize, vec_height);
}

void vecquant4matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx, int vec_height
);

void vecquant4matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_faster_cuda(vec, mat, mul, scales, zeros, g_idx, vec_height);
}

void vecquant4recons_v1_cuda(
  torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros
);

void vecquant4recons_v1(
  torch::Tensor mat, torch::Tensor res,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  vecquant4recons_v1_cuda(mat, res, scales, zeros);
}

void vecquant4recons_v2_cuda(
  torch::Tensor mat, torch::Tensor res,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant4recons_v2(
  torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros, torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  vecquant4recons_v2_cuda(mat, res, scales, zeros, g_idx);
}

void vecquant4matmul_v1_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_v1_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_v1_faster_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4matmul_seq_v2_cuda(
  torch::Tensor vec, torch::Tensor mat_t, torch::Tensor mul,
  torch::Tensor scales_t, torch::Tensor zeros_t, torch::Tensor g_idx
);

void vecquant4matmul_seq_v2(
  torch::Tensor vec, torch::Tensor mat_t, torch::Tensor mul,
  torch::Tensor scales_t, torch::Tensor zeros_t, torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_seq_v2_cuda(vec, mat_t, mul, scales_t, zeros_t, g_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant2matmul", &vecquant2matmul, "Vector 2-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant2matmul_faster", &vecquant2matmul_faster, "Vector 4-bit Quantized Matrix Multiplication (CUDA), faster version");
  m.def("vecquant3matmul_faster", &vecquant3matmul_faster, "Vector 3-bit Quantized Matrix Multiplication (CUDA), faster version");
  m.def("vecquant4matmul_faster", &vecquant4matmul_faster, "Vector 4-bit Quantized Matrix Multiplication (CUDA), faster version");

  // V1 Support for vecquant4matmul_faster
  m.def("vecquant4matmul_v1_faster", &vecquant4matmul_v1_faster, "Vector 4-bit Quantized Matrix Multiplication (CUDA), faster version, v1 support");

  // Reconstruction Kernel
  m.def("vecquant4recons_v1", &vecquant4recons_v1, "Vector 4-bit Quantized Matrix Reconstruction (CUDA)");
  m.def("vecquant4recons_v2", &vecquant4recons_v2, "Vector 4-bit Quantized Matrix Reconstruction (CUDA) with group-size support");

  // Seq Kernel (Experimental)
  m.def("vecquant4matmul_seq_v2", &vecquant4matmul_seq_v2, "Vector 4-bit Quantized Matrix Multiplication (CUDA), sequential version, v2 support");
}