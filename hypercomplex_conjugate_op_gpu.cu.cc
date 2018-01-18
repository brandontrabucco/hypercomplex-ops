#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/user_ops/hypercomplex_conjugate_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
  typedef Eigen::GpuDevice GPUDevice;

  template<typename T>
  __global__ void HypercomplexConjugateCudaKernel(
    const T* in_tensor,
    T* out_tensor,
    const int hypercomplex_size,
    const int remaining_size
  ) {
    for (
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      i < hypercomplex_size * remaining_size;
      i += blockDim.x * gridDim.x
    ) {
      if (i % hypercomplex_size == 0) { // Here we assume hypercomplex blocks are adjacent in memory
        out_tensor[i] = in_tensor[i];
      }
      else {
        out_tensor[i] = -in_tensor[i];
      }
    }
  }

  namespace functor {
    template<typename T>
    struct HypercomplexConjugate<GPUDevice, T> {
      void operator()(
          const GPUDevice& device,
          const T* in_tensor,
          T* out_tensor,
          const int hypercomplex_size,
          const int remaining_size
      ) {
        int block_count = 1024;
        int thread_per_block = 20;
        HypercomplexConjugateCudaKernel<T><<<
          block_count,
          thread_per_block,
          0,
          device.stream()
        >>>(
          in_tensor,
          out_tensor,
          hypercomplex_size,
          remaining_size
        );
      }
    };
  } // namespace functor

  template struct functor::HypercomplexConjugate<GPUDevice, uint8>;
  template struct functor::HypercomplexConjugate<GPUDevice, int8>;
  template struct functor::HypercomplexConjugate<GPUDevice, int16>;
  template struct functor::HypercomplexConjugate<GPUDevice, int32>;
  template struct functor::HypercomplexConjugate<GPUDevice, float>;
  template struct functor::HypercomplexConjugate<GPUDevice, double>;
} // namespace tensorflow

#endif // GOOGLE_CUDA
