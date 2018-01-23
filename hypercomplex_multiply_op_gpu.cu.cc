#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/user_ops/hypercomplex_multiply_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
    typedef Eigen::GpuDevice GPUDevice;

    __device__ void index_to_shape(
        int* output,
        int* shape,
        int skip_axis,
        int target,
        int dims
    ) {
        if (skip_axis == 0) {
            index_to_shape(
                &(output[1]),
                &(shape[1]),
                (skip_axis - 1),
                target,
                (dims - 1));
        } else if (dims < 1) {
            return;
        } else if (dims == 1) {
            output[0] = target;
        } else {
            output[0] = target % shape[0];
            index_to_shape(
                &(output[1]),
                &(shape[1]),
                (skip_axis - 1),
                (target / shape[0]),
                (dims - 1));
        }
    }

    __device__ int shape_to_index(
        int* shape,
        int* select,
        int dims
    ) {
        int result = 0;
        for (int i = 0; i < dims; i++) {
            int buffer = select[i];
            for (int j = i + 1; j < dims; j++) {
                buffer *= shape[j];
            } result += buffer;
        } return result;
    }

    __device__ double get_helper(
        double* input,
        int* shape,
        int hypercomplex_axis,
        int hypercomplex_index,
        int remaining_index,
        int dims
    ) {
        int* select_shape = new int[dims];
        select_shape[hypercomplex_axis] = hypercomplex_index;
        index_to_shape(
            select_shape,
            shape,
            hypercomplex_axis,
            remaining_index,
            dims);
        int index = shape_to_index(
            shape,
            select_shape,
            dims);
        return input[index];
    }

    __device__ bool* quick_conjugate_gpu(
        bool* input,
        const int length
    ) {
        for (int i = 1; i < length; i++) {
            input[i] = !input[i];
        } return input;
    }

    template<typename T>
    __device__ T partial_cayley_dickson_gpu(
        const T* left,
        const T* right,
        bool* left_sign,
        bool* right_sign,
        const int target,
        const int length
    ) {
        if (length == 1) {
            if (left_sign[0] != right_sign[0]) {
                return -left[0] * right[0];
            } else {
                return left[0] * right[0];
            }

        } else {
            bool is_left = (((float)target) /
                ((float)(length))) < 0.5f;
            if (is_left) {
                return partial_cayley_dickson_gpu<T>(
                    &(left[0]),
                    &(right[0]),
                    &(left_sign[0]),
                    &(right_sign[0]),
                    target,
                    (length / 2)
                ) - partial_cayley_dickson_gpu<T>(
                    &(right[(length / 2)]),
                    &(left[(length / 2)]),
                    quick_conjugate_gpu(
                        &(right_sign[(length / 2)]),
                        (length / 2)
                    ),
                    &(left_sign[(length / 2)]),
                    target,
                    (length / 2)
                );

            } else {
                return partial_cayley_dickson_gpu<T>(
                    &(right[(length / 2)]),
                    &(left[0]),
                    &(right_sign[(length / 2)]),
                    &(left_sign[0]),
                    (target - (length / 2)),
                    (length / 2)
                ) + partial_cayley_dickson_gpu<T>(
                    &(left[(length / 2)]),
                    &(right[0]),
                    &(left_sign[(length / 2)]),
                    quick_conjugate_gpu(
                        &(right_sign[0]),
                        (length / 2)
                    ),
                    (target - (length / 2)),
                    (length / 2)
                );
            }
        }
    } // __device__ T* partial_cayley_dickson_gpu

    template<typename T>
    __global__ void HypercomplexMultiplyCudaKernel(
        const T* in_tensor_left,
        const T* in_tensor_right,
        bool* sign_left,
        bool* sign_right,
        T* out_tensor,
        const int hypercomplex_size,
        const int remaining_size
    ) {
        for (
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < hypercomplex_size * remaining_size;
            i += blockDim.x * gridDim.x
        ) {
            int repositioned_index = (int)(i / hypercomplex_size) * hypercomplex_size;
            const T* repositioned_tensor_left = &(in_tensor_left[repositioned_index]);
            const T* repositioned_tensor_right = &(in_tensor_right[repositioned_index]);
            bool* repositioned_sign_left = &(sign_left[i * hypercomplex_size]);
            bool* repositioned_sign_right = &(sign_right[i * hypercomplex_size]);

            out_tensor[i] = partial_cayley_dickson_gpu<T>(
                repositioned_tensor_left,
                repositioned_tensor_right,
                repositioned_sign_left,
                repositioned_sign_right,
                (i % hypercomplex_size),
                hypercomplex_size);
        }
    } // __global__ void HypercomplexMultiplyCudaKernel

    namespace functor {
        template<typename T>
        struct HypercomplexMultiply<GPUDevice, T> {
            void operator()(
                const GPUDevice& device,
                const T* in_tensor_left,
                const T* in_tensor_right,
                T* out_tensor,
                const int hypercomplex_size,
                const int remaining_size
            ) {
                size_t memory_size = sizeof(bool) * hypercomplex_size * hypercomplex_size * remaining_size;
                bool* sign_left_buffer;
                bool* sign_right_buffer;
                cudaMalloc(&sign_left_buffer, memory_size);
                cudaMalloc(&sign_right_buffer, memory_size);
                cudaMemset(sign_left_buffer, 0, memory_size);
                cudaMemset(sign_right_buffer, 0, memory_size);

                int block_count = 1024;
                int thread_per_block = 20;
                HypercomplexMultiplyCudaKernel<T><<<
                    block_count,
                    thread_per_block,
                    0,
                    device.stream()
                >>>(
                    in_tensor_left,
                    in_tensor_right,
                    sign_left_buffer,
                    sign_right_buffer,
                    out_tensor,
                    hypercomplex_size,
                    remaining_size
                );

                cudaFree(sign_left_buffer);
                cudaFree(sign_right_buffer);
            }
        };
    } // namespace functor

    template struct functor::HypercomplexMultiply<GPUDevice, uint8>;
    template struct functor::HypercomplexMultiply<GPUDevice, int8>;
    template struct functor::HypercomplexMultiply<GPUDevice, int16>;
    template struct functor::HypercomplexMultiply<GPUDevice, int32>;
    template struct functor::HypercomplexMultiply<GPUDevice, float>;
    template struct functor::HypercomplexMultiply<GPUDevice, double>;
} // namespace tensorflow

#endif // GOOGLE_CUDA
