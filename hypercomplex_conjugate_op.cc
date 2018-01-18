#include "tensorflow/core/user_ops/hypercomplex_conjugate_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include <cmath>

namespace tensorflow {
    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;

    template<typename Device, typename T>
    class HypercomplexConjugateOp : public OpKernel {
        public:
        explicit HypercomplexConjugateOp(OpKernelConstruction* context)
            : OpKernel(context) {
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& input = context->input(0);
            OP_REQUIRES(
                context,
                (std::cos(
                    6.28318530718 *
                    std::log(input.dim_size(input.dims() - 1)) /
                    std::log(2)) > 0.9),
                errors::InvalidArgument(
                    "final dim must be 2**n for integer n >= 0, but got shape ",
                    input.shape().DebugString()
                )
            );

            Tensor* output = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    0,
                    input.shape(),
                    &output)
            );

            int hypercomplex_size = input.dim_size(input.dims() - 1);
            int remaining_size = 1;
            for (int i = 0; i < input.dims() - 1; i++) {
                remaining_size *= input.dim_size(i);
            }

            functor::HypercomplexConjugate<Device, T>()(
                context->eigen_device<Device>(),
                input.flat<T>().data(),
                output->flat<T>().data(),
                hypercomplex_size,
                remaining_size
            );
        }
    };

    REGISTER_OP("HypercomplexConjugate")
        .Attr("T: {uint8, int8, int16, int32, float, double} = DT_FLOAT")
        .Input("to_conjugate: T")
        .Output("conjugate: T")
        .Doc(R"doc(
            Calculates the hypercomplex conjugate along the final axis of Tensor to_conjugate by negating all elements along that axis except the fisrt element.
            )doc");

    #define REGISTER_KERNEL(T)                                                   \
    REGISTER_KERNEL_BUILDER(                                                     \
        Name("HypercomplexConjugate").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        HypercomplexConjugateOp<CPUDevice, T>);

    REGISTER_KERNEL(uint8);
    REGISTER_KERNEL(int8);
    REGISTER_KERNEL(int16);
    REGISTER_KERNEL(int32);
    REGISTER_KERNEL(float);
    REGISTER_KERNEL(double);
    #undef REGISTER_KERNEL

    #if GOOGLE_CUDA
    namespace functor {
        #define DECLARE_GPU_SPEC(T)                           \
        template <>                                           \
        void HypercomplexConjugate<GPUDevice, T>::operator()( \
            const GPUDevice& device,                          \
            const T* in_tensor,                               \
            T* out_tensor,                                    \
            int hypercomplex_size,                            \
            int remaining_size);                              \
        extern template struct HypercomplexConjugate<GPUDevice, T>;

        DECLARE_GPU_SPEC(uint8);
        DECLARE_GPU_SPEC(int8);
        DECLARE_GPU_SPEC(int16);
        DECLARE_GPU_SPEC(int32);
        DECLARE_GPU_SPEC(float);
        DECLARE_GPU_SPEC(double);
        #undef DECLARE_GPU_SPEC
    } // namespace functor

    #define REGISTER_GPU_KERNEL(T)                                               \
    REGISTER_KERNEL_BUILDER(                                                     \
        Name("HypercomplexConjugate").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        HypercomplexConjugateOp<GPUDevice, T>);

    REGISTER_GPU_KERNEL(uint8);
    REGISTER_GPU_KERNEL(int8);
    REGISTER_GPU_KERNEL(int16);
    REGISTER_GPU_KERNEL(int32);
    REGISTER_GPU_KERNEL(float);
    REGISTER_GPU_KERNEL(double);
    #undef REGISTER_GPU_KERNEL

    #endif // GOOGLE_CUDA
} // namespace tensorflow

