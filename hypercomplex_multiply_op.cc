#include "tensorflow/core/user_ops/hypercomplex_multiply_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include <cmath>

namespace tensorflow {
    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;

    void index_to_shape(
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

    int shape_to_index(
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

    double get_helper(
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

    bool* quick_conjugate(
        bool* input,
        const int length
    ) {
        for (int i = 1; i < length; i++) {
            input[i] = !input[i];
        } return input;
    }

    template<typename T>
    T partial_cayley_dickson(
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
                return partial_cayley_dickson<T>(
                    &(left[0]),
                    &(right[0]),
                    &(left_sign[0]),
                    &(right_sign[0]),
                    target,
                    (length / 2)
                ) - partial_cayley_dickson<T>(
                    &(right[(length / 2)]),
                    &(left[(length / 2)]),
                    quick_conjugate(
                        &(right_sign[(length / 2)]),
                        (length / 2)
                    ),
                    &(left_sign[(length / 2)]),
                    target,
                    (length / 2)
                );

            } else {
                return partial_cayley_dickson<T>(
                    &(right[(length / 2)]),
                    &(left[0]),
                    &(right_sign[(length / 2)]),
                    &(left_sign[0]),
                    (target - (length / 2)),
                    (length / 2)
                ) + partial_cayley_dickson<T>(
                    &(left[(length / 2)]),
                    &(right[0]),
                    &(left_sign[(length / 2)]),
                    quick_conjugate(
                        &(right_sign[0]),
                        (length / 2)
                    ),
                    (target - (length / 2)),
                    (length / 2)
                );
            }
        }
    } // T* partial_cayley_dickson

    template<typename Device, typename T>
    class HypercomplexMultiplyOp : public OpKernel {
        public:
        explicit HypercomplexMultiplyOp(OpKernelConstruction* context)
            : OpKernel(context) {
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& input_left = context->input(0);
            const Tensor& input_right = context->input(1);
                        
            OP_REQUIRES(
                context,
                ([input_left, input_right](){
                    if (input_left.dims() == input_right.dims()) {
                        for (int i = 0; i < input_left.dims(); i++) {
                            if (input_left.dim_size(i) != input_right.dim_size(i)) {
                                return false;
                            }
                        } return true;
                    } return false;
                }()),
                errors::InvalidArgument(
                    "input tensors must be the same shape, but got shapes ",
                    input_left.shape().DebugString(),
                    " and ",
                    input_right.shape().DebugString()));

            int num_dims = input_left.dims();
            int target_axis = num_dims - 1;
            int* shape = new int[num_dims];
            for (int i = 0; i < num_dims; i++) {
                shape[i] = input_left.dim_size(1);
            }

            int hypercomplex_size = input_left.dim_size(input_left.dims() - 1);
            int remaining_size = 1;
            for (int i = 0; i < input_left.dims() - 1; i++) {
                remaining_size *= input_left.dim_size(i);
            }

            OP_REQUIRES(
                context,
                (std::pow(
                    2,
                    std::round(
                        std::log(hypercomplex_size) /
                        std::log(2))) == hypercomplex_size),
                errors::InvalidArgument(
                    "final dim must be 2**n for integer n >= 0, but got shape ",
                    input_left.shape().DebugString()));

            Tensor* output = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    0,
                    input_left.shape(),
                    &output));

            functor::HypercomplexMultiply<Device, T>()(
                context->eigen_device<Device>(),
                input_left.flat<T>().data(),
                input_right.flat<T>().data(),
                output->flat<T>().data(),
                hypercomplex_size,
                remaining_size);
        }
    }; // class HypercomplexMultiplyOp

    REGISTER_OP("HypercomplexMultiply")
        .Attr("T: {uint8, int8, int16, int32, float, double} = DT_FLOAT")
        .Input("left_factor: T")
        .Input("right_factor: T")
        .Output("product: T")
        .Doc(R"doc(
            Calculates the hypercomplex product along the final axis of Tensors left_factor and right_factor by using the cayley_dickson construction.
            )doc");

    #define REGISTER_KERNEL(T)                                                  \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("HypercomplexMultiply").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        HypercomplexMultiplyOp<CPUDevice, T>);

    REGISTER_KERNEL(uint8);
    REGISTER_KERNEL(int8);
    REGISTER_KERNEL(int16);
    REGISTER_KERNEL(int32);
    REGISTER_KERNEL(float);
    REGISTER_KERNEL(double);
    #undef REGISTER_KERNEL

    #if GOOGLE_CUDA
    namespace functor {
        #define DECLARE_GPU_SPEC(T)                          \
        template <>                                          \
        void HypercomplexMultiply<GPUDevice, T>::operator()( \
            const GPUDevice& device,                         \
            const T* in_tensor_left,                         \
            const T* in_tensor_right,                        \
            T* out_tensor,                                   \
            int hypercomplex_size,                           \
            int remaining_size);                             \
        extern template struct HypercomplexMultiply<GPUDevice, T>;

        DECLARE_GPU_SPEC(uint8);
        DECLARE_GPU_SPEC(int8);
        DECLARE_GPU_SPEC(int16);
        DECLARE_GPU_SPEC(int32);
        DECLARE_GPU_SPEC(float);
        DECLARE_GPU_SPEC(double);
        #undef DECLARE_GPU_SPEC
    } // namespace functor

    #define REGISTER_GPU_KERNEL(T)                                              \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("HypercomplexMultiply").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        HypercomplexMultiplyOp<GPUDevice, T>);

    REGISTER_GPU_KERNEL(uint8);
    REGISTER_GPU_KERNEL(int8);
    REGISTER_GPU_KERNEL(int16);
    REGISTER_GPU_KERNEL(int32);
    REGISTER_GPU_KERNEL(float);
    REGISTER_GPU_KERNEL(double);
    #undef REGISTER_GPU_KERNEL

    #endif // GOOGLE_CUDA
} // namespace tensorflow

