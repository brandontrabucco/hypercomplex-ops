#ifndef TENSORFLOW_KERNELS_HYPERCOMPLEX_MULTIPLY_OP_H_
#define TENSORFLOW_KERNELS_HYPERCOMPLEX_MULTIPLY_OP_H_

namespace tensorflow {

    void index_to_shape(
        int* output,
        int* shape,
        int skip_axis,
        int target,
        int dims
    );

    int shape_to_index(
        int* shape,
        int* select,
        int dims
    );

    double get_helper(
        double* input,
        int* shape,
        int hypercomplex_axis,
        int hypercomplex_index,
        int remaining_index,
        int dims
    );

    bool* quick_conjugate(
        bool* input,
        const int length
    );

    template<typename T>
    T partial_cayley_dickson(
        const T* left,
        const T* right,
        bool* left_sign,
        bool* right_sign,
        const int target,
        const int length);

    namespace functor {
        template <typename Device, typename T>
        struct HypercomplexMultiply {
            void operator()(
                const Device& device,
                const T* in_tensor_left,
                const T* in_tensor_right,
                T* out_tensor,
                const int hypercomplex_size,
                const int remaining_size
            ) {
                int memory_size = hypercomplex_size * hypercomplex_size * remaining_size;
                bool* sign_left = new bool[memory_size];
                bool* sign_right = new bool[memory_size];

                for (
                    int i = 0;
                    i < hypercomplex_size * remaining_size;
                    i++
                ) {
                    int repositioned_index = (int)(i / hypercomplex_size) * hypercomplex_size;
                    const T* repositioned_tensor_left = &(in_tensor_left[repositioned_index]);
                    const T* repositioned_tensor_right = &(in_tensor_right[repositioned_index]);
                    bool* repositioned_sign_left = &(sign_left[i * hypercomplex_size]);
                    bool* repositioned_sign_right = &(sign_right[i * hypercomplex_size]);

                    for (int j = 0; j < hypercomplex_size; j++) {
                        repositioned_sign_left[j] = false;
                        repositioned_sign_right[j] = false;
                    }

                    out_tensor[i] = partial_cayley_dickson<T>(
                        repositioned_tensor_left,
                        repositioned_tensor_right,
                        repositioned_sign_left,
                        repositioned_sign_right,
                        (i % hypercomplex_size),
                        hypercomplex_size);
                }

                delete[] sign_left;
                delete[] sign_right;
            }
        };
    } // namespace functor
} // namespace tensorflow

#endif // TENSORFLOW_KERNELS_HYPERCOMPLEX_MULTIPLY_OP_H_
