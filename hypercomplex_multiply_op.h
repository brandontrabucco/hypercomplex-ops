#ifndef TENSORFLOW_KERNELS_HYPERCOMPLEX_MULTIPLY_OP_H_
#define TENSORFLOW_KERNELS_HYPERCOMPLEX_MULTIPLY_OP_H_

namespace tensorflow {
    bool* quick_conjugate(
        bool* input,
        const int length);

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
                for (
                    int i = 0;
                    i < hypercomplex_size * remaining_size;
                    i++
                ) {
                    int repositioned_index = (int)(i / hypercomplex_size) * hypercomplex_size;
                    const T* repositioned_tensor_left = &(in_tensor_left[repositioned_index]);
                    const T* repositioned_tensor_right = &(in_tensor_right[repositioned_index]);
                    bool* sign_left = new bool[hypercomplex_size];
                    bool* sign_right = new bool[hypercomplex_size];

                    [hypercomplex_size](bool* _sl, bool* _sr){
                        for (int j = 0; j < hypercomplex_size; j++) {
                            _sl[j] = false;
                            _sr[j] = false;
                        }
                    }(
                        sign_left,
                        sign_right);

                    out_tensor[i] = partial_cayley_dickson<T>(
                        repositioned_tensor_left,
                        repositioned_tensor_right,
                        sign_left,
                        sign_right,
                        (i % hypercomplex_size),
                        hypercomplex_size);

                    delete[] sign_left;
                    delete[] sign_right;
                }
            }
        };
    } // namespace functor
} // namespace tensorflow

#endif // TENSORFLOW_KERNELS_HYPERCOMPLEX_MULTIPLY_OP_H_
