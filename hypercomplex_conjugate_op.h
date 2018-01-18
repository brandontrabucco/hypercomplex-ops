#ifndef TENSORFLOW_KERNELS_HYPERCOMPLEX_CONJUGATE_OP_H_
#define TENSORFLOW_KERNELS_HYPERCOMPLEX_CONJUGATE_OP_H_

namespace tensorflow {
  namespace functor {
    template <typename Device, typename T>
    struct HypercomplexConjugate{
      void operator()(
        const Device& device,
        const T* in_tensor,
        T* out_tensor,
        const int hypercomplex_size,
        const int remaining_size
      ) {
        for (
          int i = 0;
          i < hypercomplex_size * remaining_size;
          i++
        ) {
          if (i % hypercomplex_size == 0) { // Here we assume hypercomplex blocks are adjacent in memory
            out_tensor[i] = in_tensor[i];
          }
          else {
            out_tensor[i] = -in_tensor[i];
          }
        }
      }
    };
  } // namespace functor
} // namespace tensorflow

#endif // TENSORFLOW_KERNELS_HYPERCOMPLEX_CONJUGATE_OP_H_
