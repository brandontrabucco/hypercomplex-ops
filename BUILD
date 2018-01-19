load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "hypercomplex.so",
    srcs = [
        "hypercomplex_conjugate_op.cc",
        "hypercomplex_conjugate_op.h",
        "hypercomplex_multiply_op.cc",
        "hypercomplex_multiply_op.h",
    ],
    gpu_srcs = [
        "hypercomplex_conjugate_op_gpu.cu.cc",
        "hypercomplex_conjugate_op.h",
        "hypercomplex_multiply_op_gpu.cu.cc",
        "hypercomplex_multiply_op.h",
    ],
)

