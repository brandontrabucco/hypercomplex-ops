#!/bin/bash

cd ../

TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
$TF_CFLAGS
$TF_LFLAGS

cd tensorflow/

bazel build --config=opt --config=cuda --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" --cxxopt="$TF_CFLAGS" --cxxopt="$TF_LFLAGS" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/core/user_ops:hypercomplex.so

python3 tensorflow/core/user_ops/hypercomplex_conjugate_op_tests.py
