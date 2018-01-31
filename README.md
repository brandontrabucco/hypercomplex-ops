# Hypercomplex Ops

Utility functions for TensorFlow that enable working with differentiable hypercomplex numbers.

The Shared Library included in this repository was compiled on 64-bit Ubuntu 16.04, using GCC 5.4, NVIDIA Graphics Driver 387, CUDA Toolkit 9.1, and cuDNN 7.0.5, and has been tested successfully on an NVIDIA GeForce 1050 Ti 4GB GPU.

## Setup

1. Download and install tensorflow.
2. Download the hypercomplex.so file.
3. Place hypercomplex.so in the same diretcory as project.

## Usage

To use this library, you must first import tensorflow, and then import the op library using load_custom_op_library.

```
import tensorflow as tf
HCX = tf.load_op_library("path/to/hypercomplex.so")
```

You may then use hypercomplex operations in your tensorflow graphs. The final axis is always hypercomplex for efficiency.

```
a = tf.Variable([0.1, -2.3, 7.2, 1.0])
b = tf.Variable([-5.2, 2.0, 1.2, -1.0])
c = HCX.hypercomplex_multiply(a, b)
d = HCX.hypercomplex_conjugate(c)
```

For tensorflow to recognize the gradients of these operations, add the following lines after load_custom_op_library.

```
@ops.RegisterGradient("HypercomplexConjugate")
def _hypercomplex_conjugate_grad(op, grad):
    return [HCX.hypercomplex_conjugate(grad)]


@ops.RegisterGradient("HypercomplexMultiply")
def _hypercomplex_multiply_grad(op, grad):
    return [
    HCX.hypercomplex_multiply(
        grad,
        HCX.hypercomplex_conjugate(op.inputs[1])),
    HCX.hypercomplex_multiply(
        HCX.hypercomplex_conjugate(op.inputs[0]),
        grad)]
```
