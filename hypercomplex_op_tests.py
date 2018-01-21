from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
from time import time

class HypercomplexTest(tf.test.TestCase):

  def npConjugate(self, a):
    if a.shape[-1] == 1:
      return a
    else:
      return np.concatenate([
        self.npConjugate(a[..., :(a.shape[-1] // 2)]),
        (-1 * a[..., (a.shape[-1] // 2):a.shape[-1]])
      ], axis=(len(a.shape) - 1))

  def npConjugateGrad(self, a):
    return self.npConjugate(np.ones(a.shape))

  def npMultiply(self, a, b):
    if a.shape[-1] == 1:
      return a * b
    else:
      def cayley_dickson(p, q, r, s):
        return np.concatenate([
          (self.npMultiply(
            p,
            r) -
           self.npMultiply(
             self.npConjugate(s),
             q)),
          (self.npMultiply(
            s,
            p) +
           self.npMultiply(
             q,
             self.npConjugate(r))),
        ], axis=(len(a.shape) - 1))
      return cayley_dickson(
        a[..., :(a.shape[-1] // 2)],
        a[..., (a.shape[-1] // 2):],
        b[..., :(a.shape[-1] // 2)],
        b[..., (a.shape[-1] // 2):])

  def npMultiplyGrad(self, a, b):
    return [
      self.npMultiply(np.ones(a.shape), self.npConjugate(b)),
      self.npMultiply(self.npConjugate(a), np.ones(a.shape))]

  def tfConjugate(self, a):
    a = tf.constant(a)
    return self.module.hypercomplex_conjugate(a).eval()

  def tfConjugateGrad(self, a):
    a = tf.constant(a)
    da = tf.gradients(self.module.hypercomplex_conjugate(a), a)[0]
    return da.eval()

  def tfMultiply(self, a, b):
    a, b = tf.constant(a), tf.constant(b)
    return self.module.hypercomplex_multiply(a, b).eval()

  def tfMultiplyGrad(self, a, b):
    a, b = tf.constant(a), tf.constant(b)
    da, db = tf.gradients(self.module.hypercomplex_multiply(a, b), [a, b])
    return da.eval(), db.eval()

  def testHypercomplexCPU(self):

    print(";;;;;;;;;;;;;;;;;;;;;;;;;")
    print("Beginning CPU test cases.")
    print(";;;;;;;;;;;;;;;;;;;;;;;;;")

    with self.test_session(use_gpu=False):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      test_input = np.random.normal(0, 1, (8))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (8))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      print(";;;;;;;;;;;;;;;;;;;;;;;;;")
      print("All CPU tests cases pass!")
      print(";;;;;;;;;;;;;;;;;;;;;;;;;")

  def testHypercomplexGPU(self):

    print(";;;;;;;;;;;;;;;;;;;;;;;;;")
    print("Beginning GPU test cases.")
    print(";;;;;;;;;;;;;;;;;;;;;;;;;")

    with self.test_session(use_gpu=True):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      test_input = np.random.normal(0, 1, (8))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (8))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npMultiply(test_input, other_input)
      tf_result = self.tfMultiply(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      print(";;;;;;;;;;;;;;;;;;;;;;;;;")
      print("All GPU tests cases pass!")
      print(";;;;;;;;;;;;;;;;;;;;;;;;;")

  def testHypercomplexSpeed(self):

    print(";;;;;;;;;;;;;;;;;;;;;;;;;;;")
    print("Beginning speed comparison.")
    print(";;;;;;;;;;;;;;;;;;;;;;;;;;;")

    test_input = np.random.normal(0, 1, (100, 100, 100, 2))
    other_input = np.random.normal(0, 1, (100, 100, 100, 2))

    with self.test_session(use_gpu=False):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      start_time = time() * 1000
      self.tfMultiply(test_input, other_input)
      end_time = time() * 1000
      cpu_time = end_time - start_time
      print("CPU took:", cpu_time, "msecs")

    with self.test_session(use_gpu=True):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      start_time = time() * 1000
      self.tfMultiply(test_input, other_input)
      end_time = time() * 1000
      gpu_time = end_time - start_time
      print("GPU took:", gpu_time, "msecs")

    if gpu_time < cpu_time:
      print("GPU is", cpu_time - gpu_time, "msecs faster than CPU!")
    else:
      print("GPU is", gpu_time - cpu_time, "msecs slower than CPU!")

    print(";;;;;;;;;;;;;;;;;;;;;;;;;;")
    print("Speed comparison finished.")
    print(";;;;;;;;;;;;;;;;;;;;;;;;;;")

  def testHypercomplexGrad(self):

    print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    print("Beginning gradient test cases.")
    print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")

    @ops.RegisterGradient("HypercomplexConjugate")
    def _hypercomplex_conjugate_grad(op, grad):
      return [self.module.hypercomplex_conjugate(grad)]

    @ops.RegisterGradient("HypercomplexMultiply")
    def _hypercomplex_multiply_grad(op, grad):
      return [
        self.module.hypercomplex_multiply(
          grad,
          self.module.hypercomplex_conjugate(op.inputs[1])),
        self.module.hypercomplex_multiply(
          self.module.hypercomplex_conjugate(op.inputs[0]),
          grad)]

    with self.test_session(use_gpu=False):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      test_input = np.random.normal(0, 1, (8))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (8))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

    with self.test_session(use_gpu=True):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      test_input = np.random.normal(0, 1, (8))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (8))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npConjugateGrad(test_input)
      tf_result = self.tfConjugateGrad(test_input)
      self.assertAllClose(np_result, tf_result)
      other_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npMultiplyGrad(test_input, other_input)
      tf_result = self.tfMultiplyGrad(test_input, other_input)
      self.assertAllClose(np_result, tf_result)

    print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    print("Finishing gradient test cases.")
    print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")

if __name__ == "__main__":
  tf.test.main()
