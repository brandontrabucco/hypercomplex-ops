import tensorflow as tf
import numpy as np

class HypercomplexTest(tf.test.TestCase):

  def npConjugate(self, a):
    if a.shape[-1] == 1:
      return a
    else:
      return np.concatenate([
        self.npConjugate(a[..., :(a.shape[-1] // 2)]),
        (-1 * a[..., (a.shape[-1] // 2):a.shape[-1]])
      ], axis=(len(a.shape) - 1))

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

  def tfConjugate(self, a):
    return self.module.hypercomplex_conjugate(a).eval()

  def testHypercomplexCPU(self):

    print("Beginning CPU test cases.")

    with self.test_session(use_gpu=False):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      test_input = np.random.normal(0, 1, (1, 1))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      print("All CPU tests cases passes!")

  def testHypercomplexGPU(self):

    print("Beginning GPU test cases.")

    with self.test_session(use_gpu=True):
      self.module = tf.load_op_library(
        '/home/brand/tensorflow/bazel-bin/' + 
        'tensorflow/core/user_ops/hypercomplex.so')

      test_input = np.random.normal(0, 1, (1, 1))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (1, 32))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (5, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)

      test_input = np.random.normal(0, 1, (4, 2, 3, 2))
      np_result = self.npConjugate(test_input)
      tf_result = self.tfConjugate(test_input)
      self.assertAllClose(np_result, tf_result)
      print("All GPU tests cases passes!")

if __name__ == "__main__":
  tf.test.main()
