import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import tensorflow as tf

from cogsci_2019_concept_learning.utils.utils_tf import cosine_distance, euclidean_distance


def unstack_apply_stack(X, Y, f):

    X = [np.squeeze(x) for x in np.split(X, X.shape[0], axis=0)]
    Y = [np.squeeze(y) for y in np.split(Y, Y.shape[0], axis=0)]

    f_out = [f(x, y) for x, y in zip(X, Y)]

    return np.stack(f_out, axis=0)


class CosineDistanceTest(tf.test.TestCase):

    def testCosineDistance(self):
        with self.test_session():
            X = np.random.rand(16, 8, 512)
            Y = np.random.rand(16, 8, 512)

            computed_distance = cosine_distance(tf.constant(X), tf.constant(Y))
            actual_distance = 1 - unstack_apply_stack(X, Y, cosine_similarity)

            self.assertAllClose(computed_distance.eval(), actual_distance)


class EuclideanDistanceTest(tf.test.TestCase):

    def testEuclideanDistance(self):
        with self.test_session():
            X = np.random.rand(16, 8, 512)
            Y = np.random.rand(16, 8, 512)

            computed_distance = euclidean_distance(tf.constant(X), tf.constant(Y))
            actual_distance = unstack_apply_stack(X, Y, euclidean_distances)

            self.assertAllClose(computed_distance.eval(), actual_distance)


if __name__ == '__main__':
    tf.test.main()
