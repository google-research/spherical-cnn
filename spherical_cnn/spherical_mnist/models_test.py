# Copyright 2025 The spherical_cnn Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for spin_spherical_cnns.models."""

import functools
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from spherical_cnn import spin_spherical_harmonics
from spherical_cnn import test_utils
from spherical_cnn.spherical_mnist import models
import tensorflow as tf


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=(4, 8, 16), spins=(0, -1, 1, 2))


class SpinSphericalClassifierTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(num_classes=3, num_filter_params=None),
                            dict(num_classes=5, num_filter_params=[3, 2]))
  def test_shape(self, num_classes, num_filter_params):
    transformer = _get_transformer()
    resolutions = [8, 4]
    spins = [[0], [0, 1]]
    channels = [1, 2]
    spectral_pooling = False
    batch_size = 2
    model = models.SpinSphericalClassifier(num_classes,
                                           resolutions,
                                           spins,
                                           channels,
                                           spectral_pooling,
                                           num_filter_params=num_filter_params,
                                           axis_name=None,
                                           input_transformer=transformer)
    resolution = resolutions[0]
    shape = [batch_size, resolution, resolution, len(spins[0]), channels[0]]
    inputs = jnp.ones(shape)
    params = model.init(jax.random.PRNGKey(0), inputs, train=False)
    outputs = model.apply(params, inputs, train=False)

    self.assertEqual(outputs.shape, (batch_size, num_classes))

  @parameterized.parameters(2, 4)
  def test_azimuthal_invariance(self, shift):
    # Make a simple two-layer classifier with pooling for testing.
    resolutions = [8, 4]
    transformer = _get_transformer()
    spins = [[0, -1], [0, 1, 2]]
    channels = [2, 3]
    shape = [2, resolutions[0], resolutions[0], len(spins[0]), channels[0]]
    sphere, _ = test_utils.get_spin_spherical(transformer, shape, spins[0])
    rotated_sphere = jnp.roll(sphere, shift, axis=2)

    model = models.SpinSphericalClassifier(num_classes=5,
                                           resolutions=resolutions,
                                           spins=spins,
                                           widths=channels,
                                           spectral_pooling=False,
                                           axis_name=None,
                                           input_transformer=transformer)
    params = model.init(jax.random.PRNGKey(0), sphere, train=False)

    output, _ = model.apply(params, sphere, train=True,
                            mutable=['batch_stats'])
    rotated_output, _ = model.apply(params, rotated_sphere, train=True,
                                    mutable=['batch_stats'])

    # The classifier should be rotation-invariant.
    self.assertAllClose(rotated_output, output, atol=1e-6)

  def test_invariance(self):
    # Make a simple two-layer classifier with pooling for testing.
    resolutions = [16, 8]
    transformer = _get_transformer()
    spins = [[0, -1], [0, 1, 2]]
    channels = [2, 3]
    shape = [2, resolutions[0], resolutions[0], len(spins[0]), channels[0]]
    pair = test_utils.get_rotated_pair(transformer, shape, spins[0],
                                       alpha=1.0, beta=2.0, gamma=3.0)

    model = models.SpinSphericalClassifier(num_classes=5,
                                           resolutions=resolutions,
                                           spins=spins,
                                           widths=channels,
                                           spectral_pooling=True,
                                           axis_name=None,
                                           input_transformer=transformer)

    params = model.init(jax.random.PRNGKey(0), pair.sphere, train=False)

    output, _ = model.apply(params, pair.sphere, train=True,
                            mutable=['batch_stats'])
    rotated_output, _ = model.apply(params, pair.rotated_sphere, train=True,
                                    mutable=['batch_stats'])

    # The classifier should be rotation-invariant. Here the tolerance is high
    # because the local pooling introduces equivariance errors.
    self.assertAllClose(rotated_output, output, atol=1e-1)
    self.assertLess(test_utils.normalized_mean_absolute_error(output,
                                                              rotated_output),
                    0.05)


class CNNClassifierTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(3, 5)
  def test_shape(self, num_classes):
    resolutions = [8, 4]
    channels = [1, 2]
    batch_size = 2
    model = models.CNNClassifier(num_classes,
                                 resolutions,
                                 channels,
                                 axis_name=None)
    resolution = resolutions[0]
    shape = [batch_size, resolution, resolution, 1, channels[0]]
    inputs = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    params = model.init(jax.random.PRNGKey(0), inputs, train=False)
    outputs = model.apply(params, inputs, train=False)

    self.assertEqual(outputs.shape, (batch_size, num_classes))

if __name__ == '__main__':
  tf.test.main()
