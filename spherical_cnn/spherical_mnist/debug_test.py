# Copyright 2023 The spherical_cnn Authors.
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

"""Tests for spin_spherical_cnns.train."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import tensorflow as tf


class DebugTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], "GPU")

  @parameterized.parameters(jnp.float32, jnp.complex64)
  def test_initializer(self, dtype):
    initializer = jax.nn.initializers.variance_scaling(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
        dtype=dtype)

    values = initializer(jax.random.PRNGKey(0), (2, 3, 4, 5), dtype)
    self.assertEqual(values.dtype, dtype)


if __name__ == "__main__":
  tf.test.main()
