# Copyright 2024 The spherical_cnn Authors.
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

"""Tests for input_pipeline."""

from absl.testing import parameterized
import jax
import numpy as np
from spherical_cnn import sphere_utils
from spherical_cnn.molecules import input_pipeline
from spherical_cnn.molecules.configs import default
import tensorflow as tf
import tensorflow_datasets as tfds


class InputPipelineTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(1, 2)
  def test_create_datasets_qm9_no_sphere_shape(self, batch_size):
    key = jax.random.PRNGKey(0)
    config = default.get_config()
    batch_size = 1
    config.dataset = 'qm9/H'
    config.per_device_batch_size = batch_size
    config.molecule_to_sphere_method = 'gaussian'

    with tfds.testing.mock_data(
        num_examples=batch_size,
    ):
      splits = input_pipeline.create_datasets(config, key)

    train_batch = iter(splits.train).next()
    validation_batch = iter(splits.validation).next()
    test_batch = iter(splits.test).next()

    # 29 max atoms, 3D position.
    expected_input_shape = (1, batch_size, 29, 3)
    expected_label_shape = (1, batch_size)

    with self.subTest('Train set'):
      # Training set contains the host dimension; val/test don't.
      self.assertEqual(train_batch['positions'].shape,
                       (1,) + expected_input_shape)
      self.assertEqual(train_batch['label'].shape, (1,) + expected_label_shape)
    with self.subTest('Validation set'):
      self.assertEqual(validation_batch['positions'].shape,
                       expected_input_shape)
      self.assertEqual(validation_batch['label'].shape, expected_label_shape)
    with self.subTest('Test set'):
      self.assertEqual(test_batch['positions'].shape, expected_input_shape)
      self.assertEqual(test_batch['label'].shape, expected_label_shape)

if __name__ == '__main__':
  tf.test.main()
