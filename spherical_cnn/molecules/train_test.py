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

"""Tests for spin_spherical_cnns.train."""

import functools
from unittest import mock

from absl import logging
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from spherical_cnn.molecules import input_pipeline
from spherical_cnn.molecules import train
from spherical_cnn.molecules.configs import default
import tensorflow as tf
import tensorflow_datasets as tfds


class TrainTest(tf.test.TestCase, parameterized.TestCase):
  """Test cases for ImageNet library."""

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], "GPU")

  @property
  def is_tpu(self):
    return jax.local_devices()[0].platform == "tpu"

  @parameterized.parameters(
      (0, 0.0),
      (1, 6.410256901290268e-05),
      (1000, 0.06410256773233414),
      (1560, 0.10000000149011612),
      (3000, 0.09927429258823395),
      (6000, 0.09324192255735397),
      (10000, 0.077022984623909))
  def test_get_learning_rate(self, step: int, expected_lr: float):
    actual_lr = train.get_learning_rate(
        step, base_learning_rate=0.1, steps_per_epoch=312, num_epochs=90)
    self.assertAllClose(expected_lr, actual_lr)

  @parameterized.parameters(
      (0, 0.0),
      (1, 6.410256901290268e-05),
      (1000, 0.06410256773233414),
      (1560, 0.10000000149011612),
      (3000, 0.09927429258823395),
      (6000, 0.09324192255735397),
      (10000, 0.077022984623909))
  def test_get_learning_rate_jitted(self, step: int, expected_lr: float):
    lr_fn = jax.jit(
        functools.partial(
            train.get_learning_rate,
            base_learning_rate=0.1,
            steps_per_epoch=312,
            num_epochs=90))
    actual_lr = lr_fn(jnp.array(step))
    self.assertAllClose(expected_lr, actual_lr)

  @parameterized.parameters(
      dict(per_batch_padding=False,
           molecule_to_sphere_method="gaussian"),
  )
  def test_train_and_evaluate_mock_qm9(self,
                                       per_batch_padding,
                                       molecule_to_sphere_method):
    config = default.get_config()
    config.model_name = "spin_spherical_regressor"
    config.resolutions = (8, 4)
    config.spins = ((0,), (0, 1))
    config.widths = (5, 3)
    config.num_filter_params = None
    config.tail_module = "deepsets"
    config.dataset = "qm9/H"
    config.sphere_resolution = 8
    config.molecule_to_sphere_method = molecule_to_sphere_method
    config.per_device_batch_size = 2
    config.checkpoint_every_steps = 1
    config.num_train_steps = 2
    config.num_eval_steps = 1
    config.num_epochs = 1
    config.warmup_epochs = 0
    config.eval_pad_last_batch = False
    config.rotation_augmentation = True
    config.per_batch_padding = per_batch_padding

    config.spectral_pooling = False
    config.use_distance_to_center = False
    config.distance_powers = (2.0,)
    config.rotation_augmentation = False
    config.shuffle_buffer_size = 1000

    workdir = self.create_tempdir().full_path
    with tfds.testing.mock_data(
        num_examples=8,
    ):
      # We mock min_atoms to avoid the overhead of recompiling `train_step` many
      # times for each test.
      with mock.patch.dict(input_pipeline.QM9_METADATA, min_atoms=28):
        train.train_and_evaluate(config, workdir)
        # Running again to load from checkpoint.
        config.num_epochs = 2
        train.train_and_evaluate(config, workdir)
        logging.info("workdir content: %s", tf.io.gfile.listdir(workdir))

  def test_jax_conjugate_gradient(self):
    """Check that current JAX version conjugates the gradients.

    The spin-weighted model has complex weights, and the training loop
    explicitly conjugates all gradients in order to make backprop work. JAX will
    possibly change this default (see https://github.com/google/jax/issues/4891)
    which may silently break our training. This checks if the convention is
    still the one we assume.
    """
    # A function that returns the imaginary part has gradient 1j in TensorFlow
    # convention, which can be directly used for gradient descent. In JAX, the
    # convention returns the conjugate of that, -1j.
    imaginary_part = lambda x: x.imag
    gradient = jax.grad(imaginary_part)
    self.assertAllClose(gradient(0.1 + 0.2j), -1j)

  def test_compute_loss_l1(self):
    labels = jnp.array([1.0, 2.0])
    predictions = jnp.array([1.0, 0.0])
    loss = train._compute_loss(predictions, labels,
                               loss_type="l1",
                               average_over_batch=True)
    self.assertAllClose(loss, 1.0)

  def test_compute_loss_l2(self):
    labels = jnp.array([1.0, 3.0])
    predictions = jnp.array([1.0, 0.0])
    loss = train._compute_loss(predictions, labels,
                               loss_type="l2",
                               average_over_batch=False)
    self.assertAllClose(loss, jnp.array([0.0, 4.5]))

  @parameterized.parameters(
      dict(x=[1.0, 2.0, 3.0], mean=1.5, standard_deviation=0.2),
      dict(x=[4.0, 5.0], mean=2.5, standard_deviation=2.0),
  )
  def test_denormalize_normalize_is_identity(self, x, mean, standard_deviation):
    """Check that `_denormalize` is the inverse of `normalize`."""
    x = jnp.array(x)
    normalized = train._normalize(x, mean, standard_deviation)
    self.assertAllClose(
        train._denormalize(normalized, mean, standard_deviation), x)

  def test_compute_metrics(self):
    prediction = jnp.array([1.0, 2.0])
    label = jnp.array([2.0, 0.0])
    mean_squared_error, mean_absolute_error = train._compute_metrics(prediction,
                                                                     label)
    # Metrics should be the MSE and MAE.
    self.assertAllClose(mean_squared_error, jnp.array([1.0, 4.0]))
    self.assertAllClose(mean_absolute_error, jnp.array([1.0, 2.0]))

if __name__ == "__main__":
  tf.test.main()
