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

"""Deterministic input pipeline."""

from typing import Callable, Dict, Optional, Union
from clu import deterministic_data
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from spherical_cnn import input_pipeline_utils
# Register spherical_mnist so that tfds.load works.
import tensorflow as tf
import tensorflow_datasets as tfds

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]


def create_datasets(
    config: ml_collections.ConfigDict, data_rng: jnp.ndarray
) -> input_pipeline_utils.DatasetSplits:
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.

  Returns:
    A DatasetSplits object containing the dataset info, and the train,
    validation, and test splits.
  """
  if config.dataset == "tiny_dummy":
    return _create_dataset_tiny_dummy(config)
  if config.dataset in ["spherical_mnist/rotated", "spherical_mnist/canonical"]:
    return _create_dataset_spherical_mnist(config, data_rng)
  else:
    raise ValueError(f"Dataset {config.dataset} not supported.")


def _create_dataset_tiny_dummy(
    config: ml_collections.ConfigDict,
) -> input_pipeline_utils.DatasetSplits:
  """Create low-resolution dataset for testing. See create_datasets()."""
  size = 100
  resolution = 8
  n_spins = 1
  n_channels = 1
  num_classes = 10
  shape = (size, resolution, resolution, n_spins, n_channels)
  entries = np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
  labels = np.resize(np.arange(num_classes), [size])
  train_dataset = tf.data.Dataset.from_tensor_slices({"input": entries,
                                                      "label": labels})
  train_dataset = train_dataset.batch(config.per_device_batch_size,
                                      drop_remainder=True)
  train_dataset = train_dataset.batch(jax.local_device_count(),
                                      drop_remainder=True)

  features = tfds.features.FeaturesDict(
      {"label": tfds.features.ClassLabel(num_classes=num_classes)})
  builder = tfds.testing.DummyDataset()
  dataset_info = tfds.core.DatasetInfo(builder=builder, features=features)

  # We don't really care about the difference between train, validation and test
  # and for dummy data.
  return input_pipeline_utils.DatasetSplits(
      info=dataset_info,
      train=train_dataset,
      validation=train_dataset.take(5),
      test=train_dataset.take(5),
  )


def _preprocess_spherical_mnist(features: Features) -> Features:
  features["input"] = tf.cast(features["image"], tf.float32) / 255.0
  # Add dummy spin dimension.
  features["input"] = features["input"][..., None, :]
  features.pop("image")
  return features


def create_train_dataset(
    config: ml_collections.ConfigDict,
    dataset_builder: tfds.core.DatasetBuilder,
    split: str,
    data_rng: jnp.ndarray,
    preprocess_fn: Optional[Callable[[Features], Features]] = None,
) -> tf.data.Dataset:
  """Create train dataset."""
  # This ensures determinism in distributed setting.
  train_split = deterministic_data.get_read_instruction_for_host(
      split, dataset_info=dataset_builder.info)
  train_dataset = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      rng=data_rng,
      preprocess_fn=preprocess_fn,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=[jax.local_device_count(), config.per_device_batch_size],
      num_epochs=config.num_epochs,
      shuffle=True,
  )
  options = tf.data.Options()
  options.experimental_external_state_policy = (
      tf.data.experimental.ExternalStatePolicy.WARN)
  train_dataset = train_dataset.with_options(options)

  return train_dataset


def _create_dataset_spherical_mnist(
    config: ml_collections.ConfigDict, data_rng: jnp.ndarray
) -> input_pipeline_utils.DatasetSplits:
  """Create Spherical MNIST. See create_datasets()."""

  dataset_loaded = False
  if not dataset_loaded:
    dataset_builder = tfds.builder("spherical_mnist")

  if config.dataset == "spherical_mnist/rotated":
    train_split = "train_rotated"
    validation_split = "validation_rotated"
    test_split = "test_rotated"
  elif config.dataset == "spherical_mnist/canonical":
    train_split = "train_canonical"
    validation_split = "validation_canonical"
    test_split = "test_canonical"
  else:
    raise ValueError(f"Unrecognized dataset: {config.dataset}")

  if config.combine_train_val_and_eval_on_test:
    train_split = f"{train_split} + {validation_split}"

  train_dataset = create_train_dataset(
      config,
      dataset_builder,
      train_split,
      data_rng,
      preprocess_fn=_preprocess_spherical_mnist)
  validation_dataset = input_pipeline_utils.create_eval_dataset(
      config,
      dataset_builder,
      validation_split,
      preprocess_fn=_preprocess_spherical_mnist,
  )
  test_dataset = input_pipeline_utils.create_eval_dataset(
      config,
      dataset_builder,
      test_split,
      preprocess_fn=_preprocess_spherical_mnist,
  )

  return input_pipeline_utils.DatasetSplits(
      info=dataset_builder.info,
      train=train_dataset,
      validation=validation_dataset,
      test=test_dataset,
  )
