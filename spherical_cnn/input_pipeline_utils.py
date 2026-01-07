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

"""Utilities to build data pipelines."""

import dataclasses
from typing import Dict, Callable, Optional, Union
from clu import deterministic_data
import jax
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]


# Dataset creation functions return info, train, validation and test sets.
@dataclasses.dataclass
class DatasetSplits:
  info: tfds.core.DatasetInfo
  train: tf.data.Dataset
  validation: tf.data.Dataset
  test: tf.data.Dataset


def create_eval_dataset(
    config: ml_collections.ConfigDict,
    dataset_builder: tfds.core.DatasetBuilder,
    split: str,
    preprocess_fn: Optional[Callable[[Features], Features]] = None,
) -> tf.data.Dataset:
  """Create evaluation dataset (validation or test sets)."""
  eval_split = tfds.split_for_jax_process(
      split, drop_remainder=False
  )

  if config.eval_pad_last_batch:
    pad_up_to_batches = "auto"
  else:
    pad_up_to_batches = None

  return deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      # Only cache dataset in distributed setup to avoid consuming a lot of
      # memory in Colab and unit tests.
      cache=jax.process_count() > 1,
      batch_dims=[jax.local_device_count(), config.per_device_batch_size],
      num_epochs=1,
      shuffle=False,
      preprocess_fn=preprocess_fn,
      pad_up_to_batches=pad_up_to_batches,
  )
