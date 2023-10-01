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

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from clu import deterministic_data
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import scipy.spatial
import scipy.spatial.transform
from spherical_cnn import input_pipeline_utils
from spherical_cnn import sphere_utils
import tensorflow as tf
import tensorflow_datasets as tfds


Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]

DatasetSplits = input_pipeline_utils.DatasetSplits

QM9_MEANS = {'A': 11.970818,
             'B': 1.4051478,
             'C': 1.1269084,
             'Cv': -22.131638777770995,
             'G': -2.6028858350372315,
             'H': -2.8300403020858766,
             'U': -2.8139546695327757,
             'U0': -2.7968806253433227,
             'alpha': 75.266045,
             'gap': 0.25202438,
             'homo': -0.24022761,
             'index': 66876.86303,
             'lumo': 0.011797028,
             'mu': 2.6746807,
             'omega1': 3504.0361,
             'r2': 1189.0505,
             'zpve': 0.14907189610350877}

QM9_STANDARD_DEVIATIONS = {'A': 2093.6948,
                           'B': 1.174844,
                           'C': 0.8806893,
                           'Cv': 6.081049081098194,
                           'G': 0.34904410237809486,
                           'H': 0.38544533170099415,
                           'U': 0.3827235925494082,
                           'U0': 0.37936310713158067,
                           'alpha': 8.172906,
                           'gap': 0.04723579,
                           'homo': 0.022003435,
                           'index': 38453.51863749076,
                           'lumo': 0.046889435,
                           'mu': 1.5014887,
                           'omega1': 266.9748,
                           'r2': 280.12296,
                           'zpve': 0.03312398802237524}

# Convert from ha to meV to follow the most commonly reported units.
QM9_ENERGIES = ['G', 'H', 'U', 'U0', 'gap', 'homo', 'lumo', 'zpve']
QM9_UNIT_CONVERSION_FACTORS = {k: 27211.4 for k in QM9_ENERGIES}

QM9_METADATA = {
    'atom_types': (1.0, 6.0, 7.0, 8.0, 9.0),
    'min_atoms': 3,
    'max_atoms': 29,
    'min_pairwise_distance': 0.9586065591971576,
    'max_pairwise_distance': 12.040386407885462,
}


def create_datasets(config: ml_collections.ConfigDict,
                    data_rng: jnp.ndarray) -> DatasetSplits:
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.

  Returns:
    A DatasetSplits object containing the dataset info, and the train,
    validation, and test splits.

  Raises:
    ValueError in case dataset is not supported.
  """
  if config.dataset.startswith('qm9'):
    return _create_dataset_qm9(config, data_rng)
  else:
    raise ValueError(f'Dataset {config.dataset} not supported.')


# This avoids splitting the data between multiple hosts, allowing per-batch
# padding for efficiency.
def _create_train_dataset(
    config: ml_collections.ConfigDict,
    dataset_builder: tfds.core.DatasetBuilder,
    split: str,
    data_rng: jnp.ndarray,
    preprocess_fn: Optional[Callable[[Features], Features]] = None,
) -> tf.data.Dataset:
  """Create train dataset without splitting over multiple hosts."""
  if config.per_batch_padding:
    # Hosts must see the data from other hosts.
    batch_dims = [jax.process_count(),
                  jax.local_device_count(),
                  config.per_device_batch_size]
  else:
    batch_dims = [jax.local_device_count(),
                  config.per_device_batch_size]
    # Dataset can be split among hosts only when not doing per_batch_padding.
    split = deterministic_data.get_read_instruction_for_host(
        split, dataset_builder.info.splits[split].num_examples)

  train_dataset = deterministic_data.create_dataset(
      dataset_builder,
      split=split,
      rng=data_rng,
      preprocess_fn=preprocess_fn,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=batch_dims,
      num_epochs=config.num_epochs,
  )

  return train_dataset


def _preprocess_no_sphere(features: Features,
                          positions_key: str,
                          charges_key: str,
                          regression_target: str,
                          metadata: Dict[str, Any]) -> Features:
  """Returns QM7/QM9 atom positions/charges and molecule properties."""
  positions = features[positions_key]
  charges = features[charges_key]
  target = features[regression_target]
  # Following Cormorant (NeurIPS'19), we subtract the thermochemical energy for
  # some targets. This happens on QM9 only.
  target = target - tf.cast(
      features.get(f'{regression_target}_thermo', 0.0), tf.float32)

  max_atoms = positions.shape[0]
  assert max_atoms == metadata['max_atoms'], 'Inconsistent `max_atoms`!'

  return {
      'positions': positions,
      'charges': charges,
      'label': target,
  }


def _create_dataset_qm9(
    config: ml_collections.ConfigDict,
    data_rng: jnp.ndarray) -> DatasetSplits:
  """Creates QM9 dataset. See create_datasets()."""
  dataset_builder = tfds.builder('qm9')

  # Add metadata.
  # Example: for dataset == 'qm9/H', the regression target will be H (enthalpy).
  dataset, regression_target = config.dataset.split('/')
  # Cormorant, EGNN and others use 100k for training, 17,748 for validation
  # and 13,083 for test. We call this version 'qm9'.
  if dataset == 'qm9':
    train_split_name = 'train'
    test_split_name = 'test'
    validation_split_name = 'validation'
  # TorchMD-Net, PaiNN and others use 110k for training, 10k for validation
  # and 10,831 for test. We call this version 'qm9+'.
  elif dataset == 'qm9+':
    train_split_name = 'train + validation[:10_000]'
    test_split_name = 'test[:10831]'
    validation_split_name = 'validation[10_000:] + test[10831:]'
  else:
    raise ValueError('Unexpected `config.dataset`! '
                     'Expected format: qm9[+]/target.')

  metadata = {
      'atom_types': QM9_METADATA['atom_types'],
      'min_atoms': QM9_METADATA['min_atoms'],
      'max_atoms': QM9_METADATA['max_atoms'],
      'min_pairwise_distance': QM9_METADATA['min_pairwise_distance'],
      'max_pairwise_distance': QM9_METADATA['max_pairwise_distance'],
      'label_mean': QM9_MEANS[regression_target],
      'label_standard_deviation': QM9_STANDARD_DEVIATIONS[regression_target],
      'unit_conversion_factor': QM9_UNIT_CONVERSION_FACTORS.get(
          regression_target, 1.0)
  }

  preprocess_args = dict(positions_key='positions',
                         charges_key='charges',
                         regression_target=regression_target,
                         metadata=metadata)

  preprocess_train = preprocess_eval = functools.partial(
      _preprocess_no_sphere, **preprocess_args)

  train_dataset = _create_train_dataset(
      config, dataset_builder, train_split_name, data_rng,
      preprocess_fn=preprocess_train)
  validation_dataset = input_pipeline_utils.create_eval_dataset(
      config, dataset_builder, validation_split_name,
      preprocess_fn=preprocess_eval)
  test_dataset = input_pipeline_utils.create_eval_dataset(
      config, dataset_builder, test_split_name,
      preprocess_fn=preprocess_eval)

  info = dataset_builder.info
  # We reconstruct info to include the metadata.
  info = tfds.core.DatasetInfo(builder=dataset_builder,
                               metadata=tfds.core.MetadataDict(metadata),
                               description=info.description,
                               features=info.features,
                               supervised_keys=info.supervised_keys,
                               disable_shuffling=info.disable_shuffling,
                               homepage=info.homepage,
                               citation=info.citation)

  return DatasetSplits(info=info,
                       train=train_dataset,
                       validation=validation_dataset,
                       test=test_dataset)
