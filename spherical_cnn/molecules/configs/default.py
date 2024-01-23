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

"""Config for training our best models on QM9 targets."""

from collections import abc
import ml_collections


TARGETS = [
    {"dataset": "qm9+/mu",
     "atom_aggregation_mode": "weighted_displacement",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/alpha",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "transformer"},

    {"dataset": "qm9+/homo",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/lumo",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/r2",
     "atom_aggregation_mode": "weighted_by_distance_squared",
     "tail_module": "transformer"},

    {"dataset": "qm9+/zpve",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/U0",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/U",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/H",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/G",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},

    {"dataset": "qm9+/Cv",
     "atom_aggregation_mode": "weighted_by_atom_type",
     "tail_module": "deepsets"},
]


def get_config():
  """Config to train on dummy data; it should be specialized for other tasks."""
  config = ml_collections.ConfigDict()

  # Model hyperparameters.
  config.model_name = "spin_spherical_residual_regressor"
  config.resolutions = (32, 32, 16, 16, 8, 8)
  config.spins = ((0,), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
  config.widths = (5, 64, 128, 128, 256, 256)
  config.num_filter_params = (8, 8, 8, 8, 4, 4)
  # The compilation cache doesn't seem to work with spectral_pooling, so
  # config.precompile should be False when using it.
  config.spectral_pooling = True
  # One of 'sum', 'deepsets' or 'transformer'.
  config.tail_module = "deepsets"
  config.use_atom_type_embedding = True
  config.use_distance_to_center = True
  config.atom_aggregation_mode = "weighted_by_atom_type"
  config.num_transformer_layers = 4
  config.num_transformer_heads = 4
  config.dropout_rate = 0.0
  config.attention_dropout_rate = 0.0

  config.dataset = "qm9+/H"
  # Resolution of the spherical representation of molecules.
  config.sphere_resolution = 32
  config.molecule_to_sphere_method = "gaussian"
  # How the input spherical feature values decay with distances. 2.0
  # corresponds to the Coulomb force.
  config.distance_powers = (2.0, 6.0)
  # Allows loading from an arbitrary checkpoint, typically used for evaluating a
  # trained model.
  config.load_checkpoint_from = ""
  # Evaluates on the test set (use sparingly).
  config.eval_on_test = False
  # When True, randomly rotate input molecules.
  config.rotation_augmentation = True
  # When True, instead of padding each batch to the maximum number of atoms in
  # the dataset, pads to the maximum along each batch. This makes
  # data-preprocessing slower but can make training faster overall.
  config.per_batch_padding = True
  # If True, XLA-compile the training steps for all possible input shapes before
  # training starts.
  config.precompile = False

  # This is the learning rate for batch size 32. The code scales it linearly
  # with the batch size.
  config.learning_rate = 1e-4
  config.learning_rate_schedule = "cosine"
  config.loss_type = "l1"
  config.warmup_epochs = 1
  config.weight_decay = 0.
  config.num_epochs = 2000
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = -1
  # Evaluates for a full epoch if num_eval_steps==-1. Set to a smaller value for
  # fast iteration when running train.train_and_eval() from a Colab.
  config.num_eval_steps = -1
  config.per_device_batch_size = 1
  # If batches should be added to evaluate the entire dataset.
  config.eval_pad_last_batch = True

  config.log_loss_every_steps = 1_000
  config.eval_every_steps = 20_000
  config.checkpoint_every_steps = 10_000
  config.shuffle_buffer_size = 100_000

  config.seed = 0

  config.trial = 0  # Dummy for repeated runs.
  config.lock()
  return config
