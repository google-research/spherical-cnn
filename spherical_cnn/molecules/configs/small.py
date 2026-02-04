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

"""Config for training a small model on QM9/H."""

from collections import abc
from spherical_cnn.molecules.configs import default


def get_config():
  config = default.get_config()
  config.num_epochs = 250
  config.dataset = "qm9/H"
  config.model_name = "spin_spherical_regressor"
  config.use_distance_to_center = False
  config.atom_aggregation_mode = "weighted_by_atom_type"

  return config
