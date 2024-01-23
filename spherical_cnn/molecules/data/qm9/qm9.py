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

"""Convert QM9 dataset to tensorflow_datasets (tfds) format.

This module converts the QM9 dataset (Ramakrishnan et al, "Quantum chemistry
structures and properties of 134 kilo molecules", Scientific Data, 2014) to
tensorflow_datasets (tfds).

To build the dataset, run the following from directory containing this file:
$ tfds build.
"""

from typing import Any, Dict, Iterable, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


_DESCRIPTION = """\
QM9 consists of computed geometric, energetic, electronic, and thermodynamic
properties for 134k stable small organic molecules made up of CHONF.
"""

_CITATION = """\
@article{ramakrishnan2014quantum,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias and von Lilienfeld, O Anatole},
  journal={Scientific Data},
  volume={1},
  year={2014},
  publisher={Nature Publishing Group}
}
"""

_HOMEPAGE = 'http://quantum-machine.org/datasets/#qm9'

_ATOMREF_URL = 'https://figshare.com/ndownloader/files/3195395'
_UNCHARACTERIZED_URL = 'https://springernature.figshare.com/ndownloader/files/3195404'
_MOLECULES_URL = 'https://springernature.figshare.com/ndownloader/files/3195389'

_SIZE = 133_885
_CHARACTERIZED_SIZE = 130_831
_TRAIN_SIZE = 100_000
_VALIDATION_SIZE = 17_748
_TEST_SIZE = 13083

_MAX_ATOMS = 29
_CHARGES = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
_LABELS = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap',
           'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']


def _process_molecule(atomref, fname):
  """Read molecule data from file."""
  with open(fname, 'r') as f:
    num_atoms = int(f.readline().rstrip())

  labels = pd.read_table(fname,
                         skiprows=1,
                         nrows=1,
                         sep=r'\s+',
                         names=_LABELS)

  atoms = pd.read_table(fname,
                        skiprows=2,
                        nrows=num_atoms,
                        sep=r'\s+',
                        names=['Z', 'x', 'y', 'z', '_'],
                        usecols=['Z', 'x', 'y', 'z'])

  # Correct exponential notation (6.8*^-6 -> 6.8e-6).
  for key in ['x', 'y', 'z']:
    if atoms[key].values.dtype == 'object':
      # there are unrecognized numbers.
      atoms[key].values[:] = np.array([
          float(x.replace('*^', 'e'))
          for i, x in enumerate(atoms[key].values)])

  charges = np.pad([_CHARGES[v] for v in atoms['Z'].values],
                   (0, _MAX_ATOMS - num_atoms))
  positions = np.stack([atoms['x'].values,
                        atoms['y'].values,
                        atoms['z'].values], axis=-1).astype(np.float32)
  positions = np.pad(positions, ((0, _MAX_ATOMS - num_atoms), (0, 0)))

  example = {'num_atoms': num_atoms,
             'charges': charges,
             'positions': positions.astype(np.float32),
             **{k: labels[k].values[0] for k in _LABELS}}

  # Create thermo targets, accumulating  thermochemical energy of each atom.
  for k, v in atomref.items():
    thermo = 0
    for z in atoms['Z'].values:
      thermo += v[z]
    example[f'{k}_thermo'] = thermo

  return example


def _get_split_ids(uncharacterized):
  """Get train/val/test split ids."""
  # Original data files are  1-indexed.
  characterized_ids = np.array(sorted(set(range(1, _SIZE + 1)) -
                                      set(uncharacterized)))
  assert len(characterized_ids) == _CHARACTERIZED_SIZE

  # We call `np.random.seed(0)` to match Cormorant's splits. However
  # this method is unreliable and not guaranteed to generate same
  # permutations in future python versions, ideally we would use
  # `np.random.default_rng` instead.
  np.random.seed(0)
  ids = np.random.permutation(_CHARACTERIZED_SIZE)

  train_ids = characterized_ids[ids[:_TRAIN_SIZE]]
  validation_ids = characterized_ids[ids[_TRAIN_SIZE:
                                         _TRAIN_SIZE+_VALIDATION_SIZE]]
  test_ids = characterized_ids[ids[_TRAIN_SIZE+_VALIDATION_SIZE:]]

  assert len(validation_ids) == _VALIDATION_SIZE
  assert len(test_ids) == _TEST_SIZE

  return {'train': train_ids,
          'validation': validation_ids,
          'test': test_ids}


class Qm9(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for QM9 dataset. See superclass for details."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'num_atoms': tfds.features.Tensor(shape=(), dtype=tf.int64),
            'charges': tfds.features.Tensor(shape=(29,), dtype=tf.int64),
            'positions': tfds.features.Tensor(shape=(29, 3), dtype=tf.float32),
            'index': tfds.features.Tensor(shape=(), dtype=tf.int64),
            'A': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'B': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'C': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'mu': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'alpha': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'homo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'lumo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'gap': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'r2': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'zpve': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'U0': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'U': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'H': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'G': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'Cv': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'zpve_thermo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'U0_thermo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'U_thermo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'H_thermo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'G_thermo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'Cv_thermo': tfds.features.Tensor(shape=(), dtype=tf.float32),
            'tag': tfds.features.Tensor(shape=(), dtype=tf.string),
        }),
        # These are returned if `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(
      self, dl_manager: tfds.download.DownloadManager) -> Dict[str, Any]:
    """Returns SplitGenerators. See superclass method for details."""
    atomref = pd.read_table(
        dl_manager.download(_ATOMREF_URL),
        skiprows=5,
        index_col='Z',
        skipfooter=1,
        sep=r'\s+',
        names=['Z', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']).to_dict()

    uncharacterized = pd.read_table(
        dl_manager.download(_UNCHARACTERIZED_URL),
        skiprows=9,
        skipfooter=1,
        sep=r'\s+',
        usecols=[0],
        names=['index']).values[:, 0]

    molecules_dir = dl_manager.download_and_extract(_MOLECULES_URL)

    split_ids = _get_split_ids(uncharacterized)

    return {
        split: self._generate_examples(split_ids[split], atomref, molecules_dir)
        for split in ['train', 'validation', 'test']}

  def _generate_examples(
      self,
      split: np.ndarray,
      atomref: Dict[str, Any],
      molecules_dir: Any) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Dataset generator. See superclass method for details."""

    for i in split:
      entry = _process_molecule(
          atomref, molecules_dir / f'dsgdb9nsd_{i:06d}.xyz')
      yield int(i), entry
