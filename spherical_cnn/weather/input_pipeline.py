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

"""Input pipeline."""

from typing import Optional, Sequence, TypeVar
import grain.python as grain
import numpy as np


def _get_constants(metadata, with_longitude=False):
  """Return the constant fields.

  Args:
    metadata: dataset metadata.
    with_longitude: whether to include the longitude in the output.

  Returns:
    Constants in (lon, lat, channels) order. Channels include land-sea
    mask, orography and sin(latitude). If `with_longitude` is true we
    also include sin(longitude) and cos(longitude).
  """
  orography = np.array(metadata['geopotential_at_surface'],
                       dtype=np.float32)
  orography /= orography.max()
  land_sea_mask = np.array(metadata['land_sea_mask'],
                           dtype=np.float32)
  # Lat is from -90 to 90 deg.
  latitude = np.sin(np.radians(metadata['latitude'],
                               dtype=np.float32))
  latitude = np.broadcast_to(latitude, land_sea_mask.shape)

  constants = [land_sea_mask, orography, latitude]
  if with_longitude:
    longitude = np.radians(metadata['longitude'],
                           dtype=np.float32)
    longitude = np.broadcast_to(longitude[:, None], land_sea_mask.shape)
    constants += [np.sin(longitude), np.cos(longitude)]

  return np.stack(constants, axis=-1)


def _encode_time(time, shape):
  """Encode hours since t0 to features."""
  hours_per_year = int(365.25 * 24)
  hour_of_year = time % hours_per_year
  hour_of_day = time % 24
  encoding = np.stack(
      [
          np.sin(2 * np.pi * hour_of_year / hours_per_year),
          np.cos(2 * np.pi * hour_of_year / hours_per_year),
          np.sin(2 * np.pi * hour_of_day / 24),
          np.cos(2 * np.pi * hour_of_day / 24),
      ],
      axis=-1,
  ).astype(np.float32)
  return np.broadcast_to(encoding[None, None, :], shape + (4,))


def _get_keisler22_idx_to_report(unroll_steps):
  """Return unrolled indices of channels to report."""
  variables = (
      'geopotential@500',
      'specific_humidity@700',
      'temperature@850',
      'u_component_of_wind@850',
      'v_component_of_wind@850',
  )
  num_variables = 78  # 6 variables at 13 vertical levels.
  idx = (7, 22, 36, 49, 62)

  unrolled_variables = []
  unrolled_idx = []
  for v, i in zip(variables, idx):
    unrolled_idx += [j * num_variables + i for j in range(unroll_steps)]
    unrolled_variables += [f'{v}_step{j+1}' for j in range(unroll_steps)]

  return variables, idx, unrolled_variables, unrolled_idx


T = TypeVar('T')


class _ShuffleWeatherSampler(grain.experimental.lazy_dataset.LazyMapDataset[T]):
  """Shuffles sets of consecutive observations.

  Attributes:
    offsets: The spacing between sequence elements. For example, (0,
      2, 4) will return sequences of 3 elements spaced by 2. If the
      dataset is ordered and sampled every 3h, for example, this
      corresponds to (t0, t0+6h, t0+12h).
    worker_count: How many processes are being used. Must match the
      grain.DataLoader `worker_count`.

  See grain.IndexSampler for other args.
  """

  def __init__(
      self,
      parent: grain.experimental.lazy_dataset.LazyMapDataset[T],
      offsets: Sequence[int],
      worker_count: int,
      shuffle: bool,
      *,
      reshuffle_each_epoch: bool = True,
      seed: int,
  ):
    super().__init__(parent)
    self._offsets = offsets
    self._seed = seed
    self._reshuffle_each_epoch = reshuffle_each_epoch
    self._shuffle = shuffle
    self._worker_count = 1 if worker_count == 0 else worker_count

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      raise NotImplementedError()

    sequence_len = len(self._offsets)
    num_records = len(self._parent)
    num_records_to_visit = num_records - (self._offsets[-1] - self._offsets[0])

    # The dataloader will assign index i to worker i % worker_count. We want
    # each worker to see the sequence according to the given offsets (for
    # example (t0, t+6h)). So for workers (w1, w2, w3) and a sequence (s1, s2),
    # we want the order to be (w1s1, w2s1, w3s1, w1s2, w2s2, w3s2, ...). In
    # terms of ids, for offsets=(0, 2), that would be (0, 1, 2, 2, 3, 4, 3, 4,
    # 5, 5, 6, 7, ...).

    # This is the sequence counter per worker.
    sequence_id = index // (self._worker_count * sequence_len)
    worker = index % self._worker_count
    offset = (index // self._worker_count) % sequence_len

    # This index is unique for each pair of (worker, sequence_id).
    index = (sequence_id * self._worker_count + worker) % num_records_to_visit

    epoch = index // num_records

    lead_time = self._offsets[offset] - self._offsets[0]

    if self._reshuffle_each_epoch:
      # index_shuffle expects 32-bit integers
      seed = (self._seed + epoch) % 2**32
    else:
      seed = self._seed

    if self._shuffle:
      index = grain.experimental.index_shuffle(
          index,
          max_index=num_records_to_visit - 1,
          seed=seed,
          rounds=4,
      )
    index += lead_time

    # `index` is numpy.int64, this fails if we don't convert to int.
    # This is given to `ShardLazyDataset`, which maps to in-shard range.
    return self._parent[int(index)]


class WeatherSampler(grain.IndexSampler):
  """Sampler that returns random sequences.

  This is mostly the same as grain._src.python.samplers.IndexSampler
  but modified to use our `ShuffleWeatherDataset`. It takes additional
  inputs `offsets` and `worker_count`, see `_ShuffleWeatherSampler` for
  descriptions.
  """

  def __init__(
      self,
      num_records: int,
      shard_options: grain.ShardOptions,
      offsets: Sequence[int],
      worker_count: int,
      num_epochs: int,
      shuffle: bool = False,
      seed: Optional[int] = None,
  ):
    super().__init__(
        num_records,
        shard_options,
        shuffle=False,
        num_epochs=num_epochs,
        seed=seed,
    )
    self._offsets = offsets
    self._num_epochs = len(offsets) * num_epochs
    self._max_index *= len(offsets)
    # Initial `_record_keys` is a ShardLazyDataset; we shuffle it.
    self._record_keys = _ShuffleWeatherSampler(
        self._record_keys,
        self._offsets,
        shuffle=shuffle,
        worker_count=worker_count,
        seed=seed,
    )
