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

from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, TypeVar
import grain.python as grain
import jax
import ml_collections
import numpy as np
from spherical_cnn.weather import input_pipeline_stats
# This is needed to be able load the custom tfds dataset.
import spherical_cnn.weather.data.era5  # pylint: disable=unused-import
import tensorflow_datasets as tfds


Dataset = grain.DataLoader


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


class WeatherBatchOperation(grain.BatchOperation):
  """Custom batch operation for ERA5-based datasets.

  We avoid using grain.Batch for efficiency. The dataset contains single
  timestamp examples but we want sequences in a specific order
  predictors/targets, as well as some pre-processing and including
  constants. Doing all of this as a single operation is more efficient.
  """

  batch_size: int
  metadata: Any

  def __init__(self, batch_size, metadata):
    super().__init__(batch_size)
    self.metadata = metadata
    self.num_times = len(metadata['offsets'])
    self.num_predictor_times = metadata['offsets'].index(0) + 1
    if self.num_predictor_times != 1:
      raise NotImplementedError(
          'Need to ensure correct ordering for unrolled predictions '
          'with more than one time sample as input.'
      )
    self.num_target_times = self.num_times - self.num_predictor_times
    self.target_fields = [
        'geopotential',
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'vertical_velocity',
    ]
    # Solar radiation, orography, land_sea_mask, lat (1), long (2), and time
    # features (4).
    self.num_constant_channels = 10

  def _init_batch(self, num_lat, num_lon, num_verticals):
    num_predictor_channels = (
        self.num_predictor_times * len(self.target_fields) * num_verticals
        + self.num_constant_channels
    )
    num_target_channels = (
        self.num_target_times * len(self.target_fields) * num_verticals
    )

    predictors = grain.SharedMemoryArray(
        (self.batch_size, num_lat, num_lon, num_predictor_channels),
        dtype=np.float32,
    )
    targets = grain.SharedMemoryArray(
        (self.batch_size, num_lat, num_lon, num_target_channels),
        dtype=np.float32,
    )
    times = grain.SharedMemoryArray((self.batch_size, self.num_times),
                                    dtype=np.int64)

    return predictors, targets, times

  def __call__(
      self, input_iterator: Iterator[grain.Record]
  ) -> Iterator[grain.Record]:
    time_i = 0
    batch_i = 0
    last_record_metadata = None
    predictors, targets, times = [], [], []
    id0, id1 = 0, 0
    for input_record in input_iterator:
      num_verticals, num_lon, num_lat = input_record.data[
          self.target_fields[0]
      ].shape
      if time_i == 0 and batch_i == 0:
        predictors, targets, times = self._init_batch(
            num_lat, num_lon, num_verticals
        )

      times[batch_i, time_i] = input_record.data['time']

      if time_i < self.num_predictor_times:
        for i, field in enumerate(self.target_fields):
          id0 = (
              num_verticals * len(self.target_fields) * time_i
              + num_verticals * i
          )
          id1 = id0 + num_verticals
          predictors[batch_i, ..., id0:id1] = input_record.data[
              field
          ].transpose(2, 1, 0)
        # Add constants last:
        if time_i == self.num_predictor_times - 1:
          id0 = id1
          id1 = id0 + 1
          # TODO(machc): This assumes only one time sample for predictors. If we
          # want more than one we should take all samples of solar radiation and
          # add at the end to play well with unrolling.
          predictors[batch_i, ..., id0:id1] = input_record.data[
              'toa_incident_solar_radiation'
          ].transpose(2, 1, 0)
          constants = _get_constants(self.metadata, with_longitude=True)
          time_features = _encode_time(
              input_record.data['time'], (num_lat, num_lon)
          )
          id0 = id1
          id1 = id0 + constants.shape[-1]
          predictors[batch_i, ..., id0:id1] = constants.transpose(1, 0, 2)
          id0 = id1
          id1 = id0 + time_features.shape[-1]
          predictors[batch_i, ..., id0:id1] = time_features
      else:
        for i, field in enumerate(self.target_fields):
          id0 = (
              num_verticals
              * len(self.target_fields)
              * (time_i - self.num_predictor_times)
              + num_verticals * i
          )
          id1 = id0 + num_verticals
          targets[batch_i, ..., id0:id1] = input_record.data[field].transpose(
              2, 1, 0
          )

      time_i += 1
      if time_i == self.num_times:
        batch_i += 1
        time_i = 0

      last_record_metadata = input_record.metadata
      if batch_i == self.batch_size:
        batch_i = 0
        time_i = 0
        if self._use_shared_memory:
          batch = {
              'predictors': predictors.metadata,
              'targets': targets.metadata,
              'times': times.metadata,
          }
        else:
          batch = {'predictors': predictors, 'targets': targets, 'times': times}

        yield grain.Record(last_record_metadata.remove_record_key(), batch)


def create_dataset_common(
    dataset_name: str,
    config: ml_collections.ConfigDict,
    seed: int,
    *,
    offsets: Sequence[int],
    train_split: str,
    validation_split: str,
    test_split: str,
    make_operations: Callable[
        [int, Dict[str, Any]], Sequence[grain.MapTransform]
    ],
    stats: Dict[str, np.ndarray],
    spin1_idx: Optional[Dict[str, Sequence[int]]] = None,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any], Dict[str, Any]]:
  """Common loader for ERA5 variants.

  Args:
    dataset_name: Dataset name.
    config: ConfigDict.
    seed: Seed for random number generator.
    offsets: Something like (-12, -6, 0, 72) will provide inputs at
      t0-12h, t0-6h, t0 and targets at t0+72h when the data is sampled
      hourly.
    train_split: Train split to be passed to `tfds.data_source`.
    validation_split: Validation split to be passed to `tfds.data_source`.
    test_split: Test split to be passed to tfds.data_source.
    make_operations: Function to return list of post-processing operations to be
      run by grain.DataLoader.
    stats: Dataset statistics to be added to `metadata['stats']`.
    spin1_idx: Dict indicating which fields are to be treated as vector fields,
      to be added to `metadata['spin1_idx']`.

  Returns:
    train/val/test data loaders and train/test metadata.
  """

  process_batch_size = jax.local_device_count() * config.per_device_batch_size

  def make_dataset(split, num_epochs, shuffle):
    # This contains "info".
    source = tfds.data_source(
        dataset_name,
        split=split,
    )
    metadata = source.dataset_info.metadata
    metadata = metadata if metadata is not None else {}
    metadata['num_elements'] = len(source) - (offsets[-1] - offsets[0])
    metadata['offsets'] = offsets

    sampler = WeatherSampler(
        num_records=len(source),
        num_epochs=num_epochs,
        offsets=offsets,
        worker_count=config.worker_count,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=shuffle,
        seed=seed,
    )

    operations = make_operations(process_batch_size, metadata)

    read_options = grain.ReadOptions(
        prefetch_buffer_size=config.prefetch_buffer_size,
        num_threads=config.num_threads,
    )
    data_loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=config.worker_count,
        worker_buffer_size=1,
        enable_profiling=False,
        read_options=read_options,
    )

    return (data_loader, metadata)

  train_dataset, metadata_train = make_dataset(
      split=train_split, num_epochs=config.num_epochs, shuffle=True
  )

  metadata_train['stats'] = stats
  metadata_train['spin1_idx'] = spin1_idx

  # Validation data:
  eval_dataset, _ = make_dataset(
      split=validation_split, num_epochs=1, shuffle=False
  )
  # Test data:
  test_dataset, metadata_test = make_dataset(
      split=test_split, num_epochs=1, shuffle=False
  )

  return (
      train_dataset,
      eval_dataset,
      test_dataset,
      metadata_train,
      metadata_test,
  )


def create_dataset_keisler22(
    config: ml_collections.ConfigDict,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any], Dict[str, Any]]:
  """Create Keisler'22 data loader.

  This follows the protocol from Keisler, Forecasting Global Weather
  with Graph Neural Networks, arXiv'22.

  Args:
    config: ConfigDict.
    seed: Seed for random number generator.

  Returns:
    train/val/test data loaders and train/test metadata.
  """

  # Data has 3h intervals and predicts 6h ahead with variable rollout, while
  # lead_time is given in hours.
  offsets = tuple(np.arange(0, config.lead_time // 3 + 1, 2))
  unroll_steps = len(offsets) - 1

  all_splits = set(range(1979, 2021))

  # These are the Keisler settings
  validation_split = set([1991, 2004, 2017])
  test_split = set([2012, 2016, 2020])
  train_split = '+'.join(
      [str(year) for year in sorted(all_splits - validation_split - test_split)]
  )
  # Non-contiguous years might result in invalid sequences due to our indexing
  # assumptions, so we use a single year for validation and test one year at a
  # time.  Fig 6 in Keisler is over 2016. WB 2 evals are over 2020.
  validation_split = '2017'
  test_split = '2016'
  # test_split = '2020'

  # These are the GraphCast settings.
  # validation_split = set([2019])
  # test_split = set([2018, 2020])
  # train_split = '+'.join([str(year) for year in
  #                         sorted(all_splits - validation_split - test_split)])

  def make_operations(process_batch_size, metadata):
    return [WeatherBatchOperation(process_batch_size, metadata)]

  # Make targets mean/stds according to # of offsets.
  def unroll(x, num_unroll_steps):
    return np.array(list(x) * num_unroll_steps, dtype=np.float32)

  stats = input_pipeline_stats.KEISLER22_STATS
  for key in [
      'targets_mean',
      'targets_std',
      'differences_mean',
      'differences_std',
  ]:
    stats[key] = unroll(stats[key], unroll_steps)

  (train_dataset, eval_dataset, test_dataset, metadata, metadata_test) = (
      create_dataset_common(
          # Version 1.0.1 includes `toa_incident_solar_radiation`.
          dataset_name=f'era5_gcs/{config.dataset}:1.0.1',
          config=config,
          seed=seed,
          offsets=offsets,
          train_split=train_split,
          validation_split=validation_split,
          test_split=test_split,
          make_operations=make_operations,
          stats=stats,
          spin1_idx=input_pipeline_stats.KEISLER22_SPIN1_IDX,
      )
  )

  # Add evaluation targets to metadata.
  _, _, unrolled_variables, unrolled_idx = _get_keisler22_idx_to_report(
      unroll_steps
  )

  metadata['metrics_idx'] = unrolled_idx
  metadata['metrics_variables'] = unrolled_variables
  # This is used to prune bad inputs due to missing years in the training data.
  metadata['expected_time_deltas'] = 6

  return train_dataset, eval_dataset, test_dataset, metadata, metadata_test


T = TypeVar('T')


class _ShuffleWeatherSampler(grain.MapDataset[T]):
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
      parent: grain.MapDataset[T],
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
