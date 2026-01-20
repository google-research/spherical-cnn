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

"""Input pipeline tests."""

from absl.testing import parameterized
import grain.python as grain
import ml_collections
import numpy as np
from spherical_cnn.weather import input_pipeline
import tensorflow as tf
import tensorflow_datasets as tfds


class InputPipelineTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      batch_size=[1, 2],
      lead_time=[6, 24],
  )
  def test_create_dataset_keisler22(self, lead_time, batch_size):
    """Test that batch contents and shape are as expected."""

    config = ml_collections.ConfigDict()
    config.dataset = 'keisler22_32x32'
    config.per_device_batch_size = batch_size
    config.num_epochs = 1
    config.lead_time = lead_time
    config.worker_count = 0
    config.prefetch_buffer_size = 1
    # Number of threads per worker
    config.num_threads = 1

    # 78 (6 fields * 13 verticals) + 10 constants.
    predictors_shape = (batch_size, 32, 32, 88)
    targets_shape = (batch_size, 32, 32, 78 * lead_time // 6)

    import os
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'testdata')

    with tfds.testing.mock_data(num_examples=100,
                                data_dir=data_dir):
      train_ds, eval_ds, test_ds, _, _ = (
          input_pipeline.create_dataset_keisler22(config, seed=0))

    for ds in [train_ds, eval_ds, test_ds]:
      batch = next(iter(ds))
      self.assertEqual(batch['predictors'].shape, predictors_shape)
      self.assertEqual(batch['targets'].shape, targets_shape)


class WeatherSamplerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      offsets=[(0, 1, 2, 3), (0, 2, 4)],
      num_records=[29, 30, 41],
      shard_count=[1, 2, 3],
      worker_count=[1, 2, 3],
      num_epochs=[1, 2, 3],
  )
  def test_offsets_and_totals(
      self, offsets, num_records, shard_count, worker_count, num_epochs
  ):
    """Check that samples have correct offsets and cover the whole dataset."""
    all_ids = []
    for shard in range(shard_count):
      shard_options = grain.ShardOptions(
          shard_index=shard, shard_count=shard_count, drop_remainder=True
      )

      sampler = input_pipeline.WeatherSampler(
          num_records=num_records,
          num_epochs=num_epochs,
          offsets=offsets,
          worker_count=worker_count,
          shard_options=shard_options,
          shuffle=True,
          seed=0,
      )

      ids = np.array([s.record_key for s in sampler])

      # This follows the way grain.DataLoader splits the indices along shards
      # and workers.
      num_global_workers = shard_count * worker_count
      local_offset = shard - num_global_workers
      last_seen_indices = {
          i: local_offset + i * shard_count for i in range(worker_count)
      }

      for worker in range(worker_count):
        first_id = last_seen_indices[worker] + num_global_workers
        ids_worker = ids[first_id::num_global_workers]
        # Take only complete sequences.
        ids_worker = ids_worker[
            : len(ids_worker) // len(offsets) * len(offsets)
        ]
        sequences = np.split(ids_worker, len(ids_worker) // len(offsets))
        # Check that every sequence has the right offsets.
        for s in sequences:
          self.assertAllEqual(np.diff(s), np.diff(offsets))

        # Store all t0 ids to ensure coverage.
        all_ids += [s[offsets.index(0)] for s in sequences]

    entries_per_shard = num_records // shard_count
    # Last entries cannot be used because of the sequence length.
    valid_entries_per_shard = entries_per_shard - (offsets[-1] - offsets[0])
    # Some entries might be missed if whole seq cannot be assigned to worker.
    min_valid_entries = (
        valid_entries_per_shard // worker_count * worker_count * shard_count
    )

    self.assertGreaterEqual(len(np.unique(all_ids)), min_valid_entries)

  @parameterized.product(
      offsets=[(0, 1)],
      num_records=[43],
      shard_count=[4],
      worker_count=[1, 3, 5],
      num_epochs=[2],
  )
  def test_sampler_with_data_loader(
      self, offsets, num_records, shard_count, worker_count, num_epochs
  ):
    """Check that order is correct with WeatherSampler in grain.DataLoader."""
    all_ids = []
    range_data_source = grain.RangeDataSource(
        start=0, stop=num_records - 1, step=1
    )
    for shard in range(shard_count):
      shard_options = grain.ShardOptions(
          shard_index=shard, shard_count=shard_count, drop_remainder=True
      )

      sampler = input_pipeline.WeatherSampler(
          num_records=num_records,
          num_epochs=num_epochs,
          offsets=offsets,
          worker_count=worker_count,
          shard_options=shard_options,
          shuffle=True,
          seed=0,
      )

      read_options = grain.ReadOptions(prefetch_buffer_size=2, num_threads=2)
      data_loader = grain.DataLoader(
          data_source=range_data_source,
          sampler=sampler,
          worker_count=worker_count,
          read_options=read_options,
      )

      ids = np.array([example for example in iter(data_loader)])

      for worker in range(worker_count):
        ids_worker = ids[worker::worker_count]
        # Take only complete sequences.
        ids_worker = ids_worker[
            : len(ids_worker) // len(offsets) * len(offsets)
        ]
        sequences = np.split(ids_worker, len(ids_worker) // len(offsets))
        # Check that every sequence has the right offsets.
        for s in sequences:
          self.assertAllEqual(np.diff(s), np.diff(offsets))

        # Store all t0 ids to ensure coverage.
        all_ids += [s[offsets.index(0)] for s in sequences]

    entries_per_shard = num_records // shard_count
    # Last entries cannot be used because of the sequence length.
    valid_entries_per_shard = entries_per_shard - (offsets[-1] - offsets[0])
    # Some entries might be missed if whole seq cannot be assigned to worker.
    min_valid_entries = (
        valid_entries_per_shard // worker_count * worker_count * shard_count
    )

    self.assertGreaterEqual(len(np.unique(all_ids)), min_valid_entries)


if __name__ == '__main__':
  tf.test.main()
