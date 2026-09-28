# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for independent Keisler normalization statistics."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
import numpy as np
from spherical_cnn.weather import input_pipeline
from spherical_cnn.weather import input_pipeline_stats

_UNROLLED = (
    'targets_mean',
    'targets_std',
    'differences_mean',
    'differences_std',
)
_BASE_STATS = {
    key: value.copy()
    for key, value in input_pipeline_stats.KEISLER22_STATS.items()
}


class _WeatherSource:
  """Small in-memory source for the real sampler, batcher and data loader."""

  def __init__(self):
    self.dataset_info = types.SimpleNamespace(
        metadata={
            'geopotential_at_surface': np.ones((3, 2), dtype=np.float32),
            'land_sea_mask': np.ones((3, 2), dtype=np.float32),
            'latitude': [-45.0, 45.0],
            'longitude': [0.0, 120.0, 240.0],
        }
    )

  def __len__(self):
    return 24

  def __getitem__(self, index):
    return {
        field: np.full((13, 3, 2), index + i, dtype=np.float32)
        for i, field in enumerate(
            (
                'geopotential',
                'specific_humidity',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
                'vertical_velocity',
            )
        )
    } | {
        'time': np.int64(index * 3),
        'toa_incident_solar_radiation': np.ones((1, 3, 2), dtype=np.float32),
    }


class KeislerStatsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.original = {key: value.copy() for key, value in _BASE_STATS.items()}
    self.shared = {key: value.copy() for key, value in self.original.items()}
    patch = mock.patch.object(
        input_pipeline_stats, 'KEISLER22_STATS', self.shared
    )
    patch.start()
    self.addCleanup(patch.stop)

  def config(self, lead_time):
    return ml_collections.ConfigDict(
        {
            'dataset': 'keisler22_32x32',
            'lead_time': lead_time,
            'per_device_batch_size': 2,
            'num_epochs': 1,
            'worker_count': 0,
            'prefetch_buffer_size': 1,
            'num_threads': 1,
        }
    )

  def make_metadata(self, lead_time):
    def common(**kwargs):
      return None, None, None, {'stats': kwargs['stats']}, {}

    with mock.patch.object(
        input_pipeline, 'create_dataset_common', side_effect=common
    ):
      return input_pipeline.create_dataset_keisler22(
          self.config(lead_time), seed=7
      )[3]

  def assert_stats(self, stats, lead_time):
    for key, original in self.original.items():
      expected = (
          np.tile(original, lead_time // 6) if key in _UNROLLED else original
      )
      np.testing.assert_array_equal(stats[key], expected)
      self.assertEqual(stats[key].dtype, np.float32)

  @parameterized.parameters(6, 12, 24)
  def test_first_call_statistics_match_the_published_values(self, lead_time):
    self.assert_stats(self.make_metadata(lead_time)['stats'], lead_time)

  @parameterized.parameters(6, 12, 24)
  def test_creation_does_not_modify_module_statistics(self, lead_time):
    metadata = self.make_metadata(lead_time)
    self.assert_stats(metadata['stats'], lead_time)
    for key in self.shared:
      np.testing.assert_array_equal(self.shared[key], self.original[key])
    self.assertIsNot(metadata['stats'], self.shared)

  @parameterized.parameters((6, 24), (24, 6), (24, 24), (12, 24), (24, 12))
  def test_repeated_creation_does_not_unroll_previous_statistics(
      self, first, second
  ):
    earlier = self.make_metadata(first)
    saved = {key: value.copy() for key, value in earlier['stats'].items()}
    later = self.make_metadata(second)
    self.assert_stats(later['stats'], second)
    self.assert_stats(earlier['stats'], first)
    self.assertIsNot(earlier['stats'], later['stats'])
    for key, value in saved.items():
      np.testing.assert_array_equal(earlier['stats'][key], value)
    self.assertLen(later['metrics_idx'], 5 * second // 6)
    self.assertEqual(later['expected_time_deltas'], 6)

  def test_failed_creation_leaves_module_statistics_unchanged(self):
    error = RuntimeError('dataset unavailable')
    with mock.patch.object(
        input_pipeline, 'create_dataset_common', side_effect=error
    ):
      with self.assertRaises(RuntimeError) as raised:
        input_pipeline.create_dataset_keisler22(self.config(24), seed=7)
    self.assertIs(raised.exception, error)
    for key in self.shared:
      np.testing.assert_array_equal(self.shared[key], self.original[key])
    self.assert_stats(self.make_metadata(6)['stats'], 6)

  @parameterized.parameters((24, 6), (12, 24))
  def test_real_loaders_match_their_normalization_shape(self, first, second):
    outputs = []
    with mock.patch.object(
        input_pipeline.tfds,
        'data_source',
        side_effect=lambda *a, **k: _WeatherSource(),
    ):
      for lead_time in (first, second):
        outputs.append(
            input_pipeline.create_dataset_keisler22(
                self.config(lead_time), seed=7
            )
        )
      for lead_time, output in zip((first, second), outputs):
        stats = output[3]['stats']
        self.assert_stats(stats, lead_time)
        for dataset in output[:3]:
          batch = next(iter(dataset))
          self.assertEqual(
              batch['targets'].shape, (2, 2, 3, 78 * lead_time // 6)
          )
          normalized = (batch['targets'] - stats['targets_mean']) / stats[
              'targets_std'
          ]
          self.assertTrue(np.isfinite(normalized).all())
          self.assertEqual(batch['predictors'].shape, (2, 2, 3, 88))


if __name__ == '__main__':
  absltest.main()
