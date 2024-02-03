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

# pylint: disable=line-too-long
r"""Build ERA5 dataset variants.

The current available datasets are set up following the description
in Keisler, "Forecasting Global Weather with Graph Neural Networks",
arXiv'22, with only changes to the input resolution.

Currently we only provide the smallest resolution (32 x 32). Each year
takes around 4min to process in a single machine and 400Mb of disk
space; there are 42 years. The code should work for higher resolutions
(and the source data is available is the same GCS directory) but it
might be unfeasible to run in a single machine.

For local dataset generation, run the following from the directory
containing pyproject.toml:

```
conda create --prefix env python=3.10 -y
conda activate env
pip install pytest
pip install -e .
cd spherical_cnn/weather/data/era5
tfds build era5 --file_format array_record
```
"""

import dataclasses
import textwrap
from typing import Tuple

import immutabledict
import numpy as np
import tensorflow_datasets as tfds
import xarray

ImmutableDict = immutabledict.immutabledict

CHUNK_SIZE = 128

# Initial and final timestamps present in the data source.
TIME_INITIAL = np.datetime64('1959-01-01T00:00:00.000000000')
TIME_FINAL = np.datetime64('2021-12-31T23:00:00.000000000')
ALL_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def _subsample_longitude_index(longitude):
  if max(longitude) <= 2*np.pi:
    raise ValueError('Expected longitude in degrees!')
  return np.linspace(0, 360, len(longitude)//2 + 1)[:-1]


def _subsample_longitudes(data):
  new_longitude = _subsample_longitude_index(data.indexes['longitude'])
  assert len(new_longitude) == len(data.indexes['latitude'])
  # Use cubic interpolation to avoid just taking every other value.
  return data.interp(longitude=new_longitude, method='cubic')


@dataclasses.dataclass
class Era5Config(tfds.core.BuilderConfig):
  """Configuration for datasets constructed from ERA5."""
  name: str
  description: str = ''
  data_source: str = ''
  resolution: Tuple[int, int] = (0, 0)
  measurements: ImmutableDict[str,
                              Tuple[int, ...]] = immutabledict.immutabledict()
  years: Tuple[int, ...] = ()
  resample: str = ''
  # Sum precipitation values over `accumulate_precipitation` hours.
  accumulate_precipitation: int = 0
  # Compute average over `average_over_hours` for all quantities.
  average_over_hours: int = 0
  # Source data dimension is typically (2x, x) but (x, x) might be better for
  # spherical harmonics computation. Enable this for subsampling.
  subsample_longitudes: bool = False
  include_metadata: bool = True


KEISLER22_BASE_CONFIG = dict(
    description=textwrap.dedent("""\
    Dataset following Keisler, Forecasting Global Weather with Graph
    Neural Networks, 2022."""),
    measurements=immutabledict.immutabledict(
        geopotential=ALL_LEVELS,
        specific_humidity=ALL_LEVELS,
        temperature=ALL_LEVELS,
        u_component_of_wind=ALL_LEVELS,
        v_component_of_wind=ALL_LEVELS,
        vertical_velocity=ALL_LEVELS,
        toa_incident_solar_radiation=(),
    ),
    years=tuple(range(1979, 2021)),
)


class Era5GCS(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ERA5 dataset, loading from GCS."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Added `toa_incident_solar_radiation` to Keisler datasets.',
  }

  BUILDER_CONFIGS = [
      Era5Config(
          name='keisler22_32x32',
          data_source=('gs://weatherbench2/datasets/era5/'
                       '1959-2022-6h-64x32_'
                       'equiangular_with_poles_conservative.zarr'),
          resolution=(32, 32),
          subsample_longitudes=True,
          resample='',
          **KEISLER22_BASE_CONFIG,
      ),
  ]

  def _get_metadata(self) -> tfds.core.MetadataDict:
    data = xarray.open_zarr(self.builder_config.data_source)
    # Metadata contains the indexes and variables that are not indexed by time.
    metadata = {}
    for v in data:
      if 'time' not in data[v].indexes:
        if self.builder_config.subsample_longitudes:
          subsampled = _subsample_longitudes(data[v])
          metadata[v] = subsampled.values.tolist()
        else:
          metadata[v] = data[v].values.tolist()
    for v in data.indexes:
      metadata[v] = data[v].values.tolist()
    # Subsample longitudes in index.
    if self.builder_config.subsample_longitudes:
      metadata['longitude'] = _subsample_longitude_index(
          metadata['longitude']).tolist()
    return tfds.core.MetadataDict(metadata)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    features = {}
    for measurement, levels in self.builder_config.measurements.items():
      shape = ((max(len(levels), 1),) +
               self.builder_config.resolution)
      features[measurement] = tfds.features.Tensor(shape=shape,
                                                   dtype=np.float32)

    metadata = (self._get_metadata() if self.builder_config.include_metadata
                else None)

    return self.dataset_info_from_configs(
        disable_shuffling=True,
        features=tfds.features.FeaturesDict({
            'time': tfds.features.Scalar(dtype=np.int64),
            **features,
        }),
        supervised_keys=None,
        homepage='https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5',
        metadata=metadata,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {str(year): self._generate_examples(str(year), dl_manager)
            for year in self.builder_config.years}

  def _generate_examples(self,
                         year: str, dl_manager: tfds.download.DownloadManager):
    """Yields examples."""
    data = xarray.open_zarr(self.builder_config.data_source,
                            chunks={'time': CHUNK_SIZE})
    data = data.sel(time=year)

    if self.builder_config.subsample_longitudes:
      data = _subsample_longitudes(data)

    if self.builder_config.average_over_hours > 0:
      data = data.rolling(
          time=self.builder_config.average_over_hours,
          min_periods=1).mean()

    if self.builder_config.accumulate_precipitation > 0:
      precipitation = data['total_precipitation'].rolling(
          time=self.builder_config.accumulate_precipitation,
          min_periods=1).sum()
      data.update({'total_precipitation': precipitation})

    if self.builder_config.resample:
      data = data.resample(time=self.builder_config.resample).nearest()
    times = data.time.values

    # Read source in batches for efficiency.
    batches = np.array_split(times, 1 + len(times) // CHUNK_SIZE)

    for batch in batches:
      data_i = data.sel(time=batch)
      measurements = {}
      for k, l in self.builder_config.measurements.items():
        if l:
          values = data_i[k].sel(level=list(l)).values.astype(np.float32)
        else:
          values = data_i[k].values
        measurements[k] = values.astype(np.float32)

      for i, time in enumerate(batch):
        if not TIME_INITIAL <= time <= TIME_FINAL:
          raise ValueError('`time` is expected to be between '
                           f'{TIME_INITIAL} and {TIME_FINAL}!')

        # Convert time delta from ns to hours.
        key = int((time - TIME_INITIAL) // int(1e9) // 3600)
        example = {'time': key}
        for measurement, values in measurements.items():
          # Expected dimensions are (vertical, long, lat).
          value = (values[i] if values[i].ndim == 3
                   else np.expand_dims(values[i], 0))
          example[measurement] = value

        yield key, example
