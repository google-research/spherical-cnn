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

"""Tests for models."""

import functools

from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from spherical_cnn import sphere_utils
from spherical_cnn import spin_spherical_harmonics
from spherical_cnn.molecules import models
import tensorflow as tf


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=(4, 8), spins=(0, 1, 2))


class TailTest(tf.test.TestCase, parameterized.TestCase):

  def test_compute_atom_counts(self):
    spins = (0, 1)
    atom_types = (1, 2, 3)
    max_atoms = 4

    tail = models._SumMLPTail(spins=spins,
                              atom_types=atom_types,
                              max_atoms=max_atoms,
                              use_atom_type_embedding=False)

    charges = jnp.array([[1, 1, 1, 1],
                         [3, 1, 2, 1],
                         [1, 2, 0, 0]])

    # Number of atoms of each type, followed by total and missing atoms.
    expected_counts = jnp.array([[4, 0, 0, 4, 0],
                                 [2, 1, 1, 4, 0],
                                 [1, 1, 0, 2, 2]])
    atom_counts = tail._compute_atom_counts(charges)

    self.assertAllClose(expected_counts, atom_counts)


class AtomTypeEmbeddingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(atom_types=(2, 3), batch_size=2, num_atoms=4),
      dict(atom_types=(1, 3, 5), batch_size=3, num_atoms=8))
  def test_embedding_value(self, atom_types, batch_size, num_atoms):
    features = jnp.zeros([batch_size, num_atoms, 1])
    charges = jnp.reshape(
        jnp.repeat(jnp.array(atom_types).astype(jnp.float32),
                   batch_size * num_atoms // len(atom_types)),
        (batch_size, num_atoms))
    # Add some null atoms.
    charges = charges.at[:, -1].set(0.0)

    # To be injected in the module's params for testing.
    atom_type_embeddings = jnp.expand_dims(
        jnp.linspace(1.0, 2.0, len(atom_types)), -1)
    # Compute expected outputs given above atom_type_feature.
    expected_outputs = jnp.zeros_like(features)
    for i, atom_type in enumerate(atom_types):
      expected_outputs = (expected_outputs.at[charges == atom_type]
                          .set(atom_type_embeddings[i]))

    embedding_module = models._AtomTypeEmbedding(atom_types=atom_types)

    key = jax.random.PRNGKey(0)
    params = embedding_module.init(key, features, charges)
    outputs_zero = embedding_module.apply(params, features, charges)

    # Now change embedding values.
    params_nonzero = flax.core.FrozenDict(
        {'params': {'atom_type_embeddings': atom_type_embeddings}})
    outputs_nonzero = embedding_module.apply(params_nonzero, features, charges)

    with self.subTest('Zero embeddings'):
      # With the default zero init, outputs don't change.
      self.assertAllClose(outputs_zero, jnp.zeros_like(outputs_zero))
    with self.subTest('Nonzero embedding'):
      self.assertAllClose(outputs_nonzero, expected_outputs)


class AtomFeatureAggregationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters('mean',
                            'weighted_by_atom_type',
                            'weighted_displacement',
                            'weighted_by_distance_squared')
  def test_shape(self, atom_aggregation_mode):
    batch_size = 2
    num_atoms = 3
    num_channels = 4
    atom_types = tuple(range(5))
    shape = (batch_size, num_atoms, num_channels)
    inputs = jnp.linspace(0, 1, np.prod(shape)).reshape(shape)
    charges = (np.random.randint(max(atom_types), size=batch_size * num_atoms)
               .reshape([batch_size, num_atoms]))
    positions = (jnp.linspace(-1, 1, batch_size * num_atoms * 3)
                 .reshape([batch_size, num_atoms, 3]))

    model = models._AtomFeatureAggregation(
        atom_types, num_atoms, atom_aggregation_mode)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, inputs, charges, positions)
    outputs = model.apply(params, inputs, charges, positions)

    self.assertEqual(outputs.shape, (batch_size, num_channels))


class SpinSphericalRegressorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(num_atoms=2,
           send_input_transformer=True,
           tail_module='sum',
           residual=True,
           spectral_pooling=False),
      dict(num_atoms=3,
           send_input_transformer=False,
           tail_module='deepsets',
           use_atom_type_embedding=True,
           residual=False,
           spectral_pooling=True),
      dict(num_atoms=2,
           send_input_transformer=False,
           tail_module='deepsets',
           use_atom_type_embedding=False,
           residual=True,
           spectral_pooling=True),
      dict(num_atoms=3,
           send_input_transformer=True,
           tail_module='transformer',
           use_atom_type_embedding=True,
           residual=False,
           spectral_pooling=False),
      dict(num_atoms=4,
           send_input_transformer=True,
           tail_module='transformer',
           use_atom_type_embedding=False,
           residual=True,
           spectral_pooling=False),
  )
  def test_shape(self,
                 num_atoms,
                 send_input_transformer,
                 tail_module,
                 residual,
                 spectral_pooling,
                 use_atom_type_embedding=True):
    transformer = _get_transformer() if send_input_transformer else None
    # Residual block is only used after the 2nd layer.
    resolutions = (8, 8, 4)
    spins = ((0,), (0, 1), (0, 1))
    channels = (1, 2, 2)
    batch_size = 2
    atom_types = (1.0, 2.0)
    metadata = nn.FrozenDict(atom_types=atom_types,
                             max_atoms=3,
                             max_pairwise_distance=1.0)
    model = models.SpinSphericalRegressor(
        resolutions=resolutions,
        spins=spins,
        widths=channels,
        residual=residual,
        spectral_pooling=spectral_pooling,
        num_filter_params=None,
        tail_module=tail_module,
        use_atom_type_embedding=use_atom_type_embedding,
        use_distance_to_center=False,
        atom_aggregation_mode='weighted_by_atom_type',
        num_transformer_layers=2,
        num_transformer_heads=1,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        metadata=metadata,
        axis_name=None,
        input_transformer=transformer)
    resolution = resolutions[0]
    shape = [batch_size, num_atoms,
             resolution, resolution,
             len(spins[0]), channels[0]]
    inputs = jnp.ones(shape)
    charges = jnp.reshape(
        jnp.repeat(jnp.array(atom_types),
                   batch_size * num_atoms // len(atom_types)),
        (batch_size, num_atoms))

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, inputs, charges,
                        train=False, positions=None)

    with self.subTest('Eval'):
      outputs = model.apply(params, inputs, charges,
                            train=False, positions=None)
      self.assertEqual(outputs.shape, (batch_size,))
    with self.subTest('Train'):
      outputs, _ = model.apply(params, inputs, charges,
                               train=True, positions=None,
                               mutable=['batch_stats'],
                               rngs={'dropout': rng})
      self.assertEqual(outputs.shape, (batch_size,))


class SpinSphericalRegressorFromPositionsTest(tf.test.TestCase,
                                              parameterized.TestCase):

  @parameterized.parameters(dict(distance=1.5,
                                 cutoff_angle=np.pi/6,
                                 min_pairwise_distance=0.5),
                            dict(distance=2.5,
                                 cutoff_angle=np.pi/4,
                                 min_pairwise_distance=1.0),
                            )
  def test_molecule_to_sphere_gaussian_simple_case(
      self, distance, cutoff_angle, min_pairwise_distance):
    """Check a simple case with two atoms."""
    # We put one atom at (0, 0, 0) and one at (0, d, 0), then check the values
    # inside and outside the cutoff angle.
    atom_types = [1, 2, 3, 4, 5]
    resolution = 16
    distance_power = 2.0
    charge_1, charge_2 = 1.0, 2.0
    # Three atoms in this molecule.
    charges = np.array([charge_1, charge_2, 0.0])
    coordinates = np.array([[0.0, 0.0, 0.0],
                            [0.0, distance, 0.0],
                            [0.0, 0.0, 0.0]])

    sphere = models._molecule_to_sphere_gaussian(
        coordinates=coordinates,
        charges=charges,
        random_seed=np.array(0.),
        resolution=resolution,
        min_pairwise_distance=min_pairwise_distance,
        cutoff_angle=cutoff_angle,
        max_atoms=3,
        atom_types=atom_types,
        distance_powers=(distance_power,),
        rotation_augmentation=False)
    # Remove spin dimension.
    sphere = sphere[..., 0, :]

    longitude, colatitude = sphere_utils.make_equiangular_grid(resolution)
    grid_coordinates = np.stack([np.sin(colatitude) * np.cos(longitude),
                                 np.sin(colatitude) * np.sin(longitude),
                                 np.cos(colatitude)],
                                axis=-1)

    # Compute angles between spherical unit vectors and the relative atom
    # position.
    angles = np.einsum('ijk,k->ij',
                       grid_coordinates, np.array([0, 1, 0]))

    interaction = charge_1 * charge_2  / distance ** distance_power
    normalization_factor = (max(atom_types)**2 /
                            min_pairwise_distance ** distance_power)
    interaction /= normalization_factor

    with self.subTest('Check type.'):
      self.assertEqual(sphere.dtype, np.float32)

    # Interactions between first atom and second atom type is below 5% of the
    # original value outside the cutoff_angle.
    with self.subTest('Outside cutoff.'):
      self.assertAllLess(sphere[0, angles < np.cos(cutoff_angle), 1],
                         0.0501 * interaction)

    # And above 5% within the angle.
    with self.subTest('Inside cutoff.'):
      self.assertAllGreater(sphere[0, angles > np.cos(cutoff_angle), 1],
                            0.0499 * interaction)

    # Interactions with self and missing atom types are zero.
    with self.subTest('Zero interactions.'):
      self.assertAllClose(sphere[0, ..., [0, 2, 3, 4]],
                          np.zeros_like(sphere[0, ..., [0, 2, 3, 4]]))

  def test_molecule_to_sphere_gaussian_is_translation_invariant(self):
    """Check that spherical representation is translation invariant."""
    atom_types = [1, 2, 3, 4, 5]
    distance_powers = (1.0, 2.0)
    # Three atoms in this molecule.
    charges = np.array([1.0, 3.0, 4.0])
    # Atoms at (0, 0, 0) and (0, 0, 3).
    coordinates = np.array([[1.0, 2.0, 3.0],
                            [0.0, 2.0, -1.0],
                            [-3.0, 2.0, 0.0]])
    options = dict(charges=charges,
                   random_seed=np.array(0.),
                   resolution=4,
                   min_pairwise_distance=1,
                   cutoff_angle=np.pi/4,
                   max_atoms=4,
                   atom_types=atom_types,
                   distance_powers=distance_powers,
                   rotation_augmentation=False)
    sphere = models._molecule_to_sphere_gaussian(
        coordinates=coordinates, **options)

    coordinates_translated = coordinates + np.array([[1., 2., 3.]])
    sphere_translated = models._molecule_to_sphere_gaussian(
        coordinates=coordinates_translated, **options)

    self.assertAllClose(sphere, sphere_translated)

  @parameterized.parameters(
      dict(rotation_augmentation=True,
           use_distance_to_center=False,
           atom_aggregation_mode='weighted_displacement'),
      dict(rotation_augmentation=False,
           use_distance_to_center=True,
           atom_aggregation_mode='weighted_by_distance_squared'),
      dict(rotation_augmentation=True,
           use_distance_to_center=True,
           atom_aggregation_mode='weighted_by_atom_type'),
  )
  def test_shape(self,
                 rotation_augmentation,
                 use_distance_to_center,
                 atom_aggregation_mode,
                 ):
    transformer = _get_transformer()
    resolutions = (8, 4)
    spins = ((0,), (0, 1))
    channels = (1, 2)
    batch_size = 2
    num_atoms = 3
    atom_types = (1.0, 2.0)
    metadata = nn.FrozenDict(atom_types=atom_types,
                             max_atoms=3,
                             min_pairwise_distance=1.0,
                             max_pairwise_distance=2.0)
    options = dict(
        resolutions=resolutions,
        spins=spins,
        widths=channels,
        residual=False,
        spectral_pooling=False,
        num_filter_params=None,
        tail_module='transformer',
        use_atom_type_embedding=False,
        use_distance_to_center=use_distance_to_center,
        atom_aggregation_mode=atom_aggregation_mode,
        num_transformer_layers=2,
        num_transformer_heads=1,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        metadata=metadata,
        axis_name=None,
        input_transformer=transformer)
    extra_options = dict(sphere_resolution=8,
                         cutoff_angle=1.0,
                         distance_powers=(2.0, 6.0),
                         rotation_augmentation=rotation_augmentation)
    model = models.SpinSphericalRegressorFromPositions(**options,
                                                       **extra_options)

    shape = [batch_size, num_atoms, 3]
    inputs = jnp.ones(shape)
    charges = jnp.reshape(
        jnp.repeat(jnp.array(atom_types),
                   batch_size * num_atoms // len(atom_types)),
        (batch_size, num_atoms))

    rngs = dict(zip(['params', 'dropout', 'augmentation'],
                    jax.random.split(jax.random.PRNGKey(0), 3)))
    params = model.init(rngs, inputs, charges, train=False)

    with self.subTest('Eval'):
      outputs = model.apply(params, inputs, charges, train=False)
      self.assertEqual(outputs.shape, (batch_size,))
    with self.subTest('Train'):
      outputs, _ = model.apply(params, inputs, charges, train=True,
                               mutable=['batch_stats'],
                               rngs=rngs)
      self.assertEqual(outputs.shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()
