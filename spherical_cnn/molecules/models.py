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

"""Spin-weighted spherical CNN models for molecular property regression."""
import abc
import functools
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from einshape.src.jax import jax_ops as einshape
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from spherical_cnn import layers
from spherical_cnn import sphere_utils
from spherical_cnn import spin_spherical_harmonics

Array = Union[np.ndarray, jnp.ndarray]
Dtype = Any
PRNGKey = Any
Shape = Tuple[int, ...]


# We use atomic masses to find the molecule's center of mass, which is useful to
# predict some quantities. These values are in g/mol and match TorchMD's.
_ATOMIC_MASSES = tuple([
    0.0, 1.008, 4.002602, 6.94, 9.0121831,
    10.81, 12.011, 14.007, 15.999, 18.998403163,
    20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983,
    40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971,
    79.904, 83.798, 85.4678, 87.62, 88.90584,
    91.224, 92.90637, 95.95, 97.90721, 101.07,
    102.9055, 106.42, 107.8682, 112.414, 114.818,
    118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766,
    144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.054, 174.9668, 178.49, 180.94788, 183.84,
    186.207, 190.23, 192.217, 195.084, 196.966569,
    200.592, 204.38, 207.2, 208.9804, 208.98243,
    209.98715, 222.01758, 223.01974, 226.02541, 227.02775,
    232.0377, 231.03588, 238.02891, 237.04817, 244.06421,
    243.06138, 247.07035, 247.07031, 251.07959, 252.083,
    257.09511, 258.09843, 259.101, 262.11, 267.122,
    268.126, 271.134, 270.133, 269.1338, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19,
    289.194, 293.204, 293.208, 294.214,
])


def _spin_mean_real(features: Array, spins: Sequence[int]) -> jnp.ndarray:
  """Spherical mean with real outputs, with special handling of spin == 0.

  Global average pooling is not rotation-equivariant for spin != 0, so we take
  the absolute values before. For spin == 0, we can normally compute the average
  then cast to real.

  Args:
    features: (batch_size, resolution, resolution, num_spins, num_channels)
      complex64 feature array.
    spins: Sequence of spin weights present in `features` second-to-last
      dimension.

  Returns:
    A float32 (batch_size, num_spins, num_channels) feature array.
  """
  mean_abs = sphere_utils.spin_spherical_mean(jnp.abs(features))
  # Taking the `abs` here is a design decision. Using the real part or
  # concatenating real and imaginary parts are also reasonable for spin == 0.
  mean = jnp.abs(sphere_utils.spin_spherical_mean(features))
  spins = einshape.einshape('s->1s1', jnp.array(spins))
  return jnp.where(spins == 0, mean, mean_abs)


# Maps _spin_mean_real appropriately when we have an extra dimension for atoms.
_spin_mean_real_atoms = jax.vmap(_spin_mean_real, in_axes=(1, None), out_axes=1)


class _Tail(nn.Module, metaclass=abc.ABCMeta):
  """Abstract class for the tail module.

  The tail takes spherical features per atom and returns a scalar prediction.

  Attributes:
    spins: A (num_spins,) tuple of spin weights.
    atom_types: A tuple with the atomic numbers present in the dataset.
    max_atoms: Maximum number of atoms in the dataset.
    use_atom_type_embedding: When True, add a learnable per atom type embedding
      to atom features.
  """
  spins: Tuple[int, ...]
  atom_types: Tuple[int, ...]
  max_atoms: int
  use_atom_type_embedding: bool
  use_distance_to_center: Optional[bool] = False
  atom_aggregation_mode: str = 'weighted_by_atom_type'

  @abc.abstractmethod
  def __call__(self, spheres: Array, charges: Array, train: bool,
               positions: Optional[Array]) -> jnp.ndarray:
    """Apply module.

    Args:
      spheres: (batch_size, num_atoms, resolution, resolution, num_spins,
        num_channels) complex64 feature array.
      charges: (batch_size, num_atoms) float32 array containing the
        charge of each atom.
      train: whether to run in training or inference mode.
      positions: (batch_size, num_atoms, 3) with atom positions per molecule.

    Returns:
      A (batch_size, 1) float32 array.
    """
    raise NotImplementedError()

  def _compute_atom_counts(self, charges: Array) -> jnp.ndarray:
    """Computes number of atoms of each type, total and missing atoms.

    Args:
      charges: (batch_size, num_atoms) array of atom types per molecule.

    Returns:
       A (batch_size, num_atom_types + 2) float32 array with the number of atoms
         of each type, the total number of atoms, and the difference between the
         maximum number of atoms on the dataset and each molecule in the batch.
    """
    atom_counts = [jnp.count_nonzero(jnp.isclose(charges, atom_type), axis=-1)
                   for atom_type in self.atom_types]
    atom_counts = jnp.stack(atom_counts, axis=1).astype(jnp.float32)
    # Append total atoms and number of missing atoms.
    total = atom_counts.sum(axis=-1, keepdims=True)

    return jnp.concatenate([atom_counts, total, self.max_atoms - total], axis=1)


class _SumMLPTail(_Tail):
  """Simple tail that sums over atoms and applies an MLP for regression."""

  @nn.compact
  def __call__(self, spheres: Array, charges: Array, train: bool,
               positions: Optional[Array]) -> jnp.ndarray:
    del train
    # Input is (batch, atoms, lat, lon, spins, channels).
    # Sum over atoms.
    mask = charges > 0
    mask = einshape.einshape('ba->ba1111', mask)
    features = jnp.sum(spheres * mask, axis=1) / self.max_atoms
    # Average over spheres.
    features = _spin_mean_real(features, self.spins)
    # Shape is now (batch, spins, channel); we merge spins and channels.
    features = einshape.einshape('bsc->b(sc)', features)

    atom_counts = self._compute_atom_counts(charges)
    features = jnp.concatenate([features,
                                atom_counts / self.max_atoms], axis=-1)
    # Apply the same MLP as `_DeepSetsTail` after aggregation.
    features = nn.Dense(256, precision='highest')(features)
    features = nn.relu(features)
    features = nn.Dense(256, precision='highest')(features)
    features = nn.relu(features)
    output = nn.Dense(1, precision='highest')(features)

    return einshape.einshape('b1->b', output)


class _AtomTypeEmbedding(nn.Module):
  """Module that returns learnable atom type embeddings.

  Attributes:
    atom_types: A tuple with the atomic numbers present in the dataset.
  """
  atom_types: Tuple[int, ...]
  initializer: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.zeros

  @nn.compact
  def __call__(self, features: Array, charges: Array) -> jnp.ndarray:
    """Adds atom type embeddings.

    Args:
      features: (batch_size, num_atoms, num_channels) array of features.
      charges: (batch_size, num_atoms) array of atom types per molecule.
    Returns:
      Array of the same size of `features` with added type embeddings.
    """
    atom_type_embeddings = self.param(
        'atom_type_embeddings',
        self.initializer,
        (len(self.atom_types), features.shape[-1]),
        jnp.float32)

    embeddings_to_features = jnp.zeros_like(features)
    for i, atom_type in enumerate(self.atom_types):
      # Each iteration will fill the positions corresponding to one atom type in
      # embeddings_to_features.
      atom_type_positions = jnp.isclose(jnp.expand_dims(charges, -1), atom_type)
      atom_type_embedding = jnp.expand_dims(atom_type_embeddings[i], (0, 1))
      embeddings_to_features = (embeddings_to_features +
                                jnp.where(atom_type_positions,
                                          atom_type_embedding,
                                          0.0))

    return embeddings_to_features


def _displacement_to_center(positions, charges):
  """Computes the displacement vector to the center of mass for each atom."""
  masses = jnp.array(_ATOMIC_MASSES)[charges.round().astype(jnp.int32)]
  total_mass = masses.sum(axis=-1)
  center_of_mass = (jnp.sum(jnp.expand_dims(masses, 2) * positions,
                            axis=1, keepdims=True) /
                    jnp.expand_dims(total_mass, (1, 2)))
  return positions - center_of_mass


def _distance_to_center(positions, charges):
  """Computes the scalar distance to center of mass for each atom."""
  displacement = _displacement_to_center(positions, charges)
  return jnp.linalg.norm(displacement, axis=-1, keepdims=True)


class _AtomFeatureAggregation(nn.Module):
  """Aggregate per-atom features into a per-molecule feature.

  Attributes:
    atom_types: A tuple with the atomic numbers present in the dataset.
    max_atoms: Maximum number of atoms in the dataset.
    atom_aggregation_mode: Method for aggregating per-atom features.
  """
  atom_types: Tuple[int, ...]
  max_atoms: int
  atom_aggregation_mode: str

  @nn.compact
  def __call__(self, features, charges, positions):
    """Apply feature aggregation.

    Args:
      features: (batch_size, num_atoms, num_channels) feature array.
      charges: (batch_size, num_atoms) array containing the charge of each atom.
      positions: (batch_size, num_atoms, 3) with atom positions per molecule.

    Returns:
      A (batch_size, num_channels) feature representing the whole molecule.
    """
    # Ignore features from zero-padded atoms.
    mask = charges > 0
    mask = einshape.einshape('ba->ba1', mask)
    features = jnp.where(mask,
                         features,
                         jnp.zeros_like(features))

    if self.atom_aggregation_mode == 'mean':
      features = jnp.sum(features, axis=1) / self.max_atoms
    elif self.atom_aggregation_mode == 'weighted_by_atom_type':
      # Using per channels weights and biases for each atom type.
      weights = _AtomTypeEmbedding(self.atom_types,
                                   initializer=nn.initializers.ones)(
                                       features, charges)
      biases = _AtomTypeEmbedding(self.atom_types,
                                  initializer=nn.initializers.zeros)(
                                      features, charges)

      features = (jnp.sum(features * weights + biases, axis=1) /
                  self.max_atoms)
    elif self.atom_aggregation_mode == 'weighted_displacement':
      # This computes the norm of 3D positions with respect to the center of
      # mass, weighted by the input features.
      displacement_to_center = _displacement_to_center(positions, charges)
      features = (jnp.expand_dims(features, -1) *
                  jnp.expand_dims(displacement_to_center, -2))
      features = jnp.sum(features, axis=1) / self.max_atoms
      squared_norm = jnp.sum(features**2, axis=-1)
      # We want the avoid NaNs on the gradient of the norm here, in case of zero
      # norm.
      squared_norm = jnp.maximum(squared_norm, 1e-8)
      features = jnp.sqrt(squared_norm)
    elif self.atom_aggregation_mode == 'weighted_by_distance_squared':
      distance_to_center = _distance_to_center(positions, charges)
      features = features * distance_to_center**2
      features = jnp.sum(features, axis=1) / self.max_atoms
    else:
      raise ValueError('Unexpected `atom_aggregation_mode`.')

    return features


class _DeepSetsTail(_Tail):
  """DeepSets tail that does independent MLPs -> aggregation -> final MLP.

  We follow the simple architecture from Cohen et al, "Spherical CNNs", ICLR'18,
  with a simple per atom MLP followed by a sum over atoms before the last
  layer. One difference is that we use the same number of units in all dense
  layers, which performs slightly better.
  """

  @nn.compact
  def __call__(self, spheres: Array, charges: Array, train: bool,
               positions: Optional[Array]) -> jnp.ndarray:
    del train
    # Input is (batch, atoms, lat, lon, spins, channels). First we average over
    # spheres and merge channels/spins dimensions.
    features = _spin_mean_real_atoms(spheres, self.spins)
    features = einshape.einshape('basc->ba(sc)', features)

    # Apply MLPs to each atom's features independently.
    features = nn.Dense(256, precision='highest')(features)
    features = nn.relu(features)

    if self.use_atom_type_embedding:
      features += _AtomTypeEmbedding(self.atom_types)(features, charges)
    if self.use_distance_to_center:
      distance_to_center = _distance_to_center(positions, charges)
      features = (features +
                  nn.Dense(features.shape[-1],
                           precision='highest')(distance_to_center))

    features = nn.Dense(256, precision='highest')(features)
    features = nn.relu(features)

    features = _AtomFeatureAggregation(
        self.atom_types, self.max_atoms, self.atom_aggregation_mode)(
            features, charges, positions)

    atom_counts = self._compute_atom_counts(charges)

    features = jnp.concatenate([features,
                                atom_counts / self.max_atoms], axis=-1)

    # Apply MLP on aggregated features.
    features = nn.Dense(256, precision='highest')(features)
    features = nn.relu(features)
    features = nn.Dense(256, precision='highest')(features)
    features = nn.relu(features)
    output = nn.Dense(1, precision='highest')(features)

    return einshape.einshape('b1->b', output)


# `_MLPBlock` and `_Encoder1DBlock` are adapted from ViT:
# https://github.com/google-research/vision_transformer/blob/1f2a4a173acf0795778f422e9e80e21d154fe299/vit_jax/models.py#L68-L155
class _MLPBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs: Array, *, deterministic: bool) -> jnp.ndarray:
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision='highest')(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision='highest')(x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class _Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self,
               inputs: Array,
               *,
               deterministic: bool,
               mask: Optional[Array] = None) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.
      mask: Boolean attention mask passed to `nn.MultiHeadDotProductAttention`.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        precision='highest',
        num_heads=self.num_heads)(
            x, x, mask=mask)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = _MLPBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + y


class _TransformerTail(_Tail):
  """Self-attention transformer tail.

  This module first averages over the spherical dimensions, then treats the
  per-atom features as a set that is processed by a transformer, which is
  followed by an average over atoms and MLP.

  Attributes:
    num_layers: Number of transformer layers.
    num_heads: Number of self-attention heads.
    mlp_dim: Number of units on each dense layer.
    dropout_rate: dropout rate after dense layer.
    attention_dropout_rate: dropout rate for attention heads.
  """
  num_layers: int = 4
  num_heads: int = 1
  mlp_dim: int = 256
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, spheres: Array, charges: Array, train: bool,
               positions: Optional[Array]) -> jnp.ndarray:
    # Input is (batch, atoms, lat, lon, spins, channels). First we average over
    # spheres and merge channels/spins dimensions.
    features = _spin_mean_real_atoms(spheres, self.spins)
    features = einshape.einshape('basc->ba(sc)', features)

    # Map to desired feature dimension inside the transformer.
    features = nn.Dense(self.mlp_dim, precision='highest')(features)
    features = nn.relu(features)

    if self.use_atom_type_embedding:
      features += _AtomTypeEmbedding(self.atom_types)(features, charges)
    if self.use_distance_to_center:
      distance_to_center = _distance_to_center(positions, charges)
      features = (features +
                  nn.Dense(features.shape[-1],
                           precision='highest')(distance_to_center))

    mask = charges > 0
    attention_mask = jnp.einsum('bx,by->bxy', mask, mask)
    # Attention mask need extra dimension for attention heads.
    attention_mask = einshape.einshape('bxy->b1xy', attention_mask)

    for layer in range(self.num_layers):
      features = _Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{layer}',
          num_heads=self.num_heads)(
              features, mask=attention_mask, deterministic=not train)
    features = nn.LayerNorm(name='encoder_norm')(features)

    # Ignore features from zero-padded atoms.
    features = _AtomFeatureAggregation(
        self.atom_types, self.max_atoms, self.atom_aggregation_mode)(
            features, charges, positions)

    atom_counts = self._compute_atom_counts(charges)

    features = jnp.concatenate([features,
                                atom_counts / self.max_atoms], axis=-1)

    # Apply MLP on aggregated features.
    features = nn.Dense(self.mlp_dim, precision='highest')(features)
    features = nn.relu(features)
    features = nn.Dense(self.mlp_dim, precision='highest')(features)
    features = nn.relu(features)
    output = nn.Dense(1, precision='highest')(features)

    return einshape.einshape('b1->b', output)


class SpinSphericalRegressor(nn.Module):
  """Spin-weighted spherical CNN for regression.

  Attributes:
    resolutions: (n_layers,) tuple of resolutions at each layer. For consecutive
      resolutions a, b, we must have either a == b or a == 2*b. The latter
      triggers inclusion of a pooling layer.
    spins: A (n_layers,) tuple of (n_spins,) tuples of spin weights per layer.
    widths: (n_layers,) tuple of width per layer (number of channels).
    residual: Whether to use residual blocks.
    spectral_pooling: When True, use spectral instead of spatial pooling.
    num_filter_params: (n_layers,) the number of filter parameters per layer.
    tail_module: Which tail module to apply.
    use_atom_type_embedding: When True, adds a learnable per atom type embedding
      to atom features.
    use_distance_to_center: When True, adds a learnable embedding
      proportional to each atom's distance to the molecule's center of mass.
    atom_aggregation_mode: Method for aggregating per-atom features.
    num_transformer_layers: Number of transformer layers.
    num_transformer_heads: Number of self-attention heads.
    dropout_rate: Dropout rate after dense layer inside transformer tail.
    attention_dropout_rate: Dropout rate for attention heads.
    metadata: Dataset metadata.
    axis_name: Identifier for the mapped axis in parallel training.
    input_transformer: None, or SpinSphericalFourierTransformer
      instance. Will be computed automatically if None.
  """
  resolutions: Tuple[int, ...]
  spins: Tuple[Tuple[int, ...], ...]
  widths: Tuple[int, ...]
  residual: bool
  spectral_pooling: bool
  num_filter_params: Optional[Tuple[int, ...]]
  tail_module: str
  use_atom_type_embedding: bool
  use_distance_to_center: bool
  atom_aggregation_mode: str
  num_transformer_layers: int
  num_transformer_heads: int
  dropout_rate: float
  attention_dropout_rate: float
  metadata: nn.FrozenDict
  axis_name: Any
  input_transformer: Optional[
      spin_spherical_harmonics.SpinSphericalFourierTransformer]

  def setup(self):
    if self.input_transformer is None:
      # Flatten spins.
      all_spins = functools.reduce(operator.concat, self.spins)
      self.transformer = (spin_spherical_harmonics.
                          SpinSphericalFourierTransformer(
                              resolutions=np.unique(self.resolutions),
                              spins=np.unique(all_spins)))
    else:
      self.transformer = self.input_transformer

    num_layers = len(self.resolutions)
    if len(self.spins) != num_layers or len(self.widths) != num_layers:
      raise ValueError('`resolutions`, `spins`, and `widths` '
                       'must be the same size!')
    model_layers = []
    for layer_id in range(num_layers - 1):
      resolution_in = self.resolutions[layer_id]
      resolution_out = self.resolutions[layer_id + 1]
      spins_in = self.spins[layer_id]
      spins_out = self.spins[layer_id + 1]
      if self.num_filter_params is None:
        num_filter_params = None
      else:
        num_filter_params = self.num_filter_params[layer_id + 1]

      num_channels = self.widths[layer_id + 1]

      # We pool before conv to avoid expensive increase of number of channels at
      # higher resolution.
      if resolution_out == resolution_in // 2:
        downsampling_factor = 2
      elif resolution_out != resolution_in:
        raise ValueError('Consecutive resolutions must be equal or halved.')
      else:
        downsampling_factor = 1

      if layer_id == 0 or not self.residual:
        model_layers.append(layers.SpinSphericalBlock(
            num_channels=num_channels,
            spins_in=spins_in,
            spins_out=spins_out,
            downsampling_factor=downsampling_factor,
            spectral_pooling=self.spectral_pooling,
            num_filter_params=num_filter_params,
            axis_name=self.axis_name,
            transformer=self.transformer,
            after_conv_module=layers.SpinSphericalBatchNormPhaseCollapse,
            name=f'spin_block_{layer_id}'))
      else:
        model_layers.append(
            layers.SpinSphericalResidualBlock(
                num_channels=num_channels,
                spins_in=spins_in,
                spins_out=spins_out,
                downsampling_factor=downsampling_factor,
                spectral_pooling=self.spectral_pooling,
                num_filter_params=num_filter_params,
                axis_name=self.axis_name,
                transformer=self.transformer,
                after_conv_module=layers.SpinSphericalBatchNormPhaseCollapse,
                name=f'spin_residual_block_{layer_id}',
            )
        )

    self.layers = model_layers
    if self.tail_module == 'sum':
      tail = _SumMLPTail
    elif self.tail_module == 'deepsets':
      tail = _DeepSetsTail
    elif self.tail_module == 'transformer':
      tail = functools.partial(
          _TransformerTail,
          num_layers=self.num_transformer_layers,
          num_heads=self.num_transformer_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate)

    self.tail = tail(
        spins=spins_out,
        atom_types=self.metadata['atom_types'],
        max_atoms=self.metadata['max_atoms'],
        use_atom_type_embedding=self.use_atom_type_embedding,
        use_distance_to_center=self.use_distance_to_center,
        atom_aggregation_mode=self.atom_aggregation_mode,
        name='tail')

  def __call__(self, spheres: Array, charges: Array, positions: Optional[Array],
               train: bool) -> jnp.ndarray:
    """Apply the model.

    Args:
      spheres: (batch_size, num_atoms, resolution, resolution, num_spins,
        num_channels) array with the spherical molecule representation.
      charges: (batch_size, num_atoms) with atom types per molecule.
      positions: (batch_size, num_atoms, 3) with atom positions per molecule.
      train: Whether to run in training or inference mode.
    Returns:
      A (batch_size,) float32 array with regression outputs.
    """
    resolution, num_spins, _ = spheres.shape[3:]
    if (resolution != self.resolutions[0] or
        num_spins != len(self.spins[0])):
      raise ValueError('Incorrect input dimensions!')

    # Detect zero-padded atoms.
    mask_valid = charges > 0

    # Combine batch and atoms dimensions.
    max_atoms = spheres.shape[1]
    features = spheres

    # Merge batch and atom dimensions.
    features = einshape.einshape('baijsc->(ba)ijsc', features)
    # This mask is used in the BatchNorms to avoid that the input zero-padded
    # dimensions influence the batch stats.
    mask_reshaped = einshape.einshape('ba->(ba)', mask_valid)

    for layer in self.layers:
      features = layer(features, train=train, weights=mask_reshaped)
    # Bring back batch and atoms dimensions.
    features = einshape.einshape('(ba)ijsc->baijsc', features, a=max_atoms)

    # Rescale positions before sending to Tail.
    if positions is not None:
      positions = positions / self.metadata['max_pairwise_distance']
    return self.tail(features, charges, train, positions)


def _random_orthogonal(key: Array,
                       n: int,
                       shape: Sequence[int] = ()) -> jnp.ndarray:
  """Same as `jax.random.orthogonal`, with smaller chance of NaNs."""
  # This computes 2x as many elements and rejects nearly singular
  # matrices to avoid NaN outputs.
  shape = (2,) + tuple(shape)
  z = jax.random.normal(key, (*shape, n, n))
  z = jnp.where(jnp.abs(jnp.linalg.det(z[0])[..., None, None]) < 1e-5,
                z[1], z[0])

  q, r = jnp.linalg.qr(z)
  d = jnp.diagonal(r, 0, -2, -1)
  return jax.lax.mul(q, jax.lax.expand_dims(
      jax.lax.div(d, abs(d).astype(d.dtype)), [-2]))


def _molecule_to_sphere_gaussian(
    coordinates: Array,
    charges: Array,
    random_seed: Optional[Array],
    resolution: int,
    min_pairwise_distance: float,
    cutoff_angle: float,
    max_atoms: int,
    atom_types: Sequence[int],
    distance_powers: Tuple[float, ...],
    rotation_augmentation: bool) -> jnp.ndarray:
  """Converts to the localized spherical molecule representation.

  This representation uses the pairwise interaction between atoms modulated by
  an unnormalized Gaussian. It results in a more localized representation that
  preserves the direction of the interaction. It does not require to define a
  radius for the spheres since only the directions matter.

  Args:
    coordinates: (max_atoms, 3) zero-padded atom positions.
    charges: (max_atoms,) atomic numbers.
    random_seed: To be used when `rotation_augmentation` is True.
    resolution: Resolution of the spheres.
    min_pairwise_distance: Minimum distance between pairs of atoms in the
      dataset.
    cutoff_angle: Angle where the representation value decays to 5% of the
      Coulomb-like interaction value.
    max_atoms: Maximum number of atoms in the molecule (over the dataset).
    atom_types: Atomic numbers present in the dataset.
    distance_powers: Tuple where each entry results in a feature channel
      1/distance**distance_power.
    rotation_augmentation: whether to augment molecule inputs with random
      rotations.

  Returns:
    A (max_atoms, resolution, resolution, len(atom_types) *
    len(distance_powers)) with the spherical representation of each atom.
  """

  if rotation_augmentation:
    # This samples from O(3), not SO(3). The properties we are interested are
    # invariant to reflections so O(3) should be better.
    random_rotation = _random_orthogonal(random_seed, 3)
    coordinates = jnp.matmul(coordinates, random_rotation, precision='high')

  num_atoms = len(coordinates)
  longitude, colatitude = sphere_utils.make_equiangular_grid(resolution)
  longitude = longitude.astype(np.float32)
  colatitude = colatitude.astype(np.float32)
  # (num_lat, num_long, 3).
  grid_coordinates = np.stack([np.sin(colatitude) * np.cos(longitude),
                               np.sin(colatitude) * np.sin(longitude),
                               np.cos(colatitude)],
                              axis=-1)

  # (num_atoms, num_atoms, 3).
  relative_positions = (jnp.expand_dims(coordinates, 0) -
                        jnp.expand_dims(coordinates, 1))
  # (num_atoms, num_atoms, 1).
  pairwise_distances = jnp.linalg.norm(relative_positions, axis=-1)
  # We set the distance to self as infinity so it doesn't contribute in the
  # final values.
  pairwise_distances = (pairwise_distances
                        .at[np.arange(num_atoms), np.arange(num_atoms)]
                        .set(jnp.inf))
  # (num_atoms, num_atoms, num_distance_powers).
  distance_factors = jnp.stack([1.0 / (pairwise_distances**distance_power)
                                for distance_power in distance_powers], axis=-1)
  # For normalization.
  max_values = np.stack(
      [max(atom_types)**2 / min_pairwise_distance**distance_power
       for distance_power in distance_powers]).astype(np.float32)

  # Interactions of the form k1 k2 / r^x.
  pairwise_charges = jnp.outer(charges, charges).astype(jnp.float32)
  pairwise_interactions = (jnp.expand_dims(pairwise_charges, 2) *
                           distance_factors /
                           np.expand_dims(max_values, (0, 1)))

  # Inner products between pairwise displacements and the spherical grid; shape
  # is (num_atoms, num_atoms, num_lat, num_long).
  inner_products = jnp.einsum(
      'ijk,lmk->ijlm',
      relative_positions / jnp.expand_dims(pairwise_distances, -1),
      grid_coordinates,
      precision='high')

  # Set Gaussian variance such that 95% of the value is cutoff at the cutoff
  # angle. Weights are not normalized.
  sigma2 = (-(np.cos(cutoff_angle) - 1)**2  / np.log(0.05)).astype(np.float32)
  gaussian_weight = jnp.exp(-(inner_products - 1)**2 / sigma2)

  # Apply Gaussian factor to interatomic interactions
  gaussian_interactions = (jnp.expand_dims(pairwise_interactions, (2, 3)) *
                           jnp.expand_dims(gaussian_weight, 4))

  # TODO(machc): Using segment_sum to replace this loop might be more efficient.
  # Sum over each atom type (across second dimension).
  combined_interactions = []
  for atom_type in atom_types:
    atom_types_ids = jnp.expand_dims(jnp.isclose(charges, atom_type),
                                     (0, 2, 3, 4))
    combined_interactions.append(
        jnp.sum(gaussian_interactions, axis=1, where=atom_types_ids))
  combined_interactions = jnp.stack(combined_interactions, axis=-1)
  # Merge atom type and distance_power dimensions.
  combined_interactions = jnp.reshape(combined_interactions,
                                      (num_atoms, resolution, resolution, -1))
  # Pad back to max_atoms.
  padded = jnp.pad(combined_interactions,
                   [(0, max_atoms-num_atoms), (0, 0), (0, 0), (0, 0)])
  # Add spin dimension.
  return jnp.expand_dims(padded, -2)


class SpinSphericalRegressorFromPositions(SpinSphericalRegressor):
  """Same as SpinSphericalRegressor, but taking atom positions.

  This will evaluate the molecule to sphere transformation on device.

  Attributes:
    sphere_resolution: resolution of the molecule spherical representation.
    cutoff_angle: Angle where the representation value decays to 5% of the
      Coulomb-like interaction value.
    distance_powers: Interaction values decay as distance**(-power).
    rotation_augmentation: Whether to randomly rotate and reflect the inputs.
  """
  sphere_resolution: int
  cutoff_angle: float
  distance_powers: Tuple[float, ...]
  rotation_augmentation: bool

  def __call__(self,
               positions: Array,
               charges: Array,
               train: bool) -> jnp.ndarray:
    """Apply the model.

    Args:
      positions: (batch_size, num_atoms, 3) with atom positions per molecule.
      charges: (batch_size, num_atoms) with atom types per molecule.
      train: Whether to run in training or inference mode.
    Returns:
      A (batch_size,) float32 array with regression outputs.
    """
    rotation_augmentation = self.rotation_augmentation and train
    partial = functools.partial(
        _molecule_to_sphere_gaussian,
        resolution=self.sphere_resolution,
        min_pairwise_distance=self.metadata['min_pairwise_distance'],
        cutoff_angle=self.cutoff_angle,
        max_atoms=positions.shape[1],
        atom_types=self.metadata['atom_types'],
        distance_powers=self.distance_powers,
        rotation_augmentation=rotation_augmentation)
    molecule_to_sphere = jax.vmap(partial, in_axes=(0, 0, 0))
    if train:
      random_seeds = jax.random.split(self.make_rng('augmentation'),
                                      len(positions))
    else:
      random_seeds = jnp.zeros((len(positions), 2))
    spheres = molecule_to_sphere(positions, charges, random_seeds)
    return super().__call__(spheres, charges, positions, train)
