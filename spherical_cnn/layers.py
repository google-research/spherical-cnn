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

"""Spin-Weighted Spherical CNN layers.

This implements the layers used in Esteves et al, "Spin-Weighted Spherical
CNNs", NeurIPS'20 [1].

Since the spin-weighted spherical convolution (SWSConv) is defined between sets
of spin-weighted spherical functions (SWSFs) of different spin weights, and in a
CNN we have a batch dimension and multiple channels per layer, in this module we
make use of 5D arrays stacking the mini-batch, spins and channels. For the
spatial equiangular representation the dimensions are (batch, lat, long, spin,
channel), and for spectral coefficients, the dimensions are (batch, ell, m,
spin, channel).
"""
import abc
import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from einshape.src.jax import jax_ops as einshape
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from spherical_cnn import sphere_utils
from spherical_cnn import spin_spherical_harmonics


Array = Union[np.ndarray, jnp.ndarray]
Initializer = Callable[[Any, Sequence[int], Any],
                       Array]
FourierTransformer = spin_spherical_harmonics.SpinSphericalFourierTransformer
DEFAULT_TRANSFORMER = FourierTransformer((4,), (0,))


def _spin_spherical_convolution_from_spectral(
    transformer, coefficients_in, filter_coefficients, spins_out,
    spectral_pooling, spectral_upsampling,
    output_representation):
  """`_spin_spherical_convolution` with spectral inputs."""
  if spectral_pooling:
    ell_max = coefficients_in.shape[0] - 1
    new_ell_max = (ell_max+1) // 2 - 1
    coefficients_in = coefficients_in[
        :new_ell_max + 1, (ell_max-new_ell_max):(ell_max+new_ell_max+1)]

  # Compute the convolution in the spectral domain.
  coefficients_out = jnp.einsum("lmic,liocd->lmod",
                                coefficients_in,
                                filter_coefficients)

  if spectral_upsampling:
    num_ell = coefficients_out.shape[0]
    coefficients_out = jnp.pad(coefficients_out,
                               [(0, num_ell), (num_ell, num_ell),
                                (0, 0), (0, 0)])

  if output_representation == "spectral":
    return coefficients_out
  elif output_representation == "spatial":
    # Convert back to the spatial domain.
    return transformer.swsft_backward_spins_channels(coefficients_out,
                                                     spins_out)
  else:
    raise ValueError("`output_representation` must be either "
                     "'spectral' or 'spatial'.")


def _spin_spherical_convolution_from_spatial(
    transformer, sphere, filter_coefficients,
    spins_in, spins_out, spectral_pooling, spectral_upsampling,
    output_representation):
  """`_spin_spherical_convolution` with spatial inputs."""
  resolution = sphere.shape[0]
  ell_max = (sphere_utils.ell_max_from_resolution(resolution // 2)
             if spectral_pooling else None)
  # Convert to the spectral domain.
  coefficients_in = transformer.swsft_forward_spins_channels(
      sphere, spins_in, ell_max=ell_max)

  # When `spectral_pooling` == True, pooling was already performed
  # during the forward transform.
  return _spin_spherical_convolution_from_spectral(
      transformer, coefficients_in, filter_coefficients, spins_out,
      spectral_pooling=False, spectral_upsampling=spectral_upsampling,
      output_representation=output_representation)


def _spin_spherical_convolution(
    transformer, sphere_or_coeffs, filter_coefficients,
    spins_in, spins_out, spectral_pooling, spectral_upsampling,
    input_representation, output_representation):
  r"""Spin-weighted spherical convolution; spatial input and spectral filters.

  This implements a multi-channel version of Eq. (13) in [1], where sphere_set
  corresponds to F and filter_coefficients to \hat{K}. For convenience, the
  inputs and outputs are in the spatial domain but the filter is defined by its
  Fourier coefficients, since this is what is learned in [1].

  The multi-channel behavior is the usual in CNNs: for a input with n_in
  channels, we have n_in * n_out filters and n_out output channels, each of
  which is the sum over n_in filter outputs.

  Args:
    transformer: SpinSphericalFourierTransformer instance.
    sphere_or_coeffs: A (resolution, resolution, n_spins_in,
      n_channels_in) array of spin-weighted spherical functions with
      equiangular sampling, or (ell_max+1, 2*ell_max+1, n_spins_in,
      n_channels_in) array of coefficients.
    filter_coefficients: (resolution // 2, n_spins_in, n_spins_out,
      n_channels_in, n_channels_out) array of filter SWSH coefficients.
    spins_in: (n_spins_in,) Sequence of int containing the input spins.
    spins_out: (n_spins_out,) Sequence of int containing the output spins.
    spectral_pooling: When True, halve input dimensions via spectral pooling.
    spectral_upsampling: When True, double input dimensions via
      spectral padding.
    input_representation: Whether inputs are in the 'spectral' or
      'spatial' domain.
    output_representation: Whether to return outputs in the 'spectral'
      or 'spatial' domain.

  Returns:
    A (resolution, resolution, n_spins_out, n_channels_out) array of
    spin-weighted spherical functions with equiangular sampling.
  """
  if input_representation == "spectral":
    return _spin_spherical_convolution_from_spectral(
        transformer, sphere_or_coeffs, filter_coefficients,
        spins_out, spectral_pooling, spectral_upsampling, output_representation)
  elif input_representation == "spatial":
    return _spin_spherical_convolution_from_spatial(
        transformer, sphere_or_coeffs, filter_coefficients,
        spins_in, spins_out,
        spectral_pooling, spectral_upsampling,
        output_representation)
  else:
    raise ValueError("`input_representation` must be either "
                     "`spectral` or `spatial`.")


# To avoid vanishing or exploding activations, we wish the spherical
# convolutions to roughly maintain the variance of the
# inputs. Assuming zero mean, the variance of the coefficients and the
# variance of the spatial filter are the same. The output for one spin
# is constructed by adding all channels for all spins, therefore we
# correct the usual `variance_scaling` by `num_spins_in`.
def spin_spherical_initializer(num_spins_in):
  return nn.initializers.variance_scaling(scale=1/num_spins_in,
                                          mode="fan_in",
                                          distribution="normal",
                                          dtype=jnp.complex64)


class SpinSphericalConvolution(nn.Module):
  """Spin-weighted spherical convolutional layer.

  Wraps _swsconv_spatial_spectral(), initializing and keeping track of the
  learnable filter.

  Attributes:
    features: int, number of output features (channels).
    spins_in: (n_spins_in,) Sequence of int containing the input spins.
    spins_out: (n_spins_out,) Sequence of int containing the output spins.
    spectral_pooling: When True, halve input dimensions via spectral pooling.
    spectral_upsampling: When True, double input dimensions via
      spectral padding.
    transformer: SpinSphericalFourierTransformer instance.
    num_filter_params: Number of parameters per filter. Fewer parameters results
      in more localized filters.
    input_representation: Whether inputs are in the 'spectral' or
      'spatial' domain.
    output_representation: Whether to return outputs in the 'spectral'
      or 'spatial' domain.
    use_bias: Whether to add learnable bias to the output. Not needed
      when followed by batch normalization.
    initializer: initializer for the filter spectrum.
  """
  features: int
  spins_in: Sequence[int]
  spins_out: Sequence[int]
  spectral_pooling: bool
  spectral_upsampling: bool
  transformer: spin_spherical_harmonics.SpinSphericalFourierTransformer
  num_filter_params: Optional[int] = None
  input_representation: str = "spatial"
  output_representation: str = "spatial"
  use_bias: bool = False
  initializer: Optional[Initializer] = None

  def _get_kernel(self, ell_max, num_channels_in):
    kernel_shape = (ell_max+1, len(self.spins_in), len(self.spins_out),
                    num_channels_in, self.features)
    init_fun = (spin_spherical_initializer(len(self.spins_in))
                if self.initializer is None else self.initializer)
    return self.param("kernel", init_fun, kernel_shape)

  def _get_localized_kernel(self, ell_max, num_channels_in):
    # We interpolate along ell to obtain all weights from the learnable weights,
    # hence it doesn't make sense to have more parameters than num_ell.
    if self.num_filter_params > ell_max + 1:
      raise ValueError("num_filter_params must be <= ell_max + 1")
    ell_in = jnp.linspace(0, 1, self.num_filter_params)
    ell_out = jnp.linspace(0, 1, ell_max + 1)
    # `vectorize` is over leading dimensions, so we put ell as the last
    # dimension and transpose it to the first later.
    learnable_shape = (len(self.spins_in), len(self.spins_out),
                       num_channels_in, self.features,
                       self.num_filter_params)
    init_fun = (spin_spherical_initializer(len(self.spins_in))
                if self.initializer is None else self.initializer)
    learnable_weights = self.param("kernel", init_fun, learnable_shape)
    # `jnp.interp` works on 1D inputs; we vectorize it to interpolate over a
    # single dimension of n-D inputs.
    vectorized_interp = jnp.vectorize(jnp.interp, signature="(m),(n),(n)->(m)")
    weights = vectorized_interp(ell_out, ell_in, learnable_weights)
    # Make ell the first dimension.
    return weights.transpose((4, 0, 1, 2, 3))

  @nn.compact
  def __call__(self, sphere_or_coeffs: Array) -> jnp.ndarray:
    """Applies convolution to inputs.

    Args:
      sphere_or_coeffs: A (batch_size, resolution, resolution,
        n_spins_in, n_channels_in) array of spin-weighted spherical
        functions (SWSF) with equiangular sampling, or a (batch_size,
        ell_max+1, 2*ell_max+1, n_spins_in, n_channels_in) array of
        spectral coefficients.

    Returns:
      A (batch_size, resolution, resolution, n_spins_out, n_channels_out)
      complex64 array of SWSF with equiangular H&W sampling.
    """
    if self.input_representation == "spectral":
      ell_max = sphere_or_coeffs.shape[1] - 1
      resolution = 2 * (ell_max + 1)
      if sphere_or_coeffs.shape[2] != 2*ell_max + 1:
        raise ValueError("Axes 1 and 2 must have dimensions "
                         "(ell_max+1, 2*ell_max+1).")
    elif self.input_representation == "spatial":
      resolution = sphere_or_coeffs.shape[1]
      ell_max = sphere_utils.ell_max_from_resolution(resolution)
      if sphere_or_coeffs.shape[2] != resolution:
        raise ValueError("Axes 1 and 2 must have the same dimensions!")
    else:
      raise ValueError("`input_representation` must be either "
                       "'spectral' or 'spatial'.")

    if sphere_or_coeffs.shape[3] != len(list(self.spins_in)):
      raise ValueError("Input axis 3 (spins_in) doesn't match layer's.")

    if self.spectral_pooling and self.spectral_upsampling:
      raise ValueError("`spectral_pooling` and `spectral_upsampling` "
                       "should not be both True.")

    if self.spectral_pooling:
      resolution //= 2
      ell_max = sphere_utils.ell_max_from_resolution(resolution)

    # Make sure constants contain all spins for input resolution.
    for spin in set(self.spins_in).union(self.spins_out):
      if not self.transformer.validate(resolution, spin):
        raise ValueError("Constants are invalid for given input!")

    num_channels_in = sphere_or_coeffs.shape[-1]
    if self.num_filter_params is None:
      kernel = self._get_kernel(ell_max, num_channels_in)
    else:
      kernel = self._get_localized_kernel(ell_max, num_channels_in)

    # Map over the batch dimension.
    vmap_convolution = jax.vmap(
        _spin_spherical_convolution,
        in_axes=(None, 0, None, None, None, None, None, None, None))
    out = vmap_convolution(self.transformer,
                           sphere_or_coeffs, kernel,
                           self.spins_in,
                           self.spins_out,
                           self.spectral_pooling,
                           self.spectral_upsampling,
                           self.input_representation,
                           self.output_representation)

    if self.use_bias:
      # Add complex bias to spin 0.
      idx_zero, _ = get_zero_nonzero_idx(self.spins_out)
      out = out.at[..., idx_zero, :].add(
          self.param("bias",
                     _complex_zeros_initializer,
                     (self.features,)))

    return out


class MagnitudeNonlinearity(nn.Module):
  """Magnitude thresholding nonlinearity, suitable for complex inputs.

  Executes the following operation pointwise: z = relu(|z|+b) * (z / |z|), where
  b is a learned bias per spin per channel.

  NOTE(machc): This operation does not preserve bandwidth and is pointwise.  It
  is only approximately equivariant for the equiangular spherical
  discretization. See `layers_test.MagnitudeNonlinearityTest` for quantitative
  evaluations of the equivariance error.

  Attributes:
    epsilon: Small float constant to avoid division by zero.
    bias_initializer: initializer for the bias (default to zeroes).
  """
  epsilon: jnp.float32 = 1e-6
  bias_initializer: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: Array) -> jnp.ndarray:
    """Applies pointwise nonlinearity to 5D inputs."""
    bias = self.param("bias", self.bias_initializer,
                      (1, 1, 1, inputs.shape[-2], inputs.shape[-1]))

    modulus_inputs = jnp.abs(inputs)
    return (nn.relu(modulus_inputs + bias) *
            (inputs / (modulus_inputs + self.epsilon)))


class MagnitudeNonlinearityLeakyRelu(nn.Module):
  """Applies MagnitudeNonlinearity to spin != 0 and leaky relu for spin == 0.

  The spin == 0 component does not change phase upon rotation, so any pointwise
  nonlinearity works. Here we choose the leaky relu.

  Attributes:
    spins: (n_spins,) Sequence of int containing the input spins.
    epsilon: Small float constant to avoid division by zero.
    bias_initializer: initializer for the spin != 0 bias (default to zeroes).
  """
  spins: Sequence[int]
  epsilon: jnp.float32 = 1e-6
  bias_initializer: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: Array) -> jnp.ndarray:
    """Applies pointwise nonlinearity to 5D inputs."""
    outputs = []
    for i, spin in enumerate(self.spins):
      inputs_spin = inputs[..., [i], :]
      if spin == 0:
        # In [1], the spin0 inputs are cast to real at every
        # layer. Here we merge this operation with the nonlinearity.
        outputs_spin = nn.leaky_relu(inputs_spin.real)
      else:
        outputs_spin = MagnitudeNonlinearity(
            self.epsilon, self.bias_initializer,
            name=f"magnitude_nonlin_{i}")(inputs_spin)
      outputs.append(outputs_spin)

    return jnp.concatenate(outputs, axis=-2)


class PhaseCollapseNonlinearity(nn.Module):
  """Nonlinearity that concatenates x to |x|.

  This is inspired by Guth et al, "Phase Collapse in Neural Networks".  We adapt
  the idea to spin-weighted functions by concatenating the magnitude of all
  spins to the value at spin == 0, then projecting it back to the input number
  of channels.

  Attributes:
    spins: (n_spins,) Sequence of int containing the input spins.
  """

  spins: Sequence[int]

  @nn.compact
  def __call__(self, inputs: Array) -> jnp.ndarray:
    """Applies the nonlinearity.

    Args:
      inputs: (batch_size, resolution, resolution, num_spins, num_channels)
        complex64 feature array.

    Returns:
      An array with the same shape as `inputs`, where only the dimension
        corresponding to spin zero changes.
    """
    idx_zero, _ = get_zero_nonzero_idx(self.spins)
    abs_inputs = einshape.einshape("bxysc->bxy1(sc)", jnp.abs(inputs))
    num_channels = inputs.shape[-1]
    dense_options = dict(
        features=num_channels, dtype=jnp.complex64, param_dtype=jnp.complex64
    )

    # We avoid a slow concatenation here by applying two dense layers.
    outputs_spin0 = (
        nn.Dense(**dense_options)(inputs[..., [idx_zero], :])
        + nn.Dense(**dense_options)(abs_inputs)
    ) * 0.5

    return inputs.at[..., [idx_zero], :].set(outputs_spin0)


class SphericalPooling(nn.Module):
  """Spherical pooling layer, accounting for cell area variation.

  Executes a weighted average pooling, with weights proportional to the H&W
  quadrature scheme. However, the pooling here is a local operation, so we don't
  use the toroidal extension.

  NOTE(machc): This operation has multiple sources of equivariance errors. A
  reasonable alternative that is perfectly equivariant is to drop high
  frequencies in the spectral domain right after the convolution. We have
  experimented with this approach long ago and found it underperforming, but it
  is probably worth revisiting.

  Attributes:
    stride: int, pooling stride and window shape are (stride, stride).
  """
  stride: int

  @nn.compact
  def __call__(self, inputs: Array) -> jnp.ndarray:
    """Applies spherical pooling.

    Args:
      inputs: An array of dimensions (batch_size, resolution, resolution,
      n_spins_in, n_channels_in).
    Returns:
      An array of dimensions (batch_size, resolution // stride, resolution //
      stride, n_spins_in, n_channels_in).
    """
    # We use variables to cache the in/out weights.
    resolution_in = inputs.shape[1]
    resolution_out = resolution_in // self.stride
    weights_in = sphere_utils.sphere_quadrature_weights(resolution_in)
    weights_out = sphere_utils.sphere_quadrature_weights(resolution_out)

    weighted = inputs * jnp.expand_dims(weights_in, (0, 2, 3, 4))
    pooled = nn.avg_pool(weighted,
                         window_shape=(self.stride, self.stride, 1),
                         strides=(self.stride, self.stride, 1))
    # This was average pooled. We multiply by stride**2 to obtain the sum
    # pooled, then divide by output weights to get the weighted average.
    pooled = (pooled * self.stride**2 /
              jnp.expand_dims(weights_out, (0, 2, 3, 4)))

    return pooled


_complex_ones_initializer = functools.partial(nn.initializers.ones,
                                              dtype=jnp.complex64)

_complex_zeros_initializer = functools.partial(nn.initializers.zeros,
                                               dtype=jnp.complex64)


class SphericalBatchNormalization(nn.Module):
  """Batch normalization for spherical functions.

  Two main changes with respect to the usual nn.BatchNorm:
    1) Subtracting a complex value is not rotation-equivariant for spin-weighted
      functions, so we add an option to not subtract the mean and only keep
      track of and divide by the variance.
    2) Mean and variance computation on the sphere must take into account the
      discretization cell areas.

  Attributes:
    use_running_stats: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    centered: When False, skips mean-subtraction step.
    epsilon: a small float added to variance to avoid dividing by zero.
    use_bias: if True, add a complex-valued learned bias.
    use_scale: if True, multiply by a complex-valued learned scale.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
  """
  use_running_stats: Optional[bool] = None
  momentum: float = 0.99
  epsilon: float = 1e-5
  centered: bool = True
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = _complex_zeros_initializer
  scale_init: Initializer = _complex_ones_initializer
  axis_name: Optional[str] = None

  @nn.compact
  def __call__(self,
               inputs: Array,
               use_running_stats: Optional[bool] = None,
               weights: Optional[Array] = None) -> jnp.ndarray:
    """Normalizes the input using batch (optional) means and variances.

    Stats are computed over the batch and spherical dimensions: (0, 1, 2).

    Args:
      inputs: An array of dimensions (batch_size, resolution, resolution,
        n_spins_in, n_channels_in).
      use_running_stats: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      weights: An array of dimensions (batch_size,) assigning weights for
        each batch element. Useful for masking.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    use_running_stats = nn.module.merge_param(
        "use_running_stats", self.use_running_stats, use_running_stats)

    # Normalization is independent per spin per channel.
    num_spins, num_channels = inputs.shape[-2:]
    feature_shape = (1, 1, 1, num_spins, num_channels)
    reduced_feature_shape = (num_spins, num_channels)

    initializing = not self.has_variable("batch_stats", "variance")

    running_variance = self.variable("batch_stats", "variance",
                                     lambda s: jnp.ones(s, jnp.float32),
                                     reduced_feature_shape)

    if self.centered:
      running_mean = self.variable("batch_stats", "mean",
                                   lambda s: jnp.zeros(s, jnp.complex64),
                                   reduced_feature_shape)

    if use_running_stats:
      variance = running_variance.value
      if self.centered:
        mean = running_mean.value
    else:
      # Compute the spherical mean over the spherical grid dimensions, then a
      # conventional mean over the batch.
      if self.centered:
        mean = sphere_utils.spin_spherical_mean(inputs)
        mean = jnp.average(mean, axis=0, weights=weights)
      # Complex variance is E[x x*] - E[x]E[x*].
      # For spin != 0, E[x] should be zero, although due to discretization this
      # is not always true. We only use E[x x*] here.
      # E[x x*]:
      mean_abs_squared = sphere_utils.spin_spherical_mean(inputs *
                                                          inputs.conj())
      mean_abs_squared = jnp.average(mean_abs_squared, axis=0, weights=weights)
      # Aggregate means over devices.
      if self.axis_name is not None and not initializing:
        if self.centered:
          mean = lax.pmean(mean, axis_name=self.axis_name)
        mean_abs_squared = lax.pmean(mean_abs_squared, axis_name=self.axis_name)

      # Imaginary part is negligible.
      variance = mean_abs_squared.real

      if not initializing:
        running_variance.value = (self.momentum * running_variance.value +
                                  (1 - self.momentum) * variance)
        if self.centered:
          running_mean.value = (self.momentum * running_mean.value +
                                (1 - self.momentum) * mean)

    if self.centered:
      outputs = inputs - mean.reshape(feature_shape)
    else:
      outputs = inputs

    factor = lax.rsqrt(variance.reshape(feature_shape) + self.epsilon)
    if self.use_scale:
      scale = self.param("scale",
                         self.scale_init,
                         reduced_feature_shape).reshape(feature_shape)
      factor = factor * scale

    outputs = outputs * factor

    if self.use_bias:
      bias = self.param("bias",
                        self.bias_init,
                        reduced_feature_shape).reshape(feature_shape)
      outputs = outputs + bias

    return outputs


def get_zero_nonzero_idx(spins: Sequence[int]) -> Tuple[int, Tuple[int, ...]]:
  """Returns zero and nonzero indices from a list of spins."""
  idx_zero = spins.index(0)
  idx_nonzero = tuple([idx for idx, spin in enumerate(spins) if spin != 0])
  if len(idx_nonzero) + 1 != len(spins):
    raise ValueError("`spins` must contain exactly one zero.")
  return idx_zero, idx_nonzero


class SpinSphericalBatchNormalization(nn.Module):
  """Batch normalization for spin-spherical functions.

  This uses the default SphericalBatchNormalization for spin == 0 and the
  centered version for other spins.


  Attributes:
    spins: (n_spins,) Sequence of int containing the input spins.
    use_running_stats: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
  """
  spins: Sequence[int]
  use_running_stats: Optional[bool] = None
  momentum: float = 0.99
  epsilon: float = 1e-5
  scale_init: Initializer = _complex_ones_initializer
  bias_init: Initializer = _complex_zeros_initializer
  axis_name: Optional[str] = None

  @nn.compact
  def __call__(self,
               inputs: Array,
               use_running_stats: Optional[bool] = None,
               weights: Optional[Array] = None) -> jnp.ndarray:
    """Call appropriate version of SphericalBatchNormalization per spin."""
    use_running_stats = nn.module.merge_param(
        "use_running_stats", self.use_running_stats, use_running_stats)

    options = dict(use_running_stats=use_running_stats,
                   momentum=self.momentum,
                   epsilon=self.epsilon,
                   scale_init=self.scale_init,
                   bias_init=self.bias_init,
                   axis_name=self.axis_name)

    idx_zero, idx_nonzero = get_zero_nonzero_idx(self.spins)
    outputs = inputs
    outputs = outputs.at[..., [idx_zero], :].set(
        SphericalBatchNormalization(
            use_bias=True,
            centered=True,
            **options)(outputs[..., [idx_zero], :],
                       weights=weights))
    outputs = outputs.at[..., idx_nonzero, :].set(
        SphericalBatchNormalization(
            use_bias=False,
            centered=False,
            **options)(outputs[..., idx_nonzero, :],
                       weights=weights))

    return outputs


class BatchNormAndActivation(nn.Module, metaclass=abc.ABCMeta):
  """Abstract class for combining batch normalization and nonlinarity.

  Subclasses of this class serve as `after_conv_module` in
  `models.SpinSphericalBlock`. The idea is that the combined
  implementation can be faster when BN and activations have different
  behaviors for spin zero and nonzero.

  Attributes:
    spins: (n_spins,) Sequence of int containing the input spins.
    use_running_stats: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    bias_initializer: initializer for activation bias (if any).
    transformer: SpinSphericalFourierTransformer instance.
  """
  spins: Sequence[int]
  use_running_stats: Optional[bool] = None
  momentum: float = 0.99
  epsilon: float = 1e-5
  axis_name: Optional[str] = None
  bias_initializer: Initializer = nn.initializers.zeros
  transformer: FourierTransformer = DEFAULT_TRANSFORMER

  @abc.abstractmethod
  def __call__(self,
               inputs: Array,
               use_running_stats: Optional[bool] = None,
               weights: Optional[Array] = None) -> jnp.ndarray:
    """Applies batch norm and activation.

    Args:
      inputs: An array of dimensions (batch_size, height, width,
        num_spins_in, num_channels_in).
      use_running_stats: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      weights: An array of dimensions (batch_size,) assigning weights for each
        batch element. Useful for masking.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    raise NotImplementedError()


class SpinSphericalBatchNormMagnitudeNonlin(BatchNormAndActivation):
  """Applies spin batch normalization and magnitude nonlinarity."""

  @nn.compact
  def __call__(self,
               inputs: Array,
               use_running_stats: Optional[bool] = None,
               weights: Optional[Array] = None) -> jnp.ndarray:
    """Calls appropriate batch normalization and nonlinearity per spin."""
    use_running_stats = nn.module.merge_param(
        "use_running_stats", self.use_running_stats, use_running_stats)

    options = dict(use_running_stats=use_running_stats,
                   momentum=self.momentum,
                   epsilon=self.epsilon,
                   axis_name=self.axis_name)
    outputs = []
    for i, spin in enumerate(self.spins):
      inputs_spin = inputs[..., [i], :]
      if spin == 0:
        outputs_spin = SphericalBatchNormalization(use_bias=True,
                                                   centered=True,
                                                   **options)(inputs_spin,
                                                              weights=weights)
        outputs_spin = nn.leaky_relu(outputs_spin.real)
      else:
        outputs_spin = SphericalBatchNormalization(use_bias=False,
                                                   centered=False,
                                                   **options)(inputs_spin,
                                                              weights=weights)
        outputs_spin = MagnitudeNonlinearity(
            bias_initializer=self.bias_initializer,
            name=f"magnitude_nonlin_{i}")(outputs_spin)
      outputs.append(outputs_spin)

    return jnp.concatenate(outputs, axis=-2)


class SpinSphericalBatchNormPhaseCollapse(BatchNormAndActivation):
  """Applies spin batch normalization and phase collapse."""

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      use_running_stats: Optional[bool] = None,
      weights: Optional[Array] = None,
  ) -> jnp.ndarray:
    """Calls appropriate batch normalization and nonlinearity per spin."""
    use_running_stats = nn.module.merge_param(
        "use_running_stats", self.use_running_stats, use_running_stats
    )

    options = dict(
        use_running_stats=use_running_stats,
        momentum=self.momentum,
        epsilon=self.epsilon,
        axis_name=self.axis_name,
    )

    idx_zero, idx_nonzero = get_zero_nonzero_idx(self.spins)
    outputs = inputs
    outputs = outputs.at[..., [idx_zero], :].set(
        SphericalBatchNormalization(use_bias=True, centered=True, **options)(
            outputs[..., [idx_zero], :], weights=weights
        )
    )
    outputs = outputs.at[..., idx_nonzero, :].set(
        SphericalBatchNormalization(use_bias=False, centered=False, **options)(
            outputs[..., idx_nonzero, :], weights=weights
        )
    )

    return PhaseCollapseNonlinearity(spins=self.spins)(outputs)


class SpinSphericalSpectralBatchNormalization(SpinSphericalBatchNormalization):
  """Batch normalization applied on the spectral domain.

  Means and variances can be computed directly from the Fourier coefficients, so
  the batch norm can be applied on the spectral domain, which is faster and more
  accurate.

  This module should match `layers.SpinSphericalBatchNormalization` exactly for
  bandlimited inputs with zero mean. There will be a slight difference in case
  of nonzero means because the spatial version use E[xx*] as the variance
  instead of E[xx*] - E[x]E[x*].
  """

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      use_running_stats: Optional[bool] = None,
      weights: Optional[Array] = None,
  ) -> jnp.ndarray:
    """Normalizes the input using batch (optional) means and variances.

    Stats are computed over the batch and spherical dimensions: (0, 1, 2).

    Args:
      inputs: An array of dimensions (batch_size, ell_max+1, 2*ell_max+1,
        num_spins_in, num_channels_in).
      use_running_stats: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      weights: An array of dimensions (batch_size,) assigning weights for each
        batch element. Useful for masking.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    use_running_stats = nn.module.merge_param(
        "use_running_stats", self.use_running_stats, use_running_stats
    )

    # Normalization is independent per spin per channel.
    num_spins, num_channels = inputs.shape[-2:]
    feature_shape = (1, 1, 1, num_spins, num_channels)
    reduced_feature_shape = (num_spins, num_channels)

    initializing = not self.has_variable("batch_stats", "variance")

    running_variance = self.variable(
        "batch_stats",
        "variance",
        lambda s: jnp.ones(s, jnp.float32),
        reduced_feature_shape,
    )

    running_mean = self.variable(
        "batch_stats",
        "mean",
        lambda s: jnp.zeros(s, jnp.complex64),
        reduced_feature_shape,
    )

    if use_running_stats:
      variance = running_variance.value
      mean = running_mean.value
    else:
      # Mean is just the coefficients at ell=0.
      mean = jnp.average(inputs[:, 0], axis=0, weights=weights)

      # Variance is the norm of the ell > 0 coefficients.
      variance = jnp.sum(
          inputs[:, 1:] * inputs[:, 1:].conj(), axis=(1, 2)
      ).real / (4 * np.pi)
      variance = jnp.average(variance, axis=0, weights=weights)

      # Aggregate means over devices.
      if self.axis_name is not None and not initializing:
        mean = lax.pmean(mean, axis_name=self.axis_name)
        variance = lax.pmean(variance, axis_name=self.axis_name)

      if not initializing:
        running_variance.value = (
            self.momentum * running_variance.value
            + (1 - self.momentum) * variance
        )
        running_mean.value = (
            self.momentum * running_mean.value + (1 - self.momentum) * mean
        )

    outputs = inputs.at[:, 0].add(-mean)
    factor = lax.rsqrt(variance.reshape(feature_shape) + self.epsilon)
    scale = self.param("scale", self.scale_init, reduced_feature_shape).reshape(
        feature_shape
    )
    outputs = outputs * factor * scale

    bias = self.param("bias", self.bias_init, (1, num_channels))
    ell_max = outputs.shape[1] - 1

    idx_zero, _ = get_zero_nonzero_idx(self.spins)
    outputs = outputs.at[:, 0, ell_max, idx_zero].add(bias)

    return outputs


class SpinSphericalSpectralBatchNormPhaseCollapse(BatchNormAndActivation):
  """Applies spectral batch normalization and phase collapse."""

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      use_running_stats: Optional[bool] = None,
      weights: Optional[Array] = None,
  ) -> jnp.ndarray:
    """Takes spectral inputs and produces spatial outputs."""
    use_running_stats = nn.module.merge_param(
        "use_running_stats", self.use_running_stats, use_running_stats
    )

    # Apply spectral BN.
    feature_maps = SpinSphericalSpectralBatchNormalization(
        spins=self.spins,
        use_running_stats=use_running_stats,
        axis_name=self.axis_name,
        momentum=self.momentum,
        epsilon=self.epsilon,
        name="spectral_batch_norm")(inputs, weights=weights)

    # To spatial domain.
    batched_backward_transform = jax.vmap(
        self.transformer.swsft_backward_spins_channels, in_axes=(0, None))
    feature_maps = batched_backward_transform(
        feature_maps, self.spins)

    # Apply nonlinearity.
    return PhaseCollapseNonlinearity(
        spins=self.spins)(feature_maps)
