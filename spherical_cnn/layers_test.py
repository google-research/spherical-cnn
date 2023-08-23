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

"""Tests for spin_spherical_cnns.layers."""

import functools
from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from spherical_cnn import layers
from spherical_cnn import sphere_utils
from spherical_cnn import spin_spherical_harmonics
from spherical_cnn import test_utils
import tensorflow as tf


# Pseudo-random number generator keys to deterministically initialize
# parameters. The initialization could cause flakiness in the unlikely event
# that JAX changes its pseudo-random algorithm.
_JAX_RANDOM_KEY = np.array([0, 0], dtype=np.uint32)

TransformerModule = spin_spherical_harmonics.SpinSphericalFourierTransformer


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=(4, 8, 16), spins=(0, -1, 1, 2))


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(resolution=8,
                                 spins_in=[0], spins_out=[0, 1, 2]),
                            dict(resolution=8, spins_in=[0], spins_out=[0]),
                            dict(resolution=8, spins_in=[1], spins_out=[1]),
                            dict(resolution=8, spins_in=[0, 1], spins_out=[0]),
                            dict(resolution=8, spins_in=[0, 1], spins_out=[1],
                                 spectral_upsampling=True),
                            dict(resolution=16,
                                 spins_in=[0, 1], spins_out=[0, 1],
                                 spectral_pooling=True),
                            dict(resolution=16, spins_in=[0], spins_out=[0, 1]),
                            dict(resolution=8, spins_in=[1], spins_out=[1],
                                 input_representation="spectral"),
                            dict(resolution=16,
                                 spins_in=[0, 1], spins_out=[0, 1],
                                 spectral_pooling=True,
                                 input_representation="spectral"),
                            )
  def test_spin_spherical_convolution_is_equivariant(
      self, resolution, spins_in, spins_out,
      spectral_pooling=False, spectral_upsampling=False,
      input_representation="spatial"):
    """Tests the SO(3)-equivariance of _swsconv_spatial_spectral()."""
    transformer = _get_transformer()
    num_channels_in, num_channels_out = 2, 3
    # Euler angles.
    alpha, beta, gamma = 1.0, 2.0, 3.0
    shape = (1, resolution, resolution, len(spins_in), num_channels_in)
    pair = test_utils.get_rotated_pair(transformer,
                                       shape=shape,
                                       spins=spins_in,
                                       alpha=alpha,
                                       beta=beta,
                                       gamma=gamma)
    # Get rid of the batch dimension.
    if input_representation == "spectral":
      inputs = pair.coefficients[0]
      rotated_inputs = pair.rotated_coefficients[0]
    else:
      inputs = pair.sphere[0]
      rotated_inputs = pair.rotated_sphere[0]

    # Filter is defined by its spectral coefficients.
    if spectral_pooling:
      ell_max = resolution // 4 - 1
    else:
      ell_max = resolution // 2 - 1
    shape = [ell_max+1,
             len(spins_in), len(spins_out),
             num_channels_in, num_channels_out]
    # Make more arbitrary reproducible complex inputs.
    filter_coefficients = jnp.linspace(-0.5 + 0.2j, 0.2,
                                       np.prod(shape)).reshape(shape)

    # We need an aux class to keep flax state now that transformer is
    # a nn.Module.
    class SphericalConvolution(nn.Module):
      transformer: spin_spherical_harmonics.SpinSphericalFourierTransformer

      def __call__(self, *args, **kwargs):
        # self.transformer.validate(resolution, spins_in)
        return layers._spin_spherical_convolution(
            self.transformer, *args, **kwargs)

    conv = SphericalConvolution(transformer)

    args = (filter_coefficients, spins_in, spins_out)
    kwargs = dict(spectral_pooling=spectral_pooling,
                  spectral_upsampling=spectral_upsampling,
                  input_representation=input_representation,
                  output_representation="spatial")
    variables = conv.init(_JAX_RANDOM_KEY, inputs, *args, **kwargs)
    sphere_out = conv.apply(variables, inputs, *args, **kwargs)

    rotated_sphere_out = conv.apply(variables, rotated_inputs, *args, **kwargs)

    # Now since the convolution is SO(3)-equivariant, the same rotation that
    # relates the inputs must relate the outputs. We apply it spectrally.
    variables = transformer.init(_JAX_RANDOM_KEY)
    coefficients_out = transformer.apply(
        variables, sphere_out, spins_out,
        method=TransformerModule.swsft_forward_spins_channels)

    # This is R(f) * g (in the spectral domain).
    rotated_coefficients_out_1 = transformer.apply(
        variables, rotated_sphere_out, spins_out,
        method=TransformerModule.swsft_forward_spins_channels)

    # And this is R(f * g) (in the spectral domain).
    rotated_coefficients_out_2 = test_utils.rotate_coefficients(
        coefficients_out, alpha, beta, gamma)

    # There is some loss of precision on the Wigner-D computation for rotating
    # the coefficients, hence we use a slighly higher tolerance.
    self.assertAllClose(rotated_coefficients_out_1, rotated_coefficients_out_2,
                        atol=1e-5)


class SpinSphericalConvolutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(batch_size=2, resolution=8,
           spins_in=[0], spins_out=[0, 1, 2],
           n_channels_in=1, n_channels_out=3,
           num_filter_params=None),
      dict(batch_size=3, resolution=8,
           spins_in=[0, 1, 2], spins_out=[0],
           n_channels_in=3, n_channels_out=1,
           num_filter_params=2,
           spectral_upsampling=True),
      dict(batch_size=2, resolution=16,
           spins_in=[0], spins_out=[0, 1],
           n_channels_in=2, n_channels_out=3,
           num_filter_params=4,
           spectral_pooling=True),
      dict(batch_size=2, resolution=16,
           spins_in=[0], spins_out=[0, 1],
           n_channels_in=2, n_channels_out=3,
           num_filter_params=4,
           output_representation="spectral"),
      dict(batch_size=2, resolution=16,
           spins_in=[0, 1], spins_out=[0, 1],
           n_channels_in=2, n_channels_out=3,
           num_filter_params=4,
           spectral_pooling=True,
           input_representation="spectral",
           use_bias=True),
  )
  def test_shape(self,
                 batch_size,
                 resolution,
                 spins_in, spins_out,
                 n_channels_in, n_channels_out,
                 num_filter_params,
                 spectral_pooling=False,
                 spectral_upsampling=False,
                 input_representation="spatial",
                 output_representation="spatial",
                 use_bias=False):
    """Checks that SpinSphericalConvolution outputs the right shape."""
    transformer = _get_transformer()
    if input_representation == "spectral":
      ell_max = sphere_utils.ell_max_from_resolution(resolution)
      shape = (batch_size, ell_max+1, 2*ell_max+1, len(spins_in), n_channels_in)
    else:
      shape = (batch_size, resolution, resolution, len(spins_in), n_channels_in)
    inputs = (jnp.linspace(-0.5, 0.7 + 0.5j, np.prod(shape))
              .reshape(shape))

    model = layers.SpinSphericalConvolution(
        transformer=transformer,
        spins_in=spins_in,
        spins_out=spins_out,
        features=n_channels_out,
        num_filter_params=num_filter_params,
        spectral_pooling=spectral_pooling,
        spectral_upsampling=spectral_upsampling,
        input_representation=input_representation,
        output_representation=output_representation,
        use_bias=use_bias)
    params = model.init(_JAX_RANDOM_KEY, inputs)
    out = model.apply(params, inputs)

    if spectral_pooling:
      resolution = resolution // 2
    elif spectral_upsampling:
      resolution = resolution * 2
    if output_representation == "spectral":
      ell_max = sphere_utils.ell_max_from_resolution(resolution)
      expected_shape = (batch_size, ell_max+1, 2*ell_max+1,
                        len(spins_out), n_channels_out)
    else:
      expected_shape = (batch_size, resolution, resolution,
                        len(spins_out), n_channels_out)
    self.assertEqual(out.shape, expected_shape)

  @parameterized.parameters(False, True)
  def test_equivariance(self, spectral_pooling):
    resolution = 16
    spins = (0, -1, 2)
    transformer = _get_transformer()
    model = layers.SpinSphericalConvolution(transformer=transformer,
                                            spins_in=spins,
                                            spins_out=spins,
                                            features=2,
                                            spectral_pooling=spectral_pooling,
                                            spectral_upsampling=False)

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins)

    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-6)

  @parameterized.parameters(1, 2, 4)
  def test_localized_kernel_shape(self, num_filter_params):
    resolution = 16
    spins_in = (0, 1)
    spins_out = (0, 1, 2)
    num_channels_in = 3
    num_channels_out = 4
    transformer = _get_transformer()
    inputs = jnp.ones((2, resolution, resolution,
                       len(spins_in), num_channels_in))

    model = layers.SpinSphericalConvolution(
        transformer=transformer,
        spins_in=spins_in,
        spins_out=spins_out,
        features=num_channels_out,
        spectral_pooling=False,
        spectral_upsampling=False,
        num_filter_params=num_filter_params)
    params = model.init(_JAX_RANDOM_KEY, inputs)

    expected_shape = (len(spins_in), len(spins_out),
                      num_channels_in, num_channels_out,
                      num_filter_params)
    self.assertEqual(params["params"]["kernel"].shape, expected_shape)

  def test_localized_with_all_params_match_nonlocalized(self):
    """Check that filters with ell_max+1 params equal nonlocalized."""
    # We use fewer params to enforce a smooth spectrum for localized
    # filters. When the number of params is maximum (ell_max+1), no smoothness
    # is enforced and the filters match their nonlocalized counterparts.
    resolution = 16
    spins_in = (0, 1)
    spins_out = (0, 1, 2)
    n_channels_out = 4
    transformer = _get_transformer()
    inputs = jnp.ones((2, resolution, resolution, len(spins_in), 3))

    model = layers.SpinSphericalConvolution(transformer=transformer,
                                            spins_in=spins_in,
                                            spins_out=spins_out,
                                            features=n_channels_out,
                                            spectral_pooling=False,
                                            spectral_upsampling=False)
    params = model.init(_JAX_RANDOM_KEY, inputs)

    # The parameters for localized filters are transposed for performance
    # reasons. This custom initializer undoes the transposing so the init is the
    # same between localized and nonlocalized.
    def _transposed_initializer(key, shape, dtype=jnp.complex64):
      del dtype
      shape = [shape[-1], *shape[:-1]]
      weights = layers.spin_spherical_initializer(len(spins_in))(key, shape)
      return weights.transpose(1, 2, 3, 4, 0)

    ell_max = resolution // 2 - 1
    model_localized = layers.SpinSphericalConvolution(
        transformer=transformer,
        spins_in=spins_in,
        spins_out=spins_out,
        features=n_channels_out,
        spectral_pooling=False,
        spectral_upsampling=False,
        num_filter_params=ell_max + 1,
        initializer=_transposed_initializer)
    params_localized = model_localized.init(_JAX_RANDOM_KEY, inputs)

    self.assertAllClose(params["params"]["kernel"].transpose(1, 2, 3, 4, 0),
                        params_localized["params"]["kernel"])


# Default initialization is zero. We include bias so that some entries are
# actually rectified.
def _magnitude_nonlinearity_nonzero_initializer(*args, **kwargs):
  return -0.1 * nn.initializers.ones(*args, **kwargs)


class MagnitudeNonlinearityTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([(1, 8, 8, 3, 4)],
                            [(2, 4, 4, 5, 6)])
  def test_magnitude_thresholding(self, input_shape):
    small_row = 3
    inputs = jnp.ones(input_shape)
    inputs = inputs.at[:, small_row].set(0.1)

    model = layers.MagnitudeNonlinearity()
    params = model.init(_JAX_RANDOM_KEY, inputs)

    # With zero bias output must match input.
    bias = params["params"]["bias"].at[:].set(0.0)
    inputs_unchanged = model.apply(params, inputs)
    self.assertAllClose(inputs, inputs_unchanged)

    # We run again with bias = -0.2; now out[small_row] must be zero.
    bias_value = -0.2
    bias = params["params"]["bias"].at[:].set(bias_value)
    params_changed = flax.core.FrozenDict({"params": {"bias": bias}})
    inputs_changed = model.apply(params_changed, inputs)
    self.assertAllEqual(inputs_changed[:, small_row],
                        np.zeros_like(inputs[:, small_row]))
    # All other rows have the bias added.
    self.assertAllClose(inputs_changed[:, :small_row],
                        inputs[:, :small_row] + bias_value)
    self.assertAllClose(inputs_changed[:, small_row+1:],
                        inputs[:, small_row+1:] + bias_value)

  @parameterized.parameters(1, 5)
  def test_azimuthal_equivariance(self, shift):
    resolution = 8
    spins = (0, -1, 2)
    transformer = _get_transformer()

    model = layers.MagnitudeNonlinearity(
        bias_initializer=_magnitude_nonlinearity_nonzero_initializer)

    output_1, output_2 = test_utils.apply_model_to_azimuthally_rotated_pairs(
        transformer, model, resolution, spins, shift)

    self.assertAllClose(output_1, output_2)

  def test_equivariance(self):
    resolution = 16
    spins = (0, -1, 2)
    transformer = _get_transformer()

    model = layers.MagnitudeNonlinearity(
        bias_initializer=_magnitude_nonlinearity_nonzero_initializer)
    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins)
    # Tolerance needs to be high here due to approximate equivariance. We also
    # check the mean absolute error.
    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-1)
    self.assertLess(abs(coefficients_1 - coefficients_2).mean(), 5e-3)


def _evaluate_magnitudenonlinearity_versions(spins):
  """Evaluates MagnitudeNonlinearity and MagnitudeNonlinearityLeakyRelu."""
  transformer = _get_transformer()
  inputs, _ = test_utils.get_spin_spherical(transformer,
                                            shape=(2, 8, 8, len(spins), 2),
                                            spins=spins)
  model = layers.MagnitudeNonlinearity(
      bias_initializer=_magnitude_nonlinearity_nonzero_initializer)
  params = model.init(_JAX_RANDOM_KEY, inputs)
  outputs = model.apply(params, inputs)

  model_relu = layers.MagnitudeNonlinearityLeakyRelu(
      spins=spins,
      bias_initializer=_magnitude_nonlinearity_nonzero_initializer)
  params_relu = model_relu.init(_JAX_RANDOM_KEY, inputs)
  outputs_relu = model_relu.apply(params_relu, inputs)

  return inputs, outputs, outputs_relu


class MagnitudeNonlinearityLeakyReluTest(tf.test.TestCase):

  def test_spin0_matches_relu(self):
    """Zero spin matches real leaky_relu, others match MagnitudeNonlinearity."""
    spins = [0, -1, 2]
    inputs, outputs, outputs_relu = _evaluate_magnitudenonlinearity_versions(
        spins)

    self.assertAllEqual(outputs[..., 1:, :], outputs_relu[..., 1:, :])
    self.assertAllEqual(outputs_relu[..., 0, :],
                        nn.leaky_relu(inputs[..., 0, :].real))


class PhaseCollapseNonlinearityTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(spins=(0,)), dict(spins=(0, 1)))
  def test_shape(self, spins):
    model = layers.PhaseCollapseNonlinearity(spins)
    batch_size, resolution, num_channels = 2, 8, 3
    input_shape = (batch_size, resolution, resolution, len(spins), num_channels)
    inputs = jnp.ones(input_shape) + 1j * jnp.ones(input_shape)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, inputs)
    outputs = model.apply(params, inputs)
    self.assertEqual(
        outputs.shape,
        (batch_size, resolution, resolution, len(spins), num_channels),
    )

  @parameterized.parameters(
      dict(spins=(0,)), dict(spins=(0, 1)), dict(spins=(-1, 0, 1))
  )
  def test_only_spin_zero_changes(self, spins):
    model = layers.PhaseCollapseNonlinearity(spins)
    batch_size, resolution, num_channels = 2, 8, 3
    input_shape = (batch_size, resolution, resolution, len(spins), num_channels)
    inputs = jnp.linspace(-1.0, 2.0, np.prod(input_shape)) + 1j * jnp.linspace(
        0.5, -1.0, np.prod(input_shape)
    )
    inputs = inputs.reshape(input_shape)

    idx_zero, idx_nonzero = layers.get_zero_nonzero_idx(spins)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, inputs)
    outputs = model.apply(params, inputs)

    with self.subTest("Zero spin changes."):
      self.assertNotAllClose(
          outputs[..., idx_zero, :], inputs[..., idx_zero, :]
      )

    with self.subTest("Nonzero spins do not change."):
      self.assertAllClose(
          outputs[..., idx_nonzero, :], inputs[..., idx_nonzero, :]
      )


class SphericalPoolingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(4, 8)
  def test_constant_latitude_values(self, resolution):
    """Average for constant-latitude values tilts towards largest area."""
    inputs = jnp.zeros([2, resolution, resolution, 1, 1])
    first_latitude = 1
    second_latitude = 2
    inputs = inputs.at[:, 0].set(first_latitude)
    inputs = inputs.at[:, 1].set(second_latitude)

    model = layers.SphericalPooling(stride=2)
    params = model.init(_JAX_RANDOM_KEY, inputs)

    pooled = model.apply(params, inputs)
    # Since both the area and the value in the second band are larger than the
    # first, the output values should be larger than the unweighted average.
    unweighted = (first_latitude + second_latitude) / 2
    self.assertAllGreater(pooled[:, 0], unweighted)

    # Now we make the second value smaller, so average must be smaller than the
    # unweighted.
    second_latitude = 0
    inputs = inputs.at[:, 1].set(second_latitude)
    unweighted = (first_latitude + second_latitude) / 2
    pooled = model.apply(params, inputs)
    self.assertAllLess(pooled[:, 0], unweighted)

  @parameterized.parameters(dict(shift=2, stride=2),
                            dict(shift=4, stride=2),
                            dict(shift=4, stride=4))
  def test_azimuthal_equivariance(self, shift, stride):
    resolution = 16
    spins = (0, -1, 2)
    transformer = _get_transformer()
    model = layers.SphericalPooling(stride=stride)

    output_1, output_2 = test_utils.apply_model_to_azimuthally_rotated_pairs(
        transformer, model, resolution, spins, shift)

    self.assertAllClose(output_1, output_2)

  @parameterized.parameters(8, 16)
  def test_SphericalPooling_matches_spin_spherical_mean(self, resolution):
    """SphericalPooling with max stride must match spin_spherical_mean."""
    shape = [2, resolution, resolution, 3, 4]
    spins = [0, -1, 2]
    inputs, _ = test_utils.get_spin_spherical(_get_transformer(), shape, spins)
    spherical_mean = sphere_utils.spin_spherical_mean(inputs)

    model = layers.SphericalPooling(stride=resolution)
    params = model.init(_JAX_RANDOM_KEY, inputs)
    pooled = model.apply(params, inputs)

    # Tolerance here is higher because of slightly different quadratures.
    self.assertAllClose(spherical_mean, pooled[:, 0, 0], atol=1e-3)


def _batched_spherical_variance(inputs):
  """Computes variances over the sphere and batch dimensions."""
  # Assumes mean=0 as in SpinSphericalBatchNormalization.
  return sphere_utils.spin_spherical_mean(inputs * inputs.conj()).mean(axis=0)


class SphericalBatchNormalizationTest(tf.test.TestCase,
                                      parameterized.TestCase):

  def test_output_and_running_variance(self):
    momentum = 0.9
    input_shape = (2, 6, 5, 4, 3, 2)
    real, imaginary = jax.random.normal(_JAX_RANDOM_KEY, input_shape)
    inputs = real + 1j * imaginary

    model = layers.SphericalBatchNormalization(momentum=momentum,
                                               use_running_stats=False,
                                               use_bias=False,
                                               centered=False)

    # Output variance must be one.
    output, initial_params = model.init_with_output(_JAX_RANDOM_KEY, inputs)
    output_variance = _batched_spherical_variance(output)
    with self.subTest(name="OutputVarianceIsOne"):
      self.assertAllClose(output_variance, jnp.ones_like(output_variance),
                          atol=1e-5)

    output, variances = model.apply(initial_params, inputs,
                                    mutable=["batch_stats"])
    # Running variance is between variance=1 and variance of input.
    input_variance = _batched_spherical_variance(inputs)
    momentum_variance = momentum * 1.0 + (1.0 - momentum) * input_variance
    with self.subTest(name="RunningVariance"):
      self.assertAllClose(momentum_variance,
                          variances["batch_stats"]["variance"])

  @parameterized.parameters(False, True)
  def test_equivariance(self, train):
    resolution = 16
    spins = (0, -1, 2)
    transformer = _get_transformer()
    model = layers.SphericalBatchNormalization(use_bias=False,
                                               centered=False)
    init_args = dict(use_running_stats=True)
    apply_args = dict(use_running_stats=not train, mutable=["batch_stats"])

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins,
        init_args=init_args,
        apply_args=apply_args)

    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-6)


class SpinSphericalBatchNormalizationTest(tf.test.TestCase,
                                          parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_equivariance(self, train):
    resolution = 16
    spins = (0, 1)
    transformer = _get_transformer()
    model = layers.SpinSphericalBatchNormalization(spins=spins)
    init_args = dict(use_running_stats=True)
    apply_args = dict(use_running_stats=not train, mutable=["batch_stats"])

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins,
        init_args=init_args,
        apply_args=apply_args)

    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-5)


class SpinSphericalBatchNormPhaseCollapseTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.parameters(
      dict(train=False, shift=3), dict(train=True, shift=2)
  )
  def test_azimuthal_equivariance(self, train, shift):
    resolution = 8
    spins = (0, 1, 2)
    transformer = _get_transformer()
    model = layers.SpinSphericalBatchNormPhaseCollapse(spins=spins)
    init_args = dict(use_running_stats=True)
    apply_args = dict(use_running_stats=not train, mutable=["batch_stats"])

    output_1, output_2 = test_utils.apply_model_to_azimuthally_rotated_pairs(
        transformer,
        model,
        resolution,
        spins,
        init_args=init_args,
        apply_args=apply_args,
        shift=shift,
    )

    self.assertAllClose(output_1, output_2)

  @parameterized.parameters(False, True)
  def test_equivariance(self, train):
    resolution = 16
    spins = (0, 1)
    transformer = _get_transformer()
    model = layers.SpinSphericalBatchNormPhaseCollapse(spins=spins)
    init_args = dict(use_running_stats=True)
    apply_args = dict(use_running_stats=not train, mutable=["batch_stats"])

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer,
        model,
        resolution,
        spins,
        init_args=init_args,
        apply_args=apply_args,
    )

    # Tolerance needs to be high here due to the approximate equivariance of the
    # pointwise operation under equiangular sampling.
    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-1)
    self.assertLess(abs(coefficients_1 - coefficients_2).mean(), 5e-3)


class SpinSphericalSpectralBatchNormalizationTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_spectral_matches_spatial(self):
    resolution = 16
    shape = (4, resolution, resolution, 2, 3)
    spins = (0, 1)
    transformer = _get_transformer()
    sphere, coefficients = test_utils.get_spin_spherical(
        transformer, shape, spins
    )
    # Ensure mean zero for spin 0 so spatial matches spectral:
    sphere_mean = sphere_utils.spin_spherical_mean(sphere[..., [0], :])
    sphere = sphere.at[..., [0], :].add(-jnp.expand_dims(sphere_mean, (1, 2)))
    coefficients = coefficients.at[:, 0].set(0.0)

    key = jax.random.PRNGKey(0)

    spatial_module = layers.SpinSphericalBatchNormalization(
        spins=spins, use_running_stats=False
    )
    spatial_params = spatial_module.init(key, sphere)

    spectral_module = layers.SpinSphericalSpectralBatchNormalization(
        spins=spins, use_running_stats=False
    )

    spectral_params = spectral_module.init(key, coefficients)

    spatial_output, _ = spatial_module.apply(
        spatial_params, sphere, mutable=["batch_stats"]
    )
    spectral_output, _ = spectral_module.apply(
        spectral_params, coefficients, mutable=["batch_stats"]
    )

    backward_transform = jax.vmap(
        functools.partial(
            transformer.apply,
            method=TransformerModule.swsft_backward_spins_channels,
        ),
        in_axes=(None, 0, None),
    )

    variables = transformer.init(jax.random.PRNGKey(0))

    self.assertAllClose(
        backward_transform(variables, spectral_output, spins), spatial_output
    )


if __name__ == "__main__":
  tf.test.main()
