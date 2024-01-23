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

"""Tests for sphere_utils."""

import functools
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from spherical_cnn import sphere_utils
from spherical_cnn import spin_spherical_harmonics
from spherical_cnn import test_utils
import sympy.physics.wigner
import tensorflow as tf


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=(4, 8, 16), spins=(0, -1, 1, 2))


class SphereUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(4, 8)
  def test_make_equiangular_grid_contains_both_poles(self, resolution):
    """Equiangular grid must contain both poles."""
    _, colatitude = sphere_utils.make_equiangular_grid(resolution)
    with self.subTest("North pole"):
      self.assertAllClose(colatitude[0], np.zeros_like(colatitude[0]))
    with self.subTest("South pole"):
      self.assertAllClose(colatitude[-1], np.full_like(colatitude[-1], np.pi))

  @parameterized.parameters(4, 8, 16, 32)
  def test_spherical_cell_area_sum(self, resolution):
    """Sum of spherical cell areas must match spherical surface area."""
    areas = sphere_utils.sphere_quadrature_weights(resolution)
    self.assertAllClose(areas.sum() * resolution, 4*np.pi)

  def test_sphere_quadrature_weights_2x2(self):
    """In a 2x2 discretization, all areas are equal."""
    areas = sphere_utils.sphere_quadrature_weights(2)
    self.assertAllClose(areas,
                        np.ones_like(areas) * np.pi)

  @parameterized.parameters(8, 16)
  def test_spin_spherical_mean(self, resolution):
    """Check that spin_spherical_mean is equivariant and Parseval holds."""
    transformer = _get_transformer()
    spins = (0, 1, -1, 2)
    shape = (2, resolution, resolution, len(spins), 2)
    alpha, beta, gamma = 1.0, 2.0, 3.0
    pair = test_utils.get_rotated_pair(transformer,
                                       shape,
                                       spins,
                                       alpha, beta, gamma)

    # Mean should be zero for spin != 0 so we compare the squared norm.
    abs_squared = lambda x: x.real**2 + x.imag**2
    norm = sphere_utils.spin_spherical_mean(abs_squared(pair.sphere))
    rotated_norm = sphere_utils.spin_spherical_mean(
        abs_squared(pair.rotated_sphere))
    with self.subTest(name="Equivariance"):
      self.assertAllClose(norm, rotated_norm)

    # Compute energy of coefficients and check that Parseval's theorem holds.
    coefficients_norm = jnp.sum(abs_squared(pair.coefficients), axis=(1, 2))
    with self.subTest(name="Parseval"):
      self.assertAllClose(norm * 4 * np.pi, coefficients_norm)

  @parameterized.parameters(4, 6, 8, 16, 20, 32)
  def test_torus_quadrature_weights_curve(self, resolution):
    """Checks that quadrature weights follow the curve in H&W, Figure 5.

    The first half of the weights corresponds to the original spherical function
    and the values resemble the naive sin(colatitude) quadrature rule: small
    near poles, max near equator. The second half consists of the extension to
    torus and has small weights.

    Args:
      resolution: int, original spherical resolution.

    Returns:
      None.
    """
    weights = sphere_utils.torus_quadrature_weights(resolution)
    # The first part of the weights has an increasing-decreasing pattern.
    increasing = np.diff(weights[:resolution]) > 0
    self.assertTrue(increasing[:resolution // 2 - 1].all())
    self.assertFalse(increasing[resolution // 2 - 1:].all())
    # The second part is the extension and has much lower weights.
    self.assertGreater(weights[:resolution].sum(),
                       weights[resolution:].sum())
    # The weights must sum up to 2, as the integral of sin(x) from 0 to pi.
    self.assertAllClose(weights.sum(), 2.)

  @parameterized.parameters((4, 0),
                            (4, 4),
                            (8, 0),
                            (8, 1))
  def test_compute_all_wigner_delta_matches_single(self, ell_max, ell):
    wigner_deltas = sphere_utils.compute_all_wigner_delta(ell_max)
    wigner_delta = sphere_utils.compute_wigner_delta(ell)
    self.assertAllClose(wigner_deltas[ell], wigner_delta)

  @parameterized.parameters(8, 16, 32)
  def test_compute_wigner_delta_matches_sympy(self, ell):
    """Wigner deltas must match up to ell==32."""
    wigner_delta_sympy = sympy.physics.wigner.wigner_d_small(ell, np.pi / 2)
    wigner_delta_sympy = np.array(wigner_delta_sympy).astype(np.float64)
    wigner_delta = sphere_utils.compute_wigner_delta(ell)
    self.assertAllClose(wigner_delta, wigner_delta_sympy)


if __name__ == "__main__":
  tf.test.main()
