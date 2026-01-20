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

"""Datasets statistics."""
import numpy as np


# Levels: (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
KEISLER22_PREDICTORS_MEAN = [
    # geopotential (0-12)
    200922.48953632, 159688.77494127, 135525.32741233, 117780.15471335,
    103569.86983515, 91567.72353888, 71720.93472939, 55504.79399355,
    41743.34173739, 29764.1839874, 14246.96566059, 7359.49653261, 934.93505382,
    # specific_humidity (13-25)
    2.70009009e-06, 2.69811931e-06, 6.43019810e-06, 2.58731720e-05,
    7.69643285e-05, 1.68000138e-04, 4.98091298e-04, 1.09909173e-03,
    1.99299043e-03, 3.15974506e-03, 6.02511843e-03, 7.99905837e-03,
    9.36163649e-03,
    # temperature (26-38)
    211.4923524, 204.8614168, 211.44848296, 218.49464795,
    225.59929419, 233.27215095, 247.487119, 258.46966558,
    266.80689765, 273.63567221, 281.09814175, 284.11381266, 288.31473935,
    # u_component_of_wind (39-51)
    3.206962, 10.18896407, 14.82815122, 15.71390932, 14.53843907,
    12.66605118, 9.20702252, 6.66528349, 4.77006387, 3.21553266,
    1.07513599, 0.18494731, -0.40246055,
    # v_component_of_wind (52-64)
    -0.00212845, 0.01765841, -0.05272861, -0.06710839, -0.04161598,
    -0.02478786, -0.01758257, -0.03257019, -0.04875721, -0.0163738,
    0.08529413, 0.17911138, 0.1811434,
    # vertical_velocity (65-77)
    5.85338678e-09, -1.90666995e-09, -1.17553189e-07, -4.34586017e-07,
    9.40738522e-07, 1.28441776e-06, 1.66364090e-06, -2.32851353e-06,
    1.61462302e-04, 8.88569999e-04, 4.64696695e-03, 7.56842361e-03,
    9.03846917e-03,
    # toa_incident_solar_radiation (78-79)
    0.0,
    # constants
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

KEISLER22_PREDICTORS_STD = [
    # geopotential
    3850.37968969, 4104.14003328, 4713.80316802, 4834.00810963,
    4605.55712732, 4217.3640495, 3407.86462919, 2735.75048692,
    2194.55288145, 1743.61756413, 1204.82552548, 1016.35363927, 900.4348462,
    # specific_humidity
    2.47705458e-07, 6.03376808e-07, 4.05595015e-06, 2.46166250e-05,
    8.20741209e-05, 1.87097077e-04, 5.66015536e-04, 1.20345386e-03,
    1.94211127e-03, 2.73246202e-03, 4.17408046e-03, 4.97445915e-03,
    5.78012839e-03,
    # temperature
    7.54188383, 11.47284544, 7.45846043, 5.29916486, 7.28008841,
    9.36649347, 10.88156901, 10.94353044, 10.90634071, 11.47755764,
    12.3265782, 12.75101061, 13.3843738,
    # u_component_of_wind
    14.97316412, 14.08450765, 17.23371765, 18.74000306, 18.54566777,
    17.2956507, 14.32534274, 11.98475486, 10.32328804, 9.14042181,
    8.12144637, 7.857296, 6.0874979,
    # v_component_of_wind
    5.68901966, 7.07634827, 9.75259499, 12.06394908, 13.07590478,
    12.63410062, 10.40650109, 8.46989169, 7.18292939, 6.3223581,
    5.79056928, 6.09763365, 5.06463825,
    # vertical_velocity
    0.01102607, 0.02351536, 0.04931006, 0.07947895, 0.10928052,
    0.1366633, 0.17474616, 0.18867781, 0.19150509, 0.19330101,
    0.17754056, 0.14144712, 0.08993197,
    # toa_incident_solar_radiation
    # Average solar radiation is 340W/m2. Dataset contains the energy
    # accumulated over 3h, so we normalize by the average energy over 3h.
    340 * 3600 * 3,
    # constants
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]

KEISLER22_DIFFERENCES_MEAN = [
    # geopotential
    -0.26591959, -0.17487696, -0.09415842, 0.08635633,
    0.25448228, 0.34333454, 0.36610768, 0.33214816,
    0.28648654, 0.26792981, 0.22587543, 0.07019159, -0.18073245,
    # specific_humidity
    1.43936581e-11, -5.84975604e-11, -2.90292185e-10, -6.66831268e-09,
    -2.87848509e-08, -4.66804735e-08, -1.27457202e-07, -3.02175153e-07,
    -3.22904310e-07, 1.71514109e-07, 1.83652249e-06, -1.08231742e-06,
    -2.36562096e-06,
    # temperature
    -0.00134389, -0.0005436, -0.00139151, -0.00268508,
    -0.00228942, -0.00103274, 0.00030122, 0.00084804,
    0.00078017, 0.00024379, 0.00317602, 0.01284331, 0.01911319,
    # u_component_of_wind
    -6.10951407e-03, -3.20943102e-03, 3.72204692e-04, -2.82389255e-03,
    -2.71747255e-03, -7.63339796e-04, -1.88660656e-04, -6.42554662e-04,
    9.12488600e-05, -7.00935200e-05, -5.42975647e-04, -3.84699935e-03,
    -2.85624451e-03,
    # v_component_of_wind
    0.00200444, 0.00319737, -0.00012477, -0.00123282,
    -0.00087224, -0.00096965, -0.0001262, 0.00074807,
    0.00096707, 0.00055797, 0.00102997, 0.00197329, 0.00120036,
    # vertical_velocity
    9.11668651e-08, -2.89434419e-07, -1.20184598e-06, -8.11465196e-07,
    -2.94753246e-07, -2.49784647e-08, -1.14913750e-06, -1.34985770e-06,
    -2.97561714e-06, -2.47290910e-05, -1.20832247e-04, -2.27861870e-04,
    -3.27192259e-04,
]

KEISLER22_DIFFERENCES_STD = [
    # geopotential
    214.20755858, 194.87469908, 216.35815883, 257.05705481,
    295.89356742, 306.09089361, 271.76399009, 227.95018849,
    198.07530672, 183.07455266, 183.11044623, 192.42403269, 204.33494944,
    # specific_humidity
    3.51138138e-08, 1.67762625e-07, 1.54536475e-06, 9.98854434e-06,
    3.46021968e-05, 7.48609712e-05, 2.16263073e-04, 4.21374778e-04,
    6.29298125e-04, 8.54095193e-04, 1.07629128e-03, 8.69433387e-04,
    7.34332408e-04,
    # temperature
    1.35752593, 1.11068324, 1.10612645, 1.44796213,
    1.35147061, 1.12685362, 1.22855886, 1.28253325,
    1.2513054, 1.22366681, 1.47208737, 1.64758503, 1.83044582,
    # u_component_of_wind
    2.69328518, 2.76646248, 3.28609124, 4.19776566,
    5.03991432, 5.21835962, 4.46984768, 3.58577347,
    3.02558072, 2.73961524, 2.75259476, 2.87694837, 2.19780391,
    # v_component_of_wind
    3.00145457, 3.06535853, 3.71521388, 4.98304028,
    6.21295548, 6.52727218, 5.60945117, 4.44941163,
    3.66542571, 3.22776516, 3.17064472, 3.36299301, 2.5381094,
    # vertical_velocity
    0.01393267, 0.02718912, 0.05303797, 0.08026392,
    0.10699478, 0.13546318, 0.17913779, 0.19579563,
    0.20007359, 0.20026567, 0.17436137, 0.12678325, 0.06542599,
]

# We want each input to be standardized with statistics computed over all
# pressure levels, so we unify KEISLER22_PREDICTORS_{MEAN, STD}.
# We want our predictions (differences) to be standardized, so we apply
# KEISLER22_DIFFERENCES_{MEAN, STD} to them and do not unify. These need to be
# corrected by KEISLER22_PREDICTORS_STD since they were computed in
# non-standardized but are applied to standardized data.
# The loss is on standardized deltas, which we will scale by the variance of the
# deltas.


# Unify stds at every height so that an error counts the same at every level.
# This makes a lot of sense for wind since vertical is noisy and not very
# informative.
def _unify_std(x):
  return [np.sqrt(np.mean([x_i**2 for x_i in x]))] * len(x)

# Geopotential.
KEISLER22_PREDICTORS_STD[:13] = _unify_std(KEISLER22_PREDICTORS_STD[:13])
# Humidity.
KEISLER22_PREDICTORS_STD[13:26] = _unify_std(KEISLER22_PREDICTORS_STD[13:26])
# Temperature.
KEISLER22_PREDICTORS_STD[26:39] = _unify_std(KEISLER22_PREDICTORS_STD[26:39])
# All wind components get the same factor.
KEISLER22_PREDICTORS_STD[39:78] = _unify_std(KEISLER22_PREDICTORS_STD[39:78])

# Skip constants. This considers a single step roll-out, needs to be tiled in
# case of more.
KEISLER22_TARGETS_MEAN = KEISLER22_PREDICTORS_MEAN[:78]
KEISLER22_TARGETS_STD = KEISLER22_PREDICTORS_STD[:78]

KEISLER22_UNIFIED_DIFFERENCES_STD = [0.0 for _ in range(78)]
# Geopotential.
KEISLER22_UNIFIED_DIFFERENCES_STD[:13] = _unify_std(
    KEISLER22_DIFFERENCES_STD[:13])
# Humidity.
KEISLER22_UNIFIED_DIFFERENCES_STD[13:26] = _unify_std(
    KEISLER22_DIFFERENCES_STD[13:26])
# Temperature.
KEISLER22_UNIFIED_DIFFERENCES_STD[26:39] = _unify_std(
    KEISLER22_DIFFERENCES_STD[26:39])
# All wind components get the same factor.
KEISLER22_UNIFIED_DIFFERENCES_STD[39:78] = _unify_std(
    KEISLER22_DIFFERENCES_STD[39:78])

KEISLER22_UNIFIED_DIFFERENCES_STD = [
    x / std
    for x, std in zip(KEISLER22_UNIFIED_DIFFERENCES_STD,
                      KEISLER22_TARGETS_STD)]

# Correct differences by std of targets:
KEISLER22_DIFFERENCES_MEAN = [x / std
                              for x, std in zip(KEISLER22_DIFFERENCES_MEAN,
                                                KEISLER22_TARGETS_STD)]
KEISLER22_DIFFERENCES_STD = [x / std
                             for x, std in zip(KEISLER22_DIFFERENCES_STD,
                                               KEISLER22_TARGETS_STD)]

KEISLER22_STATS = {
    'predictors_mean': np.array(KEISLER22_PREDICTORS_MEAN, dtype=np.float32),
    'predictors_std': np.array(KEISLER22_PREDICTORS_STD, dtype=np.float32),
    'targets_mean': np.array(KEISLER22_TARGETS_MEAN, dtype=np.float32),
    'targets_std': np.array(KEISLER22_TARGETS_STD, dtype=np.float32),
    'differences_mean': np.array(KEISLER22_DIFFERENCES_MEAN, dtype=np.float32),
    'differences_std': np.array(KEISLER22_DIFFERENCES_STD, dtype=np.float32),
    'unified_differences_std': np.array(KEISLER22_UNIFIED_DIFFERENCES_STD,
                                        dtype=np.float32),
}

KEISLER22_SPIN1_IDX = {
    'u': tuple(range(39, 52)),
    'v': tuple(range(52, 65)),
}
