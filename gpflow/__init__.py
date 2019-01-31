# Copyright 2016 alexggmatthews, James Hensman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# flake8: noqa


import tensorflow as tf

from . import (conditionals, expectations, features, kernels, likelihoods,
               logdensities, models, probability_distributions, util)

from .training import optimize
from ._settings import SETTINGS as settings
from ._version import __version__
from .base import Parameter, Module, positive, triangular

# tf.enable_eager_execution()
