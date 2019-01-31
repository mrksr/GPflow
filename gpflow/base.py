import itertools as it
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .util import default_float


positive = tfp.bijectors.Softplus
triangular = tfp.bijectors.FillTriangular


class Parameter:
    def __init__(
            self,
            data=None,
            unconstrained_tensor=None,
            dtype=None,
            name=None,
            prior=None,
            transform=None,
            trainable=True,
    ):
        if data is None and unconstrained_tensor is None:
            raise ValueError('You need to pass either initial data or an unconstrained tensor.')

        if tf.contrib.framework.is_tensor(data):
            unconstrained_tensor = data
            transform = None
            trainable = False

        if name is None:
            name = 'unnamed'

        if transform is None:
            transform = tfp.bijectors.Identity()
        self.transform = transform

        self.prior = prior

        if unconstrained_tensor is None:
            initial_value = self.transform.inverse(tf.constant(data, dtype=dtype))

            unconstrained_tensor = tf.get_variable(
                name,
                dtype=dtype,
                initializer=initial_value,
            )
        self.unconstrained_tensor = unconstrained_tensor
        self.constrained_tensor = self.transform.forward(unconstrained_tensor, name=f'{name}_constrained')

        self.trainable = trainable

    @property
    def constrained(self):
        return self.constrained_tensor

    def __call__(self):
        return self.constrained

    def log_prior(self):
        if self.prior is None:
            return tf.constant(0., dtype=self.unconstrained_tensor.dtype)

        log_prob = self.prior.log_prob(self.constrained_tensor)
        log_det_jacobian = bijector.forward_log_det_jacobian(
            self.unconstrained_tensor,
            self.unconstrained_tensor.shape.ndims,
        )
        return log_prob + log_det_jacobian


class Module(object):
    def __init__(self):
        self._modules = dict()
        self._parameters = dict()

    @property
    def parameters(self):
        return list(it.chain(
            self._parameters.values(),
            it.chain.from_iterable(m.parameters for m in self._modules.values())
        ))

    @property
    def variables(self):
        return [parameter.unconstrained_tensor for parameter in self.parameters]

    @property
    def trainable_variables(self):
        return [v for v in self.variables if v.trainable]

    @property
    def trainable(self):
        return self.trainable_variables != []

    @trainable.setter
    def trainable(self, value: bool):
        for variable in self.variables:
            variable.trainable = value

    def __getattr__(self, name):
        if name in self.__dict__['_parameters']:
            return self.__dict__['_parameters'].get(name)
        elif name in self._modules:
            return self._modules[name]
        raise AttributeError

    def __setattr__(self, name, value):
        parameters = self.__dict__.get('_parameters')
        is_parameter = isinstance(value, (Parameter,))
        if is_parameter and parameters is None:
            raise AttributeError()
        if parameters is not None and name in parameters and not is_parameter:
            raise AttributeError()
        if is_parameter and parameters is not None:
            parameters[name] = value
            return

        modules = self.__dict__.get('_modules')
        is_module = isinstance(value, (Module, ModuleList))
        if is_module and modules is None:
            raise AttributeError()
        if modules is not None and name in modules and not is_module:
            raise AttributeError()
        if is_module and modules is not None:
            self._modules[name] = value
            return

        super().__setattr__(name, value)


class ModuleList(object):
    def __init__(self, modules):
        self._modules = modules

    @property
    def variables(self):
        module_variables = sum([m.variables for m in self._modules], [])
        return module_variables

    @property
    def trainable_variables(self):
        return [v for v in self.variables if v.trainable]

    def __len__(self):
        return len(self._modules)

    def append(self, module):
        self._modules.append(module)

    def __getitem__(self, index: int):
        return self._modules[index]

    def __setitem__(self, index: int, module):
        self._modules[index] = module
