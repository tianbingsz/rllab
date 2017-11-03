

from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
# from rllab.algo.first_order_method import parse_update_method
from rllab.optimizers.minibatch_dataset import BatchDataset
from collections import OrderedDict
import tensorflow as tf
import time
from numpy import linalg as LA
from functools import partial
import pyprind
import pdb


class SteinOptimizer(Serializable):
    """
    Performs parametric stein gradient descent.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1000,
            tolerance=1e-6,
            batch_size=32,
            max_batch=10,
            alpha=0.1,
            callback=None,
            verbose=False,
            **kwargs):
        """

        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose
        self._input_vars = None
        self._train_op = None
        self._learning_rate = learning_rate
        self._max_batch = max_batch
        self._alpha = alpha

        logger.log('max_batch %d' % (self._max_batch))

    def update_opt(self, loss, target, logstd, inputs,
                   extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        self._log_std = tf.reduce_mean(logstd)

        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs

        # \partial{log \pi} / \partial{\phi} A
        # \phi is the mean_network parameters
        # pdb.set_trace()
        mean_w = target.get_mean_network().get_params(trainable=True)
        grads = tf.gradients(loss,
                             xs=target.get_mean_network().get_params(trainable=True))
        for idx, (g, param) in enumerate(zip(grads, mean_w)):
            if g is None:
                grads[idx] = tf.zeros_like(param)
        flat_grad = tensor_utils.flatten_tensor_variables(grads)

        # \sum_d \partial{logstd^d} / \partial{\phi}
        # \phi is the std_network parameters
        var_grads = tf.gradients(loss - self._alpha * self._log_std,
                                 xs=target.get_std_network().get_params(trainable=True)
                                 )
        var_w = target.get_std_network().get_params(trainable=True)
        for idx, (g, param) in enumerate(zip(var_grads, var_w)):
            if g is None:
                var_grads[idx] = tf.zeros_like(param)
        flat_var_grad = tensor_utils.flatten_tensor_variables(var_grads)

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs + extra_inputs, loss),
            f_grad=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
            ),
            f_var_grad=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_var_grad,
            ),
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(tuple(inputs) + extra_inputs))

    def optimize(self, inputs, extra_inputs=None, callback=None):
        if len(inputs) == 0:
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]
        f_grad = self._opt_fun["f_grad"]
        f_var_grad = self._opt_fun["f_var_grad"]

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        dataset = BatchDataset(
            inputs,
            self._batch_size,
            extra_inputs=extra_inputs)

        mean_w = self._target.get_mean_network().get_param_values(trainable=True)
        var_w = self._target.get_std_network().get_param_values(trainable=True)
        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))

            num_batch = 0
            loss = f_loss(*(tuple(inputs)) + extra_inputs)
            while num_batch < self._max_batch:
                batch = dataset.random_batch()
                g = f_grad(*(batch))
                # w = w - \eta g
                # pdb.set_trace()
                mean_w = mean_w - self._learning_rate * g
                self._target.get_mean_network().set_param_values(mean_w,
                                                                 trainable=True)

                # pdb.set_trace()
                g_var = f_var_grad(*(batch))
                var_w = var_w - self._learning_rate * g_var
                self._target.get_std_network().set_param_values(var_w,
                                                                trainable=True)

                new_loss = f_loss(*(tuple(inputs) + extra_inputs))
                print("mean: batch {:} grad {:}, weight {:}".format(
                    num_batch, LA.norm(g), LA.norm(mean_w)))
                print("var: batch {:}, loss {:}, diff loss{:}, grad {:}, weight {:}".format(
                    num_batch, new_loss, new_loss - loss, LA.norm(g_var), LA.norm(var_w)))
                loss = new_loss
                num_batch += 1
