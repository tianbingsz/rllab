from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
from rllab.optimizers.minibatch_dataset import BatchDataset
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import PerlmutterHvp
# from rllab.misc.ext import flatten_tensor_variables
import itertools
import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.ext import sliced_fun
import pdb


class SteinOptimizer(Serializable):
    """
    TRPO with Stein inference
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            alpha=0.0,
            max_epochs=1,
            subsample_factor=1.,
            backtrack_ratio=0.5,
            max_backtracks=10,
            debug_nan=False,
            accept_violation=False,
            hvp_approach=None,
            num_slices=1):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param debug_nan: if set to True, NanGuard will be added to the compilation, and ipdb will be invoked when
        nan is detected
        :param accept_violation: whether to accept the descent step if it violates the line search condition after
        exhausting all backtracking budgets
        :return:
        """
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._num_slices = num_slices

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._debug_nan = debug_nan
        self._accept_violation = accept_violation
        self._mean_hvp = PerlmutterHvp(num_slices)
        self._var_hvp = PerlmutterHvp(num_slices)

        self._alpha = alpha
        self._subsample_factor = subsample_factor
        self._max_epochs = max_epochs
        logger.log('temperature alpha %f' % (self._alpha))

    def update_opt(self, loss, target, logstd,
                   leq_constraint, inputs, extra_inputs=None,
                   constraint_name="constraint", *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
        that the first dimension of these inputs should correspond to the number of data points
        :param extra_inputs: A list of symbolic variables as extra inputs which should not be subsampled
        :return: No return value.
        """

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        self._target = target
        self._log_std = tf.reduce_mean(logstd)
        constraint_term, constraint_value = leq_constraint
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        # \partial{log \pi} / \partial{\phi} A
        # \phi is the mean_network parameters
        mean_w = target.get_mean_network().get_params(trainable=True)
        grads = tf.gradients(loss, xs=mean_w)
        for idx, (grad, param) in enumerate(zip(grads, mean_w)):
            if grad is None:
                grads[idx] = tf.zeros_like(param)
        flat_grad = tensor_utils.flatten_tensor_variables(grads)
        self._mean_hvp.update_opt(f=constraint_term,
                                  target=target.get_mean_network(),
                                  inputs=inputs + extra_inputs,
                                  reg_coeff=self._reg_coeff)

        # \sum_d \partial{logstd_d} / \partial{\phi}
        # \phi is the std_network parameters
        var_w = target.get_std_network().get_params(trainable=True)
        # loss = - tf.reduce_mean(logli * advantage_var)
        var_grads = tf.gradients(loss - self._alpha * self._log_std, xs=var_w)
        for idx, (g, param) in enumerate(zip(var_grads, var_w)):
            if g is None:
                var_grads[idx] = tf.zeros_like(param)
        flat_var_grad = tensor_utils.flatten_tensor_variables(var_grads)

        self._var_hvp.update_opt(f=constraint_term,
                                 target=target.get_std_network(),
                                 inputs=inputs + extra_inputs,
                                 reg_coeff=self._reg_coeff)

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                log_name="f_loss",
            ),
            f_grad=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            f_var_grad=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_var_grad,
            ),
            f_constraint=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term,
                log_name="constraint",
            ),
            f_loss_constraint=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term],
                log_name="f_loss_constraint",
            ),
        )

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)
        return self._opt_fun["f_loss"](*(inputs + extra_inputs))

    def optimize(self, inputs, extra_inputs=None,
                 subsample_grouped_inputs=None):

        if len(inputs) == 0:
            raise NotImplementedError

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        param = np.copy(self._target.get_param_values(
            trainable=True))
        logger.log("Start CG optimization: #parameters: %d, #inputs: %d, #subsample_inputs: %d"
                   % (len(param), len(inputs[0]), self._subsample_factor * len(inputs[0])))

        if self._subsample_factor < 1:
            if subsample_grouped_inputs is None:
                subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self._subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        logger.log("performing update")

        logger.log("computing gradient")
        flat_g = sliced_fun(
            self._opt_fun["f_grad"],
            self._num_slices)(
            inputs,
            extra_inputs)
        flat_var_g = sliced_fun(
            self._opt_fun["f_var_grad"],
            self._num_slices)(
            inputs,
            extra_inputs)
        logger.log("gradient computed")

        logger.log("computing descent direction")
        mean_Hx = self._mean_hvp.build_eval(subsample_inputs + extra_inputs)
        var_Hx = self._var_hvp.build_eval(subsample_inputs + extra_inputs)

        # update mean network parameters
        self._target_network = self._target.get_mean_network()
        self.conjugate_grad(flat_g, mean_Hx, inputs, extra_inputs)
        # update var network parameters
        self._target_network = self._target.get_std_network()
        self.conjugate_grad(flat_var_g, var_Hx, inputs, extra_inputs)

    def conjugate_grad(self, deltaW, Hx, inputs, extra_inputs=()):
        # s = H^-1 g
        descent_direction = krylov.cg(Hx, deltaW, cg_iters=self._cg_iters)
        # s' H s = g' s, as s = H^-1 g
        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val *
            (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")
        self.line_search(flat_descent_step, inputs, extra_inputs)

    def line_search(self, descent_step, inputs, extra_inputs=()):
        logger.log("computing loss before")
        loss_before = sliced_fun(
            self._opt_fun["f_loss"],
            self._num_slices)(
            inputs,
            extra_inputs)
        prev_param = np.copy(
            self._target_network.get_param_values(
                trainable=True))
        n_iter = 0
        for n_iter, ratio in enumerate(
                self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * descent_step
            cur_param = prev_param - cur_step
            self._target_network.set_param_values(cur_param, trainable=True)
            loss, constraint_val = sliced_fun(self._opt_fun["f_loss_constraint"],
                                              self._num_slices)(inputs, extra_inputs)

            if self._debug_nan and np.isnan(constraint_val):
                import ipdb
                ipdb.set_trace()
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
                self._max_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log(
                    "Violated because constraint %s is NaN" %
                    self._constraint_name)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                logger.log(
                    "Violated because constraint %s is violated" %
                    self._constraint_name)
            self._target_network.set_param_values(prev_param, trainable=True)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")
