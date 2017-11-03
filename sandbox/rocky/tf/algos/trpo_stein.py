from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.optimizers.stein_optimizer import SteinOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import pdb


class TRPOStein(BatchPolopt, Serializable):
    """
    TRPO with stein inference
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            optimizer=None,
            optimizer_args=None,
            delta=0.01,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = SteinOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        self.delta = delta
        super(
            TRPOStein,
            self).__init__(
            env=env,
            policy=policy,
            baseline=baseline,
            **kwargs)

    @overrides
    def init_opt(self):

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = tensor_utils.new_tensor(
            name='advantage',
            ndim=1,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k]
                                   for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k]
                                for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        # std network output
        log_std_var = dist_info_vars["log_std"]
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - tf.reduce_mean(logli * advantage_var)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        input_list = [
            obs_var,
            action_var,
            advantage_var
        ] + state_info_vars_list + old_dist_info_vars_list

        self.optimizer.update_opt(loss=surr_obj,
                                  target=self.policy,
                                  logstd=log_std_var,
                                  leq_constraint=(mean_kl, self.delta), 
                                  inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        inputs = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k]
                          for k in self.policy.distribution.dist_info_keys]
        inputs += tuple(state_info_list) + tuple(dist_info_list)
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = self.opt_info['f_kl'](
            *(list(inputs)))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
