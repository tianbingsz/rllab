

from sandbox.rocky.tf.algos.vpg import VPG
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = TfEnv(normalize(DoublePendulumEnv()))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=(100, 50, 25),
    adaptive_std=False,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    optimizer_args=dict(
        tf_optimizer_args=dict(
            learning_rate=0.005,
        )
    )
)

run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
)
