"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
from gym.envs.mujoco import HopperEnv
from gym.envs.mujoco import InvertedPendulumEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.ensemble_td3.ensemble_td3 import EnsembleTD3


def experiment(variant,env=None):
    if env is None:
        # default setting of environment
        env = NormalizedBoxEnv(HopperEnv())
    es = GaussianStrategy(
        action_space=env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = EnsembleTD3(
        env,
        qfs=[qf1,qf2],
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    env = NormalizedBoxEnv(InvertedPendulumEnv())
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=10000,
            max_path_length=1000,
            batch_size=100,
            discount=0.99,
            replay_buffer_size=int(1E6),
            policy_and_target_update_period=1,
            tau=1e-2,
            #stop_actor_training=1200,
        ),
    )
    # setup_logger('ensemble-td3-test02-normal-july-13-experiment', variant=variant)
    # setup_logger('test', variant=variant)
    setup_logger('test-InvertPendulum', variant=variant)
    experiment(variant,env=env)
