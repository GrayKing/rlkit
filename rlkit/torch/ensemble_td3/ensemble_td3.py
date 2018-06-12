from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


import abc
import pickle
import time

import gtimer as gt
import numpy as np

from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.policies.base import ExplorationPolicy
from rlkit.samplers.in_place import InPlacePathSampler

class EnsembleTD3(TorchRLAlgorithm):
    """
    Twin Delayed Deep Deterministic policy gradients

    https://arxiv.org/abs/1802.09477
    """

    def __init__(
            self,
            env,
            qfs,
            policy,
            exploration_policy,

            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
            min_num_steps_before_training=1000,

            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,

            **kwargs
    ):
        super().__init__(
            env,
            exploration_policy,
            eval_policy=policy,
            **kwargs
        )
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qfs = qfs
        self.policy = policy

        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.min_num_steps_before_training = min_num_steps_before_training

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion

        self.target_policy = policy.copy()

        self.num_q = len(qfs)

        self.target_qfs = [qf.copy() for qf in self.qfs ]

        self.qf_optimizers = [optimizer_class(
            qf.parameters(),
            lr=qf_learning_rate,
        ) for qf in self.qfs]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
        )
        self.eval_statistics = None

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        noise = torch.normal(
            torch.zeros_like(next_actions),
            self.target_policy_noise,
        )
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise


        bellman_errors = []
        for idx in range(self.num_q):
            target_q_values = self.target_qfs[idx](next_obs, noisy_next_actions)
            q_target = rewards + (1. - terminals) * self.discount * target_q_values
            q_target = q_target.detach()
            q_pred = self.qfs[idx](obs, actions)
            bellman_errors.append( (q_pred - q_target) ** 2)

        """
        Update Networks
        """
        for idx in range(self.num_q):
            self.qf_optimizers[idx].zero_grad()
            bellman_errors[idx].mean().backward()
            self.qf_optimizers[idx].step()

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            """
            Compute the gradient of taking different variables 
            """

            ensemble_q_grads = []
            ensemble_j_grads = []

            for idx in range(self.num_q):

                policy_actions = self.policy(obs)
                policy_actions.retain_grad()

                q_output = self.qfs[idx](obs, policy_actions)
                policy_loss = - q_output.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()

                ensemble_q_grads.append(torch.mean(policy_actions.grad,0).view(1,-1))

                all_grad = torch.cat([(torch.squeeze(param.grad.view(1,-1))).view(1,-1) for param in self.policy.parameters()],dim=1)
                ensemble_j_grads.append(all_grad)

            ensemble_q_grads = torch.cat(ensemble_q_grads)
            ensemble_j_grads = torch.cat(ensemble_j_grads)
            var_q_grads = ensemble_q_grads.std(dim=0) ** 2
            var_j_grads = ensemble_j_grads.std(dim=0) ** 2

            var_q_grad_sum = torch.sum(var_q_grads)
            var_j_grad_sum = torch.sum(var_j_grads)
            print(ensemble_q_grads.size())
            print(ensemble_j_grads.size())
            print(var_q_grads.size())
            print(var_j_grads.size())
            print(var_q_grad_sum.size())
            print(var_j_grad_sum.size())


            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)

            for idx in range(self.num_q):
                ptu.soft_update_from_to(self.qfs[idx], self.target_qfs[idx], self.tau)

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            exit(0)

            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qfs=self.qfs,
            policy=self.eval_policy,
            trained_policy=self.policy,
            target_policy=self.target_policy,
            exploration_policy=self.exploration_policy,
        )
        return snapshot

    def _can_train(self):
        return (
            self.replay_buffer.num_steps_can_sample() >=
            self.min_num_steps_before_training
        )

    def _can_evaluate(self):
        return (
            len(self._exploration_paths) > 0
            and self.eval_statistics is not None
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.target_policy
        ] + self.qfs + self.target_qfs

    """
    For variance test 
    """


    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            for _ in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(
                    observation,
                )
                if self.render:
                    self.training_env.render()
                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                self._n_env_steps_total += 1
                reward = raw_reward * self.reward_scale
                terminal = np.array([terminal])
                reward = np.array([reward])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal or len(self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

                gt.stamp('sample')
                self._try_to_train()
                gt.stamp('train')

            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _try_to_train(self):
        if self._can_train():
            self.training_mode(True)
            for i in range(self.num_updates_per_train_call):
                self._do_training()
                self._n_train_steps_total += 1
            self.training_mode(False)