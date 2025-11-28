from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import torch.nn as nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from dreamerv3.policy import (
    MlpPolicy,
    # CnnPolicy,
    # MultiInputPolicy,
    DreamerV3Policy,
    Actor,
    Critic,
    MLPEncoder,
    MLPDecoder,
    RecurrentModel,
    PriorNet,
    PosteriorNet,
    RewardModel,
    ContinueModel,
)
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples


SelfDreamerV3 = TypeVar("SelfDreamerV3", bound="DreamerV3")


class ReplayBufferSequence(ReplayBuffer):

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None, sequence_length: int = 24
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            assert (
                upper_bound - sequence_length > 0
            ), "not enough data in the buffer to sample. you should increase the learning_starts"

            batch_inds = np.random.randint(0, upper_bound - sequence_length, size=batch_size)
            return self._get_samples(batch_inds, env=env, sequence_length=sequence_length)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            assert (
                self.buffer_size - sequence_length > 0
            ), "not enough data in the buffer to sample. you should increase the buffer_size"

            batch_inds = (
                np.random.randint(1, self.buffer_size - sequence_length, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            assert (
                upper_bound - sequence_length > 0
            ), "not enough data in the buffer to sample. you should increase the learning_starts"
            batch_inds = np.random.randint(0, self.pos - sequence_length, size=batch_size)
        return self._get_samples(batch_inds, env=env, sequence_length=sequence_length)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None, sequence_length: int = 24
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx

        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        batch_sequence = np.array([np.arange(start, start + sequence_length) for start in batch_inds])

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                np.array(
                    [self.next_observations[batch_sequence[i], env_indices[i], :] for i in range(len(batch_inds))]
                )
            )
            # next_obs = self._normalize_obs(
            #     self.observations[(batch_sequence + 1) % self.buffer_size, env_indices, :], env
            # )
        else:
            next_obs = self._normalize_obs(
                np.array(
                    [self.next_observations[batch_sequence[i], env_indices[i], :] for i in range(len(batch_inds))]
                )
            )

        obs = self._normalize_obs(
            np.array([self.observations[batch_sequence[i], env_indices[i], :] for i in range(len(batch_inds))])
        )

        actions = np.array([self.actions[batch_sequence[i], env_indices[i], :] for i in range(len(batch_inds))])
        dones = np.expand_dims(
            np.array(
                [
                    self.dones[batch_sequence[i], env_indices[i]]
                    * (1 - self.timeouts[batch_sequence[i], env_indices[i]])
                    for i in range(len(batch_inds))
                ]
            ),
            axis=-1,
        )
        rewards = self._normalize_reward(
            np.expand_dims([self.rewards[batch_sequence[i], env_indices[i]] for i in range(len(batch_inds))], axis=-1),
            env,
        )

        # #print("batch_sequence", batch_sequence.shape)
        # #print("env_indices", env_indices)
        # #print("next_obs", next_obs.shape)
        # #print("self.observations", self.observations.shape)
        # #print("obs", obs.shape)
        # #print("self.actions", self.actions.shape)
        # #print("actions", actions.shape)
        # #print("batch_inds", batch_inds)
        # #print("dones", dones.shape)
        # #print("reward", rewards.shape)

        data = (
            obs,
            actions,
            next_obs,
            dones,
            rewards,
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class Moments(nn.Module):
    def __init__(self, device, decay=0.99, min_=1, percentileLow=0.05, percentileHigh=0.95):
        super().__init__()
        self._decay = decay
        self._min = th.tensor(min_)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", th.zeros((), dtype=th.float32, device=device))
        self.register_buffer("high", th.zeros((), dtype=th.float32, device=device))

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        x = x.detach()
        low = th.quantile(x, self._percentileLow)
        high = th.quantile(x, self._percentileHigh)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        inverse_scale = th.max(self._min, self.high - self.low)
        return self.low.detach(), inverse_scale.detach()


class DreamerV3(OffPolicyAlgorithm):
    """
    DreamerV3
    Off-Policy model based with actor-critic learned on imagined trajectories.
    This implementation borrows code from original implementation (https://github.com/danijar/dreamerv3)
    and a simplified implementation ( https://github.com/InexperiencedMe/NaturalDreamer) and from Stable Baselines
    (https://github.com/hill-a/stable-baselines).

    Paper: https://arxiv.org/abs/2301.04104
    Introduction to DreamerV3: https://www.youtube.com/watch?v=viXppDhx4R0

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original DreamerV3 paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`dreamerv3_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        # "CnnPolicy": CnnPolicy,
        # "MultiInputPolicy": MultiInputPolicy,
    }
    policy: DreamerV3Policy
    actor: Actor
    critic: Critic
    critic_target: Critic
    # encoder: MLPEncoder
    # decoder: MLPDecoder
    # recurrentModel: RecurrentModel
    # priorNet: PriorNet
    # posteriorNet: PosteriorNet
    # rewardPredictor: RewardModel
    # continuePredictor: ContinueModel

    def __init__(
        self,
        policy: Union[str, type[DreamerV3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 1000,
        batch_size: int = 256,
        sequence_length: int = 32,  # sequence_length determines the loop length for recurent state model
        tau: float = 0.005,
        gamma: float = 0.997,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBufferSequence]] = ReplayBufferSequence,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        # ent_coef: Union[str, float] = "auto",
        ent_coef: float = 3.0e-4,
        # target_update_interval: int = 1,
        # target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        free_nats: float = 1.0,
        beta_prior: float = 1.0,
        beta_posterior: float = 0.1,
        use_continuation_prediction: bool = True,
        gradient_clip: float = 100,
        gradient_norm_type: int = 2,
        imagination_horizon: int = 16,
        gae_lambda: float = 0.95,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.sequence_length = sequence_length
        # self.target_entropy = target_entropy
        # self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        # self.target_update_interval = target_update_interval
        # self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.free_nats = free_nats
        self.beta_prior = beta_prior
        self.beta_posterior = beta_posterior
        self.use_continuation_prediction = use_continuation_prediction
        self.gradient_clip = gradient_clip
        self.gradient_norm_type = gradient_norm_type
        self.imagination_horizon = imagination_horizon
        self.gae_lambda = gae_lambda
        self.value_moments = Moments(self.device)

        if _init_setup_model:
            self._setup_model()

        print("DreamerV3 policy :", self.policy)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # # Target entropy is used when learning the entropy coefficient
        # if self.target_entropy == "auto":
        #     # automatically set target entropy if needed
        #     self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        # else:
        #     # Force conversion
        #     # this will also throw an error for unexpected string
        #     self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        # if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
        #     # Default initial value of ent_coef when learned
        #     init_value = 1.0
        #     if "_" in self.ent_coef:
        #         init_value = float(self.ent_coef.split("_")[1])
        #         assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

        #     # Note: we optimize the log of the entropy coeff which is slightly different from the paper
        #     # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        #     self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
        #     self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        # else:
        # Force conversion to float
        # this will throw an error if a malformed string (different from 'auto')
        # is passed

        self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.policy.world_model_optimizer]
        # if self.ent_coef_optimizer is not None:
        #     optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        # ent_coef_losses, ent_coefs = [], []
        actor_loss_list, critic_loss_list = [], []

        world_model_loss_list, reconstruction_loss_list, reward_loss_list, kl_loss_list = [], [], [], []

        for gradient_step in range(gradient_steps):

            # ==================================================================
            # ==================================================================
            # Off-policy Model based training with sequences
            # ==================================================================
            # ==================================================================

            # *** Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env, sequence_length=self.sequence_length)  # type: ignore[union-attr]

            # *** We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # *** Transform [batch_size, sequence_length, obs.shape] → [batch_size * sequence_length, obs_shape]
            flat_observations = replay_data.observations.view(-1, 2)

            observation_shape = flat_observations[0].shape

            # *** encode flat obs and transform [batch_size *sequence_length, features_dim] → [batch_size , sequence_length, features_dim]
            encoded_observations = self.policy.encoder(flat_observations).view(
                self.batch_size, self.sequence_length, -1
            )

            # *** init previous recurrent_state and discrete_latent_state
            previous_recurrent_state = th.zeros(self.batch_size, self.policy.recurrent_size, device=self.device)
            previous_discrete_latent_state = th.zeros(self.batch_size, self.policy.latent_size, device=self.device)

            recurrent_states_list, priors_logits_list, posteriors_list, posteriors_logits_list = [], [], [], []

            # *** loop in the sequence to get all recurrent states and prior/posterior discrete latent states
            for t in range(1, self.sequence_length):

                recurrent_state = self.policy.recurrent_model(
                    previous_recurrent_state, previous_discrete_latent_state, replay_data.actions[:, t - 1]
                )
                # *** prior     output is a *discrete latent state*, which only comes from *recurrent_state* (but logits are contineous).
                _, prior_logits = self.policy.prior_net(recurrent_state)

                # *** posterior output is a discrete latent state*, which comes from *both recurrent_state AND encoded_observations*
                # *** posterior is more relyable because it contains encoded_observations in addition
                posterior, posterior_logits = self.policy.posterior_net(
                    th.cat((recurrent_state, encoded_observations[:, t]), -1)
                )

                recurrent_states_list.append(recurrent_state)
                priors_logits_list.append(prior_logits)
                posteriors_list.append(posterior)
                posteriors_logits_list.append(posterior_logits)

                # *** update the states for next discrete_latent_state and recurrent_state
                previous_recurrent_state = recurrent_state
                previous_discrete_latent_state = posterior

            # *** stack recurrent and prior/posterior discrete latent states
            recurrent_states_list = th.stack(recurrent_states_list, dim=1)
            # (batch_size, sequence_length-1, recurrentSize)
            priors_logits_list = th.stack(priors_logits_list, dim=1)
            # (batch_size, sequence_length-1, latent_length, latent_classes)
            posteriors_list = th.stack(posteriors_list, dim=1)
            # (batch_size, sequence_length-1, latent_length*latent_classes)
            posteriors_logits_list = th.stack(posteriors_logits_list, dim=1)
            # (batch_size, sequence_length-1, latent_length, latent_classes)

            # *** cat recurrent_states and posteriors discrete latent states
            full_state = th.cat((recurrent_states_list, posteriors_list), dim=-1)
            # (batch_size, sequence_length-1, recurrentSize + latent_length*latent_classes)

            # *** decode the MEAN of full_state into *original observation* into reconstruction_means
            reconstruction_means = self.policy.decoder(full_state.view(-1, self.policy.full_state_size)).view(
                self.batch_size, self.sequence_length - 1, *observation_shape
            )
            # *** impose a non learning variance = 1 for reconstruction_distribution.
            # *** Independent is necessarry to replace *n* Normal distributions 1D into *1* multivariate Normal distributions
            #     with *n* independants components.
            # *** reconstruction_loss is the *neg log_prob* for minimum optimisation
            reconstruction_distribution = Independent(Normal(reconstruction_means, 1), len(observation_shape))
            reconstruction_loss = -reconstruction_distribution.log_prob(replay_data.observations[:, 1:]).mean()

            # *** RewardModel returns directly a distrib with mean AND variance trainable.
            # *** reward_loss is the *neg log_prob* for minimum optimisation
            reward_distribution = self.policy.reward_predictor(full_state)
            reward_loss = -reward_distribution.log_prob(replay_data.rewards[:, 1:].squeeze(-1)).mean()

            # *** OneHotCategoricalStraightThrough forward : transform logits into OneHot vectors for discrete latent states
            # *** OneHotCategoricalStraightThrough backward is differentiable softmax
            # *** Stop Gradient (SG) is needed to calcule kl divergence from a frozen distrib. Each distrib is alternatively frozen
            prior_distribution = Independent(OneHotCategoricalStraightThrough(logits=priors_logits_list), 1)
            prior_distribution_SG = Independent(
                OneHotCategoricalStraightThrough(logits=priors_logits_list.detach()), 1
            )
            posterior_distribution = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits_list), 1)
            posterior_distribution_SG = Independent(
                OneHotCategoricalStraightThrough(logits=posteriors_logits_list.detach()), 1
            )

            prior_loss = kl_divergence(posterior_distribution_SG, prior_distribution)
            posterior_loss = kl_divergence(posterior_distribution, prior_distribution_SG)
            # *** Map the free_nats coef to prior_loss shape
            free_nats = th.full_like(prior_loss, self.free_nats)

            # *** scaled prior loss > beta_posterior : prior distrib must learn more because less informed (lack of encoded state)
            # *** training is stopped if kl prior ou posterior loss is too high
            prior_loss = self.beta_prior * th.maximum(prior_loss, free_nats)
            posterior_loss = self.beta_posterior * th.maximum(posterior_loss, free_nats)
            kl_loss = (prior_loss + posterior_loss).mean()

            world_model_loss = reconstruction_loss + reward_loss + kl_loss

            if self.use_continuation_prediction:
                # *** continue_predictor ouputs a Bernoulli distribution, because target is True or False
                # *** continue_loss is the Binary Cross Entropy with the dones as target
                continue_distribution = self.policy.continue_predictor(full_state)
                continue_probs = continue_distribution.probs.squeeze(-1)
                dones_target = (1 - replay_data.dones[:, 1:]).squeeze(-1)
                criterion = th.nn.BCELoss()
                continue_loss = criterion(continue_probs, dones_target)
                world_model_loss += continue_loss.mean()

            # *** training with clip gradients:
            self.policy.world_model_optimizer.zero_grad()
            world_model_loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.policy.world_model_parameters, self.gradient_clip, norm_type=self.gradient_norm_type
            )
            self.policy.world_model_optimizer.step()

            # *** append losses for control
            kl_loss_shift_for_graphing = (self.beta_prior + self.beta_posterior) * self.free_nats
            world_model_loss_list.append(world_model_loss.item() - kl_loss_shift_for_graphing)
            reconstruction_loss_list.append(reconstruction_loss.item())
            reward_loss_list.append(reward_loss.item())
            kl_loss_list.append(kl_loss.item() - kl_loss_shift_for_graphing)

            # ==================================================================
            # ==================================================================
            # Policy training with imagination trajectories based from the model
            # ==================================================================
            # ==================================================================

            #                                   *****
            # For imagination trajectories, we start with all states of sequences and use prior
            # to imagine trajectories from these states as initial state for each trajectories
            # that's why we flat sequences (.view ) into individuals states.
            #                                   *****

            full_state = full_state.view(-1, self.policy.full_state_size).detach()
            # (batch_size * (sequence_length-1), recurrentSize + latent_length*latent_classes)

            # *** separate recurrent_state and discrete_latent_state from full_state
            recurrent_state, discrete_latent_state = th.split(
                full_state, (self.policy.recurrent_size, self.policy.latent_size), -1
            )
            # recurrent_state (batch_size * (sequence_length-1), recurrentSize)
            # discrete_latent_state (batch_size * (sequence_length-1), latent_length*latent_classes)

            full_states_list, log_probs_list, entropies_list = [], [], []

            # *** Rollout imagination transitions with prior_net from all states in the sequences
            # *** => storage of full_state, and log_prob/entropy actor list
            for _ in range(self.imagination_horizon):
                action, logprob = self.actor.action_log_prob(full_state.detach())
                entropy = -logprob

                recurrent_state = self.policy.recurrent_model(recurrent_state, discrete_latent_state, action)
                # *** discrete_latent_state is overwritten by the new discrete_latent_state
                discrete_latent_state, _ = self.policy.prior_net(recurrent_state)

                # *** full_state is overwritten by the new recurrent_state and discrete_latent_state
                full_state = th.cat((recurrent_state, discrete_latent_state), -1)

                # full_state log_prob and entropy actor list storage
                full_states_list.append(full_state)
                log_probs_list.append(logprob)
                entropies_list.append(entropy)

            full_states_sequences = th.stack(full_states_list, dim=1)
            # (batchSize*batchLength, imagination_horizon, recurrentSize + latent_length*latent_classes)

            log_probs_list = th.stack(log_probs_list[1:], dim=1)  # (batchSize*batchLength, imagination_horizon-1)
            entropies_list = th.stack(entropies_list[1:], dim=1)  # (batchSize*batchLength, imagination_horizon-1)

            predicted_rewards = self.policy.reward_predictor(full_states_sequences[:, :-1]).mean

            # *** Critic returns a Normal distribution with mean and variance trainable
            values = self.critic(full_states_sequences).mean

            # *** predict the dones. = self.gamma if not use_continuation_prediction
            continues = (
                self.policy.continue_predictor(full_states_sequences).mean.squeeze(-1)
                if self.use_continuation_prediction
                else th.full_like(predicted_rewards, self.gamma)
            )

            lambda_values = self.compute_lambda_values(predicted_rewards, values, continues, self.gae_lambda)

            # TODO : test lambda_values_bis if it IMPROVES RESULTS
            # lambda_values_bis = self.compute_lambda_values_bis(predicted_rewards, values, continues, self.gae_lambda)
            # print("lambda_values", lambda_values)
            # print("lambda_values_bis", lambda_values_bis)
            # print("lambda_values - lambda_values_bis", lambda_values - lambda_values_bis)

            _, inverse_scale = self.value_moments(lambda_values)
            # *** inverse_scale = max ( Per95(Return) - Per05(Return) )
            # with Exponential Moving Average (EMA) for perceptiles : EMAₜ = β · EMAₜ₋₁ + (1 - β) · xₜ
            #     => scalar ~= max(Return) - min(return)

            advantages = (lambda_values - values[:, :-1]) / inverse_scale
            # *** advantages is not simply A = returns - values[:, :-1] but A is normalize by inverse_scale

            actor_loss = -th.mean(advantages.detach() * log_probs_list + self.ent_coef * entropies_list)
            # *** advantages is not simply A = returns - values[:, :-1] but A is normalize by inverse_scale

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip, norm_type=self.gradient_norm_type)
            self.actor.optimizer.step()
            actor_loss_list.append(actor_loss.item())

            # *********************************************
            # *** critic loss ***
            # Categorical distribution with exponentially spaced bins     IS MISSING !!!
            # -------------------------------------------------------     ==========
            #
            # This part below with both imagination and replay trajectory IS MISSING !!!
            # ----------------------------------------------------------- ==========
            #
            # To improve value prediction in environments where rewards are challenging
            # to predict, we apply the critic loss both to imagined trajectories with
            # loss scale βval = 1 and to trajectories sampled from the replay buffer
            # with loss scale βrepval = 0.3. The critic replay loss uses the imagination
            # returns Rλ  t at the start states of the imagination rollouts as on-policy
            #  value annotations for the replay trajectory to then compute λ-returns over
            # the replay rewards.
            # *********************************************

            value_distributions = self.critic(full_states_sequences[:, :-1].detach())
            critic_loss = -th.mean(value_distributions.log_prob(lambda_values.detach()))
            # *** estimates V using the maximum likelihood loss with target = return beyond the prediction horizon

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip, norm_type=self.gradient_norm_type)
            self.critic.optimizer.step()
            critic_loss_list.append(critic_loss.item())

            # return full_states.view(-1, self.policy.full_state_size).detach(), metrics

            # # =============================================================================

            # # =============================================================================
            # # --- Policy training

            # # Action by the current actor for the sampled state
            # actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            # log_prob = log_prob.reshape(-1, 1)

            # ent_coef_loss = None
            # if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            #     # Important: detach the variable from the graph
            #     # so we don't change it with other losses
            #     # see https://github.com/rail-berkeley/softlearning/issues/60
            #     ent_coef = th.exp(self.log_ent_coef.detach())
            #     assert isinstance(self.target_entropy, float)
            #     ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            #     ent_coef_losses.append(ent_coef_loss.item())
            # else:
            #     ent_coef = self.ent_coef_tensor

            # ent_coefs.append(ent_coef.item())

            # # Optimize entropy coefficient, also called
            # # entropy temperature or alpha in the paper
            # if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            #     self.ent_coef_optimizer.zero_grad()
            #     ent_coef_loss.backward()
            #     self.ent_coef_optimizer.step()

            # with th.no_grad():
            #     # Select action according to policy
            #     next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            #     # Compute the next Q values: min over all critics targets
            #     next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            #     next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            #     # add entropy term
            #     next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            #     # td error + entropy term
            #     target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # # Get current Q-values estimates for each critic network
            # # using action from the replay buffer
            # current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # # Compute critic loss
            # critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            # assert isinstance(critic_loss, th.Tensor)  # for type checker
            # critic_loss_list.append(critic_loss.item())  # type: ignore[union-attr]

            # # Optimize the critic
            # self.critic.optimizer.zero_grad()
            # critic_loss.backward()
            # self.critic.optimizer.step()

            # # Compute actor loss
            # # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # # Min over all critic networks
            # q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            # min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            # actor_loss_list.append(actor_loss.item())

            # # Optimize the actor
            # self.actor.optimizer.zero_grad()
            # actor_loss.backward()
            # self.actor.optimizer.step()

            # # Update target networks
            # if gradient_step % self.target_update_interval == 0:
            #     polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            #     # Copy running stats, see GH issue #996
            #     polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/ent_coef", np.mean(ent_coefs))

        self.logger.record("train/world_model_loss", np.mean(world_model_loss_list))
        self.logger.record("train/world_reconstruction_loss", np.mean(reconstruction_loss_list))
        self.logger.record("train/world_reward_loss", np.mean(reward_loss_list))
        self.logger.record("train/world_kl_loss", np.mean(kl_loss_list))

        self.logger.record("train/actor_loss", np.mean(actor_loss_list))
        self.logger.record("train/critic_loss", np.mean(critic_loss_list))
        # self.logger.record("train/actor_loss", np.mean(actor_loss_list))
        # self.logger.record("train/critic_loss", np.mean(critic_loss_list))
        # if len(ent_coef_losses) > 0:
        #     self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def compute_lambda_values(self, rewards, values, continues, lambda_=0.95):
        returns = th.zeros_like(rewards)
        bootstrap = values[:, -1]
        for i in reversed(range(rewards.shape[-1])):
            returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
            bootstrap = returns[:, i]
        return returns

    def compute_lambda_values_bis(self, rewards, values, continues, lambda_=0.95):
        advantages = th.zeros_like(rewards)
        last_gae_lam = 0
        for step in reversed(range(rewards.shape[-1])):
            delta = rewards[:, step] + self.gamma * values[:, step + 1] * continues[:, step + 1] - values[:, step]
            last_gae_lam = delta + self.gamma * lambda_ * continues[:, step + 1] * last_gae_lam
            advantages[:, step] = last_gae_lam
        returns = advantages + values[:, :-1]
        return returns

    def tranform_obs_to_tensor(self, observation):
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        return obs_tensor, vectorized_env

    def tranform_action_to_np(self, actions, vectorized_env):

        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.policy.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions  # type: ignore[return-value]

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """

        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"

            # =============================================================================
            # *****************************************************************************
            # specific add for recurrent and latent state for DreamerV3

            last_observations, vectorized_env = self.tranform_obs_to_tensor(self._last_obs)
            with th.no_grad():
                self.recurrent_states = self.policy.recurrent_model(
                    self.recurrent_states, self.discrete_latent_states, self.last_buffer_actions
                )
                encoded_last_obs = self.policy.encoder(last_observations)
                self.discrete_latent_states, _ = self.policy.posterior_net(
                    th.cat((self.recurrent_states, encoded_last_obs), dim=1)
                )
                actions = self.policy._predict(self.recurrent_states, self.discrete_latent_states, deterministic=False)
            unscaled_action = self.tranform_action_to_np(actions, vectorized_env)

            # *****************************************************************************
            # =============================================================================

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action

        # =============================================================================
        # *****************************************************************************
        # Need to save previous actions for DreamerV3

        self.last_buffer_actions = th.tensor(buffer_action, device=self.device)

        # *****************************************************************************
        # =============================================================================
        return action, buffer_action

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBufferSequence,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBufferSequence``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()

                    # =============================================================================
                    # *****************************************************************************
                    # Need to init recurrent_states, discrete_latent_states, last_buffer_actions
                    self.init_recurents_latents_actions(idx)
                    # *****************************************************************************
                    # =============================================================================

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def init_recurents_latents_actions(self, env_num=None):

        if env_num is None:
            self.recurrent_states, self.discrete_latent_states = th.zeros(
                self.n_envs, self.policy.recurrent_size, device=self.device
            ), th.zeros(self.n_envs, self.policy.latent_size, device=self.device)
            self.last_buffer_actions = th.zeros(self.n_envs, self.policy.action_dim).to(self.device)
        else:
            self.recurrent_states[env_num] = 0
            self.discrete_latent_states[env_num] = 0
            self.last_buffer_actions[env_num] = 0

    def learn(
        self: SelfDreamerV3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DreamerV3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDreamerV3:

        self.init_recurents_latents_actions()

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        # if self.ent_coef_optimizer is not None:
        #     saved_pytorch_variables = ["log_ent_coef"]
        #     state_dicts.append("ent_coef_optimizer")
        # else:
        # saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, []
