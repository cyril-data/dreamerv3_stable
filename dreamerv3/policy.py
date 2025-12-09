from typing import Any, Optional, Union, Dict
from stable_baselines3.common.preprocessing import preprocess_obs

import torch as th
from gymnasium import spaces
from torch import nn
import gymnasium as gym
import numpy as np

from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
    Distribution,
)
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict

from torch.distributions.utils import probs_to_logits
from torch.distributions import Independent, OneHotCategoricalStraightThrough, Normal, Bernoulli
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import warnings

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLPEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        net_arch: list[int] = [64, 64],
        features_dim: int = 64,
        activation_fn: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.encoder = create_mlp(input_dim, self.features_dim, net_arch, activation_fn=activation_fn)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, observations: th.Tensor) -> th.Tensor:

        return self.encoder(observations)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        net_arch: list[int] = [64, 64],
        features_dim: int = 64,
        activation_fn: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__()

        self.observation_space = observation_space
        self.output_dim = observation_space.shape[0]
        decoder_net_arch = list(reversed(net_arch))  # get the sym of net_arch encoder
        self.decoder = create_mlp(
            input_dim=features_dim,
            output_dim=self.output_dim,
            net_arch=decoder_net_arch,
            activation_fn=activation_fn,  # Convert to class for create_mlp
        )
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, features_dim: th.Tensor) -> th.Tensor:
        return self.decoder(features_dim)


class NatureCNNEncoder(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper with configurable architecture.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
    :param normalized_image: Whether to assume that the image is already normalized
    :param channels: List of output channels for each conv layer
    :param kernels: List of kernel sizes for each conv layer
    :param strides: List of strides for each conv layer
    :param paddings: List of paddings for each conv layer
    :param activation: Activation function to use
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        channels: list = [32, 64, 64],
        kernels: list = [8, 4, 3],
        strides: list = [4, 2, 1],
        paddings: list = [0, 0, 0],
        activation: callable = nn.Tanh,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        # Validation des paramètres
        assert (
            len(channels) == len(kernels) == len(strides) == len(paddings)
        ), "channels, kernels, strides and paddings must have the same length"

        n_input_channels = observation_space.shape[0]

        # Construction des couches convolutives
        conv_layers = []
        in_channels = n_input_channels

        for i, (out_channels, kernel_size, stride, padding) in enumerate(zip(channels, kernels, strides, paddings)):
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    activation(),
                ]
            )
            in_channels = out_channels

        conv_layers.append(nn.Flatten())

        self.cnn = nn.Sequential(*conv_layers)

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), activation())

        # Stockage des paramètres pour référence
        self.architecture_params = {
            "channels": channels,
            "kernels": kernels,
            "strides": strides,
            "paddings": paddings,
            "input_shape": observation_space.shape,
            "flatten_dim": n_flatten,
        }

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class NatureCNNDecoder(BaseFeaturesExtractor):
    """
    Decoder corresponding to the NatureCNN architecture with same parameters as encoder.

    :param observation_space: The observation space that we want to reconstruct
    :param features_dim: Dimension of the latent input vector
    :param normalized_image: Whether to assume that the image is already normalized
    :param channels: List of channels for each conv layer in the encoder
    :param kernels: List of kernel sizes for each conv layer in the encoder
    :param strides: List of strides for each conv layer in the encoder
    :param paddings: List of paddings for each conv layer in the encoder
    :param activation: Activation function to use
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        channels: list = [32, 64, 64],
        kernels: list = [8, 4, 3],
        strides: list = [4, 2, 1],
        paddings: list = [0, 0, 0],
        activation: callable = nn.Tanh,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureDecoder must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        # For the decoder, features_dim is the latent input dimension
        super().__init__(observation_space, features_dim)

        # Validate parameters
        assert (
            len(channels) == len(kernels) == len(strides) == len(paddings)
        ), "channels, kernels, strides and paddings must have the same length"

        self.latent_dim = features_dim
        self.output_channels = observation_space.shape[0]
        self.input_height = observation_space.shape[1]
        self.input_width = observation_space.shape[2]

        # Calculate spatial dimensions at each step of the encoder
        self.encoder_spatial_dims = self._compute_encoder_output_shape(
            self.input_height, self.input_width, kernels, strides, paddings
        )

        # The last spatial dimension corresponds to the decoder input
        final_height, final_width = self.encoder_spatial_dims[-1]

        self.final_channels = channels[-1]  # Last channel of encoder = first of decoder
        self.n_flatten = self.final_channels * final_height * final_width

        # Linear layer to project from latent space to spatial flattened space
        self.linear = nn.Sequential(nn.Linear(features_dim, self.n_flatten), activation())

        # Build transpose convolution layers (reverse order of encoder)
        tconv_layers = []

        # Reverse parameters for the decoder
        # We need channels in reverse order: [last_encoder_channel, ..., first_encoder_channel, output_channels]
        decoder_channels = [channels[-1]] + channels[:-1] + [self.output_channels]

        # Reverse other parameters
        decoder_kernels = kernels[::-1]  # [8, 4, 3] -> [3, 4, 8]
        decoder_strides = strides[::-1]  # [4, 2, 1] -> [1, 2, 4]
        decoder_paddings = paddings[::-1]  # [0, 0, 0] -> [0, 0, 0]

        # Calculate output_padding to get exact original dimensions
        decoder_output_paddings = self._compute_output_padding(
            self.encoder_spatial_dims, decoder_kernels, decoder_strides, decoder_paddings
        )

        in_channels = decoder_channels[0]

        # Build transpose conv layers
        for i, (out_channels, kernel_size, stride, padding, output_padding) in enumerate(
            zip(decoder_channels[1:], decoder_kernels, decoder_strides, decoder_paddings, decoder_output_paddings)
        ):
            # Create ConvTranspose2d layer
            tconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )

            # No activation after the final layer (output layer)
            if i < len(decoder_channels) - 2:
                tconv_layers.append(activation())

            in_channels = out_channels

        self.decoder = nn.Sequential(*tconv_layers)

        # Store parameters for reference
        self.architecture_params = {
            "channels": channels,
            "kernels": kernels,
            "strides": strides,
            "paddings": paddings,
            "encoder_spatial_dims": self.encoder_spatial_dims,
            "output_shape": observation_space.shape,
            "decoder_channels": decoder_channels,
            "decoder_kernels": decoder_kernels,
            "decoder_strides": decoder_strides,
            "decoder_paddings": decoder_paddings,
            "decoder_output_paddings": decoder_output_paddings,
        }

    def _compute_encoder_output_shape(self, input_h, input_w, kernels, strides, paddings):
        """Calculate spatial dimensions at each step of the encoder"""
        spatial_dims = [(input_h, input_w)]

        h, w = input_h, input_w
        for kernel, stride, padding in zip(kernels, strides, paddings):
            h = (h + 2 * padding - kernel) // stride + 1
            w = (w + 2 * padding - kernel) // stride + 1
            spatial_dims.append((h, w))

        return spatial_dims

    def _compute_output_padding(self, encoder_dims, decoder_kernels, decoder_strides, decoder_paddings):
        """
        Calculate the output_padding needed for ConvTranspose2d layers
        to get exactly the target dimensions (reverse of encoder steps).
        """
        output_paddings = []

        # Start from the last encoder dimension and go backward
        current_h, current_w = encoder_dims[-1]

        for i in range(len(decoder_kernels)):
            # Target dimension is the encoder dimension at the corresponding step (from the end)
            target_h, target_w = encoder_dims[-(i + 2)]

            # Calculate output dimension without output_padding
            out_h = (current_h - 1) * decoder_strides[i] - 2 * decoder_paddings[i] + decoder_kernels[i]
            out_w = (current_w - 1) * decoder_strides[i] - 2 * decoder_paddings[i] + decoder_kernels[i]

            # Calculate needed output_padding
            output_padding_h = target_h - out_h
            output_padding_w = target_w - out_w

            # Convert to tuple for ConvTranspose2d
            output_paddings.append((output_padding_h, output_padding_w))

            # Update for next iteration
            current_h, current_w = target_h, target_w

        return output_paddings

    def forward(self, latent: th.Tensor) -> th.Tensor:
        batch_size = latent.shape[0]

        # Linear projection from latent space to flattened spatial space
        x = self.linear(latent)

        # Reshape to the final spatial dimensions of the encoder
        final_h, final_w = self.encoder_spatial_dims[-1]
        x = x.view(batch_size, self.final_channels, final_h, final_w)

        # Apply transpose convolutions to reconstruct the image
        x = self.decoder(x)

        return x

class Actor(nn.Module):
    """
    Actor network (policy) for DreamerV3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        action_space: spaces.Box,
        net_arch: list[int],
        features_dim: int,
        activation_fn: type[nn.Module] = nn.Tanh,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        random: bool = True,
    ):
        super().__init__()

        # Save arguments to re-create object at loading
        self.action_space = action_space
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.random = random

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn=activation_fn)

        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) / 2 * (th.tanh(log_std) + 1)  # (-1, 1) to (min, max)
        # log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        if self.random:
            log_std = th.tensor(LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, features: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(features)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(features)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def entropy(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(features)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, features: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(features, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


class Critic(nn.Module):
    """
    Critic network(s) for DreamerV3
    It represents the value function.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        net_arch: list[int],
        features_dim: int,
        activation_fn: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()

        self.value_network = create_mlp(features_dim, 2, net_arch, activation_fn=activation_fn)
        self.value_network = nn.Sequential(*self.value_network)

    def forward(
        self,
        features: th.Tensor,
    ) -> th.Tensor:
        mean, log_std = self.value_network(features).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), th.exp(log_std).squeeze(-1))

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


class RecurrentModel(nn.Module):
    def __init__(
        self,
        latent_size: int,
        action_space: spaces.Box,
        recurrent_size: int = 512,
        hidden_size: int = 200,
        activation_fn: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.activation = activation_fn()
        self.recurrent_size = recurrent_size
        action_dim = get_action_dim(action_space)

        self.linear = nn.Linear(latent_size + action_dim, hidden_size)
        self.recurrent = nn.GRUCell(hidden_size, recurrent_size)

    def forward(self, recurrent_state, latent_state, action):
        # print("recurrent_state", recurrent_state.shape)
        # print("latent_state", latent_state.shape)
        # print("action", action.shape)
        # print("th.cat((latent_state, action), dim=1)", th.cat((latent_state, action), dim=1).shape)
        # print(
        #     "self.linear(th.cat((latent_state, action), dim=1))",
        #     self.linear(th.cat((latent_state, action), dim=1)).shape,
        # )
        # print(
        #     "self.activation(self.linear(th.cat((latent_state, action), dim=1)))",
        #     self.activation(self.linear(th.cat((latent_state, action), dim=1))).shape,
        # )
        return self.recurrent(self.activation(self.linear(th.cat((latent_state, action), dim=1))), recurrent_state)


class PriorNet(nn.Module):
    def __init__(
        self,
        inputSize: int,
        latent_length: int = 16,
        latent_classes: int = 16,
        net_arch: list[int] = [200, 200],
        activation_fn: type[nn.Module] = nn.Tanh,
        uniform_mix: float = 0.01,
    ):
        super().__init__()

        self.latent_length = latent_length
        self.latent_classes = latent_classes
        self.latent_size = latent_length * latent_classes
        self.uniform_mix = uniform_mix

        self.network = create_mlp(inputSize, self.latent_size, net_arch, activation_fn=activation_fn)
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        # print("forward Prior x", x.shape)
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latent_length, self.latent_classes).softmax(-1)
        uniform = th.ones_like(probabilities) / self.latent_classes
        finalProbabilities = (1 - self.uniform_mix) * probabilities + self.uniform_mix * uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latent_size), logits


class PosteriorNet(PriorNet):
    pass


class RewardModel(nn.Module):
    def __init__(
        self,
        inputSize: int,
        net_arch: list[int] = [400, 400],
        activation_fn: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.network = create_mlp(inputSize, 2, net_arch, activation_fn=activation_fn)
        self.network = nn.Sequential(*self.network)

    def forward(self, x: th.Tensor) -> th.Tensor:
        mean, log_std = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), th.exp(log_std).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, net_arch: list[int] = [400, 400, 400], activation_fn: type[nn.Module] = nn.Tanh):
        super().__init__()
        self.network = create_mlp(inputSize, 1, net_arch, activation_fn=activation_fn)
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        return Bernoulli(logits=self.network(x))


class DreamerV3Policy(BasePolicy):
    """
    Policy class (with both actor, critic, and model) for DreamerV3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    critic: Critic
    critic_target: Critic
    encoder: MLPEncoder
    decoder: MLPDecoder
    recurrent_model: RecurrentModel
    prior_net: PriorNet
    posterior_net: PosteriorNet
    reward_predictor: RewardModel
    continue_predictor: ContinueModel

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        # net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        # activation_fn: type[nn.Module] = nn.Tanh,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = MLPEncoder,
        features_extractor_kwargs: Optional[dict[str, Any]] = {
            "features_dim": 64,
        },
        normalize_images: bool = False,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        features_decoder_class: type[BaseFeaturesExtractor] = MLPDecoder,
        latent_classes: int = 16,
        recurrent_size: int = 512,
        latent_length: int = 16,
        recurrent_model_kwargs={
            "hidden_size": 200,
            "activation_fn": nn.Tanh,
        },
        prior_net_kwargs={
            "net_arch": [200],
            "activation_fn": nn.Tanh,
            "uniform_mix": 0.01,
        },
        posterior_net_kwargs={
            "net_arch": [200],
            "activation_fn": nn.Tanh,
            "uniform_mix": 0.01,
        },
        reward_net_kwargs={
            "net_arch": [400, 400],
            "activation_fn": nn.Tanh,
        },
        continue_net_kwargs={
            "net_arch": [400, 400, 400],
            "activation_fn": nn.Tanh,
        },
        actor_kwargs={
            "net_arch": [400, 400],
            "activation_fn": nn.Tanh,
        },
        critic_kwargs={
            "net_arch": [400, 400, 400],
            "activation_fn": nn.Tanh,
        },
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # =============================================================================
        #    Model Based initalisation
        self.recurrent_size = recurrent_size
        self.latent_size = latent_length * latent_classes
        self.full_state_size = recurrent_size + self.latent_size
        self.action_dim = get_action_dim(self.action_space)

        # -----------------------------------------------------------------------------
        # init encoder
        self.encoder = features_extractor_class(
            observation_space,
            # net_arch=net_arch, activation_fn=activation_fn,
            **features_extractor_kwargs,
        ).to(self.device)

        if not isinstance(self.encoder, MLPEncoder):
            warnings.warn(
                """
                In DreamerV3 features_extractor_class works like an encoder. But features_extractor_class is not 
                an `MLPEncoder`. 

                => Please make sur that you're decoder **has the good architecture to decode** 
                    it means Fullstatefeatures (comming from the recurrent_model + prior/posterior_net) must be 
                    decoded into the "original" observation (In fact : it's not the "original" 
                    but rather the `preprocess_obs`)  
                """
            )

        self.encoded_obs_size = self.encoder.features_dim

        # # init decoder
        # self.decoder = features_decoder_class(
        #     observation_space=self.encoder.observation_space,
        #     net_arch=self.encoder.net_arch,
        #     features_dim=self.full_state_size,
        #     activation_fn=self.encoder.activation_fn,
        # ).to(self.device)

        # -----------------------------------------------------------------------------
        # init decoder
        features_extractor_kwargs["features_dim"] = self.full_state_size
        print("features_extractor_kwargs", features_extractor_kwargs)

        self.decoder = features_decoder_class(
            observation_space,
            # net_arch=net_arch,
            # activation_fn=activation_fn,
            **features_extractor_kwargs,
        ).to(self.device)

        # -----------------------------------------------------------------------------
        # init recurrent_model
        self.recurrent_model = RecurrentModel(
            latent_size=self.latent_size,
            action_space=self.action_space,
            recurrent_size=self.recurrent_size,
            **recurrent_model_kwargs,
        ).to(self.device)

        # -----------------------------------------------------------------------------
        # init prior_net
        self.prior_net = PriorNet(
            self.recurrent_model.recurrent_size,
            latent_classes=latent_classes,
            latent_length=latent_length,
            **prior_net_kwargs,
        ).to(self.device)

        # -----------------------------------------------------------------------------
        # init posterior_net
        self.posterior_net = PosteriorNet(
            self.recurrent_model.recurrent_size + self.encoded_obs_size,
            latent_classes=latent_classes,
            latent_length=latent_length,
            **posterior_net_kwargs,
        ).to(self.device)

        # -----------------------------------------------------------------------------
        # init reward_predictor
        self.reward_predictor = RewardModel(self.full_state_size, **reward_net_kwargs).to(self.device)

        # -----------------------------------------------------------------------------
        # init continue_predictor
        self.continue_predictor = ContinueModel(self.full_state_size, **continue_net_kwargs).to(self.device)

        # -----------------------------------------------------------------------------
        # init world_model_optimizer

        self.world_model_parameters = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.recurrent_model.parameters())
            + list(self.prior_net.parameters())
            + list(self.posterior_net.parameters())
            + list(self.reward_predictor.parameters())
            + list(self.continue_predictor.parameters())
        )

        self.world_model_optimizer = self.optimizer_class(
            self.world_model_parameters, lr=lr_schedule(1), **self.optimizer_kwargs  # type: ignore[call-arg]
        )

        # =============================================================================

        # =============================================================================
        #     Actor Critic initalisation
        # -----------------------------------------------------------------------------
        # --- Initalisation actor
        self.actor_kwargs = actor_kwargs
        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)

        # print("self.actor_kwargs", self.actor_kwargs)

        self.actor = Actor(action_space, features_dim=self.full_state_size, **self.actor_kwargs).to(self.device)
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # -----------------------------------------------------------------------------
        # --- Initalisation critic
        self.critic_kwargs = critic_kwargs
        self.critic = Critic(features_dim=self.full_state_size, **self.critic_kwargs).to(self.device)
        critic_parameters = list(self.critic.parameters())
        self.critic_target = Critic(features_dim=self.full_state_size, **self.critic_kwargs).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

        # =============================================================================

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                # features_extractor_class=self.features_extractor_class,
                # features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # ================ A MODIF AVEC TRANSFORMATION OBS -> FULL LATENT STATE  ==============
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    def forward(
        self,
        latent_state: PyTorchObs,
        recurrent_state: th.tensor = None,
        deterministic: bool = False,
    ) -> th.Tensor:

        return self._predict(latent_state, recurrent_state, deterministic=deterministic)

    # def predict(
    #     self,
    #     observation: Union[np.ndarray, dict[str, np.ndarray]],
    #     state: Optional[tuple[np.ndarray, ...]] = None,
    #     episode_start: Optional[np.ndarray] = None,
    #     deterministic: bool = False,
    # ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
    #     """
    #     Get the policy action from an observation (and optional hidden state).
    #     Includes sugar-coating to handle different observations (e.g. normalizing images).

    #     :param observation: the input observation
    #     :param state: The last hidden states (can be None, used in recurrent policies)
    #     :param episode_start: The last masks (can be None, used in recurrent policies)
    #         this correspond to beginning of episodes,
    #         where the hidden states of the RNN must be reset.
    #     :param deterministic: Whether or not to return deterministic actions.
    #     :return: the model's action and the next hidden state
    #         (used in recurrent policies)
    #     """
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.set_training_mode(False)

    #     # Check for common mistake that the user does not mix Gym/VecEnv API
    #     # Tuple obs are not supported by SB3, so we can safely do that check
    #     if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
    #         raise ValueError(
    #             "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
    #             "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
    #             "vs `obs = vec_env.reset()` (SB3 VecEnv). "
    #             "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
    #             "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
    #         )

    #     obs_tensor, vectorized_env = self.obs_to_tensor(observation)

    #     with th.no_grad():
    #         recurrent_states, latent_states, last_buffer_actions = state
    #         recurrent_states = self.recurrent_model(recurrent_states, latent_states, last_buffer_actions)
    #         encoded_last_obs = self.encoder(obs_tensor)
    #         latent_states, _ = self.posterior_net(th.cat((recurrent_states, encoded_last_obs), dim=1))
    #         actions = self._predict(recurrent_states, latent_states, deterministic=False)

    #     # Convert to numpy, and reshape to the original action shape
    #     actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

    #     if isinstance(self.action_space, spaces.Box):
    #         if self.squash_output:
    #             # Rescale to proper domain when using squashing
    #             actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
    #         else:
    #             # Actions could be on arbitrary scale, so clip the actions to avoid
    #             # out of bound error (e.g. if sampling from a Gaussian distribution)
    #             actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

    #     # Remove batch dimension if needed
    #     if not vectorized_env:
    #         assert isinstance(actions, np.ndarray)
    #         actions = actions.squeeze(axis=0)

    #     return actions, state  # type: ignore[return-value]

    def _predict(
        self,
        latent_state: PyTorchObs,
        recurrent_state: th.tensor = None,
        deterministic: bool = False,
    ) -> th.Tensor:

        return self.actor(th.cat((recurrent_state, latent_state), dim=1), deterministic)

    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = DreamerV3Policy


class CnnPolicy(DreamerV3Policy):
    """
    CNN from DQN Nature paper with configurable architecture.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
    :param normalized_image: Whether to assume that the image is already normalized
    :param channels: List of output channels for each conv layer
    :param kernels: List of kernel sizes for each conv layer
    :param strides: List of strides for each conv layer
    :param paddings: List of paddings for each conv layer
    :param activation: Activation function to use
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNNEncoder,
        features_extractor_kwargs: Optional[dict[str, Any]] = {
            "features_dim": 64,
        },
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        features_decoder_class: type[BaseFeaturesExtractor] = NatureCNNDecoder,
        latent_classes: int = 16,
        recurrent_size: int = 512,
        latent_length: int = 16,
        recurrent_model_kwargs={
            "hidden_size": 200,
            "activation_fn": nn.Tanh,
        },
        prior_net_kwargs={
            "net_arch": [200],
            "activation_fn": nn.Tanh,
            "uniform_mix": 0.01,
        },
        posterior_net_kwargs={
            "net_arch": [200],
            "activation_fn": nn.Tanh,
            "uniform_mix": 0.01,
        },
        reward_net_kwargs={
            "net_arch": [400, 400],
            "activation_fn": nn.Tanh,
        },
        continue_net_kwargs={
            "net_arch": [400, 400, 400],
            "activation_fn": nn.Tanh,
        },
        actor_kwargs={
            "net_arch": [400, 400],
            "activation_fn": nn.Tanh,
        },
        critic_kwargs={
            "net_arch": [400, 400, 400],
            "activation_fn": nn.Tanh,
        },
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNNEncoder must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            features_decoder_class,
            latent_classes,
            recurrent_size,
            latent_length,
            recurrent_model_kwargs,
            prior_net_kwargs,
            posterior_net_kwargs,
            reward_net_kwargs,
            continue_net_kwargs,
            actor_kwargs,
            critic_kwargs,
        )


# class CombinedEncoder(BaseFeaturesExtractor):
#     """
#     Combined features extractor for Dict observation spaces.
#     Builds a features extractor for each key of the space. Input from each space
#     is fed through a separate submodule (CNN or MLP, depending on input shape),
#     the output features are concatenated and fed through additional MLP network ("combined").

#     :param observation_space:
#     :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
#         256 to avoid exploding network sizes.
#     :param normalized_image: Whether to assume that the image is already normalized
#         or not (this disables dtype and bounds checks): when True, it only checks that
#         the space is a Box and has 3 dimensions.
#         Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         cnn_output_dim: int = 256,
#         normalized_image: bool = False,
#     ) -> None:
#         # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
#         super().__init__(observation_space, features_dim=1)

#         extractors: dict[str, nn.Module] = {}

#         total_concat_size = 0
#         for key, subspace in observation_space.spaces.items():
#             if is_image_space(subspace, normalized_image=normalized_image):
#                 extractors[key] = NatureCNNEncoder(
#                     subspace, features_dim=cnn_output_dim, normalized_image=normalized_image
#                 )
#                 total_concat_size += cnn_output_dim
#             else:
#                 # The observation key is a vector, flatten it if needed
#                 extractors[key] = nn.Flatten()
#                 total_concat_size += get_flattened_obs_dim(subspace)

#         self.extractors = nn.ModuleDict(extractors)

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations: TensorDict) -> th.Tensor:
#         encoded_tensor_list = []

#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         return th.cat(encoded_tensor_list, dim=1)


# class CombinedDecoder(nn.Module):
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         cnn_output_dim: int = 256,
#         normalized_image: bool = False,
#     ) -> None:
#         super().__init__()

#         self.observation_space = observation_space
#         self.cnn_output_dim = cnn_output_dim
#         self.normalized_image = normalized_image

#         decoders = {}
#         subspace_sizes = {}

#         total_concat_size = 0
#         for key, subspace in observation_space.spaces.items():
#             if is_image_space(subspace, normalized_image=normalized_image):
#                 decoders[key] = NatureCNNDecoder(
#                     subspace, features_dim=cnn_output_dim, normalized_image=normalized_image
#                 )
#                 subspace_sizes[key] = cnn_output_dim
#                 total_concat_size += cnn_output_dim
#             else:
#                 subspace_size = get_flattened_obs_dim(subspace)
#                 decoders[key] = nn.Linear(subspace_size, subspace_size)
#                 subspace_sizes[key] = subspace_size
#                 total_concat_size += subspace_size

#         self.decoders = nn.ModuleDict(decoders)
#         self.subspace_sizes = subspace_sizes
#         self.latent_dim = total_concat_size

#         # Store split indices
#         self.split_indices = []
#         current_idx = 0
#         for key in observation_space.spaces.keys():
#             size = subspace_sizes[key]
#             self.split_indices.append((current_idx, current_idx + size))
#             current_idx += size

#     def forward(self, latent: th.Tensor) -> TensorDict:
#         if latent.shape[1] != self.latent_dim:
#             raise ValueError(
#                 f"Latent dimension mismatch: got {latent.shape[1]}, expected {self.latent_dim}. "
#                 f"Subspace sizes: {self.subspace_sizes}"
#             )

#         reconstructed = {}
#         for (key, decoder), (start, end) in zip(self.decoders.items(), self.split_indices):
#             reconstructed[key] = decoder(latent[:, start:end])

#         return reconstructed

#     @property
#     def required_latent_dim(self) -> int:
#         return self.latent_dim


# class MultiInputPolicy(DreamerV3Policy):
#     """
#     Policy class (with both actor and critic) for DreamerV3.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
#     :param features_extractor_class: Features extractor to use.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param n_critics: Number of critic networks to create.
#     :param share_features_extractor: Whether to share or not the features extractor
#         between the actor and the critic (this saves computation time)
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
#         activation_fn: type[nn.Module] = nn.Tanh,
#         use_sde: bool = False,
#         log_std_init: float = -3,
#         use_expln: bool = False,
#         clip_mean: float = 2.0,
#         features_extractor_class: type[BaseFeaturesExtractor] = CombinedEncoder,
#         features_extractor_kwargs: Optional[dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[dict[str, Any]] = None,
#         features_decoder_class: type[BaseFeaturesExtractor] = CombinedDecoder,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             use_sde,
#             log_std_init,
#             use_expln,
#             clip_mean,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#             features_decoder_class,
#         )
