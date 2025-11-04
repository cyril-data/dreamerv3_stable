import numpy as np
import random
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha=0.6, beta=0.4, **kwargs):
        """
        Prioritized Replay Buffer.

        :param alpha: coefficient de priorisation (0: pas de priorisation, 1: priorisation maximale)
        :param beta: coefficient de correction pour l'importance (0: pas de correction, 1: correction complète)
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.beta_increment_per_sampling = (1.0 - beta) / self.n_steps
        self.t = 0  # compteur des étapes pour ajuster beta

    def add(self, observation, action, reward, next_observation, done, info, episode_length):
        """
        Ajoute une nouvelle transition avec une priorité initiale (provisoire).
        """
        idx = self._next_idx
        super().add(observation, action, reward, next_observation, done, info, episode_length)
        # Initialiser la priorité à un maximum pour les nouvelles transitions
        self.priorities[idx] = np.max(self.priorities) if len(self.priorities) > 0 else 1.0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Échantillonne des transitions en fonction des priorités, avec une correction de l'importance.
        """
        if self._size == 0:
            return None

        # Calcul des probabilités basées sur les priorités
        priorities = self.priorities[: self._size] ** self.alpha
        prob = priorities / priorities.sum()

        indices = np.random.choice(self._size, batch_size, p=prob)

        # Correction de l'importance pour compenser les priorités biaisées
        weights = (self._size * prob[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalisation des poids

        # Incrémentation de beta pour la prochaine itération
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        # Échantillonner les transitions
        batch = super()._encode_sample(indices)

        return ReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
            dones=batch.dones,
            weights=weights,
            batch_indices=indices,
        )

    def update_priorities(self, indices, priorities):
        """
        Met à jour les priorités des transitions échantillonnées.
        """
        self.priorities[indices] = priorities
