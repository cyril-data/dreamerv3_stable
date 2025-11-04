from stable_baselines3.common.callbacks import EvalCallback
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    sync_envs_normalization,
)

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from minigrid.core.world_object import Wall, Goal  # Door, Key, Floor
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from stable_baselines3.common.buffers import ReplayBufferSamples


import yaml
from argparse import Namespace
from stable_baselines3.common.callbacks import BaseCallback
import yaml
from argparse import Namespace
from stable_baselines3.common.callbacks import BaseCallback
from env.nav_2d import draw_walls
import gymnasium as gym
from src.utils import make_file_path
import time


class SaveConfigCallback(BaseCallback):
    """
    Callback to save the configuration to a YAML file.

    :param config: (Namespace) The configuration to save.
    :param path: (str) The path to save the YAML file.
    :param verbose: (int) The verbosity level (0: none, 1: info).
    """

    def __init__(self, config: Namespace, dir_path: str = None, verbose=1):
        super(SaveConfigCallback, self).__init__(verbose)
        self.config = config
        self.dir_path = dir_path
        self.path = os.path.join(dir_path, "config.yml")

    def _on_training_start(self) -> None:
        """
        Method called at the beginning of training to save the configuration.
        """
        config_dict = vars(self.config)  # Convert Namespace to dictionary
        try:
            with open(self.path, "w") as file:
                yaml.dump(config_dict, file, default_flow_style=False)
            if self.verbose > 0:
                print(f"Configuration saved to {self.dir_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def _on_step(self) -> bool:
        return True


# def plot_eigen(vertical_holes, horizontal_holes, eigenvect, phi, n, dir_path, step_count, name="phi"):

#     for k in range(len(eigenvect)):

#         eigen_projection = eigenvect[k] @ phi.T

#         Z = eigen_projection.reshape(n, n).cpu()

#         # Plot en heatmap
#         plt.imshow(Z, extent=[0, 1, 0, 1], origin="lower", cmap="viridis", aspect="auto")
#         plt.colorbar(label=f"eigen_projection n°{k}")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.title(f"step {step_count}: eigenvect[{k}] @ phi.T")

#         # Ajout des murs
#         ax = plt.gca()
#         draw_walls(
#             ax,
#             wall_x=0.5,
#             wall_y=0.5,
#             wall_thickness=0.02,
#             hole_thickness=0.02,
#             vertical_holes=vertical_holes,
#             horizontal_holes=horizontal_holes,
#             color="red",
#             alpha=0.6,
#         )

#         plt.savefig(os.path.join(dir_path, f"eigen_step{step_count:05d}proj_{k:02d}_{name}"))

#         plt.close()


# class StateCoverageCallback(BaseCallback):
#     """
#     Callback pour mesurer la couverture de l'espace d'état
#     via une discrétisation en grille.
#     """

#     def __init__(self, env, dir_path: str = None, verbose: int = 0, n_step: int = 100, grid_size: int = 20):
#         super(StateCoverageCallback, self).__init__(verbose)
#         self.env = env
#         self.dir_path = dir_path
#         self.n_step = n_step
#         self.grid_size = grid_size
#         self.step_count = 0

#         # Obtenir les bornes de l’espace d’observation
#         obs_space = env.observation_space
#         assert len(obs_space.shape) == 1, "Seulement supporté pour obs vectorielles"
#         self.low = obs_space.low
#         self.high = obs_space.high
#         self.dim = obs_space.shape[0]

#         # Initialiser une table de cases visitées (booléen)
#         self.visited = np.zeros([grid_size] * self.dim, dtype=bool)

#         if self.dir_path is not None:
#             os.makedirs(self.dir_path, exist_ok=True)

#     def _obs_to_cell(self, obs: np.ndarray):
#         """
#         Mapper un état continu vers une cellule discrète de la grille
#         """
#         # Normaliser entre [0,1]
#         norm_obs = (obs - self.low) / (self.high - self.low + 1e-8)
#         # Discrétiser
#         idx = np.floor(norm_obs * self.grid_size).astype(int)
#         idx = np.clip(idx, 0, self.grid_size - 1)
#         return tuple(idx)

#     def _on_step(self) -> bool:
#         self.step_count += 1

#         obs = self.locals["new_obs"]  # état courant (array shape [batch, dim])
#         if isinstance(obs, np.ndarray) and obs.ndim == 1:
#             obs = obs[None, :]  # rendre batchable

#         for o in obs:
#             cell = self._obs_to_cell(o)
#             self.visited[cell] = True

#         if self.step_count % self.n_step == 0:
#             start_time = time.time()

#             coverage = self.visited.sum() / self.visited.size

#             if self.verbose > 0:
#                 print(f"[Coverage] Step {self.step_count} - coverage={coverage:.3f}")

#             # Logger pour tensorboard
#             if self.logger:
#                 self.logger.record("exploration/state_coverage", coverage)

#             # Sauvegarde optionnelle
#             if self.dir_path is not None:
#                 np.save(os.path.join(self.dir_path, f"coverage_{self.step_count}.npy"), self.visited)

#             # print("callback coverage time : %s seconds" % (time.time() - start_time))

#         return True


# class EigenSFOption(BaseCallback):
#     """
#     A post to visualize eigen SF vectors
#     """

#     def __init__(self, dir_path: str = None, verbose: int = 0, n_step: int = 2500):
#         super(EigenSFOption, self).__init__(verbose)
#         self.dir_path = dir_path
#         self.n_step = n_step
#         self.step_count = 0

#         # Créer le répertoire de sauvegarde s'il n'existe pas
#         if self.dir_path is not None:
#             os.makedirs(self.dir_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         self.step_count += 1

#         # Check if it's time to evaluate and save the model
#         if self.step_count % self.n_step == 0:
#             start_time = time.time()

#             with th.no_grad():
#                 eigenvalue, eigenvect = self.model.critic_sf.compute_eigen()

#                 n = 20
#                 x = np.linspace(0, 1, n)
#                 y = np.linspace(0, 1, n)

#                 # Grille complète
#                 X, Y = np.meshgrid(x, y)
#                 # Obs contient tous les couples (x,y) -> shape (n*n, 2)
#                 obs = np.column_stack([X.ravel(), Y.ravel()])

#                 phi = self.model.critic_sf.phi(th.from_numpy(obs).to("cuda")).squeeze(0).squeeze(0)

#                 # action_space = self.training_env.get_attr("action_space")
#                 # action = np.array([[action_space[0].high[0], 0] for k in range(len(obs))], dtype=np.float32)

#                 obs = th.from_numpy(obs).to("cuda")
#                 # action = th.from_numpy(action).to("cuda")

#                 action = self.model.actor(phi)

#                 next_phi = self.model.critic_sf.next_phi(obs, action).squeeze(0).squeeze(0)

#                 vertical_holes = self.training_env.get_attr("vertical_holes")[0]
#                 horizontal_holes = self.training_env.get_attr("horizontal_holes")[0]


#                 plot_eigen(
#                     vertical_holes,
#                     horizontal_holes,
#                     eigenvect,
#                     next_phi,
#                     n,
#                     self.dir_path,
#                     self.step_count,
#                     name="next_phi'",
#                     box_low=box_low,
#                     box_high=box_high,
#                     wall_x=(box_high - box_low) / 2 + box_low,
#                     wall_y=(box_high - box_low) / 2 + box_low,
#                     wall_thickness=0.02 * (box_high - box_low),
#                     hole_thickness=0.02 * (box_high - box_low),
#                 )


#             print("callback eigen time : %s seconds" % (time.time() - start_time))

#         return True


# def plot_eigen(
#     vertical_holes,
#     horizontal_holes,
#     eigenvect,
#     phi,
#     n,
#     dir_path,
#     step_count,
#     name="phi",
#     box_low=0.0,
#     box_high=1.0,
#     wall_x=0.5,
#     wall_y=0.5,
#     wall_thickness=0.02,
#     hole_thickness=0.02,
# ):
#     """
#     Trace les projections des vecteurs propres sur phi et ajoute les murs.
#     Sauvegarde chaque projection en image dans dir_path.
#     """
#     for k in range(len(eigenvect)):
#         eigen_projection = eigenvect[k] @ phi.T
#         Z = eigen_projection.reshape(n, n).cpu()

#         # Heatmap
#         plt.imshow(
#             Z,
#             extent=[box_low, box_high, box_low, box_high],
#             origin="lower",
#             cmap="viridis",
#             aspect="auto",
#         )
#         plt.colorbar(label=f"eigen_projection n°{k}")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.title(f"step {step_count}: eigenvect[{k}] @ phi.T")

#         # Ajout des murs
#         ax = plt.gca()
#         draw_walls(
#             ax,
#             wall_x=wall_x,
#             wall_y=wall_y,
#             wall_thickness=wall_thickness,
#             hole_thickness=hole_thickness,
#             vertical_holes=vertical_holes,
#             horizontal_holes=horizontal_holes,
#             box_low=box_low,
#             box_high=box_high,
#             color="red",
#             alpha=0.6,
#         )

#         # Sauvegarde
#         plt.savefig(os.path.join(dir_path, f"eigen_step{step_count:05d}proj_{k:02d}_{name}"))
#         plt.close()


# class EigenSFOption(BaseCallback):
#     """
#     Callback pour visualiser les vecteurs propres (Successor Features).
#     """

#     def __init__(
#         self, dir_path: str = None, verbose: int = 0, n_step: int = 2500, grid_size: int = 20, device: str = "cuda"
#     ):
#         super(EigenSFOption, self).__init__(verbose)
#         self.dir_path = dir_path
#         self.n_step = n_step
#         self.step_count = 0
#         self.grid_size = grid_size
#         self.device = device

#         if self.dir_path is not None:
#             os.makedirs(self.dir_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         self.step_count += 1

#         if self.step_count % self.n_step == 0:
#             start_time = time.time()

#             # Récupérer les bornes de l'espace d'observation
#             obs_space = self.training_env.observation_space
#             assert isinstance(obs_space, gym.spaces.Box), "Observation space must be Box"
#             box_low = float(obs_space.low[0])
#             box_high = float(obs_space.high[0])

#             with th.no_grad():
#                 # Décomposition en vecteurs propres
#                 eigenvalue, eigenvect = self.model.critic_sf.compute_eigen()

#                 # Grille régulière entre box_low et box_high
#                 x = np.linspace(box_low, box_high, self.grid_size)
#                 y = np.linspace(box_low, box_high, self.grid_size)
#                 X, Y = np.meshgrid(x, y)
#                 obs = np.column_stack([X.ravel(), Y.ravel()])

#                 # Feature phi(obs)
#                 obs_tensor = th.from_numpy(obs).float().to(self.device)
#                 phi = self.model.critic_sf.phi(obs_tensor).squeeze(0).squeeze(0)

#                 print("phi", phi.shape)

#                 # Action de l'acteur
#                 action = self.model.actor(phi)

#                 # Next feature
#                 next_phi = self.model.critic_sf.next_phi(obs_tensor, action).squeeze(0).squeeze(0)

#                 # Infos sur les murs
#                 vertical_holes = self.training_env.get_attr("vertical_holes")[0]
#                 horizontal_holes = self.training_env.get_attr("horizontal_holes")[0]

#                 # Visualisation
#                 plot_eigen(
#                     vertical_holes,
#                     horizontal_holes,
#                     eigenvect,
#                     next_phi,
#                     self.grid_size,
#                     self.dir_path,
#                     self.step_count,
#                     name="next_phi'",
#                     box_low=box_low,
#                     box_high=box_high,
#                     wall_x=(box_high - box_low) / 2 + box_low,
#                     wall_y=(box_high - box_low) / 2 + box_low,
#                     wall_thickness=0.02 * (box_high - box_low),
#                     hole_thickness=0.02 * (box_high - box_low),
#                 )

#             if self.verbose > 0:
#                 print(f"[EigenSF] Step {self.step_count} - done in {time.time() - start_time:.2f}s")

#         return True


class StateCoverageCallback(BaseCallback):
    """
    Callback pour mesurer la couverture de l'espace d'état
    via une discrétisation en grille.
    """

    def __init__(self, env, dir_path: str = None, verbose: int = 0, n_step: int = 100, grid_size: int = 20):
        super(StateCoverageCallback, self).__init__(verbose)
        self.dir_path = dir_path
        self.n_step = n_step
        self.grid_size = grid_size
        self.step_count = 0

        # Attributs initialisés à l'entraînement (car training_env n'est dispo qu'après .learn)
        self.low, self.high, self.dim = None, None, None
        self.visited = None

        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def _init_grid(self):
        """Initialiser la grille de couverture en fonction de l'espace d'observation de l'env."""
        obs_space = self.training_env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), "Seulement supporté pour espaces continus (Box)"
        assert len(obs_space.shape) == 1, "Seulement supporté pour obs vectorielles"

        self.low = obs_space.low
        self.high = obs_space.high
        self.dim = obs_space.shape[0]

        # Grille de cases visitées
        self.visited = np.zeros([self.grid_size] * self.dim, dtype=bool)

    def _obs_to_cell(self, obs: np.ndarray):
        """
        Mapper un état continu vers une cellule discrète de la grille.
        """
        # Normaliser entre [0,1]
        norm_obs = (obs - self.low) / (self.high - self.low + 1e-8)
        # Discrétiser
        idx = np.floor(norm_obs * self.grid_size).astype(int)
        idx = np.clip(idx, 0, self.grid_size - 1)
        return tuple(idx)

    def _on_step(self) -> bool:
        self.step_count += 1

        # Initialiser la grille au premier appel (quand training_env est dispo)
        if self.visited is None:
            self._init_grid()

        obs = self.locals["new_obs"]  # état courant (np.ndarray shape [batch, dim])
        if isinstance(obs, np.ndarray) and obs.ndim == 1:
            obs = obs[None, :]  # rendre batchable

        for o in obs:
            cell = self._obs_to_cell(o)
            self.visited[cell] = True

        if self.step_count % self.n_step == 0:
            start_time = time.time()

            coverage = self.visited.sum() / self.visited.size

            if self.verbose > 0:
                print(f"[Coverage] Step {self.step_count} - coverage={coverage:.3f}")

            # Logger pour tensorboard
            if self.logger:
                self.logger.record("exploration/state_coverage", coverage)

            # # Sauvegarde optionnelle
            # if self.dir_path is not None:
            #     np.save(os.path.join(self.dir_path, f"coverage_{self.step_count}.npy"), self.visited)

            if self.verbose > 1:
                print(f"callback coverage time: {time.time() - start_time:.2f}s")

        return True


def plot_eigen(
    vertical_walls,
    horizontal_walls,
    eigenvect,
    features,
    grid_size,
    dir_path,
    step_count,
    name="phi",
    box_low=0.0,
    box_high=1.0,
    add_arrows=None,
):
    """
    Trace les projections des vecteurs propres sur phi et ajoute les murs.
    Sauvegarde chaque projection en image dans dir_path.
    """
    for k in range(len(eigenvect)):
        eigen_projection = eigenvect[k] @ features.T

        # print("k", k)
        # print("eigen_projection", eigen_projection)

        Z = eigen_projection.reshape(grid_size, grid_size).cpu()

        # if add_arrows is not None:
        #     print("add_arrows", add_arrows[k].shape, add_arrows[k][0], add_arrows[k][1])
        plt.figure(figsize=(10, 10))
        # Heatmap

        plt.imshow(
            Z,
            extent=[box_low, box_high, box_low, box_high],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(label=f"eigen_projection n°{k}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"step {step_count}: eigenvect[{k}] @ phi.T")

        # Ajout des murs
        ax = plt.gca()
        # for wall in vertical_walls:
        draw_walls(ax, vertical_walls, horizontal_walls, box_low=0.0, box_high=1.0, color="red", alpha=0.6)

        if add_arrows is not None:
            add_arrows = [a / 0.1 for a in add_arrows]

            # === Ajout des flèches ===
            if add_arrows is not None:
                arrows = add_arrows[k].cpu().numpy()  # [400, 2]
                print(
                    "arrows",
                )

                # positions de chaque point sur la grille
                xs = np.linspace(box_low, box_high, grid_size)
                ys = np.linspace(box_low, box_high, grid_size)
                X, Y = np.meshgrid(xs, ys)

                # attention: reshape pour correspondre à la grille
                U = arrows[:, 0].reshape(grid_size, grid_size)  # composante x
                V = arrows[:, 1].reshape(grid_size, grid_size)  # composante y

                ax.quiver(X, Y, U, V, color="white", scale=5.0, width=0.005)

        plt.savefig(os.path.join(dir_path, f"eigen_step{step_count:05d}proj_{k:02d}_{name}.svg"), bbox_inches="tight")
        # plt.show()

        plt.close()


class EigenSFOption(BaseCallback):
    """
    Callback pour visualiser les vecteurs propres (Successor Features).
    """

    def __init__(
        self,
        dir_path: str = None,
        verbose: int = 0,
        n_step: int = 2500,
        grid_size: int = 20,
        device: str = "cuda",
        last_step=None,
    ):
        super(EigenSFOption, self).__init__(verbose)
        self.dir_path = dir_path
        self.n_step = n_step
        self.step_count = 0
        self.grid_size = grid_size
        self.device = device
        self.last_step = last_step
        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def plot_eigen_value(self, prefix="") -> bool:

        start_time = time.time()

        # Récupérer les bornes de l'espace d'observation
        obs_space = self.training_env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), "Observation space must be Box"
        box_low = float(obs_space.low[0])
        box_high = float(obs_space.high[0])

        with th.no_grad():
            # Décomposition en vecteurs propres
            eigenvalue, eigenvect = self.model.critic_sf.compute_eigen()

            # Grille régulière entre box_low et box_high
            x = np.linspace(box_low, box_high, self.grid_size)
            y = np.linspace(box_low, box_high, self.grid_size)
            X, Y = np.meshgrid(x, y)
            obs = np.column_stack([X.ravel(), Y.ravel()])

            # Feature phi(obs)
            obs_tensor = th.from_numpy(obs).float().to(self.device)
            phi = self.model.critic_sf.phi(obs_tensor).squeeze(0).squeeze(0)

            # Infos sur les murs (nouvelle API avec plusieurs murs)
            vertical_walls = self.training_env.get_attr("vertical_walls")[0]
            horizontal_walls = self.training_env.get_attr("horizontal_walls")[0]

            # low, high = self.training_env.action_space.low, self.training_env.action_space.high
            # mid = (low + high) / 2.0
            # mid = th.as_tensor(mid, device=phi.device, dtype=phi.dtype)
            # action_default = mid.unsqueeze(0).expand(phi.shape[0], -1).clone()
            # action_default.requires_grad_(True)

            # with th.enable_grad():
            # phi_next = self.model.critic_sf.next_phi_from_feature(phi, action_default)

            # phi_proj = []
            # # grad_list = []
            # # grad_norm_list = []
            # # grad_norm_list_neg = []
            # for eigen in eigenvect:
            #     phi_proj.append(phi @ eigen)
            #     # grad = th.autograd.grad(test.sum(), action_default, retain_graph=True)[0]
            #     # grad_list.append(grad)

            #     # grad_norm = grad.norm(dim=-1, keepdim=True)  # [batch,1]

            #     # eta_adapt = self.model.actor.eta * th.tanh(grad_norm)  # borné entre [0, self.eta]
            #     # # mean = gradient ascent step
            #     # mean_actions = (action_default + eta_adapt * grad).detach()
            #     # grad_norm_list.append(grad / grad_norm)
            #     # grad_norm_list_neg.append(-grad / grad_norm)

            # Visualisation
            # plot_eigen(
            #     vertical_walls=vertical_walls,
            #     horizontal_walls=horizontal_walls,
            #     eigenvect=eigenvect,
            #     features=phi,
            #     grid_size=self.grid_size,
            #     dir_path=self.dir_path,
            #     step_count=self.step_count,
            #     name=prefix + "phi_proj_pos",
            #     box_low=box_low,
            #     box_high=box_high,
            #     add_arrows=None,
            # )
            plot_eigen(
                vertical_walls=vertical_walls,
                horizontal_walls=horizontal_walls,
                eigenvect=-eigenvect,
                features=phi,
                grid_size=self.grid_size,
                dir_path=self.dir_path,
                step_count=self.step_count,
                name=prefix + "phi_proj_neg",
                box_low=box_low,
                box_high=box_high,
                add_arrows=None,
            )

            # # Feature phi(obs)
            # phi = self.model.critic_sf_target.phi(obs_tensor).squeeze(0).squeeze(0)

            # # Visualisation
            # plot_eigen(
            #     vertical_walls=vertical_walls,
            #     horizontal_walls=horizontal_walls,
            #     eigenvect=eigenvect,
            #     features=phi,
            #     grid_size=self.grid_size,
            #     dir_path=self.dir_path,
            #     step_count=self.step_count,
            #     name=prefix + "phi_proj_pos_target",
            #     box_low=box_low,
            #     box_high=box_high,
            #     add_arrows=None,
            # )
            # plot_eigen(
            #     vertical_walls=vertical_walls,
            #     horizontal_walls=horizontal_walls,
            #     eigenvect=-eigenvect,
            #     features=phi,
            #     grid_size=self.grid_size,
            #     dir_path=self.dir_path,
            #     step_count=self.step_count,
            #     name=prefix + "phi_proj_neg_target",
            #     box_low=box_low,
            #     box_high=box_high,
            #     add_arrows=None,
            # )

            # plot_eigen(
            #     vertical_walls=vertical_walls,
            #     horizontal_walls=horizontal_walls,
            #     eigenvect=-eigenvect,
            #     features=phi,
            #     grid_size=self.grid_size,
            #     dir_path=self.dir_path,
            #     step_count=self.step_count,
            #     name="grad_norm_inv",
            #     box_low=box_low,
            #     box_high=box_high,
            #     add_arrows=grad_norm_list_neg,
            # )

            # plot_eigen(
            #     vertical_walls=vertical_walls,
            #     horizontal_walls=horizontal_walls,
            #     eigenvect=eigenvect,
            #     features=phi,
            #     grid_size=self.grid_size,
            #     dir_path=self.dir_path,
            #     step_count=self.step_count,
            #     name="grad",
            #     box_low=box_low,
            #     box_high=box_high,
            #     add_arrows=grad_list,
            # )

        if self.verbose > 0:
            print(f"[EigenSF] Step {self.step_count} - done in {time.time() - start_time:.2f}s")

    def _on_step(self) -> bool:
        self.step_count += 1

        if self.step_count % self.n_step == 0:
            self.plot_eigen_value()

        if self.last_step:
            if self.step_count == self.last_step:
                self.plot_eigen_value()
        return True

    # def on_training_end(self):
    #     self.plot_eigen_value(prefix="final")


class SaveBestNQModels(BaseCallback):
    """
    A custom callback that saves the 2 best models based on the lowest loss.
    """

    def __init__(self, dir_path: str = None, verbose: int = 0, n_step: int = 100):
        super(SaveBestNQModels, self).__init__(verbose)
        self.dir_path = dir_path
        self.n_step = n_step
        self.step_count = 0
        self.best_losses = [float("inf"), float("inf")]  # Track the 2 best losses
        self.best_models = [
            "best_lossq_model_1_initial",
            "best_lossq_model_2_initial",
        ]  # Initial paths for the 2 best models

        # Créer le répertoire de sauvegarde s'il n'existe pas
        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def _delete_old_model(self, pattern: str):
        """
        Deletes the old model file matching the given pattern.
        """
        if os.path.exists(self.dir_path):
            for file_name in os.listdir(self.dir_path):
                if file_name.startswith(pattern):
                    os.remove(os.path.join(self.dir_path, file_name))

    def _on_step(self) -> bool:
        self.step_count += 1

        # Check if it's time to evaluate and save the model
        if self.step_count % self.n_step == 0:
            # Get the current loss from the logger
            current_loss = self.logger.name_to_value.get("train/loss_q", None)

            if current_loss is not None:

                # Check if the current loss is better than the 2nd best loss
                if current_loss < self.best_losses[1]:
                    # Determine if it's the new best or second best
                    if current_loss < self.best_losses[0]:
                        # New best model
                        self.best_losses = [current_loss, self.best_losses[0]]
                        self.best_models = [f"best_lossq_model_1_{self.step_count}", self.best_models[0]]
                        self._delete_old_model("best_lossq_model_1_")

                    else:
                        # Second best model
                        self.best_losses[1] = current_loss
                        self.best_models[1] = f"best_lossq_model_2_{self.step_count}"
                        self._delete_old_model("best_lossq_model_2_")

                    # Save the new model
                    model_path = os.path.join(
                        self.dir_path,
                        self.best_models[0] if current_loss < self.best_losses[0] else self.best_models[1],
                    )
                    self.model.save(model_path)
                    print(
                        f"Model saved at {model_path} with loss {current_loss}"
                    )  # Debug: Afficher le chemin du modèle sauvegardé

        return True


class MyEvalCallback(EvalCallback):

    def _on_step(self) -> bool:
        # Active Q-learning avant d'exécuter le code original
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            start = time.time()
            # Exécute le comportement original de _on_step

            if hasattr(self.model, "only_qlearning") and self.model.only_qlearning == False:
                self.model.only_qlearning = True
                continue_training = super()._on_step()
                self.model.only_qlearning = False
            else:
                continue_training = super()._on_step()

            end = time.time()
            print(f"--- eval time for {self.n_eval_episodes} episode : \t {end - start :.3E} \n")

        return continue_training

    def on_training_end(self):
        """Called at the end of training; saves the final model."""
        if self.verbose >= 1:
            print("Training has ended, saving the final model.")
        final_model_path = os.path.join(self.best_model_save_path, "final_model")
        self.model.save(final_model_path)
        print(f"Final model saved at {self.best_model_save_path}")

        # Check if best_model.zip exists; if not, save it
        best_model_path = os.path.join(self.best_model_save_path, "final_model.zip")
        if not os.path.exists(best_model_path):
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))


class SaveNStepModel(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, dir_path: str = None, verbose: int = 0, n_step: int = 50000):
        super(SaveNStepModel, self).__init__(verbose)

        self.dir_path = dir_path
        self.n_step = n_step  # episode counter
        self.step_count = 0  # episode counter

    def _on_step(self) -> bool:
        self.step_count += 1

        if self.step_count % self.n_step == 0:

            self.model.save(os.path.join(self.dir_path, f"best_model_{self.step_count}"))

            print(f"model saved in best_model_{self.step_count}")
        return True


class SaveBestNModelsCriticLoss(BaseCallback):
    """
    A custom callback that saves the 2 best models based on the lowest loss.
    """

    def __init__(self, dir_path: str = None, verbose: int = 0, n_step: int = 1000):
        super(SaveBestNModelsCriticLoss, self).__init__(verbose)
        self.dir_path = dir_path
        self.n_step = n_step
        self.step_count = 0
        self.best_losses = [float("inf"), float("inf")]  # Track the 2 best losses
        self.best_models = ["best_model_1_initial", "best_model_2_initial"]  # Initial paths for the 2 best models

        # Créer le répertoire de sauvegarde s'il n'existe pas
        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def _delete_old_model(self, pattern: str):
        """
        Deletes the old model file matching the given pattern.
        """
        if os.path.exists(self.dir_path):
            for file_name in os.listdir(self.dir_path):
                if file_name.startswith(pattern):
                    os.remove(os.path.join(self.dir_path, file_name))

    def _on_step(self) -> bool:
        self.step_count += 1

        # Check if it's time to evaluate and save the model
        if self.step_count % self.n_step == 0:
            # Get the current loss from the logger
            current_loss = self.logger.name_to_value.get("train/critic_loss", None)

            if current_loss is not None:

                # Check if the current loss is better than the 2nd best loss
                if current_loss < self.best_losses[1]:
                    # Determine if it's the new best or second best
                    if current_loss < self.best_losses[0]:
                        # New best model
                        self.best_losses = [current_loss, self.best_losses[0]]
                        self.best_models = [f"best_model_1_{self.step_count}", self.best_models[0]]
                        self._delete_old_model("best_model_1_")

                    else:
                        # Second best model
                        self.best_losses[1] = current_loss
                        self.best_models[1] = f"best_model_2_{self.step_count}"
                        self._delete_old_model("best_model_2_")

                    # Save the new model
                    model_path = os.path.join(
                        self.dir_path,
                        self.best_models[0] if current_loss < self.best_losses[0] else self.best_models[1],
                    )
                    self.model.save(model_path)
                    self.model.save(os.path.join(self.dir_path, "best_model"))

                    print(
                        f"Model saved at {model_path} with loss {current_loss}"
                    )  # Debug: Afficher le chemin du modèle sauvegardé

        return True

    def on_training_end(self):
        """Called at the end of training; saves the final model."""
        if self.verbose >= 1:
            print("Training has ended, saving the final model.")
        final_model_path = os.path.join(self.dir_path, "final_model")
        self.model.save(final_model_path)
        print(f"Final model saved at {final_model_path}")

        # # Check if best_model.zip exists; if not, save it
        # best_model_path = os.path.join(self.dir_path, "final_model.zip")
        # if not os.path.exists(best_model_path):
        #     self.model.save(os.path.join(self.best_model_save_path, "best_model"))


class SaveBestNModelsValueLoss(BaseCallback):
    """
    A custom callback that saves the 2 best models based on the lowest loss.
    """

    def __init__(self, dir_path: str = None, verbose: int = 0, n_step: int = 1000):
        super(SaveBestNModelsValueLoss, self).__init__(verbose)
        self.dir_path = dir_path
        self.n_step = n_step
        self.step_count = 0
        self.best_losses = [float("inf"), float("inf")]  # Track the 2 best losses
        self.best_models = ["best_model_1_initial", "best_model_2_initial"]  # Initial paths for the 2 best models

        # Créer le répertoire de sauvegarde s'il n'existe pas
        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def _delete_old_model(self, pattern: str):
        """
        Deletes the old model file matching the given pattern.
        """
        if os.path.exists(self.dir_path):
            for file_name in os.listdir(self.dir_path):
                if file_name.startswith(pattern):
                    os.remove(os.path.join(self.dir_path, file_name))

    def _on_step(self) -> bool:
        self.step_count += 1

        # Check if it's time to evaluate and save the model
        if self.step_count % self.n_step == 0:
            # Get the current loss from the logger
            current_loss = self.logger.name_to_value.get("train/value_loss", None)

            if current_loss is not None:

                # Check if the current loss is better than the 2nd best loss
                if current_loss < self.best_losses[1]:
                    # Determine if it's the new best or second best
                    if current_loss < self.best_losses[0]:
                        # New best model
                        self.best_losses = [current_loss, self.best_losses[0]]
                        self.best_models = [f"best_value_1_{self.step_count}", self.best_models[0]]
                        self._delete_old_model("best_value_1_")

                    else:
                        # Second best model
                        self.best_losses[1] = current_loss
                        self.best_models[1] = f"best_value_2_{self.step_count}"
                        self._delete_old_model("best_value_2_")

                    # Save the new model
                    model_path = os.path.join(
                        self.dir_path,
                        self.best_models[0] if current_loss < self.best_losses[0] else self.best_models[1],
                    )
                    self.model.save(model_path)
                    self.model.save(os.path.join(self.dir_path, "best_model"))

                    print(
                        f"Model saved at {model_path} with loss {current_loss}"
                    )  # Debug: Afficher le chemin du modèle sauvegardé

        return True

    def on_training_end(self):
        """Called at the end of training; saves the final model."""
        if self.verbose >= 1:
            print("Training has ended, saving the final model.")
        final_model_path = os.path.join(self.dir_path, "final_model")
        self.model.save(final_model_path)
        print(f"Final model saved at {final_model_path}")

        # # Check if best_model.zip exists; if not, save it
        # best_model_path = os.path.join(self.dir_path, "final_model.zip")
        # if not os.path.exists(best_model_path):
        #     self.model.save(os.path.join(self.best_model_save_path, "best_model"))


class SaveBestNPsiModels(BaseCallback):
    """
    A custom callback that saves the 2 best models based on the lowest loss.
    """

    def __init__(self, dir_path: str = None, verbose: int = 0, n_step: int = 100):
        super(SaveBestNPsiModels, self).__init__(verbose)
        self.dir_path = dir_path
        self.n_step = n_step
        self.step_count = 0
        self.best_losses = [float("inf"), float("inf")]  # Track the 2 best losses
        self.best_models = ["best_model_1_initial", "best_model_2_initial"]  # Initial paths for the 2 best models

        # Créer le répertoire de sauvegarde s'il n'existe pas
        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def _delete_old_model(self, pattern: str):
        """
        Deletes the old model file matching the given pattern.
        """
        if os.path.exists(self.dir_path):
            for file_name in os.listdir(self.dir_path):
                if file_name.startswith(pattern):
                    os.remove(os.path.join(self.dir_path, file_name))

    def _on_step(self) -> bool:
        self.step_count += 1

        # Check if it's time to evaluate and save the model
        if self.step_count % self.n_step == 0:
            # Get the current loss from the logger
            current_loss = self.logger.name_to_value.get("train/sf_loss", None)

            if current_loss is not None:

                # Check if the current loss is better than the 2nd best loss
                if current_loss < self.best_losses[1]:
                    # Determine if it's the new best or second best
                    if current_loss < self.best_losses[0]:
                        # New best model
                        self.best_losses = [current_loss, self.best_losses[0]]
                        self.best_models = [f"best_sf_1_{self.step_count}", self.best_models[0]]
                        self._delete_old_model("best_sf_1_")

                    else:
                        # Second best model
                        self.best_losses[1] = current_loss
                        self.best_models[1] = f"best_sf_2_{self.step_count}"
                        self._delete_old_model("best_sf_2_")

                    # Save the new model
                    model_path = os.path.join(
                        self.dir_path,
                        self.best_models[0] if current_loss < self.best_losses[0] else self.best_models[1],
                    )
                    self.model.save(model_path)
                    self.model.save(os.path.join(self.dir_path, "best_sf"))
                    print(
                        f"Model saved at {model_path} with loss {current_loss}"
                    )  # Debug: Afficher le chemin du modèle sauvegardé

        return True
