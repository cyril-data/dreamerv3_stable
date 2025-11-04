import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


# class Continuous2DNavigationEnv(gym.Env):
#     def __init__(self, render_mode=False, max_steps=500):
#         self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(2,), dtype=np.float32)
#         self.goal_pos = np.array([0.2, 0.15])
#         self.max_steps = max_steps
#         self.current_step = 0  # Ajoutez ce compteur
#         self.reset()

#     def reset(self, seed=None, options=None):
#         self.agent_pos = np.random.rand(2)
#         self.current_step = 0  # Réinitialisez le compteur
#         return self.agent_pos, {}

#     def step(self, action):
#         self.agent_pos = np.clip(self.agent_pos + action, 0, 1)
#         self.current_step += 1  # Incrémentez le compteur

#         dist = np.linalg.norm(self.agent_pos - self.goal_pos)
#         reward = -(dist**2)

#         # Condition d'arrêt : goal atteint OU limite de steps
#         terminated = dist < 0.02
#         truncated = self.current_step >= self.max_steps

#         return self.agent_pos, reward, terminated, truncated, {}

#     def render(self):
#         plt.clf()
#         ax = plt.gca()
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.set_aspect("equal")

#         # Dessiner le goal (cercle vert)
#         goal_circle = plt.Circle(self.goal_pos, 0.03, color="green", label="Goal")
#         ax.add_patch(goal_circle)

#         # Dessiner l'agent (cercle bleu)
#         agent_circle = plt.Circle(self.agent_pos, 0.03, color="blue", label="Agent")
#         ax.add_patch(agent_circle)

#         # Légende et affichage
#         plt.legend(loc="upper right")
#         plt.pause(0.01)

#     def close(self):
#         plt.close()


# class Continuous2DNavigationEnv(gym.Env):
#     def __init__(self, render_mode=False, max_steps=500):
#         self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(2,), dtype=np.float32)

#         self.goal_pos = np.array([0.2, 0.15])
#         self.wall_x = 0.5
#         self.wall_thickness = 0.02  # demi-largeur = 0.01
#         self.max_steps = max_steps
#         self.current_step = 0

#         self.render_mode = render_mode
#         self.reset()

#     def reset(self, seed=None, options=None):
#         self.agent_pos = np.random.rand(2)
#         self.current_step = 0
#         return self.agent_pos, {}

#     def step(self, action):
#         proposed_pos = np.clip(self.agent_pos + action, 0, 1)

#         # Empêcher le passage à travers le mur
#         if self._collides_with_wall(self.agent_pos, proposed_pos):
#             proposed_pos = self._slide_along_wall(self.agent_pos, proposed_pos)

#         self.agent_pos = proposed_pos
#         self.current_step += 1

#         dist = np.linalg.norm(self.agent_pos - self.goal_pos)
#         reward = -(dist**2)

#         terminated = dist < 0.02
#         truncated = self.current_step >= self.max_steps

#         return self.agent_pos, reward, terminated, truncated, {}

#     def _collides_with_wall(self, old_pos, new_pos):
#         # Le mur est vertical entre y=0 et y=1, centré sur wall_x
#         half_thickness = self.wall_thickness / 2
#         # Si l'agent "passe" de l'autre côté du mur
#         crosses_wall = (old_pos[0] < self.wall_x - half_thickness and new_pos[0] >= self.wall_x - half_thickness) or (
#             old_pos[0] > self.wall_x + half_thickness and new_pos[0] <= self.wall_x + half_thickness
#         )
#         # Vérifier si la position finale est dans la zone interdite
#         in_wall_zone = self.wall_x - half_thickness <= new_pos[0] <= self.wall_x + half_thickness
#         return crosses_wall or in_wall_zone

#     def _slide_along_wall(self, old_pos, new_pos):
#         # On bloque juste le mouvement horizontal vers le mur
#         half_thickness = self.wall_thickness / 2
#         if old_pos[0] < self.wall_x:
#             new_pos[0] = min(new_pos[0], self.wall_x - half_thickness)
#         else:
#             new_pos[0] = max(new_pos[0], self.wall_x + half_thickness)
#         return new_pos

#     def render(self):
#         plt.clf()
#         ax = plt.gca()
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.set_aspect("equal")

#         # Mur vertical (rectangle rouge)
#         half_thickness = self.wall_thickness / 2
#         wall_rect = plt.Rectangle(
#             (self.wall_x - half_thickness, 0),  # coin inférieur gauche
#             self.wall_thickness,  # largeur
#             1,  # hauteur
#             color="red",
#             alpha=0.6,
#             label="Wall",
#         )
#         ax.add_patch(wall_rect)

#         # Goal (cercle vert)
#         goal_circle = plt.Circle(self.goal_pos, 0.03, color="green", label="Goal")
#         ax.add_patch(goal_circle)

#         # Agent (cercle bleu)
#         agent_circle = plt.Circle(self.agent_pos, 0.03, color="blue", label="Agent")
#         ax.add_patch(agent_circle)

#         plt.legend(loc="upper right")
#         plt.pause(0.01)

#     def close(self):
#         plt.close()


# class Continuous2DNavigationEnv(gym.Env):
#     def __init__(self, render_mode=False, max_steps=500, hole_center=0.5, hole_height=10.0, terminated_cond=0.05):
#         self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(2,), dtype=np.float32)

#         self.terminated_cond = terminated_cond
#         self.goal_pos = np.array([0.2, 0.15])
#         self.wall_x = 0.5
#         self.wall_thickness = 0.02  # largeur totale
#         self.max_steps = max_steps
#         self.current_step = 0

#         # Paramètres du trou
#         self.hole_center = hole_center
#         self.hole_height = hole_height
#         if self.hole_height is None:
#             self.hole_height = 10.0
#         self.render_mode = render_mode
#         self.reset()

#     def reset(self, seed=None, options=None):
#         self.agent_pos = np.random.rand(2)
#         self.current_step = 0
#         return self.agent_pos, {}

#     def step(self, action):
#         proposed_pos = np.clip(self.agent_pos + action, 0, 1)

#         if self._collides_with_wall(self.agent_pos, proposed_pos):
#             proposed_pos = self._slide_along_wall(self.agent_pos, proposed_pos)

#         self.agent_pos = proposed_pos
#         self.current_step += 1

#         dist = np.linalg.norm(self.agent_pos - self.goal_pos)
#         reward = -(dist**2)

#         terminated = dist < self.terminated_cond
#         truncated = self.current_step >= self.max_steps

#         return self.agent_pos, reward, terminated, truncated, {}

#     def _collides_with_wall(self, old_pos, new_pos):
#         half_thickness = self.wall_thickness / 2
#         # Zone verticale du trou

#         hole_y_min = self.hole_center - self.hole_height / 2
#         hole_y_max = self.hole_center + self.hole_height / 2

#         crosses_wall = (old_pos[0] < self.wall_x - half_thickness and new_pos[0] >= self.wall_x - half_thickness) or (
#             old_pos[0] > self.wall_x + half_thickness and new_pos[0] <= self.wall_x + half_thickness
#         )
#         in_wall_zone = self.wall_x - half_thickness <= new_pos[0] <= self.wall_x + half_thickness and not (
#             hole_y_min <= new_pos[1] <= hole_y_max
#         )  # autoriser si dans le trou
#         return crosses_wall and in_wall_zone or in_wall_zone

#     def _slide_along_wall(self, old_pos, new_pos):
#         half_thickness = self.wall_thickness / 2
#         if old_pos[0] < self.wall_x:
#             new_pos[0] = min(new_pos[0], self.wall_x - half_thickness)
#         else:
#             new_pos[0] = max(new_pos[0], self.wall_x + half_thickness)
#         return new_pos

#     def render(self):
#         plt.clf()
#         ax = plt.gca()
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.set_aspect("equal")

#         half_thickness = self.wall_thickness / 2

#         # Dessiner le mur avec un trou
#         hole_y_min = self.hole_center - self.hole_height / 2
#         hole_y_max = self.hole_center + self.hole_height / 2

#         # Partie basse du mur
#         if hole_y_min > 0:
#             wall_bottom = plt.Rectangle(
#                 (self.wall_x - half_thickness, 0), self.wall_thickness, hole_y_min, color="red", alpha=0.6
#             )
#             ax.add_patch(wall_bottom)

#         # Partie haute du mur
#         if hole_y_max < 1:
#             wall_top = plt.Rectangle(
#                 (self.wall_x - half_thickness, hole_y_max), self.wall_thickness, 1 - hole_y_max, color="red", alpha=0.6
#             )
#             ax.add_patch(wall_top)

#         # Goal
#         goal_circle = plt.Circle(self.goal_pos, 0.03, color="green", label="Goal")
#         ax.add_patch(goal_circle)

#         # Agent
#         agent_circle = plt.Circle(self.agent_pos, 0.03, color="blue", label="Agent")
#         ax.add_patch(agent_circle)

#         plt.legend(loc="upper right")
#         plt.pause(0.01)

#     def close(self):
#         plt.close()


# def draw_walls(
#     ax, wall_x, wall_y, wall_thickness, hole_thickness, vertical_holes, horizontal_holes, color="red", alpha=0.6
# ):
#     """
#     Dessine les murs vertical et horizontal avec leurs trous sur un axe matplotlib.
#     """
#     v_half = wall_thickness / 2
#     h_half = hole_thickness / 2

#     # Mur vertical avec trous
#     y_segments = [(0, 1)]

#     for hole in vertical_holes:
#         center, height = hole["center"], hole["height"]
#         hole_min, hole_max = center - height / 2, center + height / 2
#         new_segments = []
#         for seg_min, seg_max in y_segments:
#             if seg_min < hole_min:
#                 new_segments.append((seg_min, min(seg_max, hole_min)))
#             if seg_max > hole_max:
#                 new_segments.append((max(seg_min, hole_max), seg_max))
#         y_segments = new_segments

#     for seg_min, seg_max in y_segments:
#         ax.add_patch(
#             plt.Rectangle((wall_x - v_half, seg_min), wall_thickness, seg_max - seg_min, color=color, alpha=alpha)
#         )

#     # Mur horizontal avec trous
#     x_segments = [(0, 1)]
#     for hole in horizontal_holes:
#         center, width = hole["center"], hole["width"]
#         hole_min, hole_max = center - width / 2, center + width / 2
#         new_segments = []
#         for seg_min, seg_max in x_segments:
#             if seg_min < hole_min:
#                 new_segments.append((seg_min, min(seg_max, hole_min)))
#             if seg_max > hole_max:
#                 new_segments.append((max(seg_min, hole_max), seg_max))
#         x_segments = new_segments

#     for seg_min, seg_max in x_segments:
#         ax.add_patch(
#             plt.Rectangle((seg_min, wall_y - h_half), seg_max - seg_min, hole_thickness, color=color, alpha=alpha)
#         )


# class Continuous2DNavigationEnv(gym.Env):

#     def __init__(
#         self,
#         render_mode=False,
#         max_steps=500,
#         vertical_holes=None,
#         horizontal_holes=None,
#         terminated_cond=0.05,
#         box_low=0.0,
#         box_high=1.0,
#     ):

#         self.observation_space = spaces.Box(low=box_low, high=box_high, shape=(2,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(2,), dtype=np.float32)

#         self.terminated_cond = terminated_cond
#         self.goal_pos = np.array([0.8, 0.8])

#         # Mur vertical
#         self.wall_x = 0.5
#         self.wall_thickness = 0.02
#         self.vertical_holes = vertical_holes or []  # [{center: float, height: float}, ...]

#         # Mur horizontal
#         self.wall_y = 0.5
#         self.hole_thickness = 0.02
#         self.horizontal_holes = horizontal_holes or []  # [{center: float, width: float}, ...]

#         self.max_steps = max_steps
#         self.current_step = 0
#         self.render_mode = render_mode
#         self.reset()

#     def _collides_with_wall(self, old_pos, new_pos):
#         v_half = self.wall_thickness / 2
#         h_half = self.hole_thickness / 2

#         # Mur vertical
#         if self.wall_x - v_half <= new_pos[0] <= self.wall_x + v_half:
#             allowed = False
#             for hole in self.vertical_holes:
#                 center, height = hole["center"], hole["height"]
#                 if center - height / 2 <= new_pos[1] <= center + height / 2:
#                     allowed = True
#                     break
#             if not allowed:
#                 return True

#         # Mur horizontal
#         if self.wall_y - h_half <= new_pos[1] <= self.wall_y + h_half:
#             allowed = False
#             for hole in self.horizontal_holes:
#                 center, width = hole["center"], hole["width"]
#                 if center - width / 2 <= new_pos[0] <= center + width / 2:
#                     allowed = True
#                     break
#             if not allowed:
#                 return True

#         return False

#     def reset(self, seed=None, options=None):
#         self.agent_pos = np.random.rand(2)
#         self.current_step = 0
#         return self.agent_pos, {}

#     def step(self, action):
#         proposed_pos = np.clip(self.agent_pos + action, 0, 1)

#         if self._collides_with_wall(self.agent_pos, proposed_pos):
#             proposed_pos = self._slide_along_wall(self.agent_pos, proposed_pos)

#         self.agent_pos = proposed_pos
#         self.current_step += 1

#         dist = np.linalg.norm(self.agent_pos - self.goal_pos)
#         reward = -(dist**2)

#         terminated = dist < self.terminated_cond
#         truncated = self.current_step >= self.max_steps

#         return self.agent_pos, reward, terminated, truncated, {}

#     def _slide_along_wall(self, old_pos, new_pos):
#         v_half = self.wall_thickness / 2
#         h_half = self.hole_thickness / 2

#         # Mur vertical
#         if self.wall_x - v_half <= new_pos[0] <= self.wall_x + v_half:
#             if old_pos[0] < self.wall_x:
#                 new_pos[0] = self.wall_x - v_half
#             else:
#                 new_pos[0] = self.wall_x + v_half

#         # Mur horizontal
#         if self.wall_y - h_half <= new_pos[1] <= self.wall_y + h_half:
#             if old_pos[1] < self.wall_y:
#                 new_pos[1] = self.wall_y - h_half
#             else:
#                 new_pos[1] = self.wall_y + h_half

#         return new_pos

#     def render(self):
#         plt.clf()
#         ax = plt.gca()
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.set_aspect("equal")

#         # Murs
#         draw_walls(
#             ax,
#             self.wall_x,
#             self.wall_y,
#             self.wall_thickness,
#             self.hole_thickness,
#             self.vertical_holes,
#             self.horizontal_holes,
#         )

#         # Goal
#         ax.add_patch(plt.Circle(self.goal_pos, 0.03, color="green", label="Goal"))

#         # Agent
#         ax.add_patch(plt.Circle(self.agent_pos, 0.03, color="blue", label="Agent"))

#         plt.legend(loc="upper right")
#         plt.pause(0.01)

#     def close(self):
#         plt.close()


# def draw_walls(
#     ax,
#     wall_x,
#     wall_y,
#     wall_thickness,
#     hole_thickness,
#     vertical_holes,
#     horizontal_holes,
#     box_low=0.0,
#     box_high=1.0,
#     color="red",
#     alpha=0.6,
# ):
#     """
#     Dessine les murs vertical et horizontal avec leurs trous
#     entre box_low et box_high sur un axe matplotlib.
#     """
#     v_half = wall_thickness / 2
#     h_half = hole_thickness / 2

#     # Mur vertical avec trous
#     y_segments = [(box_low, box_high)]
#     for hole in vertical_holes:
#         center, height = hole["center"], hole["height"]
#         hole_min, hole_max = center - height / 2, center + height / 2
#         new_segments = []
#         for seg_min, seg_max in y_segments:
#             if seg_min < hole_min:
#                 new_segments.append((seg_min, min(seg_max, hole_min)))
#             if seg_max > hole_max:
#                 new_segments.append((max(seg_min, hole_max), seg_max))
#         y_segments = new_segments

#     for seg_min, seg_max in y_segments:
#         ax.add_patch(
#             plt.Rectangle((wall_x - v_half, seg_min), wall_thickness, seg_max - seg_min, color=color, alpha=alpha)
#         )

#     # Mur horizontal avec trous
#     x_segments = [(box_low, box_high)]
#     for hole in horizontal_holes:
#         center, width = hole["center"], hole["width"]
#         hole_min, hole_max = center - width / 2, center + width / 2
#         new_segments = []
#         for seg_min, seg_max in x_segments:
#             if seg_min < hole_min:
#                 new_segments.append((seg_min, min(seg_max, hole_min)))
#             if seg_max > hole_max:
#                 new_segments.append((max(seg_min, hole_max), seg_max))
#         x_segments = new_segments

#     for seg_min, seg_max in x_segments:
#         ax.add_patch(
#             plt.Rectangle((seg_min, wall_y - h_half), seg_max - seg_min, hole_thickness, color=color, alpha=alpha)
#         )


# class Continuous2DNavigationEnv(gym.Env):

#     def __init__(
#         self,
#         render_mode=False,
#         max_steps=500,
#         vertical_holes=None,
#         horizontal_holes=None,
#         terminated_cond=0.05,
#         box_low=0.0,
#         box_high=1.0,
#     ):

#         self.box_low = np.array([box_low, box_low], dtype=np.float32)
#         self.box_high = np.array([box_high, box_high], dtype=np.float32)

#         self.observation_space = spaces.Box(low=self.box_low, high=self.box_high, shape=(2,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(2,), dtype=np.float32)

#         self.terminated_cond = terminated_cond
#         self.goal_pos = np.array([0.8, 0.8]) * (box_high - box_low) + box_low  # mise à l’échelle

#         # Mur vertical
#         self.wall_x = (box_high - box_low) / 2 + box_low
#         self.wall_thickness = 0.02 * (box_high - box_low)
#         self.vertical_holes = vertical_holes or []

#         # Mur horizontal
#         self.wall_y = (box_high - box_low) / 2 + box_low
#         self.hole_thickness = 0.02 * (box_high - box_low)
#         self.horizontal_holes = horizontal_holes or []

#         self.max_steps = max_steps
#         self.current_step = 0
#         self.render_mode = render_mode
#         self.reset()

#     def reset(self, seed=None, options=None):
#         rng = np.random.default_rng(seed)
#         self.agent_pos = rng.uniform(self.box_low, self.box_high)
#         self.current_step = 0
#         return self.agent_pos, {}

#     def step(self, action):
#         proposed_pos = np.clip(self.agent_pos + action, self.box_low, self.box_high)

#         if self._collides_with_wall(self.agent_pos, proposed_pos):
#             proposed_pos = self._slide_along_wall(self.agent_pos, proposed_pos)

#         self.agent_pos = proposed_pos
#         self.current_step += 1

#         dist = np.linalg.norm(self.agent_pos - self.goal_pos)
#         reward = -(dist**2)

#         terminated = dist < self.terminated_cond
#         truncated = self.current_step >= self.max_steps

#         return self.agent_pos, reward, terminated, truncated, {}

#     def _slide_along_wall(self, old_pos, new_pos):
#         half_thickness = self.wall_thickness / 2
#         if old_pos[0] < self.wall_x:
#             new_pos[0] = min(new_pos[0], self.wall_x - half_thickness)
#         else:
#             new_pos[0] = max(new_pos[0], self.wall_x + half_thickness)
#         return new_pos

#     def _collides_with_wall(self, old_pos, new_pos):
#         v_half = self.wall_thickness / 2
#         h_half = self.hole_thickness / 2

#         # Mur vertical
#         if self.wall_x - v_half <= new_pos[0] <= self.wall_x + v_half:
#             allowed = False
#             for hole in self.vertical_holes:
#                 center, height = hole["center"], hole["height"]
#                 if center - height / 2 <= new_pos[1] <= center + height / 2:
#                     allowed = True
#                     break
#             if not allowed:
#                 return True

#         # Mur horizontal
#         if self.wall_y - h_half <= new_pos[1] <= self.wall_y + h_half:
#             allowed = False
#             for hole in self.horizontal_holes:
#                 center, width = hole["center"], hole["width"]
#                 if center - width / 2 <= new_pos[0] <= center + width / 2:
#                     allowed = True
#                     break
#             if not allowed:
#                 return True

#         return False

#     def render(self):
#         plt.clf()
#         ax = plt.gca()
#         ax.set_xlim(self.box_low[0], self.box_high[0])
#         ax.set_ylim(self.box_low[1], self.box_high[1])
#         ax.set_aspect("equal")

#         # Murs
#         draw_walls(
#             ax,
#             self.wall_x,
#             self.wall_y,
#             self.wall_thickness,
#             self.hole_thickness,
#             self.vertical_holes,
#             self.horizontal_holes,
#             box_low=self.box_low[0],
#             box_high=self.box_high[0],
#         )

#         # Goal
#         ax.add_patch(
#             plt.Circle(self.goal_pos, 0.03 * (self.box_high[0] - self.box_low[0]), color="green", label="Goal")
#         )

#         # Agent
#         ax.add_patch(
#             plt.Circle(self.agent_pos, 0.03 * (self.box_high[0] - self.box_low[0]), color="blue", label="Agent")
#         )

#         plt.legend(loc="upper right")
#         plt.pause(0.01)


class Continuous2DNavigationEnv(gym.Env):
    def __init__(
        self,
        render_mode=False,
        max_steps=500,
        vertical_walls=None,
        horizontal_walls=None,
        terminated_cond=0.07,
        box_low=0.0,
        box_high=1.0,
        action_low=-0.03,
        action_high=0.03,
        dim_action=2,
        goal_pos=[0.9, 0.9],
        init_pos=None,
    ):

        if init_pos:
            self.agent_pos_init = np.array(init_pos) * (box_high - box_low) + box_low
        self.init_pos = init_pos

        self.box_low = np.array([box_low, box_low], dtype=np.float32)
        self.box_high = np.array([box_high, box_high], dtype=np.float32)

        self.observation_space = spaces.Box(low=self.box_low, high=self.box_high, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(dim_action,), dtype=np.float32)

        self.thickness = np.max([np.abs(action_low), np.abs(action_high)])

        self.terminated_cond = terminated_cond

        self.goal_pos = np.array(goal_pos) * (box_high - box_low) + box_low

        # Définition des murs
        self.vertical_walls = vertical_walls or []
        self.horizontal_walls = horizontal_walls or []

        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        self.reset()

    def reset(self, seed=None, options=None):
        if self.init_pos:
            self.agent_pos = self.agent_pos_init
        else:
            rng = np.random.default_rng(seed)
            self.agent_pos = rng.uniform(self.box_low, self.box_high)
        self.current_step = 0
        return self.agent_pos, {}

    def step(self, action):
        proposed_pos = np.clip(self.agent_pos + action, self.box_low, self.box_high)

        if self._collides_with_walls(proposed_pos):
            proposed_pos = self._slide_along_walls(self.agent_pos, proposed_pos)

        self.agent_pos = proposed_pos
        self.current_step += 1

        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -(dist**2)

        terminated = dist < self.terminated_cond
        truncated = self.current_step >= self.max_steps

        return self.agent_pos, reward, terminated, truncated, {}

    def _collides_with_walls(self, pos):
        # Vérifie collision avec tous les murs
        for wall in self.vertical_walls:
            x, holes = wall["x"], wall.get("holes", [])
            half = self.thickness / 2
            if x - half <= pos[0] <= x + half:
                # Vérifie si pos est dans un trou
                if not any(
                    c - h / 2 <= pos[1] <= c + h / 2 for c, h in [(hole["center"], hole["height"]) for hole in holes]
                ):
                    return True

        for wall in self.horizontal_walls:
            y, holes = wall["y"], wall.get("holes", [])
            half = self.thickness / 2
            if y - half <= pos[1] <= y + half:
                if not any(
                    c - w / 2 <= pos[0] <= c + w / 2 for c, w in [(hole["center"], hole["width"]) for hole in holes]
                ):
                    return True
        return False

    def _slide_along_walls(self, old_pos, new_pos):
        # Stratégie simple : si collision, on bloque sur l’axe du mur
        for wall in self.vertical_walls:
            x = wall["x"]
            half = self.thickness / 2
            if x - half <= new_pos[0] <= x + half:
                if old_pos[0] < x:
                    new_pos[0] = x - half
                else:
                    new_pos[0] = x + half

        for wall in self.horizontal_walls:
            y = wall["y"]
            half = self.thickness / 2
            if y - half <= new_pos[1] <= y + half:
                if old_pos[1] < y:
                    new_pos[1] = y - half
                else:
                    new_pos[1] = y + half
        return new_pos

    def render(self):
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(self.box_low[0], self.box_high[0])
        ax.set_ylim(self.box_low[1], self.box_high[1])
        ax.set_aspect("equal")

        # Dessine murs verticaux
        for wall in self.vertical_walls:
            x, holes = wall["x"], wall.get("holes", [])
            segments = self._wall_segments(self.box_low[1], self.box_high[1], holes, "vertical")
            for y0, y1 in segments:
                ax.add_patch(plt.Rectangle((x - self.thickness / 2, y0), self.thickness, y1 - y0, color="black"))

        # Dessine murs horizontaux
        for wall in self.horizontal_walls:
            y, holes = wall["y"], wall.get("holes", [])
            segments = self._wall_segments(self.box_low[0], self.box_high[0], holes, "horizontal")
            for x0, x1 in segments:
                ax.add_patch(plt.Rectangle((x0, y - self.thickness / 2), x1 - x0, self.thickness, color="black"))

        # Goal
        ax.add_patch(plt.Circle(self.goal_pos, 0.03 * (self.box_high[0] - self.box_low[0]), color="green"))
        # Agent
        ax.add_patch(plt.Circle(self.agent_pos, 0.03 * (self.box_high[0] - self.box_low[0]), color="blue"))

        plt.pause(0.01)

    def _wall_segments(self, low, high, holes, orientation="vertical"):
        # Construit les segments pleins d’un mur avec trous
        holes = sorted(holes, key=lambda h: h["center"])
        segments = []
        cursor = low
        for hole in holes:
            c = hole["center"]
            size = hole["height"] if orientation == "vertical" else hole["width"]
            segments.append((cursor, c - size / 2))
            cursor = c + size / 2
        segments.append((cursor, high))
        return [(a, b) for (a, b) in segments if b > a]


def draw_walls(
    ax,
    vertical_walls,
    horizontal_walls,
    box_low=0.0,
    box_high=1.0,
    color="red",
    alpha=0.6,
    thickness=0.05,
):
    """
    Dessine tous les murs verticaux et horizontaux avec leurs trous
    entre box_low et box_high sur un axe matplotlib.

    Args:
        vertical_walls (list): liste de murs verticaux
            ex: [{"x": 0.5, "thickness": 0.02, "holes": [{"center":0.3,"height":0.2}]}]
        horizontal_walls (list): liste de murs horizontaux
            ex: [{"y": 0.5, "thickness": 0.02, "holes": [{"center":0.4,"width":0.2}]}]
    """
    # --- MURS VERTICAUX ---
    for wall in vertical_walls:
        x = wall["x"]
        holes = wall.get("holes", [])
        v_half = thickness / 2

        # Découpe en segments
        y_segments = [(box_low, box_high)]
        for hole in sorted(holes, key=lambda h: h["center"]):
            center, height = hole["center"], hole["height"]
            hole_min, hole_max = center - height / 2, center + height / 2
            new_segments = []
            for seg_min, seg_max in y_segments:
                if seg_min < hole_min:
                    new_segments.append((seg_min, min(seg_max, hole_min)))
                if seg_max > hole_max:
                    new_segments.append((max(seg_min, hole_max), seg_max))
            y_segments = new_segments

        # Dessine les segments
        for seg_min, seg_max in y_segments:
            ax.add_patch(
                plt.Rectangle(
                    (x - v_half, seg_min),
                    thickness,
                    seg_max - seg_min,
                    color=color,
                    alpha=alpha,
                )
            )

    # --- MURS HORIZONTAUX ---
    for wall in horizontal_walls:
        y = wall["y"]
        holes = wall.get("holes", [])
        h_half = thickness / 2

        # Découpe en segments
        x_segments = [(box_low, box_high)]
        for hole in sorted(holes, key=lambda h: h["center"]):
            center, width = hole["center"], hole["width"]
            hole_min, hole_max = center - width / 2, center + width / 2
            new_segments = []
            for seg_min, seg_max in x_segments:
                if seg_min < hole_min:
                    new_segments.append((seg_min, min(seg_max, hole_min)))
                if seg_max > hole_max:
                    new_segments.append((max(seg_min, hole_max), seg_max))
            x_segments = new_segments

        # Dessine les segments
        for seg_min, seg_max in x_segments:
            ax.add_patch(
                plt.Rectangle(
                    (seg_min, y - h_half),
                    seg_max - seg_min,
                    thickness,
                    color=color,
                    alpha=alpha,
                )
            )
