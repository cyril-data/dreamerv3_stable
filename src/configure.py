import argparse
import yaml
import sys
import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from sac_sf.sac_sf import SAC_SF
from ppo_sf.ppo_sf import PPO_SF

from env.nav_2d import Continuous2DNavigationEnv

# from lsfm.lsfm import LSFM
# from sacd.sacd import SACD
from src.loggers import MLflowOutputFormat
from stable_baselines3.common.logger import HumanOutputFormat, Logger

from stable_baselines3.common.policies import BasePolicy

from gymnasium.spaces import Discrete, Box, MultiDiscrete, MultiBinary

from src.policy_feature_extractor import LSFMFeaturesExtractor

from src.callbacks import (
    make_file_path,
    SaveConfigCallback,
    SaveBestNModelsCriticLoss,
    EigenSFOption,
    StateCoverageCallback,
    SaveBestNModelsValueLoss,
    SaveBestNPsiModels,
)
from minigrid.manual_control import ManualControl
import os
import torch as th
from torch import nn
import mlflow
from torchsummary import summary

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import get_device

from stable_baselines3.common.callbacks import EvalCallback


def inspect_action_space(env):
    action_space = env.action_space
    print("--- ACTION DIM ---")
    if isinstance(action_space, Discrete):
        print("Type d'espace d'action : Discrete")
        print("Nombre d'actions possibles :", action_space.n)
        action_dim = action_space.n

    elif isinstance(action_space, Box):
        print("Type d'espace d'action : Box (continu)")
        print("Shape de l'action :", action_space.shape)
        print("Valeurs min :", action_space.low)
        print("Valeurs max :", action_space.high)
        action_dim = action_space.shape

    elif isinstance(action_space, MultiDiscrete):
        print("Type d'espace d'action : MultiDiscrete")
        print("Nombres d'actions possibles par dimension :", action_space.nvec)
        action_dim = action_space.nvec

    elif isinstance(action_space, MultiBinary):
        print("Type d'espace d'action : MultiBinary")
        print("Nombre total de bits :", action_space.n)
        action_dim = action_space.n

    else:
        print("Type d'espace d'action inconnu")
        action_dim = None

    return action_dim


def configure_callback(env, args, model_path: str = None):

    callbacks = [
        SaveConfigCallback(args, model_path),
        SaveBestNModelsCriticLoss(dir_path=model_path),
        StateCoverageCallback(env, dir_path=model_path),
        SaveBestNModelsValueLoss(dir_path=model_path),
        SaveBestNPsiModels(dir_path=model_path),
    ]

    if args.alg == "sac_sf":
        callbacks.append(EigenSFOption(dir_path=model_path, last_step=args.it))

    return callbacks


def configure_logger(args):
    """Sets up logging with MLflow and console output."""

    # experince name
    if args.xp is not None:
        experiment_name = args.xp
        mlflow.set_experiment(experiment_name)

    # run name
    if args.run is not None:
        run_name = args.run
    else:
        run_name = None

    return (
        Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(sys.stdout),
                MLflowOutputFormat(vars(args)),
            ],
        ),
        run_name,
    )


def configure_policy(args, env):
    if args.alg == "sac_sf":
        extractor_kwargs = {"features_dim": args.fd, "value_critic_no_share_sf_features_extractor": True}

        policy_kwargs = {
            "normalize_images": False,
            "features_extractor_kwargs": extractor_kwargs,
            "net_arch": [args.fd],
        }
        # if args.lf:
        #     if args.lf > 0.0:
        #         policy_kwargs["uniform_policy"] = True

    elif args.alg == "ppo_sf":
        extractor_kwargs = {"features_dim": args.fd, "num_layers": args.nl}
        if "ffn" in args:
            if args.ffn is not None:
                extractor_kwargs["activation_fn"] = eval(args.ffn)

        if "ll" in args:
            if args.ll is not "[]":
                extractor_kwargs["list_layers"] = eval(args.ll)
                extractor_kwargs["features_dim"] = eval(args.ll + "[-1]")

        policy_kwargs = {
            "normalize_images": False,
            "net_arch": eval(args.na),
            "share_features_extractor": args.sfe,
            "features_extractor_kwargs": extractor_kwargs,
            "freeze_feature": args.ff,
        }

        if "fn" in args:
            if args.fn is not None:
                policy_kwargs["activation_fn"] = eval(args.fn)

    else:
        policy_kwargs = {"normalize_images": False, "net_arch": eval(args.na), "share_features_extractor": args.sfe}

    return policy_kwargs  # , policy_kwargs_lsfm


def configure_env(args):
    """
    Initializes the environment based on the specified configurations.

    Parameters:
        env_name (str): The name of the environment.
        args (dict): Additional arguments to configure the environment.

    Returns:
        env: The initialized environment.
    """
    env_name = args.env
    # Determine the render mode
    render_mode = "human" if "eval" in args or ("manual" in args and args.manual) else None

    # Map environment names to their corresponding classes
    custom_envs = {
        "nav_2d": Continuous2DNavigationEnv,
    }

    # Create the appropriate environment
    if env_name in custom_envs:
        env_class = custom_envs[env_name]
        env_kwarg = {"render_mode": render_mode}

        if "hw" in args:
            env_kwarg["horizontal_walls"] = args.hw
        if "vw" in args:
            env_kwarg["vertical_walls"] = args.vw
        if "bl" in args:
            env_kwarg["box_low"] = args.bl
        if "bh" in args:
            env_kwarg["box_high"] = args.bh

        if "gp" in args:
            env_kwarg["goal_pos"] = eval(args.gp)

        if "al" in args:
            env_kwarg["action_low"] = -abs(args.al)
            env_kwarg["action_high"] = abs(args.al)

        if "ip" in args:
            if args.ip is not None:
                env_kwarg["init_pos"] = eval(args.ip)

        env = env_class(**env_kwarg)
    else:
        env = gym.make(env_name, render_mode=render_mode) if render_mode else gym.make(env_name)

    action_dim = inspect_action_space(env)

    # Enable manual control if specified
    if "manual" in args and args.manual == True:
        print("Manual control enabled")
        manual_control = ManualControl(env, seed=42)
        manual_control.start()

    # Parcours de la chaîne des wrappers
    current_env = env
    while hasattr(current_env, "env"):
        print(type(current_env))
        current_env = current_env.env  # Passe au wrapper interne

    return env


def find_and_set_device(policy):
    # attributs name in class model.policy
    for attr_name in dir(policy):
        if "net" in attr_name:
            attr_value = getattr(policy, attr_name)
            # tcheck if it's a BasePolicy instance
            if isinstance(attr_value, BasePolicy):
                # Get the first parameter to know the device
                device = next(attr_value.parameters()).device
                print(f"Device for '{attr_name}': {device}")
                return device
    print("No suitable 'net' attribute found.")
    return None


def replace_obs_wrapper(env, target_wrapper_type, new_wrapper_factory, *args, **kwargs):
    """
    Replace the last occurrence of a specific wrapper with a new wrapper.

    Args:
        env: The environment to modify.
        target_wrapper_type: The type of the wrapper to replace.
        new_wrapper_factory: A function to create the new wrapper.
        *args, **kwargs: Additional arguments to pass to the new wrapper's constructor.

    Returns:
        The modified environment.
    """
    if isinstance(env, target_wrapper_type):
        # Replace the wrapper directly if it's the target type
        env = new_wrapper_factory(env.env, *args, **kwargs)
    elif isinstance(env, gym.Wrapper):
        # Recursively replace in the wrapped environment
        env.env = replace_obs_wrapper(env.env, target_wrapper_type, new_wrapper_factory, *args, **kwargs)
    return env


def configure_model(env, policy_kwargs, args):
    """Instantiates the appropriate model with provided configurations."""
    alg_name = args.alg

    model_args = {
        "policy": "MlpPolicy",  # if args.ob == "pos" else "CnnPolicy",
        "env": env,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
    }

    # Dictionnaire pour associer chaque attribut `args` avec une clé `model_args`
    attributes = {
        "lr": "learning_rate",
        "bs": "batch_size",
        "ef": "exploration_fraction",
        "ui": "target_update_interval",
        "bf": "buffer_size",
        "ga": "gamma",
    }

    if alg_name == "sac" or alg_name == "ppo_sf":
        attributes["ec"] = "ent_coef"

    if alg_name == "ppo_sf":
        attributes["sl"] = "is_sf_loss_train"

        if "sl" in args:
            if "vl" in args:
                if args.vl:
                    args.sl = False
    # Ajouter les attributs dépendants de alg_name
    if alg_name == "sacd" or alg_name == "sac_sf":
        attributes["ec"] = "ent_coef"
        attributes["lf"] = "lambda_sf_q"
        attributes["ns"] = "n_step_change_eigenvector"
        attributes["et"] = "eta"  # lenght explore step
        attributes["em"] = "eigen_max_option"  # eigen_max_option

    if alg_name == "lsfm":
        attributes["ql"] = "only_qlearning"
        attributes["pl"] = "n_step_planification"
        attributes["qi"] = "policy_optimization_step"
        attributes["wa"] = "warmup_DQN_learning_it"
        attributes["ta"] = "tau"
        attributes["bm"] = "beta_model_sample"
        attributes["ae"] = "alpha_exploration_rollout"
        attributes["dl"] = "dyna_lossphi_criterium"
        attributes["an"] = "alpha_n"
        attributes["ap"] = "alpha_psi"
        attributes["ar"] = "alpha_r"
        attributes["af"] = "alpha_next_phi"

    # Boucle sur les attributs pour les ajouter ou les supprimer si None
    for attr, key in attributes.items():
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value is not None:
                model_args[key] = value
                print(f"attr {attr}  key {key}, value {value}")
            else:
                model_args.pop(key, None)

    # Select and instantiate the model
    model_cls = {"ppo": PPO, "dqn": DQN, "sac": SAC, "sac_sf": SAC_SF, "ppo_sf": PPO_SF}.get(
        alg_name
    )  # , "dqn_feature": DQN_feature
    if model_cls is None:
        raise ValueError(f"Unsupported algorithm: {alg_name}")

    print("\n\n\n model_args", model_args)

    model = model_cls(**model_args)

    observation_shape = env.observation_space.shape

    # Obtenez l'espace d'action
    action_space = env.action_space

    # Affichez la forme de l'action
    if hasattr(action_space, "shape"):
        print("Action Shape :", action_space.shape)
    else:
        print("Discrete Action space. Possible actions :", action_space.n)

    # Accéder à l'attribut "net" dans model.policy
    if hasattr(model.policy, "net"):
        net = getattr(model.policy, "net")
    else:
        print("no net attribute in model.policy")

    # Parcourir les modules enfants pour trouver un attribut contenant "net"
    for name, module in model.policy.named_children():
        if "net" in name:
            print(f"Net module : {name} = {module}")

    if args.load is not None:
        if os.path.isdir(args.load):
            model_path = os.path.join(args.load, "best_model")
        else:
            model_path = os.path.splitext(args.load)[0] if args.load.endswith(".zip") else args.load

        assert os.path.isfile(
            model_path + ".zip"
        ), "arg --load must be a dir with best_model.zip inside, or the zip model file do not exist"

        # set device to cpu if cuda is not available
        device = get_device(device="auto")
        data, params, pytorch_variables = load_from_zip_file(
            os.path.join(model_path),
            device=device,
            custom_objects=None,
            print_system_info=False,
        )

        model.set_parameters(params, exact_match=False, device=device)

        print(f"Model weight are loadded from {os.path.join(model_path)}.zip")

    num_of_parameters = sum(map(th.numel, model.policy.parameters()))
    print("Total number of parameters", num_of_parameters, "\n")

    # print("self.actor", model.policy.actor)
    # print("self.critic", model.policy.critic)
    # print("self.critic_sf", model.policy.critic_sf)

    return model


def configure_checkpoint_model(model, run_dir, filename):
    """Saves the model to a specified path."""
    model_path = make_file_path(run_dir, filename)
    model.save(model_path)
    print(f"Model saved at {model_path}")


def custom_type(value):
    try:
        # Essayer de convertir en float
        return float(value)
    except ValueError:
        # Si ça échoue, retourner la chaîne de caractères
        return value


def parse_args():
    """Parse and return command line arguments and YAML configuration."""

    # Étape 1 : Charger uniquement l'argument `--conf` pour obtenir le fichier de configuration YAML
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        default="config/config_sac.yml",
        help="Path to the YAML configuration file. Default: 'config/config_ppo.yml'.",
    )
    args, remaining_argv = parser.parse_known_args()

    # Étape 2 : Charger le YAML spécifié dans `--conf`
    with open(args.conf, "r") as file:
        param_yml = yaml.safe_load(file)

    # Étape 3 : Créer le parser principal avec les valeurs par défaut du YAML si elles existent
    parser = argparse.ArgumentParser()

    def add_argument_with_default(name, default_value, **kwargs):
        """Add default value only if there is no argument in the yaml file"""
        parser.add_argument(name, default=param_yml.get(name.lstrip("--"), default_value), **kwargs)

    def add_boolean_argument_with_default(name, default_value, **kwargs):
        """Add bool argument only if there is no argument in the yaml file."""
        yaml_value = param_yml.get(name.lstrip("--"), default_value)

        # Vérification avant d'ajouter l'argument
        if name not in {opt.option_strings[0] for opt in parser._actions}:
            if yaml_value is False:
                parser.add_argument(name, action="store_true", help="Active l'option ql")
            else:
                parser.add_argument(
                    name,
                    action="store_true",
                    default=True,
                    help=f"Active l'option {name} avec une valeur par défaut à True",
                )

    # Add arguments with défaut value if specified in the YAML
    add_argument_with_default("--env", None, type=str, help="Name of the environment (e.g., 'MiniGrid-Empty-5x5-v0').")

    add_argument_with_default("--bl", 0.0, type=float, help="env : box_low")
    add_argument_with_default("--bh", 1.0, type=float, help="env : box_high")

    add_argument_with_default(
        "--alg", None, type=str, help="Algorithm name to use for training (options: 'ppo', 'dqn', 'lsfm', 'sacd')."
    )

    add_argument_with_default("--lr", None, type=float, help="Learning rate for the model (e.g., 0.0003).")
    add_argument_with_default("--it", None, type=int, help="Total number of iterations for training (e.g., 1000000).")
    add_argument_with_default("--bs", None, type=int, help="Batch size for training (e.g., 64).")
    add_argument_with_default("--ef", None, type=float, help="Fraction of exploration during training (e.g., 0.1).")
    add_argument_with_default("--ec", None, type=float, help="Entropy coefficient for SAC algorithm (e.g., 0.01).")
    add_argument_with_default("--ob", None, type=str, help="Observation type. Options: 'partial', 'full', or 'pos'.")
    add_argument_with_default("--ac", None, type=str, help="Action type. Options: 'reduced' or 'full'.")
    add_argument_with_default("--rw", None, type=str, help="Reward type. Options: 'constant' or 'None'.")
    add_argument_with_default("--ui", None, type=int, help="Target update interval (e.g., every 1000 iterations).")
    add_argument_with_default("--load", None, type=str, help="Directory path to load the model checkpoint from.")
    add_argument_with_default(
        "--fm", None, type=str, help="Model.pth path to load the feature model in env_obs_wrapper for dqn."
    )
    add_argument_with_default("--na", "[]", type=str, help="net_arch of the policy.")
    add_argument_with_default("--ll", "[]", type=str, help="list layer for feature architecture.")
    add_argument_with_default("--ffn", None, type=str, help="activation_fn of the feature.")

    add_argument_with_default("--fn", None, type=str, help="activation_fn of the policy.")
    add_argument_with_default("--fd", 32, type=int, help="feature dim")
    add_argument_with_default("--fb", None, type=int, help="feature exctrator dim before lsfm")
    add_argument_with_default("--xp", None, type=str, help="experience name")
    add_argument_with_default("--run", None, type=str, help="run name")

    add_boolean_argument_with_default("--manual", False, help="Enable manual control of the environment.")
    add_boolean_argument_with_default("--ql", False, help="Learn a linear Q-value from latent space.")

    add_argument_with_default("--pl", None, type=int, help="n step planification for each real sample batch")
    add_argument_with_default("--qi", None, type=int, help="q_value iteration for each real sample batch")
    add_argument_with_default("--wa", None, type=int, help="iteration from which qlearning start")

    add_argument_with_default("--bf", None, type=int, help="Buffer size for training (e.g., 10000).")
    add_argument_with_default("--ga", None, type=float, help="gamma (e.g., 0.99).")
    add_argument_with_default(
        "--ta",
        1.0,
        type=float,
        help="the soft update coefficient 'Polyak update', between 0 and 1) default 1 for hard update",
    )
    add_argument_with_default("--nl", 2, type=int, help="num layer for feature dim")
    add_argument_with_default(
        "--ea",
        10,
        type=int,
        help="num  n_eval_episodes: specifying the number of episodes to average the reported success rate in evaluation mode",
    )
    add_argument_with_default("--bm", None, type=float, help="Model/Real tradeoff for sample batch (alg MBMFPO)")
    add_argument_with_default(
        "--ae", None, type=float, help="Exploration/exploitation tradeoff for rollout (alg MBMFPO)"
    )
    add_argument_with_default("--dl", None, type=float, help="Philoss criterium to start Dyna training")
    add_argument_with_default(
        "--lf",
        0.0,
        type=float,
        help="lambda_sf_q criterium to orient actor from exploration (=1) or exploitation (=0) ",
    )

    add_boolean_argument_with_default("--vs", False, help="value_critic_no_share_sf_features_extractor.")

    add_boolean_argument_with_default("--sfe", False, help="share_features_extractor.")
    add_boolean_argument_with_default("--ff", False, help="freeze_feature")

    add_boolean_argument_with_default("--sl", False, help="is_sf_loss_train")
    add_boolean_argument_with_default("--vl", False, help="is_vf_loss_train")

    add_argument_with_default("--vw", None, help="vertical_walls ")
    add_argument_with_default("--hw", None, help="horizontal_walls  ")

    add_argument_with_default("--an", 1.0, type=float, help="alpha_n : loss coef for n")
    add_argument_with_default("--ap", 0.01, type=float, help="alpha_psi : loss coef for psi")
    add_argument_with_default("--ar", 0.10, type=float, help="alpha_r : loss coef for r")
    add_argument_with_default("--af", 0.01, type=float, help="alpha_next_phi : loss coef for next_phi")

    add_argument_with_default("--ns", 200, type=int, help="n_step_change_eigenvector")
    add_argument_with_default("--et", 0.1, type=float, help="eta")
    add_argument_with_default("--em", 10, type=int, help="eigen_max_option")

    add_argument_with_default("--gp", "[0.9,0.9]", type=str, help="goal_pos.")
    add_argument_with_default("--al", 0.03, type=float, help="action limit")
    add_argument_with_default("--ip", None, type=str, help="initial position. random if None")

    # joined = "_" + "_".join([f"{k.lstrip('-')}{v}" for k, v in zip(remaining_argv[::2], remaining_argv[1::2])])
    joined = "_" + "_".join(
        [
            k.lstrip("-") if k == "--load" or k == "--fm" or k == "--run" else f"{k.lstrip('-')}{v}"
            for k, v in zip(remaining_argv[::2], remaining_argv[1::2])
        ]
    )
    joined = joined.removesuffix("_run")
    if joined != "_":
        if "--run" not in remaining_argv:

            remaining_argv.append("--run")
            remaining_argv.append(joined)
        else:
            index = remaining_argv.index("--run")
            run_name = remaining_argv[index + 1]
            remaining_argv[index + 1] = "".join([run_name, joined])

    print("remaining_argv", remaining_argv)

    # Étape 4 : Parser tous les arguments finaux avec priorité à la ligne de commande
    args = parser.parse_args(remaining_argv)

    # Conversion en dictionnaire pour afficher la configuration complète
    params = vars(args)
    print("Configuration parameters:", params)
    print("alg :", params["alg"])
    print("run :", params["run"])

    if params["run"]:
        params["run"] = params["env"] + params["alg"] + params["run"]
    else:
        params["run"] = params["env"] + params["alg"]

    return argparse.Namespace(**params)
