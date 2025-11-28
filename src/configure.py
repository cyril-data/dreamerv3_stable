import argparse
import yaml
import sys
import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from src.utils import inspect_action_space
from dreamerv3.dreamerv3 import DreamerV3
from env.nav_2d import Continuous2DNavigationEnv
from src.loggers import MLflowOutputFormat
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import get_device
from src.callbacks import (
    SaveConfigCallback,
    SaveBestNModelsCriticLoss,
    StateCoverageCallback,
    SaveBestNModelsValueLoss,
)
from minigrid.manual_control import ManualControl
import os
import torch as th
import mlflow
from stable_baselines3.common.env_util import make_vec_env


def configure_callback(env, args, model_path: str = None):
    callbacks = [
        SaveConfigCallback(args, model_path),
        SaveBestNModelsCriticLoss(dir_path=model_path),
        StateCoverageCallback(env, dir_path=model_path),
        SaveBestNModelsValueLoss(dir_path=model_path),
    ]
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

    policy_kwargs = {
        "normalize_images": False,
        "net_arch": eval(args.na),
    }
    if args.alg == "dreamerv3":
        if "latent_classes" in args:
            policy_kwargs["latent_classes"] = args.latent_classes
        if "recurrent_size" in args:
            policy_kwargs["recurrent_size"] = args.recurrent_size
        if "latent_length" in args:
            policy_kwargs["latent_length"] = args.latent_length
        if "recurrent_model" in args:
            policy_kwargs["recurrent_model_kwargs"] = args.recurrent_model
        if "prior_net" in args:
            policy_kwargs["prior_net_kwargs"] = args.prior_net
        if "posterior_net" in args:
            policy_kwargs["posterior_net_kwargs"] = args.posterior_net
        if "reward_net" in args:
            policy_kwargs["reward_net_kwargs"] = args.reward_net
        if "continue_net" in args:
            policy_kwargs["continue_net_kwargs"] = args.continue_net
        if "actor" in args:
            policy_kwargs["actor_kwargs"] = args.actor
        if "critic" in args:
            policy_kwargs["critic_kwargs"] = args.critic

    return policy_kwargs


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
    # arg env
    env_kwargs = {"render_mode": render_mode}
    if "hw" in args:
        env_kwargs["horizontal_walls"] = args.hw
    if "vw" in args:
        env_kwargs["vertical_walls"] = args.vw
    if "bl" in args:
        env_kwargs["box_low"] = args.bl
    if "bh" in args:
        env_kwargs["box_high"] = args.bh
    if "gp" in args:
        env_kwargs["goal_pos"] = eval(args.gp)
    if "al" in args:
        env_kwargs["action_low"] = -abs(args.al)
        env_kwargs["action_high"] = abs(args.al)
    if "ip" in args:
        if args.ip is not None:
            env_kwargs["init_pos"] = eval(args.ip)

    # Get number of environments (default to 1 if not specified)
    n_envs = getattr(args, "nenv", 1)

    # Create vectorized environment
    if env_name in custom_envs:
        env_class = custom_envs[env_name]
        vec_env = make_vec_env(env_class, n_envs=n_envs, seed=0, env_kwargs=env_kwargs)
    else:
        vec_env = make_vec_env(
            env_name, n_envs=n_envs, seed=0, env_kwargs={"render_mode": render_mode} if render_mode else {}
        )

    # Note: Manual control and inspection work differently with vectorized envs
    if n_envs == 1:
        # For single env, you can still access the underlying env
        single_env = vec_env.envs[0]
        inspect_action_space(single_env)

        if "manual" in args and args.manual:
            print("Manual control enabled")
            manual_control = ManualControl(single_env, seed=42)
            manual_control.start()
    else:
        # For multiple envs, inspect the first one
        inspect_action_space(vec_env.envs[0])
        if "manual" in args and args.manual:
            print("Warning: Manual control not supported with multiple environments")

    print("env_kwargs", env_kwargs)

    return vec_env

    # # Parcours de la chaîne des wrappers
    # current_env = env
    # while hasattr(current_env, "env"):
    #     print(type(current_env))
    #     current_env = current_env.env  # Passe au wrapper interne

    # return env


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
    model_cls = {"ppo": PPO, "dqn": DQN, "sac": SAC, "dreamerv3": DreamerV3}.get(  # , "dreamer": DreamerV3
        alg_name
    )  # , "dqn_feature": DQN_feature
    if model_cls is None:
        raise ValueError(f"Unsupported algorithm: {alg_name}")
    model = model_cls(**model_args)
    # Load model
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
    return model


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
    # env
    add_argument_with_default("--env", None, type=str, help="Name of the environment (e.g., 'MiniGrid-Empty-5x5-v0').")
    add_argument_with_default("--nenv", None, type=int, help="n_envs")
    add_argument_with_default("--bl", 0.0, type=float, help="env : box_low")
    add_argument_with_default("--bh", 1.0, type=float, help="env : box_high")
    add_boolean_argument_with_default("--manual", False, help="Enable manual control of the environment.")
    add_argument_with_default("--gp", "[0.9,0.9]", type=str, help="goal_pos.")
    add_argument_with_default("--al", 0.03, type=float, help="action limit")
    add_argument_with_default("--ip", None, type=str, help="initial position. random if None")
    add_argument_with_default("--vw", None, help="vertical_walls ")
    add_argument_with_default("--hw", None, help="horizontal_walls  ")

    # alg
    add_argument_with_default(
        "--alg", None, type=str, help="Algorithm name to use for training (options: 'ppo', 'dqn', 'lsfm', 'sacd')."
    )

    # model
    add_argument_with_default("--lr", None, type=float, help="Learning rate for the model (e.g., 0.0003).")
    add_argument_with_default("--it", None, type=int, help="Total number of iterations for training (e.g., 1000000).")
    add_argument_with_default("--bs", None, type=int, help="Batch size for training (e.g., 64).")
    add_argument_with_default("--load", None, type=str, help="Directory path to load the model checkpoint from.")
    add_argument_with_default("--na", "[64,64]", type=str, help="net_arch of the policy.")
    add_argument_with_default("--xp", None, type=str, help="experience name")
    add_argument_with_default("--run", None, type=str, help="run name")
    add_argument_with_default("--ga", None, type=float, help="gamma (e.g., 0.99).")
    add_argument_with_default("--latent_classes", None, help="latent_classes")
    add_argument_with_default("--latent_length", None, help="latent_length")
    add_argument_with_default("--recurrent_size", None, help="recurrent_size")
    add_argument_with_default("--recurrent_model", None, help="recurrent_model")
    add_argument_with_default("--prior_net", None, help="prior_net")
    add_argument_with_default("--posterior_net", None, help="posterior_net")
    add_argument_with_default("--reward_net", None, help="reward_net")
    add_argument_with_default("--continue_net", None, help="continue_net")
    add_argument_with_default("--actor", None, help="actor")
    add_argument_with_default("--critic", None, help="critic")

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

    # Étape 4 : Parser tous les arguments finaux avec priorité à la ligne de commande
    args = parser.parse_args(remaining_argv)

    # Conversion en dictionnaire pour afficher la configuration complète
    params = vars(args)
    print("Configuration parameters:", params)
    if params["run"]:
        params["run"] = params["env"] + params["alg"] + params["run"]
    else:
        params["run"] = params["env"] + params["alg"]

    return argparse.Namespace(**params)
