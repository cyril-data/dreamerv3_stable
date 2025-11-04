import yaml
import argparse
import os
import numpy as np

from src.configure import (
    parse_args,
    configure_env,
    configure_model,
    configure_policy,
)
import torch
import matplotlib.pyplot as plt
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs


# Fonction pour afficher la chaîne de wrappers
def afficher_wrappers(env):
    current_env = env
    i = 0
    while hasattr(current_env, "env"):
        print(f"Wrapper {i}: {type(current_env).__name__}")
        current_env = current_env.env
        i += 1
    return f"Environnement de base : {type(current_env).__name__}"


def main(args):

    # Retrieve environment and algorithm names
    alg_name = args.env, args.alg

    # Initialize the environment
    env = configure_env(args)

    # Configure policy
    policy_kwargs = configure_policy(args, env)

    # Initialize and configure model
    model = configure_model(env, policy_kwargs, args)

    # print("\n\n afficher_wrappers ", afficher_wrappers(env))

    obs, _ = env.reset()

    rew_pred = []
    rew_true = []
    step_ep = 0

    for k in range(10000):
        step_ep += 1
        # while True:

        # print("obs", obs.shape)
        # print("env.observation_space", env.observation_space)
        # obs = maybe_transpose(obs, env.observation_space)

        # print("obs", obs.shape)

        # print("obs transpose", obs.shape)

        # if obs.shape[0] and env.spec.id.startswith("MiniGrid"):
        # obs = np.transpose(obs, (2, 0, 1))

        action, _states = model.predict(obs, deterministic=True)
        # device = next(model.policy.sf_net.parameters()).device

        # # phi_pred = output["Ma"][action]
        # print("action", action)

        # print("obs", obs[0])

        # # print("obs", torch.tensor(obs).unsqueeze(0).shape)
        # output = model.policy.sf_net(torch.tensor(obs).unsqueeze(0).to(device))
        # phi = output["phi"]
        # print("phi", phi)

        obs, reward, terminated, truncated, info = env.step(action)

        # print("*", action, k)
        # print("\n")

        # next_phi = output["phi"]

        # # print(f"\n *** distance = (phi - phi_pred) \n         : {abs(next_phi[0] - phi_pred[0])}")

        # reward_predict = output["ra"][0, action].item()
        # rew_pred.append(reward_predict)
        # rew_true.append(reward)

        # # print("action", action)

        # # print(f"psi: {output['fa'][0]}")

        # # print("reward_predict reward", reward_predict, reward)
        # if reward == 1:
        #     print("reward_predict POS reward", reward_predict, reward)
        env.render()

        if step_ep > 100:
            truncated = True
            step_ep = 0

        if terminated or truncated:
            obs, info = env.reset()

    # x = np.linspace(0, 50, 50)

    # fig, ax = plt.subplots()
    # ax.plot(x, rew_pred)
    # ax.plot(x, rew_true)
    # plt.show()


if __name__ == "__main__":

    config_base = parse_args()

    """ Load model config and model weight for running human render mode"""
    parser = argparse.ArgumentParser(prog="PROG", argument_default=argparse.SUPPRESS)
    parser.add_argument("--load", type=str, default="ppo_model_trained", help="Configuration file path")
    args = parser.parse_args()
    # Load YAML configuration and combine with command line arguments

    if os.path.isdir(args.load):
        config_path = os.path.join(args.load, "config.yml")

    elif args.load.endswith(".yaml") or args.load.endswith(".yml"):
        config_path = args.load

    elif args.load.endswith(".zip"):
        print("endswith zip")
        config_path = os.path.join(os.path.dirname(args.load), "config.yml")
    else:
        config_path = os.path.join(os.path.dirname(args.load + ".zip"), "config.yml")

    with open(config_path, "r") as file:
        param_yml = yaml.safe_load(file)

    config = {**vars(config_base), **param_yml}
    if "ql" in config:
        config["ql"] = True
    config = argparse.Namespace(**config)

    print("Loaded configuration parameters :", vars(config))

    config.eval = True  # Setting eval to True in the Namespace
    config.load = args.load  # Setting the model dir
    main(config)
