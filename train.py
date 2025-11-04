import mlflow
import time
from src.configure import (
    parse_args,
    configure_callback,
    configure_env,
    configure_logger,
    configure_model,
    configure_policy,
)
import gymnasium as gym
import os


# Fonction pour lister les wrappers
def list_wrappers(env):
    wrappers = []
    while isinstance(env, gym.Wrapper):
        wrappers.append(type(env).__name__)
        env = env.env  # Accéder à l'environnement enveloppé
    return wrappers


def analyser_parametres_partages(model):
    policy = model.policy

    print("=== FEATURES EXTRACTORS ===")
    print(f"pi_features_extractor id: {id(policy.pi_features_extractor)}")
    print(f"vf_features_extractor id: {id(policy.vf_features_extractor)}")
    print(f"Partagé: {id(policy.pi_features_extractor) == id(policy.vf_features_extractor)}")

    print("\n=== MLP EXTRACTOR ===")
    print(f"mlp_extractor id: {id(policy.mlp_extractor)}")
    print(f"latent policy_net params: {sum(p.numel() for p in policy.mlp_extractor.latent_policy_net.parameters())}")
    print(f"latent value_net params: {sum(p.numel() for p in policy.mlp_extractor.latent_value_net.parameters())}")
    print(f"latent sf_net params: {sum(p.numel() for p in policy.mlp_extractor.latent_sf_net.parameters())}")

    print("\n=== OPTIMISEUR ===")
    print(f"Optimiseur unique: {id(policy.optimizer)}")
    print(f"Nombre total de paramètres: {sum(p.numel() for p in policy.parameters())}")

    print("\n=== DÉTAIL DES PARAMÈTRES ===")
    for name, param in policy.named_parameters():
        print(f"{name}: {param.shape} - {param.numel()} params")

    pytorch_total_params = sum(p.numel() for p in policy.parameters())

    print("Total params", pytorch_total_params)

    print("model.policy", model.policy)


def main(args):

    step_start_time = time.time()

    # Set up logger
    logger, run_name = configure_logger(args)

    # --- Learning Phase ---
    with mlflow.start_run(run_name=run_name) as run:
        # Initialize the environment
        env = configure_env(args)

        # Configure policy
        policy_kwargs = configure_policy(args, env)

        # Initialize and configure model
        model = configure_model(env, policy_kwargs, args)

        # Configure callbacks
        model_path = run.info.artifact_uri.replace("file://", "")
        callbacks = configure_callback(env, args, model_path=model_path)

        # logger
        model.set_logger(logger)

        print("-" * 80)
        print("-" * 80)
        print(
            f"Même objet features_extractor: {id(model.policy.pi_features_extractor) == id(model.policy.vf_features_extractor)}"
        )
        analyser_parametres_partages(model)
        # Begin training
        model.learn(total_timesteps=args.it, callback=callbacks)

    print("Running time :", time.time() - step_start_time)

    with open("temp_model_path.txt", "w") as f:
        f.write(model_path)

    del model


if __name__ == "__main__":
    config = parse_args()
    main(config)
