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

        # Begin training
        model.learn(total_timesteps=args.it, callback=callbacks)

    print("Running time :", time.time() - step_start_time)

    with open("temp_model_path.txt", "w") as f:
        f.write(model_path)

    del model


if __name__ == "__main__":
    config = parse_args()
    main(config)
