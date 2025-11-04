from gymnasium.spaces import Discrete, Box, MultiDiscrete, MultiBinary
import gymnasium as gym


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


def make_file_path(path_dir, file_name):
    if path_dir.startswith("file://"):
        input_string = path_dir[7:]

    file_path = "/".join((input_string, file_name))
    return file_path


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


# Fonction pour lister les wrappers
def list_wrappers(env):
    wrappers = []
    while isinstance(env, gym.Wrapper):
        wrappers.append(type(env).__name__)
        env = env.env  # Accéder à l'environnement enveloppé
    return wrappers
