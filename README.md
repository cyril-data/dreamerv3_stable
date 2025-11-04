# Dreamer V3
 
This project provides a reinforcement learning training framework for action continouous GridWorld environments using Stable Baselines3. It supports various algorithms and allows for customizable configurations through command line arguments and YAML configuration files.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Environment Wrappers](#environment-wrappers)
- [Callbacks](#callbacks)
- [Logger](#logger)
- [Policy Configuration](#policy-configuration)
- [Model Configuration](#model-configuration)
- [License](#license)

## Installation

To run this project, you will need Python 3.10 or higher. It is recommended to create a virtual environment for managing dependencies. You can install the required packages using pip:

```bash
pip install -r requirements.txt
conda install -c conda-forge box2d-py # need instal in conda env 
```

Ensure you have the necessary libraries, including `stable_baselines3`, `gymnasium`, and `minigrid`.


Warning : if 
```
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
X Error of failed request:  BadValue (integer parameter out of range for operation)
  Major opcode of failed request:  152 (GLX)
  Minor opcode of failed request:  3 (X_GLXCreateContext)
  Value in failed request:  0x0
  Serial number of failed request:  186
  Current serial number in output stream:  187
```
Try `conda install -c conda-forge libstdcxx-ng` if you're working with conda, or tcheck solution on : 
https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris 


## Usage

To run the training process, execute the main script with appropriate command line arguments. Here’s a basic command to start training:

```bash
python train.py --conf config/config_ppo.yml
```

You can overright parameters by adding some parameters. 

You can customize the arguments based on your requirements. Use `--help` to see all available options:

```bash
python train.py --help
```

## Configuration

The training parameters can be specified through a YAML configuration file or via command line arguments. The following parameters are available:

<!-- - `--conf`: Path to the YAML configuration file.
- `--env`: Name of the environment (e.g., `MiniGrid-Empty-5x5-v0`).
- `--alg`: The reinforcement learning algorithm to use (`ppo`, `dqn`, `lsfm`, `sacd`).
- `--lr`: Learning rate for the optimizer.
- `--it`: Total number of iterations for training.
- `--bs`: Batch size for the training.
- `--ef`: Exploration fraction.
- `--ec`: Entropy coefficient (if applicable).
- `--ob`: Observation type: `partial`, `full`, or `pos`.
- `--ac`: Action type: `reduced` or `full`.
- `--rw`: Reward type: `constant` or `None`.
- `--ui`: Target update interval.
- `--manual`: Enable manual control.
- `--load`: Directory path to load model checkpoint.
- `--na`: net_arch of the policy.
- `--fn` : activation_fn of the policy.
- `--fd` : feature dim
- `--fb` : feature exctrator dim before lsfm
- `--xp` : experience name
- `--run` : run name
- `--ql` : Learn a linear Q-value from latent space
- `--pl` : , help="n step planification for each real sample batch
- `--qi` : , help="q_value iteration for each real sample batch
- `--wa` : , help="iteration from which qlearning start -->



## Callbacks

The framework includes several callback classes to enhance training:

- **MyEvalCallback**: Evaluates the model during training and saves the best model.
- **SaveConfigCallback**: Saves the configuration used for training.

## Logger

Logging is set up using `MLflow` and console output. You can monitor training progress through the logs, which include both human-readable formats and MLflow tracking.

## Policy Configuration

<!-- You can configure policy feature extractors based on the observation type. The framework uses `MinigridFeaturesExtractor` and `MinigridFeaturesPositionExtractor` to handle feature extraction. -->

## Model Configuration

The framework allows you to configure and instantiate various reinforcement learning models. It supports algorithms such as PPO, SAC, **but built to test DreamerV3**. Models can be loaded from saved checkpoints to continue training.

## Run on supercomputers

You would need to load this specific modules : 
```
module load python/3.10.4
module load openmpi
```

<!-- And you can adapt your own script in `./script` dir.  -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

