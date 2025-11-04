from stable_baselines3.common.logger import KVWriter
import mlflow
import numpy as np
from typing import Any, Dict, Tuple, Union


# class MLflowOutputFormat(KVWriter):
#     """
#     Dumps key/value pairs into MLflow's numeric format.
#     """

#     def __init__(self, params: Dict[str, Any] = {}) -> None:
#         self.params = params

#     def write(
#         self,
#         key_values: Dict[str, Any],
#         key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
#         step: int = 0,
#     ) -> None:

#         if len(self.params) > 0:
#             for k, v in self.params.items():
#                 mlflow.log_param(k, v)

#         for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

#             if excluded is not None and "mlflow" in excluded:
#                 continue

#             if isinstance(value, np.ScalarType):
#                 if not isinstance(value, str):
#                     mlflow.log_metric(key, value, step)


class MLflowOutputFormat(KVWriter):
    def __init__(self, params: Dict[str, Any] = {}) -> None:
        self.params = params

    def write(self, key_values, key_excluded, step=0) -> None:
        if len(self.params) > 0:
            for k, v in self.params.items():
                mlflow.log_param(k, v)

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "mlflow" in excluded:
                continue
            if isinstance(value, np.ScalarType) and not isinstance(value, str):
                mlflow.log_metric(key, value, step)

    # méthode spéciale qui bypass HumanOutputFormat
    def flush_only(self, key_values, key_excluded, step=0) -> None:
        self.write(key_values, key_excluded, step)
