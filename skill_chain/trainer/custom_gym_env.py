import os
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Type

import gym
import hydra
import numpy as np
import torch
from gym import spaces
from habitat import ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat.config import read_write
from habitat.gym import make_gym_from_config
from habitat_baselines.common.env_factory import VectorEnvFactory

if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_gym_env_from_config(config, rank: int) -> gym.Env:
    return make_gym_from_config(config)


class CustomVectorEnvFactory(VectorEnvFactory):
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
    ) -> VectorEnv:
        configs = []

        if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
            logger.warn(
                "Using the debug Vector environment interface. Expect slower performance."
            )
            vector_env_cls = ThreadedVectorEnv
        else:
            vector_env_cls = VectorEnv

        return vector_env_cls(
            make_env_fn=make_gym_env_from_config,
            env_fn_args=tuple(
                (config.habitat, env_rank)
                for env_rank in range(config.habitat_baselines.num_environments)
            ),
            workers_ignore_signals=workers_ignore_signals,
        )
