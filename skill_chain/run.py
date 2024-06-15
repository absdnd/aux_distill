"""
Script to launch Habitat Baselines trainer.
"""

import os
import random

import gym
import hydra
import numpy as np
import torch
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default_structured_configs import \
    HabitatBaselinesConfigPlugin


import sys
sys.path.append(os.getcwd())
import skill_chain.trainer
from skill_chain import default_structured_configs

gym.logger.set_level(40)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg):
    cfg = patch_config(cfg)
    random.seed(cfg.habitat.seed)
    np.random.seed(cfg.habitat.seed)
    torch.manual_seed(cfg.habitat.seed)

    if cfg.habitat_baselines.force_torch_single_threaded and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(cfg.habitat_baselines.trainer_name)
    trainer = trainer_init(cfg)

    if cfg.habitat_baselines.evaluate:
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    main()
