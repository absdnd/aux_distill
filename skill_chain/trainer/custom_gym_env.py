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
from omegaconf import OmegaConf
from habitat.config import read_write
from omegaconf.dictconfig import DictConfig
# from tests import dataset_test
import functools


if TYPE_CHECKING:
    from omegaconf import DictConfig

def make_gym_env_from_config(config, rank: int) -> gym.Env:
    return make_gym_from_config(config)

class CustomVectorEnvFactory(VectorEnvFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def assign_env_id_to_task(self, config, env_skills, allowed_skills, eval_allowed_skills):
        task_configs = []

        def rgetattr(obj, attr, *args):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)
            return functools.reduce(_getattr, [obj] + attr.split("."))
        

        def update_task_config(task_config, update_measure_k, update_measure_v):
            if isinstance(update_measure_v, (dict, DictConfig)):
                cur_k = update_measure_k
                for k, v in update_measure_v.items():
                    if isinstance(v, (dict, DictConfig)):
                        cur_k = f"{cur_k}.{k}"
                        update_task_config(task_config, cur_k, v)
                    else: 
                        rgetattr(task_config, update_measure_k)[k] = v

                return task_config
            else: 
                raise ValueError("update_measure_v must be a dict")

        for env_id, env_k in enumerate(allowed_skills):
            env_task = env_skills[env_k]
            task_config = config.copy()
            OmegaConf.set_readonly(task_config, False)
            task_config.task.task_id = env_task.task_id
            task_config.task.type = env_task.type
            task_config.task.reward_measure = env_task.reward_measure
            task_config.task.success_measure = env_task.success_measure
            task_config.task.task_spec_base_path = env_task.task_spec_base_path
            task_config.task.use_marker_options = env_task.get("use_marker_options", [])
            if env_task.get("pddl_def", None) is not None:
                task_config.task.start_template = env_task.pddl_def.start_template
                task_config.task.goal_template = env_task.pddl_def.goal_template
                task_config.task.sample_entities = env_task.pddl_def.sample_entities
            # During evaluation we want to allow the sensor to have all skills. # 
            if len(allowed_skills) <= 1:
                allowed_skills = eval_allowed_skills
            
            task_config.task.lab_sensors.hier_skill_sensor.allowed_skills = allowed_skills
            task_config.task.lab_sensors.meta_hier_skill_sensor.allowed_skills = allowed_skills
            task_config.task.measurements.task_stage_steps.allowed_skills = [*env_skills.keys()]
            task_config.task.measurements.task_stage_steps.allowed_skill_ids = [env_skills[skill_name].task_id for skill_name in env_skills]
            task_config.task.measurements.task_stage_steps.max_episode_steps = [env_skills[skill_name].max_episode_steps for skill_name in env_skills]

            
            for update_measure_k, update_measure_v in env_task.update_measures.items():
                task_config = update_task_config(task_config, update_measure_k, update_measure_v)
                # if isinstance(update_measure_v, (dict, DictConfig)):
                #     cur_k = update_measure_k
                #     for k, v in update_measure_v.items():
                #         if isinstance(v, (dict, DictConfig)):
                #             cur_k = f"{cur_k}.{k}"
                #             rgetattr(task_config, cur_k)[k] = v
                #         else:
                #             rgetattr(task_config, update_measure_k)[k] = v
                # else: 
                #     raise ValueError("update_measure_v must be a dict")
            # 
            task_config.task.task_spec = env_task.task_spec
            task_config.environment.max_episode_steps = env_task.max_episode_steps
            OmegaConf.set_readonly(task_config, True)
            task_configs.append(task_config)
        return task_configs
    
    def get_world_rank(self):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size

    def assign_seed_to_envs(self, num_environments, configs_list, scene_splits, is_first_rank, random_seed_by_rank):
        ret_config_list = []
        envs_per_task = num_environments // len(configs_list)
        
        rank, _ = self.get_world_rank()        
        for overall_env_idx in range(num_environments):
            task_id = overall_env_idx // envs_per_task
            task_config = configs_list[task_id].copy()
            env_idx = overall_env_idx % envs_per_task
            with read_write(task_config):
                if random_seed_by_rank:
                    task_config.seed = task_config.seed * (rank + 1)  + overall_env_idx
                else: 
                    task_config.seed = task_config.seed + overall_env_idx

                remove_measure_names = []
                if not is_first_rank:
                    remove_measure_names.extend(
                        task_config.task.rank0_measure_names
                    )
                if (env_idx != 0) or not is_first_rank:
                    remove_measure_names.extend(
                            task_config.task.rank0_env0_measure_names
                    )
                task_config.task.measurements = {
                    k: v
                    for k, v in task_config.task.measurements.items()
                    if k not in remove_measure_names
                }
                if len(scene_splits) > 0:
                    task_config.dataset.content_scenes = scene_splits[env_idx]
                ret_config_list.append(task_config)
        
        return ret_config_list

    def create_scene_splits(self, config, num_skills,enforce_scenes_greater_eq_environments):
        
        num_environments = config.habitat_baselines.num_environments // num_skills

        print("Num Environments: ", num_environments)
        print("Num Skills: ", num_skills)        
        dataset = make_dataset(config.habitat.dataset.type)
        scenes = config.habitat.dataset.content_scenes
        if "*" in config.habitat.dataset.content_scenes:
            scenes = dataset.get_scenes_to_load(config.habitat.dataset)

        if num_environments < 1:
            raise RuntimeError("num_environments must be strictly positive")

        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        random.shuffle(scenes)

        scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
        
        if len(scenes) < num_environments:
            msg = f"There are less scenes ({len(scenes)}) than environments ({num_environments}). "
            if enforce_scenes_greater_eq_environments:
                logger.warn(
                    msg
                    + "Reducing the number of environments to be the number of scenes."
                )
                num_environments = len(scenes)
                scene_splits = [[s] for s in scenes]
            else:
                logger.warn(
                    msg
                    + "Each environment will use all the scenes instead of using a subset."
                )
            for scene in scenes:
                for split in scene_splits:
                    split.append(scene)
        else:
            for idx, scene in enumerate(scenes):
                scene_splits[idx % len(scene_splits)].append(scene)
            assert sum(map(len, scene_splits)) == len(scenes)

        return scene_splits
    
    # Construct Environments #     
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
    ) -> VectorEnv:
        configs = []

        # Scene Splitting from the environments # 
        print("Allowed_Skills", config.env_tasks.allowed_skills)
        scene_splits = self.create_scene_splits(config, len(config.env_tasks.allowed_skills), enforce_scenes_greater_eq_environments)
        if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
            logger.warn(
                "Using the debug Vector environment interface. Expect slower performance."
            )
            vector_env_cls = ThreadedVectorEnv
        else:
            vector_env_cls = VectorEnv
        
        max_episode_steps = {
            k: v.max_episode_steps for k, v in config.env_tasks.skills.items()
        }
        
        habitat_env_configs = self.assign_env_id_to_task(
            config.habitat, 
            env_skills = config.env_tasks.skills, 
            allowed_skills = config.env_tasks.allowed_skills, 
            eval_allowed_skills=config.env_tasks.eval_allowed_skills, 
        )
        
        config_list = self.assign_seed_to_envs(
            num_environments = config.habitat_baselines.num_environments,  
            configs_list=habitat_env_configs, 
            scene_splits=scene_splits,
            is_first_rank = is_first_rank, 
            random_seed_by_rank = config.env_tasks.random_seed_by_rank
        )

        # Running tests in the beginning of training # 
        envs = vector_env_cls(
            make_env_fn=make_gym_env_from_config,
            env_fn_args=tuple(
                (config_list[env_rank], env_rank)
                for env_rank in range(config.habitat_baselines.num_environments)
            ),
            workers_ignore_signals=workers_ignore_signals,
        )
        
        # envs = vector_env_cls(
        #     make_env_fn=make_gym_env_from_config,
        #     env_fn_args=tuple(
        #         (habitat_env_configs[env_rank % len(config.env_tasks.allowed_skills)], env_rank)
        #         for env_rank in range(config.habitat_baselines.num_environments)
        #     ),
        #     workers_ignore_signals=workers_ignore_signals,
        # )

        if config.habitat.simulator.renderer.enable_batch_renderer:
            envs.initialize_batch_renderer(config)

        return envs
    

