
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.common.env_factory import VectorEnvFactory
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only
from typing import Dict
import torch
from habitat.utils import profiling_wrapper
from habitat_baselines.utils.timing import g_timer
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
)
import re
import numpy as np
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_infos,
)
import os
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
import functools
from copy import deepcopy
from skill_chain.common.normalized_rollout_storage import NormalizedReturnRolloutStorage


@baseline_registry.register_trainer(name="aux_trainer")
class AuxDistillPPOTrainer(PPOTrainer): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = len(self.config.env_tasks)
        print("Num tasks: ", self.num_tasks)
        print("Max skills: ", self.config.habitat.task.lab_sensors.env_task_sensor.max_skills)
        print("Force-Terminate", self.config.habitat.task.measurements.force_terminate.max_accum_force)
        print("Using Dataset: ", self.config.habitat.dataset.data_path)
        print("Observation Keys: ", self.config.habitat.gym.obs_keys)
        # print("Force-Terminate", self.config.habitat.task.measurements.force_terminate.max_accum_force)
        self._id_to_task_name = {}
        self._log_art_list = {}
        self._id_to_rl_config = {}
        for skill_name, skill_config in self.config.env_tasks.skills.items():
            if skill_name in self.config.env_tasks.allowed_skills:
                self._id_to_task_name[skill_config.task_id] = skill_name
                self._log_art_list[skill_config.task_id] = skill_config.get('log_art_list', [])
                self._id_to_rl_config[skill_config.task_id] = skill_config.get('update_rl_params', None)
        
        self._is_art_task = {
            0: "easy", 
            1: "fridge",
            2: "kitchen_counter" 
        }

            
    def _init_train(self, *args, **kwargs):
        super()._init_train(*args, **kwargs)
            
        # self.running_episode_stats = {
            # ""
        # }
        self.running_art_episode_stats = {
            0: {
                "count": torch.zeros(self.envs.num_envs, 1),
                "reward": torch.zeros(self.envs.num_envs, 1)
            },
            1:{
                "count": torch.zeros(self.envs.num_envs, 1),
                "reward": torch.zeros(self.envs.num_envs, 1)
            }, 
            2: {
                "count": torch.zeros(self.envs.num_envs, 1),
                "reward": torch.zeros(self.envs.num_envs, 1)
            }
        }

        # Env Task ID Sensor #
        env_task_id_sensor = torch.argmax(self._agent.rollouts.buffers["observations"][0]['env_task_id_sensor'], dim = -1)
        self._gamma = torch.ones(self.envs.num_envs, 1, device=self.device) * self._ppo_cfg.gamma
        
        for task_id, rl_config in self._id_to_rl_config.items():
            if rl_config is not None:
                self._gamma[env_task_id_sensor == task_id] = rl_config.gamma
        

    @profiling_wrapper.RangeContext("MonoPPOTrainer.update")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        with inference_mode():
            step_batch = self._agent.rollouts.get_last_step()
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }

            next_value = self._agent.actor_critic.get_value(
                step_batch["observations"],
                step_batch.get("recurrent_hidden_states", None),
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )
        if isinstance(self._agent.rollouts, NormalizedReturnRolloutStorage):
            next_value = self._agent.rollouts.unnormalize_returns(next_value)

        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._gamma,
            self._ppo_cfg.tau,
        )
        
        self._agent.train()
        losses = self._agent.updater.update(self._agent.rollouts, id_to_task_name=self._id_to_task_name)

        self._agent.rollouts.after_update()
        self._agent.after_update()

        del_k = []
        losses_per_task = {}
        for k, v in losses.items():
            if len(k.split('.')) > 1:
                [metric_name, task_id] = k.split('.')
                task_name = self._id_to_task_name[int(task_id)]
                losses_per_task[f"{task_name}/{metric_name}"] = v
                del_k.append(k)
        
        losses.update(losses_per_task)
        for k in del_k:
            del losses[k]
    
        return losses    

    # Include articulated statistics in the running episode stats #
    def _include_art_stats(self, extracted_infos, current_ep_reward, env_slice, done_masks):
        prev_step_batch = self._agent.rollouts.get_last_step()
        if "is_art_sensor" not in prev_step_batch["observations"]:
            return
        art_sensor_obs = prev_step_batch["observations"]["is_art_sensor"]
        for art_id in self._is_art_task.keys():
            art_mask  = (art_sensor_obs == art_id).cpu()
            hard_mask = (art_sensor_obs != 0).cpu()
            art_name = self._is_art_task[art_id]
            for k, v_k in extracted_infos.items():
                art_k = art_name + "." + k
                hard_k = "hard." + k
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if art_k not in self.running_episode_stats:
                    self.running_episode_stats[art_k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                if hard_k not in self.running_episode_stats:
                    self.running_episode_stats[hard_k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[hard_k][env_slice] += v.where(done_masks * hard_mask, v.new_zeros(()))
                self.running_episode_stats[art_k][env_slice] += v.where(done_masks * art_mask, v.new_zeros(()))

            # Include the hard and easy stats #
            art_reward_k = art_name + ".reward"
            art_count_k = art_name + ".count"
            if art_reward_k not in self.running_episode_stats:
                self.running_episode_stats[art_reward_k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            if art_count_k not in self.running_episode_stats:
                self.running_episode_stats[art_count_k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            
            hard_reward_k = "hard.reward"
            hard_count_k = "hard.count"
            
            if hard_reward_k not in self.running_episode_stats:
                self.running_episode_stats["hard.reward"] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            if hard_count_k not in self.running_episode_stats:
                self.running_episode_stats["hard.count"] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            self.running_episode_stats[hard_reward_k][env_slice] += current_ep_reward.where(done_masks * hard_mask, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats[hard_count_k][env_slice] += (done_masks * hard_mask).float()  # type: ignore

            self.running_episode_stats[art_reward_k][env_slice] += current_ep_reward.where(done_masks * art_mask, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats[art_count_k][env_slice] += (done_masks * art_mask).float()  # type: ignore
            
    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            

        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=self._rank0_keys
            )
            for k, v_k in extracted_infos.items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

            # Include articulated stats before logging in reward # 
            self._include_art_stats(extracted_infos, current_ep_reward, env_slice, done_masks)

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )


        if self._is_static_encoder:
            with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                batch[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self._encoder(batch)

        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start
    
    
    def compute_model_mem_size(self):
        func = lambda v: v.element_size() * v.nelement()
        param_space = [func(v) for v in self._agent.actor_critic.parameters()]
        return sum(param_space)
    

    def extract_metric_from_name(self, name, metric):
        pattern = f"{metric}=(-?\d+\.\d+)"
        match = re.search(pattern, name)
        if match:
            return match.group(1)
        return None

    def extract_ckpt_from_name(self, name):
        pattern = "ckpt=(\d+)"
        match = re.search(pattern, name)
        if match:
            return match.group(1)
        return None
    
    def run_post_eval(self, writer, video_dir, checkpoint_index, step_id: int = 0):
        metric_list=self.config.env_tasks.eval_group_measures
        group_by=self.config.env_tasks.eval_group_by
        metric_by_group = {"hard": {}}
        
        for video_name in os.listdir(video_dir):
            for metric in metric_list:
                metric_value = self.extract_metric_from_name(video_name, metric)
                group_by_value = self.extract_metric_from_name(video_name, group_by)
                cur_ckpt_idx = self.extract_ckpt_from_name(video_name)

                if cur_ckpt_idx is None:
                    continue
                
                if (
                    group_by_value is None \
                    or metric_value is None or \
                    float(cur_ckpt_idx) != checkpoint_index):
                    continue            

                group_by_value = int(float(group_by_value))
                group_name = self._is_art_task[group_by_value]
                if group_name not in metric_by_group:
                    metric_by_group[group_name] = {}
                
                if group_by_value != 0:
                    metric_by_group["hard"][metric] = metric_by_group["hard"].get(metric, []) + [float(metric_value)]
                
                metric_by_group[group_name][metric] = metric_by_group[group_name].get(metric, []) + [float(metric_value)]
        
        # Group values by metric and group name #
        for group_name, metric_dict in metric_by_group.items():
            max_len = 0 
            for metric_name in metric_dict:
                values = metric_dict[metric_name]
                max_len = max(max_len, len(values))
                writer.add_scalar(f"{group_name}/{metric_name}", np.mean(values), step_id)
            writer.add_scalar(f"{group_name}/count", max_len, step_id)
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer, 
        checkpoint_index: int = 0,
    ) -> None:
        print("Observation Keys: ", self.config.habitat.gym.obs_keys)
        print("Evaluating Checkpoint: ", checkpoint_path)
        video_dir = self.config.habitat_baselines.video_dir
        i = 0
        while os.path.exists(video_dir):
            if video_dir.endswith("/"):
                video_dir = video_dir[:-1]
            video_dir +=  "_" + str(i) + "/"
            i += 1
        

        super()._eval_checkpoint(checkpoint_path, writer, checkpoint_index)
        
        step_id = 0
        if self.config.habitat_baselines.eval.should_load_ckpt:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
        self.run_post_eval(writer, video_dir, checkpoint_index, step_id)
            
            
    def compute_rollout_mem_size(self):
        func = lambda v: v.element_size() * v.nelement()
        obs_dict = self._agent.rollouts.buffers._map_apply_func(func, self._agent.rollouts.buffers['observations'])
        obs_space = sum([*obs_dict.values()])
        rnn_space = self._agent.rollouts.buffers['recurrent_hidden_states'].element_size() * self._agent.rollouts.buffers['recurrent_hidden_states'].nelement()
        total_space = obs_space + rnn_space
        return total_space

    # Training log for losses and other metrics # 
    @rank0_only
    def _training_log(
            self, writer, losses: Dict[str, float], prev_time: int = 0
    ): 
        
        super()._training_log(writer, losses, prev_time)
        env_task_id = torch.tensor(self._single_proc_infos['env_task_measure'])
        is_art_task = torch.tensor(self._single_proc_infos['is_art_measure'])
        '''
        Collecting metrics based on task and whether articulated or not 
        => The metrics are added on the tensorboard
        '''
        
        def log_metrics(self, mask, prefix_k, measures_list, art_prefix = "", exclude_measures = []):
            task_deltas = {
                k: (
                    ((v[-1] - v[0]) * mask).sum().item()
                    if len(v) > 1
                    else (v[0] * mask).sum().item()
                )
                for k, v in self.window_episode_stats.items()
            }
            count_k = art_prefix + "count"
            reward_k = art_prefix + "reward"

            task_deltas[count_k] = max(task_deltas[count_k], 1)
            
            writer.add_scalar(
                f"{prefix_k}/reward",
                task_deltas[reward_k] / task_deltas[count_k],
                self.num_steps_done,
            )
            metrics = {
                f"{prefix_k}/{k}": task_deltas[art_prefix + k] / task_deltas[count_k]
                for k in measures_list
                if (k not in ["reward", "count"] and k not in exclude_measures)
            }

            for k, v in metrics.items():
                writer.add_scalar(k, v, self.num_steps_done)

        # Logging metrics for each task #
        for task_id, task_name in self._id_to_task_name.items():
            env_task_mask = (env_task_id == task_id).view(-1, 1).float()
            log_metrics(
                self, 
                env_task_mask, 
                prefix_k=task_name,
                measures_list=self.config.env_tasks.skills[task_name]["measures_list"], 
                exclude_measures=self.config.env_tasks.skills[task_name].get("exclude_measures", [])
            )
            for art_k in self._log_art_list[task_id]: 
                if art_k not in self._log_art_list[task_id]:
                    continue
                log_metrics(
                    self, 
                    env_task_mask, 
                    prefix_k=f"{art_k}/{task_name}",
                    measures_list=self.config.env_tasks.skills[task_name]["measures_list"], 
                    art_prefix = art_k + ".",
                    exclude_measures=self.config.env_tasks.skills[task_name].get("exclude_measures", [])
                )

           


            
        
            



        