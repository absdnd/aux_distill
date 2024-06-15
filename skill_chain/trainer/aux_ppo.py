from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.common.rollout_storage import RolloutStorage
import torch
import collections
from typing import Dict, List, Any
from habitat.utils import profiling_wrapper
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)
from habitat_baselines.utils.timing import g_timer
import torch.nn.functional as F
from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.updater import Updater
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)
from torch.distributions.kl import kl_divergence
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin 
from habitat_baselines.utils.timing import g_timer
from skill_chain.common.normalized_rollout_storage import NormalizedReturnRolloutStorage
from habitat_baselines.utils.common import CustomNormal
EPS_PPO = 1e-5
@baseline_registry.register_updater
class AuxPredPPOUpdater(PPO):
    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config):
        return cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
            dist_loss_coef = config.get('dist_loss_coef', 0.0),
            dist_from_id = config.get('dist_from_id', -1),
            inc_dist_per_update = config.get('inc_dist_per_update', 0.0),
            dist_skill_id = config.get('dist_skill_id', 5),
            meta_dist_skill_id = config.get('meta_dist_skill_id', -1),
            normalize_dist_by_task = config.get('normalize_dist_by_task', False),
            distill_only_base_actions = config.get('distill_only_base_actions', False),
            noise_to_kl_mask = config.get('noise_to_kl_mask', 0.0),
        )
    
    def __init__(
        self,
        dist_skill_id: int = 5,
        dist_from_id = -1,
        meta_dist_skill_id: int = -1, 
        dist_loss_coef: float = 0.0,
        inc_dist_per_update: float = 0.0,
        num_steps: int = 128,
        normalize_dist_by_task: bool = False,
        distill_only_base_actions: bool = False,
        noise_to_kl_mask:float = 0.0,
        *args, 
        **kwargs
    ) -> None:
        self._dist_from = dist_from_id
        self._normalize_dist_by_task = normalize_dist_by_task
        self._distill_only_base_actions = distill_only_base_actions
        self.dist_skill_id = dist_skill_id
        self.meta_dist_skill_id = meta_dist_skill_id
        self._max_dist_coef = dist_loss_coef
        self._dist_loss_coef = 0.0
        self._updater_counter = 0.0
        self._inc_dist_per_update = inc_dist_per_update
        self._noise_to_kl_mask = noise_to_kl_mask
        if self._inc_dist_per_update == 0.0:
            logger.warn("Distillation loss coefficient will not be increased: Setting maximum value immediately")
            self._dist_loss_coef = self._max_dist_coef
        super().__init__(*args, **kwargs)


        
    def update(
        self, 
        rollouts : RolloutStorage,  
        id_to_task_name: Dict[int, str] = None, 
    ):
        critic = self.actor_critic.critic
        orig_returns = rollouts.buffers["returns"]
        task_ids = torch.argmax(rollouts.buffers["observations"]["env_task_id_sensor"], dim=-1)
        if torch.distributed.is_initialized():
            before_reduce = task_ids.clone()
            task_ids = torch.argmax(rollouts.buffers["observations"]["env_task_id_sensor"], dim=-1)
            if torch.any(task_ids[0] != before_reduce[0]):
                raise AssertionError(
                    f"All task_ids should be the same across processes got {task_ids[0]} and {before_reduce[0]}"
                )
        advantages, adv_mean_var = self.get_advantages(rollouts)
        rollouts.buffers["returns"] = critic.post_process_returns(orig_returns, task_ids)
        rollouts.buffers["value_preds"] = critic.post_process_returns(rollouts.buffers["value_preds"], task_ids) 

        learner_metrics: Dict[str, List[Any]] = collections.defaultdict(list)
        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.data_generator(
                advantages, self.num_mini_batch
            )
            for _bid, batch in enumerate(data_generator):
                self._update_from_batch(
                     batch, epoch, rollouts, learner_metrics, id_to_task_name=id_to_task_name
                )

            profiling_wrapper.range_pop()  # PPO.update epoch

        self._add_adv_to_metrics(learner_metrics, adv_mean_var)
        self._dist_loss_coef = min(self._max_dist_coef, self._dist_loss_coef + self._inc_dist_per_update)
        self._set_grads_to_none()
        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }
    
    def _add_adv_to_metrics(self, learner_metrics, adv_mean_var):
        for task_id, (mean, var) in adv_mean_var.items():
            learner_metrics[f"adv_mean.{task_id}"].append(mean)
            learner_metrics[f"adv_var.{task_id}"].append(var)

    def get_advantages(self, rollouts):
        raw_advantages = rollouts.buffers["returns"] - rollouts.buffers["value_preds"]
        env_task_ids = torch.argmax(rollouts.buffers["observations"]["env_task_id_sensor"], dim=-1)

        if torch.distributed.is_initialized():
            before_reduce = env_task_ids.clone()
            torch.distributed.all_reduce(env_task_ids, op=torch.distributed.ReduceOp.MAX)
            if torch.any(env_task_ids[0] != before_reduce[0]):
                raise AssertionError(
                    f"All env_task_ids should be the same across processes got {env_task_ids[0]} and {before_reduce[0]}"
                )
        
        unique_env_task_ids = torch.unique(env_task_ids)
        normalized_advantages = torch.zeros_like(raw_advantages)

        expanded_task_ids = env_task_ids.unsqueeze(-1)
        adv_mean_var = {}
        for task_id in unique_env_task_ids:
            task_id_mask = expanded_task_ids == task_id
            task_advantages = raw_advantages * task_id_mask.float()
            masked_advantages = task_advantages.masked_select(task_id_mask)
            task_var, task_mean = self._compute_var_mean(masked_advantages[torch.isfinite(masked_advantages)])
            
            if self.use_normalized_advantage:          
                norm_task_advantages = (masked_advantages - task_mean) / (task_var + EPS_PPO)
                normalized_advantages.masked_scatter_(task_id_mask, norm_task_advantages)

        if self.use_normalized_advantage:
            return normalized_advantages, adv_mean_var

        return raw_advantages, adv_mean_var


    
    def _compute_all_task_distribution(self, batch, task_ids):
        # Compute all task ids in the mix # 
        # task_ids = torch.unique(torch.argmax(batch['observations']['env_task_id_sensor'], dim = 1)).tolist()
        action_distrib_dict = {}
        batch_size, num_tasks = batch['observations']['env_task_id_sensor'].shape[0], batch['observations']['env_task_id_sensor'].shape[1]
        with torch.no_grad():
            for task_id in task_ids:
                if task_id == self.dist_skill_id:
                    continue
                copy_batch = batch.copy()
                task_id_sensor = torch.eye(num_tasks)[task_id].to(self.device)
                task_id_sensor = task_id_sensor.unsqueeze(0).repeat(batch_size, 1)
                copy_batch['observations']['env_task_id_sensor'] = task_id_sensor
                _, _, _, _, _, action_distrib = self.actor_critic.evaluate_actions(
                    copy_batch["observations"],
                    copy_batch["recurrent_hidden_states"],
                    copy_batch["prev_actions"],
                    copy_batch["masks"],
                    copy_batch["actions"],
                    copy_batch.get("rnn_build_seq_info", None),
                )
                action_distrib_dict[task_id] = action_distrib

        return action_distrib_dict

    def _check_nav_success(self, gps_compass): 
        pos_dist = gps_compass[:, 0] < 1.5
        angle_dist = abs(gps_compass[:, 1]) < 0.2617
        return pos_dist * angle_dist

    # Articulated Task masks # 
    def _art_task_masks(self, is_art_sensor, task_id):
        if task_id == [6, 8]:
            return is_art_sensor == 2
        elif task_id == [7, 9]:
            return is_art_sensor == 1
        else:
            return is_art_sensor == 0

    def extract_kl_mask(self, task_id, batch):

        if 'hier_skill_sensor' in batch['observations']:
            kl_mask = batch['observations']['hier_skill_sensor'] == task_id
            
            if self._noise_to_kl_mask > 0.0:
                noise_mask = torch.normal(mean = 0.0, std = 1.0, size = kl_mask.shape).to(self.device)
                kl_mask = kl_mask  +  (kl_mask > 0) * noise_mask * self._noise_to_kl_mask
                kl_mask = kl_mask.clamp(0.0, 2.0)

            if task_id == 3 or task_id == 15 or self._distill_only_base_actions:
                action_mask = torch.zeros_like(batch['actions'])
                action_mask[:, -2:] = 1
            else:
                action_mask = torch.ones_like(batch['actions'])
        
        else: 
            raise ValueError('Hierarchical Skill Sensor not found in the batch')
        
        meta_kl_mask = torch.zeros_like(kl_mask)
        if 'meta_hier_skill_sensor' in batch['observations']:
            meta_kl_mask = batch['observations']['meta_hier_skill_sensor'] == task_id
        else: 
            if self.meta_dist_skill_id != -1:
                raise ValueError('Meta Hierarchical Skill Sensor not found in the batch')
        
        
        return kl_mask, meta_kl_mask, action_mask.float()
    
    # Update PPO from batch of data # 
    @g_timer.avg_time("ppo.update_from_batch", level=1)
    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics, id_to_task_name: Dict[int, str]):
        """ 
        Performs a gradient update from the minibatch.
        """

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))
        
        # env_task_ids = torch.Tensor([*id_to_task_name.keys()]).to(self.device)
        env_task_ids = torch.argmax(batch['observations']['env_task_id_sensor'], dim = 1).to(self.device) 
        self._set_grads_to_none()

        # Creating an evaluate actions script for computing the KL loss #
        (
            values,
            action_log_probs,
            dist_entropy,
            _,
            aux_loss_res,
            distribution 
        ) = self._evaluate_actions(
            batch["observations"],
            batch["recurrent_hidden_states"],
            batch["prev_actions"],
            batch["masks"],
            batch["actions"],
            batch.get("rnn_build_seq_info", None),
        )

        action_distrib_dict = self._compute_all_task_distribution(batch, task_ids = [*id_to_task_name.keys()])
        total_kl_loss = 0.0
        
        if self._dist_loss_coef > 0.0:      
            num_tasks = len(action_distrib_dict)
            
            batch_size, action_size = batch['actions'].shape[0], batch['actions'].shape[1]    
            src_distrib = distribution.expand((num_tasks, batch_size, action_size))
            sum_fn = lambda x: x.sum(-1).sum(-1)
            with torch.no_grad():
                targ_distrib_loc = torch.zeros_like(src_distrib.loc)
                targ_distrib_scale = torch.ones_like(src_distrib.scale)
                
                targ_kl_mask = torch.zeros(num_tasks, batch_size, 1, device = self.device)
                targ_action_mask =  torch.zeros(num_tasks, batch_size, action_size, device = self.device)
                task_mask = (env_task_ids == self.dist_skill_id).unsqueeze(-1).expand_as(targ_kl_mask)
                meta_task_mask = (env_task_ids == self.meta_dist_skill_id).unsqueeze(-1).expand_as(targ_kl_mask)
                
                meta_targ_kl_mask = torch.zeros(num_tasks, batch_size, 1, device = self.device)

                mask_id = 0
                for idx, (task_id, task_action_distrib) in enumerate(action_distrib_dict.items()):
                    if self._dist_from != -1 and task_id not in self._dist_from:
                        continue
                    
                    mask_id += 1
                    kl_mask, meta_kl_mask, action_mask = self.extract_kl_mask(task_id, batch)
                    targ_kl_mask[idx] = kl_mask
                    targ_action_mask[idx] = action_mask
                    targ_distrib_loc[idx] = task_action_distrib.loc
                    targ_distrib_scale[idx] = task_action_distrib.scale
                    
                    
                    meta_targ_kl_mask[idx] = meta_kl_mask
                
            
            # env_meta_task_mask=meta_task_mask[0]
            # idxs, _ = torch.where(env_meta_task_mask)
            # hier_values_at_idx = batch['observations']['meta_hier_skill_sensor'][idxs]
        
            
            targ_distrib = CustomNormal(loc = targ_distrib_loc, scale = targ_distrib_scale)
            kl_div_loss = kl_divergence(src_distrib, targ_distrib)
            kl_loss_by_task = sum_fn(kl_div_loss * task_mask * targ_kl_mask * targ_action_mask) 
            kl_mask_by_task = sum_fn(task_mask * targ_kl_mask * targ_action_mask)
            
            # Meta KL Loss by task #
            # idxs, _ = torch.where(meta_targ_kl_mask[3] * (~task_mask[3]))
            meta_kl_loss_by_task = sum_fn(kl_div_loss * meta_task_mask * meta_targ_kl_mask * targ_action_mask)
            meta_kl_mask_by_task = sum_fn(meta_task_mask * meta_targ_kl_mask * targ_action_mask)

            # Normalizing the kl loss on a task specific level #
            if self._normalize_dist_by_task:
                task_kl_loss = kl_loss_by_task / (kl_mask_by_task + EPS_PPO)
                meta_task_kl_loss = meta_kl_loss_by_task / (meta_kl_mask_by_task + EPS_PPO)
            else: 
                task_kl_loss = kl_loss_by_task / (kl_mask_by_task.sum() + EPS_PPO)
                meta_task_kl_loss = meta_kl_loss_by_task / (meta_kl_mask_by_task.sum() + EPS_PPO)

            # Run an evaluation script to check performance #
            for idx, task_id in enumerate(action_distrib_dict.keys()):
                learner_metrics[f"{id_to_task_name[task_id]}/dist_loss"].append(task_kl_loss[idx])
                learner_metrics[f"{id_to_task_name[task_id]}/meta_dist_loss"].append(meta_task_kl_loss[idx])
            
            total_kl_loss = task_kl_loss.sum() + meta_task_kl_loss.sum()

        
        learner_metrics["dist_loss"].append(total_kl_loss)
        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)

        values = values.float()
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

            
        value_loss = 0.5 * F.mse_loss(
            values, batch["returns"], reduction="none"
        )

        if "is_coeffs" in batch:
            assert isinstance(batch["is_coeffs"], torch.Tensor)
            ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)
            mean_fn = lambda t: torch.mean(ver_is_coeffs * t)
        else:
            mean_fn = torch.mean


        for task_id in torch.unique(env_task_ids):
            task_id_mask = env_task_ids == task_id
            learner_metrics[f"action_loss.{task_id}"].append(mean_fn(action_loss[task_id_mask]))
            learner_metrics[f"value_loss.{task_id}"].append(mean_fn(value_loss[task_id_mask]))
            learner_metrics[f"dist_entropy.{task_id}"].append(mean_fn(dist_entropy[task_id_mask]))
            # learner_metrics[f"kl_loss.{task_id}"].append(mean_fn(kl_loss_dict[int(task_id)]))
            # Logging in the mean and variance of the returns #     
            if isinstance(rollouts, NormalizedReturnRolloutStorage):
                rollout_mean, rollout_var = rollouts.get_mean_var(int(task_id))
                learner_metrics[f"rollout_mean.{task_id}"].append(rollout_mean)
                learner_metrics[f"rollout_var.{task_id}"].append(rollout_var)
        
        # Distillation Loss coefficient: The difference between source and target distribution #
        learner_metrics["dist_loss_coef"].append(self._dist_loss_coef)
        action_loss, value_loss, dist_entropy = map(
            mean_fn,
            (action_loss, value_loss, dist_entropy),
        )

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]

        # Entropy coefficient usage for evaluation #
        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        if self._dist_loss_coef > 0.0:
            all_losses.append(self._dist_loss_coef * total_kl_loss)

        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )

            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (
                        rollouts.current_policy_version
                        - batch["policy_version"]
                    ).float(),
                    "policy_version_difference",
                )



@baseline_registry.register_updater
class AuxPredDDPPOUpdater(DecentralizedDistributedMixin, AuxPredPPOUpdater):
    pass
