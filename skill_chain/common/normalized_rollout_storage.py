from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.baseline_registry import baseline_registry
from typing import Optional
import torch
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar

@baseline_registry.register_storage
class NormalizedReturnRolloutStorage(RolloutStorage):
    def __init__(
            self, 
            actor_critic, 
            **kwargs,
    ):
        super().__init__(actor_critic=actor_critic,**kwargs)
        # self.num_tasks = self.buffers['observations']['env_task_id_sensor'].shape[-1]
        self.return_running_mean_and_var = {}
        self._env_to_task_id = None
    
    def insert_first_observations(self, batch):
        super().insert_first_observations(batch)
        unique_task_ids = torch.unique(torch.argmax(batch['env_task_id_sensor'], dim = 1))
        num_tasks = len(unique_task_ids)
        assert (self._num_envs % num_tasks == 0, "The number of environments should be a multiple of num-tasks")
        for task_id in unique_task_ids: 
            self.return_running_mean_and_var[int(task_id)] = RunningMeanAndVar(self._num_envs // num_tasks)
            self.return_running_mean_and_var[int(task_id)] = self.return_running_mean_and_var[int(task_id)].to(self.device)

    def get_mean_var(self, task_int_id):
        return self.return_running_mean_and_var[task_int_id]._mean.item(), self.return_running_mean_and_var[task_int_id]._var.item()

    def unnormalize_returns(self, value_preds):
        if self._env_to_task_id is None:
            self.assign_env_to_task_id()
        
        norm_value_preds = value_preds.clone()
        for task_id, task_envs in self._env_to_task_id.items():
            task_var = self.return_running_mean_and_var[task_id]._var.view(len(task_envs), 1)
            task_mean = self.return_running_mean_and_var[task_id]._mean.view(len(task_envs), 1)
            norm_value_preds[task_envs] = task_var * value_preds[task_envs] + task_mean
        return norm_value_preds
    
    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))
        self.device = device



        
    def assign_env_to_task_id(self):
        env_task_id_observations = torch.argmax(
            self.buffers['observations'][0]['env_task_id_sensor'], axis = 1
        )
        self._env_to_task_id = {}
        for env_id, env_task_id in enumerate(env_task_id_observations):
            int_task_id = int(env_task_id)
            self._env_to_task_id[int_task_id] = self._env_to_task_id.get(int_task_id, []) + [env_id]
        
    def compute_returns(self, *args, **kwargs):
        super().compute_returns(*args, **kwargs)
        if self._env_to_task_id is None:
            self.assign_env_to_task_id()
        
        # Compute running mean and variance of each task's returns # 
        for task_id, task_envs in self._env_to_task_id.items():
            self.buffers["returns"][:, task_envs] = \
                self.return_running_mean_and_var[task_id](
                self.buffers["returns"][:, task_envs].unsqueeze(-1)
            ).squeeze(-1)
    
    def data_generator(
            self, 
            advantages: Optional[torch.Tensor],
            num_mini_batch: int,
    ): 
        assert isinstance(self.buffers["returns"], torch.Tensor)
        num_environments = self.buffers["returns"].size(1)
        assert num_environments >= num_mini_batch, (
            "PPO requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_environments, num_mini_batch)
        )

        if num_environments % num_mini_batch != 0:
            pass
        
        dones_cpu = (
            torch.logical_not(self.buffers["masks"].bool())
            .cpu()
            .view(-1, self._num_envs)
            .numpy()
        )

        # Compute the maximum and minimum return here for normalization
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            curr_slice = (slice(0, self.current_rollout_step_idx), inds)

            # Computing a batch of data and normalizing it. # 
            batch = self.buffers[curr_slice]
            
            if advantages is not None:
                batch["advantages"] = advantages[curr_slice]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            batch.map_in_place(lambda v: v.flatten(0, 1))

            batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                device=self.device,
                build_fn_result=build_pack_info_from_dones(
                    dones_cpu[
                        0 : self.current_rollout_step_idx, inds.numpy()
                    ].reshape(-1, len(inds)),
                ),
            )

            yield batch.to_tree()

    
    

