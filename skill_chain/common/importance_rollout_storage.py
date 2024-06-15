from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.baseline_registry import baseline_registry
from typing import Optional
import torch
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)

@baseline_registry.register_storage
class ImportanceWeightedRolloutStorage(RolloutStorage):
    def __init__(
            self, 
            actor_critic, 
            **kwargs,
    ): 
        super().__init__(actor_critic=actor_critic,**kwargs)
        self.skill_weights = {}
        for k, v in actor_critic.skill_weights.items():
            self.skill_weights[float(k)] = v

        self.buffers["is_coeffs"] = torch.ones_like(
                self.buffers["returns"]
        )
        

    def batch_by_task(self, task_id_obs):
        task_ids = torch.unique(task_id_obs)
        task_batches = {}
        for task_id in task_ids:
            task_id_mask = task_id_obs == task_id.item()
            task_id_inds, = torch.nonzero(task_id_mask, as_tuple = True)
            indexes = torch.randperm(task_id_inds.shape[0])
            task_batches[task_id.item()] = task_id_inds[indexes].cpu()
        return task_batches

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
        
        task_id_obs = self.buffers['observations'][0]['env_task_id_sensor'].view(-1)
        task_batches = self.batch_by_task(task_id_obs)
        
        for task_id, task_batch in task_batches.items():
            self.buffers["is_coeffs"][:, task_batch] = self.skill_weights[task_id]
                    
        dones_cpu = (
            torch.logical_not(self.buffers["masks"].bool())
            .cpu()
            .view(-1, self._num_envs)
            .numpy()
        )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            curr_slice = (slice(0, self.current_rollout_step_idx), inds)

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

    
    

